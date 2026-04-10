from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsi_sae.data.layout import extract_slide_id, resolve_slide_path_from_mapping


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract prototype tiles from local slides using a synced viewer bundle.")
    ap.add_argument("--bundle", type=Path, required=True, help="Bundle directory or bundle_manifest.json path.")
    ap.add_argument("--data-root", type=Path, required=True, help="Canonical local data root that contains registry/ and wsi_slides/.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write extracted tile images and summaries.")
    ap.add_argument("--tile-size", type=int, default=256, help="Tile size in level-0 pixels.")
    ap.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"], help="Output image format.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of rows to extract; 0 means all.")
    ap.add_argument("--skip-contact-sheets", action="store_true", help="Skip latent-level contact sheet images.")
    return ap


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _resolve_bundle_paths(bundle: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = bundle / "bundle_manifest.json" if bundle.is_dir() else bundle
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    csv_name = artifacts.get("prototype_tiles_csv", "prototype_tiles.csv")
    csv_path = manifest_path.parent / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"prototype_tiles.csv not found: {csv_path}")
    return manifest_path, csv_path, manifest


def _load_rows(csv_path: Path, *, limit: int) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _open_backends():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("extract-tiles requires Pillow on the local PC.") from exc
    try:
        import openslide  # type: ignore
    except ImportError:
        openslide = None
    return Image, openslide


def _read_tile(path: Path, *, x: int, y: int, tile_size: int, image_module, openslide_module):
    suffix = path.suffix.lower()
    if openslide_module is not None and suffix in {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}:
        slide = openslide_module.OpenSlide(str(path))
        try:
            return slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
        finally:
            slide.close()
    with image_module.open(path) as img:
        return img.convert("RGB").crop((x, y, x + tile_size, y + tile_size))


def _save_tile_image(tile, out_path: Path, *, image_format: str, image_module) -> None:
    # Rebuild the image onto a fresh RGB canvas so embedded metadata such as
    # problematic ICC profiles from source slides is not carried into saved tiles.
    rgb_tile = tile.convert("RGB")
    clean_tile = image_module.new("RGB", rgb_tile.size)
    clean_tile.paste(rgb_tile)
    save_kwargs: dict[str, Any] = {}
    if str(image_format).lower() == "png":
        save_kwargs["compress_level"] = 1
    clean_tile.save(out_path, **save_kwargs)


def _make_contact_sheet(rows: list[dict[str, str]], *, out_path: Path, tile_size: int, image_module) -> None:
    if not rows:
        return
    cols = min(4, max(1, len(rows)))
    row_count = (len(rows) + cols - 1) // cols
    canvas = image_module.new("RGB", (cols * tile_size, row_count * tile_size), color=(255, 255, 255))
    for idx, row in enumerate(rows):
        img_path = Path(row["tile_path"])
        if not img_path.exists():
            continue
        try:
            with image_module.open(img_path) as img:
                img = img.convert("RGB")
                col = idx % cols
                row_idx = idx // cols
                canvas.paste(img, (col * tile_size, row_idx * tile_size))
        except Exception:
            continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path, csv_path, bundle_manifest = _resolve_bundle_paths(args.bundle)
    rows = _load_rows(csv_path, limit=int(args.limit))
    Image, openslide = _open_backends()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = args.out_dir / "tiles"
    sheet_dir = args.out_dir / "contact_sheets"

    extracted_rows: list[dict[str, str]] = []
    rows_by_latent: dict[str, list[dict[str, str]]] = defaultdict(list)
    success = 0
    missing_slide = 0

    for row in rows:
        slide_key = extract_slide_id(row.get("slide_key", "") or row.get("h5_path", ""))
        mapping_row, slide_path = resolve_slide_path_from_mapping(args.data_root, slide_key=slide_key)
        out_row = dict(row)
        out_row["slide_key"] = slide_key
        out_row["slide_path"] = str(slide_path) if slide_path else ""
        out_row["status"] = "missing_slide"
        out_row["tile_path"] = ""
        out_row["error"] = ""
        if mapping_row is not None:
            out_row["mapping_cohort"] = mapping_row.get("cohort", "")
        else:
            out_row["mapping_cohort"] = ""

        if slide_path is None:
            missing_slide += 1
            out_row["error"] = "Unable to resolve slide via registry/mapping.csv"
            extracted_rows.append(out_row)
            continue

        latent_idx = int(row.get("latent_idx", 0))
        rank = int(row.get("prototype_rank", 0))
        tile_index = int(row.get("tile_index", row.get("tile_idx", -1)))
        coord_x = int(float(row.get("coord_x", row.get("x", 0))))
        coord_y = int(float(row.get("coord_y", row.get("y", 0))))

        out_name = f"latent_{latent_idx:04d}__rank_{rank:03d}__{slide_key}__tile_{tile_index:06d}.{args.image_format}"
        out_path = tile_dir / f"latent_{latent_idx:04d}" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tile = _read_tile(slide_path, x=coord_x, y=coord_y, tile_size=int(args.tile_size), image_module=Image, openslide_module=openslide)
            _save_tile_image(tile, out_path, image_format=str(args.image_format), image_module=Image)
            out_row["status"] = "ok"
            out_row["tile_path"] = str(out_path)
            success += 1
            rows_by_latent[str(latent_idx)].append(out_row)
        except Exception as exc:
            out_row["status"] = "extract_error"
            out_row["error"] = str(exc)
        extracted_rows.append(out_row)

    fieldnames = list(dict.fromkeys([*rows[0].keys()] if rows else []))
    for extra in ["slide_path", "mapping_cohort", "status", "tile_path", "error"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    manifest_rows_path = args.out_dir / "extracted_tiles.csv"
    with manifest_rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in extracted_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not args.skip_contact_sheets:
        for latent_idx, latent_rows in rows_by_latent.items():
            _make_contact_sheet(
                sorted(latent_rows, key=lambda item: int(item.get("prototype_rank", 0))),
                out_path=sheet_dir / f"latent_{int(latent_idx):04d}.{args.image_format}",
                tile_size=int(args.tile_size),
                image_module=Image,
            )

    summary = {
        "bundle_manifest": str(manifest_path),
        "bundle_csv": str(csv_path),
        "source_schema_version": bundle_manifest.get("schema_version", ""),
        "data_root": str(args.data_root),
        "tile_size": int(args.tile_size),
        "rows_total": len(rows),
        "rows_extracted": success,
        "rows_missing_slide": missing_slide,
        "rows_output_csv": str(manifest_rows_path),
        "tiles_dir": str(tile_dir),
        "contact_sheets_dir": str(sheet_dir),
    }
    with (args.out_dir / "extract_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
