from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from wsi_sae.commands.extract_tiles import _make_contact_sheet, _open_backends, _read_tile
from wsi_sae.data import resolve_feature_path, resolve_slide_path_from_mapping


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Materialize representative-latent bundle rows locally using local feature H5s and local WSIs."
    )
    ap.add_argument("--bundle", type=Path, required=True, help="Bundle directory or bundle_manifest.json path.")
    ap.add_argument("--data-root", type=Path, required=True, help="Canonical local data root that contains registry/, wsi_features/, and wsi_slides/.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for materialized rows, features, and extracted tiles.")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-contact-sheets", action="store_true")
    return ap


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _resolve_bundle_paths(bundle: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    manifest_path = bundle / "bundle_manifest.json" if bundle.is_dir() else bundle
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    rep_csv = manifest_path.parent / str(artifacts.get("representative_latents_csv", "representative_latents.csv"))
    support_csv = manifest_path.parent / str(artifacts.get("representative_support_tiles_csv", "representative_support_tiles.csv"))
    if not rep_csv.exists():
        raise FileNotFoundError(f"representative_latents.csv not found: {rep_csv}")
    if not support_csv.exists():
        raise FileNotFoundError(f"representative_support_tiles.csv not found: {support_csv}")
    return manifest_path, rep_csv, support_csv, manifest


def _load_rows(rep_csv: Path, support_csv: Path, *, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in [rep_csv, support_csv]:
        with path.open("r", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _read_feature_row(feature_path: Path, tile_index: int) -> np.ndarray:
    import h5py  # type: ignore

    with h5py.File(feature_path, "r") as f:
        ds = f["features"]
        if ds.ndim == 2:
            return np.asarray(ds[int(tile_index)], dtype=np.float32)
        if ds.ndim == 3 and int(ds.shape[0]) == 1:
            return np.asarray(ds[0, int(tile_index)], dtype=np.float32)
        raise ValueError(f"Unsupported features shape: {ds.shape} in {feature_path}")


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path, rep_csv, support_csv, bundle_manifest = _resolve_bundle_paths(args.bundle)
    rows = _load_rows(rep_csv, support_csv, limit=int(args.limit))
    Image, openslide = _open_backends()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.out_dir / "tiles"
    sheets_dir = args.out_dir / "contact_sheets"

    feature_vectors: list[np.ndarray] = []
    feature_index_rows: list[dict[str, Any]] = []
    feature_cache: dict[tuple[str, int], int] = {}
    tile_cache: dict[tuple[str, int, int, int], str] = {}
    materialized_rows: list[dict[str, Any]] = []
    rows_by_sheet: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    feature_hits = 0
    tile_hits = 0

    for row in rows:
        feature_relpath = str(row.get("feature_relpath", ""))
        slide_key = str(row.get("slide_key", ""))
        encoder = str(row.get("encoder", ""))
        tile_index = int(float(row.get("tile_index", -1)))
        coord_x = int(float(row.get("coord_x", 0)))
        coord_y = int(float(row.get("coord_y", 0)))

        resolved_feature_row, local_feature_path = resolve_feature_path(
            args.data_root,
            feature_relpath=feature_relpath,
            slide_key=slide_key,
            encoder=encoder,
        )
        mapping_row, local_slide_path = resolve_slide_path_from_mapping(args.data_root, slide_key=slide_key)

        out_row = dict(row)
        out_row["local_feature_path"] = str(local_feature_path) if local_feature_path else ""
        out_row["local_slide_path"] = str(local_slide_path) if local_slide_path else ""
        out_row["status"] = "ok"
        out_row["error"] = ""
        out_row["tile_image_path"] = ""
        out_row["tile_path"] = ""
        out_row["feature_vector_row"] = ""

        errors: list[str] = []
        if local_feature_path is None:
            errors.append("missing_feature")
        if local_slide_path is None:
            errors.append("missing_slide")

        if local_feature_path is not None:
            feature_key = (str(local_feature_path), int(tile_index))
            try:
                if feature_key not in feature_cache:
                    feature_vec = _read_feature_row(local_feature_path, tile_index)
                    feature_cache[feature_key] = len(feature_vectors)
                    feature_vectors.append(feature_vec.astype(np.float32, copy=False))
                    feature_index_rows.append(
                        {
                            "feature_vector_row": feature_cache[feature_key],
                            "local_feature_path": str(local_feature_path),
                            "feature_relpath": feature_relpath,
                            "slide_key": slide_key,
                            "case_id": row.get("case_id", ""),
                            "encoder": encoder,
                            "tile_index": tile_index,
                            "coord_x": coord_x,
                            "coord_y": coord_y,
                        }
                    )
                out_row["feature_vector_row"] = int(feature_cache[feature_key])
                feature_hits += 1
            except Exception as exc:
                errors.append(f"feature_error:{type(exc).__name__}")

        if local_slide_path is not None:
            tile_key = (str(local_slide_path), coord_x, coord_y, int(args.tile_size))
            try:
                if tile_key not in tile_cache:
                    tile = _read_tile(
                        local_slide_path,
                        x=coord_x,
                        y=coord_y,
                        tile_size=int(args.tile_size),
                        image_module=Image,
                        openslide_module=openslide,
                    )
                    tile_name = (
                        f"{slide_key}__tile_{tile_index:06d}__x_{coord_x}__y_{coord_y}.{args.image_format}"
                    )
                    tile_path = tiles_dir / slide_key / tile_name
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    tile.save(tile_path)
                    tile_cache[tile_key] = str(tile_path)
                out_row["tile_image_path"] = tile_cache[tile_key]
                out_row["tile_path"] = tile_cache[tile_key]
                tile_hits += 1
            except Exception as exc:
                errors.append(f"extract_error:{type(exc).__name__}")

        if errors:
            out_row["status"] = "+".join(errors)
            out_row["error"] = "; ".join(errors)

        materialized_rows.append(out_row)
        if out_row["tile_image_path"]:
            rows_by_sheet[
                (
                    str(out_row.get("latent_strategy", "")),
                    int(float(out_row.get("latent_idx", 0))),
                    str(out_row.get("representative_method", "")),
                )
            ].append(out_row)

    materialized_fieldnames = list(dict.fromkeys([*rows[0].keys()] if rows else []))
    for extra in [
        "local_feature_path",
        "local_slide_path",
        "status",
        "error",
        "tile_image_path",
        "tile_path",
        "feature_vector_row",
    ]:
        if extra not in materialized_fieldnames:
            materialized_fieldnames.append(extra)
    materialized_csv = args.out_dir / "materialized_rows.csv"
    _write_csv(materialized_csv, materialized_rows, materialized_fieldnames)

    feature_index_csv = args.out_dir / "encoder_feature_index.csv"
    _write_csv(
        feature_index_csv,
        feature_index_rows,
        [
            "feature_vector_row",
            "local_feature_path",
            "feature_relpath",
            "slide_key",
            "case_id",
            "encoder",
            "tile_index",
            "coord_x",
            "coord_y",
        ],
    )

    if feature_vectors:
        encoder_features = np.stack(feature_vectors, axis=0).astype(np.float32, copy=False)
    else:
        encoder_features = np.empty((0, 0), dtype=np.float32)
    encoder_features_npy = args.out_dir / "encoder_features.npy"
    np.save(encoder_features_npy, encoder_features)

    if not args.skip_contact_sheets:
        for (latent_strategy, latent_idx, representative_method), sheet_rows in rows_by_sheet.items():
            ordered = sorted(
                [row for row in sheet_rows if str(row.get("row_kind", "")) == "support"],
                key=lambda row: int(float(row.get("method_rank", 0))),
            )
            if not ordered:
                ordered = sorted(sheet_rows, key=lambda row: int(float(row.get("method_rank", 0))))
            sheet_path = sheets_dir / f"{latent_strategy}__latent_{latent_idx:04d}__{representative_method}.{args.image_format}"
            _make_contact_sheet(ordered, out_path=sheet_path, tile_size=int(args.tile_size), image_module=Image)

    summary = {
        "bundle_manifest": str(manifest_path),
        "representative_latents_csv": str(rep_csv),
        "representative_support_tiles_csv": str(support_csv),
        "source_schema_version": bundle_manifest.get("schema_version", ""),
        "data_root": str(args.data_root),
        "tile_size": int(args.tile_size),
        "rows_total": len(rows),
        "rows_with_feature_vector": feature_hits,
        "rows_with_tile_image": tile_hits,
        "unique_feature_vectors": len(feature_vectors),
        "unique_tile_images": len(tile_cache),
        "materialized_rows_csv": str(materialized_csv),
        "encoder_features_npy": str(encoder_features_npy),
        "encoder_feature_index_csv": str(feature_index_csv),
        "tiles_dir": str(tiles_dir),
        "contact_sheets_dir": str(sheets_dir),
    }
    with (args.out_dir / "materialize_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
