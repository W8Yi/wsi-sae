from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import h5py


DEFAULT_DATA_ROOT = "/research/projects/mllab/WSI"
DEFAULT_PROJECT = "TCGA"
SUPPORTED_ENCODERS = ("uni2", "seal", "gigapath", "virchow2")
SLIDE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}-DX[0-9A-Z]+)", re.IGNORECASE)
CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)

ENCODER_ALIASES = {
    "uni2": "uni2",
    "uni2h": "uni2",
    "features_uni2": "uni2",
    "seal": "seal",
    "features_seal": "seal",
    "gigapath": "gigapath",
    "features_gigapath": "gigapath",
    "virchow2": "virchow2",
    "features_virchow2": "virchow2",
}
LEGACY_ENCODER_DIRS = {
    "uni2": "features_uni2",
    "seal": "features_seal",
    "gigapath": "features_gigapath",
    "virchow2": "features_virchow2",
}


def canonicalize_encoder_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in ENCODER_ALIASES:
        raise ValueError(f"Unsupported encoder '{name}'. Supported encoders: {', '.join(SUPPORTED_ENCODERS)}")
    return ENCODER_ALIASES[key]


def parse_encoder_list(text: str | Iterable[str] | None) -> list[str]:
    if text is None:
        values: list[str] = list(SUPPORTED_ENCODERS)
    elif isinstance(text, str):
        values = [part.strip() for part in text.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in text if str(part).strip()]
    out: list[str] = []
    for value in values:
        enc = canonicalize_encoder_name(value)
        if enc not in out:
            out.append(enc)
    return out


def normalize_project_name(project: str | None) -> str:
    value = str(project or DEFAULT_PROJECT).strip().upper()
    return value or DEFAULT_PROJECT


def normalize_cohort_name(cohort: str) -> str:
    value = str(cohort or "").strip().upper()
    if value.startswith("TCGA-"):
        value = value[5:]
    return value


def legacy_cohort_dir_name(project: str, cohort: str) -> str:
    return f"{normalize_project_name(project)}-{normalize_cohort_name(cohort)}"


def extract_slide_id(text: str) -> str:
    match = SLIDE_ID_RE.search(str(text))
    if match:
        return match.group(1).upper()
    return Path(str(text)).stem.split(".")[0].upper()


def extract_case_id(slide_id: str) -> str:
    match = CASE_ID_RE.search(str(slide_id))
    return match.group(1).upper() if match else str(slide_id).upper()


def infer_cohort_from_path(path: str | Path) -> str | None:
    for part in Path(path).parts:
        upper = part.upper()
        if upper.startswith("TCGA-") and len(upper) > 5:
            return normalize_cohort_name(upper)
    return None


def data_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("WSI_DATA_ROOT", "").strip()
    return Path(env) if env else Path(DEFAULT_DATA_ROOT)


def canonical_feature_h5_dir(root: str | Path, *, encoder: str, project: str, cohort: str) -> Path:
    return data_root(root) / "wsi_features" / canonicalize_encoder_name(encoder) / normalize_project_name(project) / normalize_cohort_name(cohort) / "h5"


def canonical_feature_index_path(root: str | Path, *, encoder: str, project: str, cohort: str) -> Path:
    return data_root(root) / "wsi_features" / canonicalize_encoder_name(encoder) / normalize_project_name(project) / normalize_cohort_name(cohort) / "index.csv"


def canonical_slide_dir(root: str | Path, *, project: str, cohort: str) -> Path:
    return data_root(root) / "wsi_slides" / normalize_project_name(project) / normalize_cohort_name(cohort) / "slides"


def canonical_slide_list_path(root: str | Path, *, project: str, cohort: str) -> Path:
    return data_root(root) / "wsi_slides" / normalize_project_name(project) / normalize_cohort_name(cohort) / "metadata" / "slide_list.csv"


def registry_dir(root: str | Path) -> Path:
    return data_root(root) / "registry"


def metadata_dir(root: str | Path) -> Path:
    return data_root(root) / "metadata"


def init_layout(root: str | Path, *, project: str = DEFAULT_PROJECT, encoders: Iterable[str] = SUPPORTED_ENCODERS) -> dict[str, Any]:
    root_path = data_root(root)
    project_name = normalize_project_name(project)
    created: list[str] = []
    dirs = [
        root_path / "wsi_slides" / project_name,
        root_path / "registry",
        root_path / "metadata" / "indexes",
        root_path / "metadata" / "manifests",
        root_path / "metadata" / "splits",
    ]
    for encoder in parse_encoder_list(encoders):
        dirs.append(root_path / "wsi_features" / encoder / project_name)
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    return {
        "root": str(root_path),
        "project": project_name,
        "encoders": parse_encoder_list(encoders),
        "created_dirs": created,
    }


def read_h5_stats(path: str | Path) -> tuple[int, int]:
    with h5py.File(path, "r") as f:
        ds = f["features"]
        if ds.ndim == 3 and ds.shape[0] == 1:
            return int(ds.shape[1]), int(ds.shape[2])
        if ds.ndim == 2:
            return int(ds.shape[0]), int(ds.shape[1])
        raise ValueError(f"Unsupported features shape {ds.shape} in {path}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return len(materialized)


def _relative_symlink_target(dst: Path, src: Path) -> str:
    return os.path.relpath(src, start=dst.parent)


def ensure_link(dst: Path, src: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.is_symlink():
        existing = dst.resolve()
        if existing == src:
            return "kept"
        dst.unlink()
    elif dst.exists():
        if dst.resolve() == src:
            return "kept"
        raise FileExistsError(f"Refusing to overwrite non-link existing file: {dst}")
    dst.symlink_to(_relative_symlink_target(dst, src))
    return "linked"


def _iter_legacy_feature_paths(
    legacy_root: str | Path,
    *,
    project: str,
    encoders: Iterable[str],
) -> list[dict[str, str]]:
    legacy_root = Path(legacy_root)
    project_name = normalize_project_name(project)
    out: list[dict[str, str]] = []
    for cohort_dir in sorted(p for p in legacy_root.glob(f"{project_name}-*") if p.is_dir()):
        cohort = normalize_cohort_name(cohort_dir.name)
        for encoder in parse_encoder_list(encoders):
            src_dir = cohort_dir / LEGACY_ENCODER_DIRS[encoder]
            if not src_dir.is_dir():
                continue
            for h5_path in sorted(src_dir.glob("*.h5")):
                out.append(
                    {
                        "project": project_name,
                        "cohort": cohort,
                        "encoder": encoder,
                        "source_path": str(h5_path.resolve()),
                        "slide_id": extract_slide_id(h5_path.name),
                    }
                )
    return out


def _merge_slide_rows(existing_rows: list[dict[str, str]], new_slide_ids: Iterable[str], *, cohort: str) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in existing_rows:
        slide_id = extract_slide_id(row.get("slide_id", "") or row.get("slide_filename", ""))
        if not slide_id:
            continue
        row_copy = dict(row)
        row_copy["slide_id"] = slide_id
        row_copy.setdefault("slide_filename", "")
        row_copy.setdefault("case_id", extract_case_id(slide_id))
        row_copy.setdefault("dataset", "TCGA")
        row_copy.setdefault("cohort", cohort)
        row_copy.setdefault("wsi_path", "")
        merged[slide_id] = row_copy
    for slide_id in sorted({extract_slide_id(x) for x in new_slide_ids}):
        merged.setdefault(
            slide_id,
            {
                "slide_id": slide_id,
                "slide_filename": "",
                "case_id": extract_case_id(slide_id),
                "dataset": "TCGA",
                "cohort": cohort,
                "wsi_path": "",
            },
        )
    return [merged[key] for key in sorted(merged.keys())]


def build_feature_index_rows(h5_dir: str | Path) -> list[dict[str, Any]]:
    h5_dir = Path(h5_dir)
    rows: list[dict[str, Any]] = []
    for h5_path in sorted(h5_dir.glob("*.h5")):
        num_tiles, feature_dim = read_h5_stats(h5_path)
        rows.append(
            {
                "slide_id": extract_slide_id(h5_path.name),
                "path": str(h5_path.resolve()),
                "num_tiles": num_tiles,
                "feature_dim": feature_dim,
            }
        )
    return rows


def ensure_feature_indexes(root: str | Path, *, project: str, encoders: Iterable[str]) -> dict[str, int]:
    root_path = data_root(root)
    counts: dict[str, int] = {}
    for encoder in parse_encoder_list(encoders):
        project_dir = root_path / "wsi_features" / encoder / normalize_project_name(project)
        if not project_dir.exists():
            continue
        for cohort_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            h5_dir = cohort_dir / "h5"
            if not h5_dir.is_dir():
                continue
            rows = build_feature_index_rows(h5_dir)
            count = write_csv(
                cohort_dir / "index.csv",
                ["slide_id", "path", "num_tiles", "feature_dim"],
                rows,
            )
            counts[f"{encoder}:{cohort_dir.name}"] = count
    return counts


def ingest_tcga_features(
    root: str | Path,
    *,
    legacy_root: str | Path,
    project: str = DEFAULT_PROJECT,
    encoders: Iterable[str] = SUPPORTED_ENCODERS,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    if link_mode != "symlink":
        raise ValueError("V1 ingestion only supports --link-mode symlink.")
    summary = init_layout(root, project=project, encoders=encoders)
    entries = _iter_legacy_feature_paths(legacy_root, project=project, encoders=encoders)
    by_cohort: dict[str, set[str]] = {}
    linked = 0
    kept = 0
    for entry in entries:
        cohort = str(entry["cohort"])
        encoder = str(entry["encoder"])
        source_path = Path(str(entry["source_path"]))
        target_dir = canonical_feature_h5_dir(root, encoder=encoder, project=project, cohort=cohort)
        target_dir.mkdir(parents=True, exist_ok=True)
        action = ensure_link(target_dir / source_path.name, source_path)
        if action == "linked":
            linked += 1
        else:
            kept += 1
        by_cohort.setdefault(cohort, set()).add(str(entry["slide_id"]))

    index_counts = ensure_feature_indexes(root, project=project, encoders=encoders)
    slide_list_counts: dict[str, int] = {}
    slide_fields = ["slide_id", "slide_filename", "case_id", "dataset", "cohort", "wsi_path"]
    for cohort, slide_ids in sorted(by_cohort.items()):
        slide_list_path = canonical_slide_list_path(root, project=project, cohort=cohort)
        merged_rows = _merge_slide_rows(_read_csv_rows(slide_list_path), slide_ids, cohort=cohort)
        slide_list_counts[cohort] = write_csv(slide_list_path, slide_fields, merged_rows)

    return {
        **summary,
        "legacy_root": str(Path(legacy_root)),
        "link_mode": link_mode,
        "feature_links_created": linked,
        "feature_links_kept": kept,
        "feature_entries_seen": len(entries),
        "index_counts": index_counts,
        "slide_list_counts": slide_list_counts,
    }


def _load_slide_rows(root: str | Path, *, project: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    slide_project_dir = data_root(root) / "wsi_slides" / normalize_project_name(project)
    if not slide_project_dir.exists():
        return rows
    for cohort_dir in sorted(p for p in slide_project_dir.iterdir() if p.is_dir()):
        slide_list = cohort_dir / "metadata" / "slide_list.csv"
        for row in _read_csv_rows(slide_list):
            slide_id = extract_slide_id(row.get("slide_id", "") or row.get("slide_filename", ""))
            rows.append(
                {
                    "slide_id": slide_id,
                    "slide_filename": row.get("slide_filename", ""),
                    "case_id": row.get("case_id", extract_case_id(slide_id)),
                    "dataset": row.get("dataset", normalize_project_name(project)),
                    "cohort": row.get("cohort", cohort_dir.name),
                    "wsi_path": row.get("wsi_path", ""),
                    "wsi_ext": Path(row.get("wsi_path", "")).suffix.lower() if row.get("wsi_path", "") else "",
                }
            )
    return rows


def _load_feature_rows(root: str | Path, *, project: str, encoders: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ensure_feature_indexes(root, project=project, encoders=encoders)
    for encoder in parse_encoder_list(encoders):
        project_dir = data_root(root) / "wsi_features" / encoder / normalize_project_name(project)
        if not project_dir.exists():
            continue
        for cohort_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            index_path = cohort_dir / "index.csv"
            for row in _read_csv_rows(index_path):
                slide_id = extract_slide_id(row.get("slide_id", "") or row.get("path", ""))
                feature_path = row.get("path", "")
                rows.append(
                    {
                        "slide_id": slide_id,
                        "case_id": extract_case_id(slide_id),
                        "dataset": normalize_project_name(project),
                        "cohort": cohort_dir.name,
                        "encoder": encoder,
                        "feature_path": feature_path,
                        "feature_ext": Path(feature_path).suffix.lower(),
                    }
                )
    return rows


def build_registry(root: str | Path, *, project: str = DEFAULT_PROJECT, encoders: Iterable[str] = SUPPORTED_ENCODERS) -> dict[str, Any]:
    root_path = data_root(root)
    slide_rows = _load_slide_rows(root_path, project=project)
    feature_rows = _load_feature_rows(root_path, project=project, encoders=encoders)

    slide_fields = ["slide_id", "slide_filename", "case_id", "dataset", "cohort", "wsi_path", "wsi_ext"]
    feature_fields = ["slide_id", "case_id", "dataset", "cohort", "encoder", "feature_path", "feature_ext"]
    write_csv(registry_dir(root_path) / "slides.csv", slide_fields, slide_rows)
    write_csv(registry_dir(root_path) / "features.csv", feature_fields, feature_rows)

    slides_by_id: dict[str, list[dict[str, str]]] = {}
    for row in slide_rows:
        slides_by_id.setdefault(row["slide_id"], []).append(row)
    features_by_id: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        features_by_id.setdefault(row["slide_id"], {})[row["encoder"]] = row["feature_path"]

    mapping_rows: list[dict[str, str]] = []
    missing_slides: list[dict[str, str]] = []
    missing_features: list[dict[str, str]] = []
    ambiguous: list[dict[str, str]] = []

    all_ids = sorted(set(slides_by_id.keys()) | set(features_by_id.keys()))
    for slide_id in all_ids:
        slide_matches = slides_by_id.get(slide_id, [])
        feature_map = features_by_id.get(slide_id, {})
        if len(slide_matches) > 1:
            ambiguous.append(
                {
                    "slide_id": slide_id,
                    "candidate_count": str(len(slide_matches)),
                    "candidate_paths": "|".join(row.get("wsi_path", "") for row in slide_matches if row.get("wsi_path", "")),
                }
            )
            continue

        slide_row = slide_matches[0] if slide_matches else {}
        mapping_row = {
            "slide_id": slide_id,
            "case_id": slide_row.get("case_id", extract_case_id(slide_id)),
            "dataset": slide_row.get("dataset", normalize_project_name(project)),
            "cohort": slide_row.get("cohort", ""),
            "slide_filename": slide_row.get("slide_filename", ""),
            "wsi_path": slide_row.get("wsi_path", ""),
            "uni_path": feature_map.get("uni2", ""),
            "seal_path": feature_map.get("seal", ""),
        }
        mapping_rows.append(mapping_row)
        if not slide_matches and feature_map:
            missing_slides.append(mapping_row)
        if slide_matches and not feature_map:
            missing_features.append(mapping_row)

    write_csv(
        registry_dir(root_path) / "mapping.csv",
        ["slide_id", "case_id", "dataset", "cohort", "slide_filename", "wsi_path", "uni_path", "seal_path"],
        mapping_rows,
    )
    write_csv(
        registry_dir(root_path) / "missing_slides.csv",
        ["slide_id", "case_id", "dataset", "cohort", "slide_filename", "wsi_path", "uni_path", "seal_path"],
        missing_slides,
    )
    write_csv(
        registry_dir(root_path) / "missing_features.csv",
        ["slide_id", "case_id", "dataset", "cohort", "slide_filename", "wsi_path", "uni_path", "seal_path"],
        missing_features,
    )
    write_csv(
        registry_dir(root_path) / "ambiguous_slides.csv",
        ["slide_id", "candidate_count", "candidate_paths"],
        ambiguous,
    )

    return {
        "root": str(root_path),
        "project": normalize_project_name(project),
        "slides": len(slide_rows),
        "features": len(feature_rows),
        "mapping_rows": len(mapping_rows),
        "missing_slides": len(missing_slides),
        "missing_features": len(missing_features),
        "ambiguous_slides": len(ambiguous),
    }


def validate_layout(root: str | Path, *, project: str = DEFAULT_PROJECT, encoders: Iterable[str] = SUPPORTED_ENCODERS) -> dict[str, Any]:
    root_path = data_root(root)
    errors: list[str] = []
    warnings: list[str] = []

    for rel in ["wsi_slides", "wsi_features", "registry", "metadata/indexes", "metadata/manifests", "metadata/splits"]:
        if not (root_path / rel).exists():
            errors.append(f"Missing required directory: {root_path / rel}")

    validated_features = 0
    for encoder in parse_encoder_list(encoders):
        project_dir = root_path / "wsi_features" / encoder / normalize_project_name(project)
        if not project_dir.exists():
            warnings.append(f"Missing encoder project directory: {project_dir}")
            continue
        for cohort_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            h5_dir = cohort_dir / "h5"
            index_path = cohort_dir / "index.csv"
            if not h5_dir.is_dir():
                errors.append(f"Missing H5 directory: {h5_dir}")
                continue
            if not index_path.exists():
                warnings.append(f"Missing index.csv, regenerating is recommended: {index_path}")
            for h5_path in sorted(h5_dir.glob("*.h5")):
                try:
                    read_h5_stats(h5_path)
                    validated_features += 1
                except Exception as exc:
                    errors.append(f"Unreadable feature file {h5_path}: {exc}")
            slide_list = canonical_slide_list_path(root_path, project=project, cohort=cohort_dir.name)
            if not slide_list.exists():
                warnings.append(f"Missing slide list for cohort {cohort_dir.name}: {slide_list}")

    summary = {
        "root": str(root_path),
        "project": normalize_project_name(project),
        "encoders": parse_encoder_list(encoders),
        "validated_feature_files": validated_features,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise RuntimeError("\n".join(errors))
    return summary


def promote_links(root: str | Path, *, project: str = DEFAULT_PROJECT, encoders: Iterable[str] = SUPPORTED_ENCODERS) -> dict[str, Any]:
    root_path = data_root(root)
    moved = 0
    skipped = 0
    for encoder in parse_encoder_list(encoders):
        project_dir = root_path / "wsi_features" / encoder / normalize_project_name(project)
        if not project_dir.exists():
            continue
        for cohort_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            h5_dir = cohort_dir / "h5"
            if not h5_dir.is_dir():
                continue
            for entry in sorted(h5_dir.iterdir()):
                if not entry.is_symlink():
                    skipped += 1
                    continue
                target = entry.resolve(strict=True)
                entry.unlink()
                shutil.move(str(target), str(entry))
                moved += 1
    index_counts = ensure_feature_indexes(root_path, project=project, encoders=encoders)
    return {
        "root": str(root_path),
        "project": normalize_project_name(project),
        "moved_files": moved,
        "skipped_files": skipped,
        "index_counts": index_counts,
    }


def _scan_entries_from_legacy_root(
    legacy_root: str | Path,
    *,
    project: str,
    encoders: Iterable[str],
) -> list[dict[str, str]]:
    return _iter_legacy_feature_paths(legacy_root, project=project, encoders=encoders)


def _scan_entries_from_canonical_root(
    root: str | Path,
    *,
    project: str,
    encoders: Iterable[str],
) -> list[dict[str, str]]:
    root_path = data_root(root)
    entries: list[dict[str, str]] = []
    for encoder in parse_encoder_list(encoders):
        project_dir = root_path / "wsi_features" / encoder / normalize_project_name(project)
        if not project_dir.exists():
            continue
        for cohort_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            h5_dir = cohort_dir / "h5"
            if not h5_dir.is_dir():
                continue
            for h5_path in sorted(h5_dir.glob("*.h5")):
                entries.append(
                    {
                        "project": normalize_project_name(project),
                        "cohort": cohort_dir.name,
                        "encoder": encoder,
                        "source_path": str(h5_path),
                        "slide_id": extract_slide_id(h5_path.name),
                    }
                )
    return entries


def scan_h5_health(
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
    project: str = DEFAULT_PROJECT,
    encoders: Iterable[str] = SUPPORTED_ENCODERS,
    source: str = "canonical",
    legacy_root: str | Path | None = None,
    out_dir: str | Path | None = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    source = str(source).strip().lower()
    if source not in {"canonical", "legacy"}:
        raise ValueError("--source must be one of: canonical, legacy")

    normalized_encoders = parse_encoder_list(encoders)
    if source == "legacy":
        legacy_base = Path(legacy_root) if legacy_root is not None else data_root(root) / "TCGA_features"
        entries = _scan_entries_from_legacy_root(legacy_base, project=project, encoders=normalized_encoders)
        scan_root = str(legacy_base)
    else:
        entries = _scan_entries_from_canonical_root(root, project=project, encoders=normalized_encoders)
        scan_root = str(data_root(root))

    rows: list[dict[str, str]] = []
    by_encoder: dict[str, dict[str, int]] = {}
    bad_count = 0
    for entry in entries:
        encoder = str(entry["encoder"])
        cohort = str(entry["cohort"])
        path = Path(str(entry["source_path"]))
        status = "ok"
        error = ""
        num_tiles = ""
        feature_dim = ""
        try:
            n, d = read_h5_stats(path)
            num_tiles = str(n)
            feature_dim = str(d)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            bad_count += 1
            if stop_on_error:
                raise
        row = {
            "encoder": encoder,
            "cohort": cohort,
            "slide_id": str(entry["slide_id"]),
            "path": str(path),
            "status": status,
            "error": error,
            "num_tiles": num_tiles,
            "feature_dim": feature_dim,
        }
        rows.append(row)
        stats = by_encoder.setdefault(encoder, {"ok": 0, "error": 0})
        stats[status] = int(stats.get(status, 0)) + 1

    summary = {
        "project": normalize_project_name(project),
        "source": source,
        "scan_root": scan_root,
        "encoders": normalized_encoders,
        "files_scanned": len(rows),
        "files_ok": len(rows) - bad_count,
        "files_error": bad_count,
        "by_encoder": by_encoder,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "h5_health_report.csv",
            ["encoder", "cohort", "slide_id", "path", "status", "error", "num_tiles", "feature_dim"],
            rows,
        )
        (out_dir / "h5_health_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        summary["report_csv"] = str(out_dir / "h5_health_report.csv")
        summary["summary_json"] = str(out_dir / "h5_health_summary.json")

    return summary


def resolve_slide_path_from_mapping(data_root_path: str | Path, *, slide_key: str) -> tuple[dict[str, str] | None, Path | None]:
    mapping_path = registry_dir(data_root_path) / "mapping.csv"
    if not mapping_path.exists():
        return None, None
    slide_key = extract_slide_id(slide_key)
    for row in _read_csv_rows(mapping_path):
        if extract_slide_id(row.get("slide_id", "")) != slide_key:
            continue
        wsi_path = row.get("wsi_path", "")
        if wsi_path:
            path = Path(wsi_path)
            if path.exists():
                return row, path
        cohort = normalize_cohort_name(row.get("cohort", ""))
        filename = row.get("slide_filename", "")
        if cohort and filename:
            candidate = canonical_slide_dir(data_root_path, project=row.get("dataset", DEFAULT_PROJECT), cohort=cohort) / filename
            if candidate.exists():
                return row, candidate
        if cohort:
            slide_dir = canonical_slide_dir(data_root_path, project=row.get("dataset", DEFAULT_PROJECT), cohort=cohort)
            if slide_dir.is_dir():
                matches = sorted(slide_dir.glob(f"{slide_key}.*"))
                if matches:
                    return row, matches[0]
        return row, None
    return None, None


def resolve_feature_path(
    data_root_path: str | Path,
    *,
    feature_relpath: str = "",
    slide_key: str = "",
    encoder: str = "",
) -> tuple[dict[str, str] | None, Path | None]:
    root_path = data_root(data_root_path)
    rel = str(feature_relpath).strip()
    if rel:
        candidate = (root_path / rel).resolve()
        if candidate.exists():
            return None, candidate

    slide_key = extract_slide_id(slide_key)
    encoder_key = str(encoder).strip().lower()
    encoder_col = {
        "uni2": "uni_path",
        "seal": "seal_path",
        "gigapath": "gigapath_path",
        "virchow2": "virchow2_path",
    }.get(encoder_key, f"{encoder_key}_path" if encoder_key else "")

    if slide_key and encoder_col:
        mapping_path = registry_dir(root_path) / "mapping.csv"
        if mapping_path.exists():
            for row in _read_csv_rows(mapping_path):
                if extract_slide_id(row.get("slide_id", "")) != slide_key:
                    continue
                mapped = str(row.get(encoder_col, "")).strip()
                if mapped:
                    candidate = Path(mapped)
                    if candidate.exists():
                        return row, candidate
                break

    if slide_key and encoder_key:
        features_path = registry_dir(root_path) / "features.csv"
        if features_path.exists():
            for row in _read_csv_rows(features_path):
                if extract_slide_id(row.get("slide_id", "")) != slide_key:
                    continue
                if str(row.get("encoder", "")).strip().lower() != encoder_key:
                    continue
                mapped = str(row.get("feature_path", "")).strip()
                if mapped:
                    candidate = Path(mapped)
                    if candidate.exists():
                        return row, candidate
                break

    return None, None
