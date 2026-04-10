from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from wsi_sae.data.dataloader import _resolve_h5_path
from wsi_sae.data.layout import extract_case_id, extract_slide_id, infer_cohort_from_path


REPRESENTATIVE_METHODS = (
    "max_activation",
    "median_activation",
    "diverse_support",
    "slide_spread",
)
DEFAULT_LATENT_STRATEGIES = (
    "top_activation",
    "top_variance",
    "top_sparsity",
    "sdf_parent_balanced",
)
REPRESENTATIVE_BUNDLE_SCHEMA_VERSION = "2.0"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _feature_relpath_from_path(path: Path, *, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except Exception:
        pass
    parts = list(path.parts)
    if "wsi_features" in parts:
        idx = parts.index("wsi_features")
        return str(Path(*parts[idx:]))
    return path.name


def infer_latent_group(latent_idx: int, sdf_hierarchy: dict[str, Any] | None) -> str:
    if isinstance(sdf_hierarchy, dict):
        parent_map = sdf_hierarchy.get("level1_to_level2_parent_selected")
        if isinstance(parent_map, dict):
            parent = parent_map.get(str(int(latent_idx)), parent_map.get(int(latent_idx)))
            if parent is not None:
                return f"parent_{int(parent)}"
    return "selected"


def build_source_support_rows(
    raw_support_rows: list[dict[str, Any]],
    *,
    data_root: Path,
    encoder: str,
    dataset: str,
    run_name: str,
    stage: str,
    data_split: str,
    latent_strategy: str,
    latent_idx: int,
    latent_group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_rank, item in enumerate(raw_support_rows, start=1):
        raw_h5 = str(item.get("h5_path", ""))
        resolved = Path(_resolve_h5_path(raw_h5))
        slide_key = extract_slide_id(str(item.get("slide_key", "")) or resolved.name or raw_h5)
        case_id = extract_case_id(slide_key)
        activation = float(item.get("activation", item.get("score", 0.0)))
        tile_index = int(item.get("tile_index", item.get("tile_idx", -1)))
        coord_x = int(item.get("coord_x", item.get("x", 0)))
        coord_y = int(item.get("coord_y", item.get("y", 0)))
        feature_relpath = _feature_relpath_from_path(resolved, data_root=data_root)
        rows.append(
            {
                "run_name": run_name,
                "stage": stage,
                "dataset": dataset,
                "encoder": encoder,
                "data_split": data_split,
                "latent_strategy": latent_strategy,
                "latent_idx": int(latent_idx),
                "latent_group": latent_group,
                "representative_method": "",
                "row_kind": "support",
                "method_rank": 0,
                "source_rank": int(source_rank),
                "case_id": case_id,
                "slide_key": slide_key,
                "cohort": infer_cohort_from_path(resolved) or "",
                "tile_index": tile_index,
                "coord_x": coord_x,
                "coord_y": coord_y,
                "feature_relpath": feature_relpath,
                "feature_h5_name": resolved.name,
                "legacy_h5_path": raw_h5,
                "activation": activation,
                "method_score": activation,
                "slide_support_count": 0,
                "slide_max_activation": 0.0,
                "slide_mean_activation": 0.0,
            }
        )
    return rows


def attach_slide_support_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slide[str(row["slide_key"])].append(row)
    out: list[dict[str, Any]] = []
    for row in rows:
        slide_rows = by_slide[str(row["slide_key"])]
        activations = [float(item["activation"]) for item in slide_rows]
        enriched = dict(row)
        enriched["slide_support_count"] = int(len(slide_rows))
        enriched["slide_max_activation"] = float(max(activations))
        enriched["slide_mean_activation"] = float(sum(activations) / len(activations))
        out.append(enriched)
    return out


def rank_support_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method not in REPRESENTATIVE_METHODS:
        raise ValueError(f"Unsupported representative method: {method}")
    if not rows:
        return []

    source_rows = sorted(
        [dict(row) for row in rows],
        key=lambda row: (-float(row["activation"]), int(row["source_rank"]), str(row["slide_key"])),
    )

    if method == "max_activation":
        ordered = source_rows
        for row in ordered:
            row["method_score"] = float(row["activation"])

    elif method == "median_activation":
        median = float(np.median(np.asarray([float(row["activation"]) for row in source_rows], dtype=np.float32)))
        ordered = sorted(
            source_rows,
            key=lambda row: (
                abs(float(row["activation"]) - median),
                -float(row["activation"]),
                int(row["source_rank"]),
            ),
        )
        for row in ordered:
            row["method_score"] = -abs(float(row["activation"]) - median)

    elif method == "diverse_support":
        first_by_slide: list[dict[str, Any]] = []
        remainder: list[dict[str, Any]] = []
        seen_slides: set[str] = set()
        for row in source_rows:
            slide_key = str(row["slide_key"])
            if slide_key not in seen_slides:
                seen_slides.add(slide_key)
                first_by_slide.append(row)
            else:
                remainder.append(row)
        ordered = first_by_slide + remainder
        for row in ordered:
            row["method_score"] = float(row["activation"])

    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            grouped[str(row["slide_key"])].append(row)
        slide_order = sorted(
            grouped.items(),
            key=lambda item: (
                -int(item[1][0]["slide_support_count"]),
                -float(item[1][0]["slide_max_activation"]),
                -float(item[1][0]["slide_mean_activation"]),
                str(item[0]),
            ),
        )
        ordered = []
        for _slide_key, slide_rows in slide_order:
            ordered.extend(
                sorted(
                    slide_rows,
                    key=lambda row: (-float(row["activation"]), int(row["source_rank"])),
                )
            )
        for row in ordered:
            row["method_score"] = (
                float(row["slide_support_count"]) * 1_000_000.0
                + float(row["slide_max_activation"]) * 1_000.0
                + float(row["slide_mean_activation"])
            )

    out: list[dict[str, Any]] = []
    for method_rank, row in enumerate(ordered, start=1):
        ranked = dict(row)
        ranked["representative_method"] = method
        ranked["method_rank"] = int(method_rank)
        ranked["row_kind"] = "support"
        out.append(ranked)
    return out


def build_latent_summary_row(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    stage: str,
    dataset: str,
    encoder: str,
    data_split: str,
    latent_strategy: str,
    latent_idx: int,
    latent_group: str,
    global_stats_selected: dict[str, float] | None,
) -> dict[str, Any]:
    activations = np.asarray([float(row["activation"]) for row in rows], dtype=np.float32) if rows else np.asarray([], dtype=np.float32)
    unique_slides = sorted({str(row["slide_key"]) for row in rows})
    unique_cases = sorted({str(row["case_id"]) for row in rows})
    stats = global_stats_selected or {}
    return {
        "run_name": run_name,
        "stage": stage,
        "dataset": dataset,
        "encoder": encoder,
        "data_split": data_split,
        "latent_strategy": latent_strategy,
        "latent_idx": int(latent_idx),
        "latent_group": latent_group,
        "support_tile_count": int(len(rows)),
        "unique_slide_count": int(len(unique_slides)),
        "unique_case_count": int(len(unique_cases)),
        "activation_max": float(np.max(activations)) if activations.size else 0.0,
        "activation_mean": float(np.mean(activations)) if activations.size else 0.0,
        "activation_p50": float(np.percentile(activations, 50)) if activations.size else 0.0,
        "activation_p90": float(np.percentile(activations, 90)) if activations.size else 0.0,
        "max_activation_global": float(stats.get("max_activation", 0.0)),
        "variance_global": float(stats.get("variance", 0.0)),
        "sparsity_score_global": float(stats.get("sparsity_score", 0.0)),
    }


def build_bundle_summary(
    *,
    run_name: str,
    stage: str,
    dataset: str,
    encoder: str,
    selection_split: str,
    export_split: str,
    representative_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    latent_summary_rows: list[dict[str, Any]],
    latent_strategies: list[str],
) -> dict[str, Any]:
    methods = sorted({str(row["representative_method"]) for row in representative_rows if str(row.get("representative_method", ""))})
    unique_slides = sorted({str(row["slide_key"]) for row in support_rows if str(row.get("slide_key", ""))})
    unique_cases = sorted({str(row["case_id"]) for row in support_rows if str(row.get("case_id", ""))})
    unique_features = sorted({str(row["feature_relpath"]) for row in support_rows if str(row.get("feature_relpath", ""))})
    activations = [float(row["activation"]) for row in support_rows]
    return {
        "run_name": run_name,
        "stage": stage,
        "dataset": dataset,
        "encoder": encoder,
        "selection_split": selection_split,
        "export_split": export_split,
        "available_latent_strategies": latent_strategies,
        "available_representative_methods": methods,
        "total_latent_rows": len(latent_summary_rows),
        "total_representative_rows": len(representative_rows),
        "total_support_rows": len(support_rows),
        "total_slides": len(unique_slides),
        "total_cases": len(unique_cases),
        "total_feature_files": len(unique_features),
        "max_activation": max(activations) if activations else 0.0,
        "mean_activation": (sum(activations) / len(activations)) if activations else 0.0,
    }


def build_wsi_bench_model_entry(
    *,
    model_id: str,
    model_name: str,
    encoder: str,
    dataset: str,
    slides_root: str,
) -> dict[str, Any]:
    return {
        "models": [
            {
                "model_id": model_id,
                "model_name": model_name,
                "encoder": encoder,
                "dataset": dataset,
                "slides_root": slides_root or "__SET_LOCAL_SLIDES_ROOT__",
                "representative_latents_csv": "representative_latents.csv",
                "representative_support_tiles_csv": "representative_support_tiles.csv",
                "latent_summary_csv": "latent_summary.csv",
                "bundle_summary_json": "bundle_summary.json",
                "tile_size": 256,
            }
        ]
    }


__all__ = [
    "DEFAULT_LATENT_STRATEGIES",
    "REPRESENTATIVE_BUNDLE_SCHEMA_VERSION",
    "REPRESENTATIVE_METHODS",
    "_write_json",
    "attach_slide_support_stats",
    "build_bundle_summary",
    "build_latent_summary_row",
    "build_source_support_rows",
    "build_wsi_bench_model_entry",
    "infer_latent_group",
    "rank_support_rows",
]
