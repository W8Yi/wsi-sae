from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SLIDE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}-DX\d+)", re.IGNORECASE)
CASE_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
BUNDLE_SCHEMA_VERSION = "1.0"


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _safe_copy(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst.name


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def parse_slide_key(name_or_path: str) -> str:
    text = str(name_or_path)
    match = SLIDE_ID_RE.search(text)
    if match:
        return match.group(1).upper()
    stem = Path(text).stem
    return stem.split(".")[0].upper()


def parse_case_id(slide_key: str) -> str:
    match = CASE_ID_RE.search(str(slide_key))
    return match.group(1).upper() if match else str(slide_key).upper()


def infer_latent_group(latent_idx: int, payload: dict[str, Any]) -> str:
    sdf_hierarchy = payload.get("sdf_hierarchy")
    if isinstance(sdf_hierarchy, dict):
        parent_map = sdf_hierarchy.get("level1_to_level2_parent_selected")
        if isinstance(parent_map, dict):
            parent = parent_map.get(str(int(latent_idx)), parent_map.get(int(latent_idx)))
            if parent is not None:
                return f"parent_{int(parent)}"
    return "selected"


def build_prototype_rows(
    payload: dict[str, Any],
    *,
    encoder: str,
    dataset: str,
    data_split: str,
    run_name: str,
) -> list[dict[str, Any]]:
    top_tiles = payload.get("top_tiles", {})
    if not isinstance(top_tiles, dict):
        raise ValueError("pass2 payload must contain dict top_tiles")

    rows: list[dict[str, Any]] = []
    for latent_key, items in top_tiles.items():
        latent_idx = int(latent_key)
        latent_group = infer_latent_group(latent_idx, payload)
        if not isinstance(items, list):
            continue
        sorted_items = sorted(
            [item for item in items if isinstance(item, dict)],
            key=lambda item: float(item.get("score", item.get("activation", 0.0))),
            reverse=True,
        )
        for rank, item in enumerate(sorted_items, start=1):
            h5_path = str(item.get("h5_path", ""))
            slide_key = str(item.get("slide_key") or parse_slide_key(h5_path))
            tile_index = int(item.get("tile_index", item.get("tile_idx", -1)))
            coord_x = int(item.get("coord_x", item.get("x", 0)))
            coord_y = int(item.get("coord_y", item.get("y", 0)))
            activation = float(item.get("activation", item.get("score", 0.0)))
            row = {
                "latent_idx": latent_idx,
                "latent_group": latent_group,
                "prototype_rank": rank,
                "activation": activation,
                "attention": float(item.get("attention", 0.0)),
                "label": item.get("label", ""),
                "pred": item.get("pred", ""),
                "prob_pos": item.get("prob_pos", ""),
                "case_id": str(item.get("case_id") or parse_case_id(slide_key)),
                "slide_key": slide_key,
                "tile_index": tile_index,
                "coord_x": coord_x,
                "coord_y": coord_y,
                "h5_path": h5_path,
                "dataset": dataset,
                "encoder": encoder,
                "data_split": data_split,
                "run_name": run_name,
                "score": activation,
            }
            rows.append(row)
    rows.sort(key=lambda row: (int(row["latent_idx"]), int(row["prototype_rank"])))
    return rows


def build_latent_summary_rows(payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top_tiles = payload.get("top_tiles", {})
    stats = payload.get("global_stats_selected", {}) if isinstance(payload.get("global_stats_selected"), dict) else {}
    max_activation = stats.get("max_activation", {}) if isinstance(stats.get("max_activation"), dict) else {}
    variance = stats.get("variance", {}) if isinstance(stats.get("variance"), dict) else {}
    sparsity_score = stats.get("sparsity_score", {}) if isinstance(stats.get("sparsity_score"), dict) else {}

    rows_by_latent: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_latent.setdefault(int(row["latent_idx"]), []).append(row)

    out: list[dict[str, Any]] = []
    for latent_idx, latent_rows in sorted(rows_by_latent.items()):
        latent_rows = sorted(latent_rows, key=lambda item: float(item.get("activation", 0.0)), reverse=True)
        best = latent_rows[0]
        out.append(
            {
                "latent_idx": int(latent_idx),
                "latent_group": str(best.get("latent_group", "selected")),
                "top_tile_count": len(top_tiles.get(str(latent_idx), [])) if isinstance(top_tiles, dict) else len(latent_rows),
                "unique_slide_count": len({str(row.get("slide_key", "")) for row in latent_rows}),
                "best_activation": float(best.get("activation", 0.0)),
                "best_slide_key": str(best.get("slide_key", "")),
                "best_case_id": str(best.get("case_id", "")),
                "best_tile_index": int(best.get("tile_index", -1)),
                "best_coord_x": int(best.get("coord_x", 0)),
                "best_coord_y": int(best.get("coord_y", 0)),
                "max_activation_global": float(max_activation.get(str(latent_idx), max_activation.get(int(latent_idx), 0.0))),
                "variance_global": float(variance.get(str(latent_idx), variance.get(int(latent_idx), 0.0))),
                "sparsity_score_global": float(sparsity_score.get(str(latent_idx), sparsity_score.get(int(latent_idx), 0.0))),
            }
        )
    return out


def build_bundle_summary(
    *,
    rows: list[dict[str, Any]],
    latent_summary_rows: list[dict[str, Any]],
    selected_latents: list[int],
    encoder: str,
    dataset: str,
    data_split: str,
) -> dict[str, Any]:
    slide_keys = {str(row.get("slide_key", "")) for row in rows if str(row.get("slide_key", ""))}
    activations = [float(row.get("activation", 0.0)) for row in rows]
    return {
        "encoder": encoder,
        "dataset": dataset,
        "data_split": data_split,
        "selected_latents_count": len(selected_latents),
        "latent_rows_count": len(latent_summary_rows),
        "prototype_rows_count": len(rows),
        "slides_count": len(slide_keys),
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
    prototype_tiles_csv: str,
) -> dict[str, Any]:
    return {
        "models": [
            {
                "model_id": model_id,
                "model_name": model_name,
                "encoder": encoder,
                "dataset": dataset,
                "slides_root": slides_root or "__SET_LOCAL_SLIDES_ROOT__",
                "prototype_tiles_csv": prototype_tiles_csv,
                "top_attention_tiles_csv": "",
                "tile_size": 256,
            }
        ]
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Export server-side SAE mining artifacts into a machine-agnostic bundle plus "
            "a wsi-bench-compatible prototype_tiles.csv."
        )
    )
    ap.add_argument("--pass2-json", type=Path, required=True, help="Pass2 mining JSON to export.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Destination bundle directory.")
    ap.add_argument("--run-config", type=Path, default=None, help="Optional train run_config.json to copy into the bundle.")
    ap.add_argument("--prototypes-npz", type=Path, default=None, help="Optional prototype NPZ to copy into the bundle.")
    ap.add_argument("--prototypes-json", type=Path, default=None, help="Optional prototype stats JSON to copy into the bundle.")
    ap.add_argument("--latent-targets-json", type=Path, default=None, help="Optional latent target JSON to copy into the bundle.")
    ap.add_argument("--probe-summary-json", type=Path, default=None, help="Optional probe summary JSON to copy into the bundle.")
    ap.add_argument("--model-id", type=str, default="", help="Model identifier for the exported bundle.")
    ap.add_argument("--model-name", type=str, default="", help="Human-readable model name.")
    ap.add_argument("--encoder", type=str, default="", help="Feature encoder name.")
    ap.add_argument("--dataset", type=str, default="", help="Dataset/cohort name.")
    ap.add_argument("--experiment-name", type=str, default="", help="Experiment name override.")
    ap.add_argument("--stage", type=str, default="", help="Stage override.")
    ap.add_argument("--data-split", type=str, default="", help="Optional split label to store in CSV and manifest (for example train/test).")
    ap.add_argument("--wsi-bench-slides-root", type=str, default="", help="Optional slides_root placeholder/value for the generated wsi-bench model snippet.")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    payload = _load_json(args.pass2_json)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected dict payload at {args.pass2_json}")

    pass1_cfg = payload.get("config_pass1", {}) if isinstance(payload.get("config_pass1"), dict) else {}
    pass2_cfg = payload.get("config_pass2", {}) if isinstance(payload.get("config_pass2"), dict) else {}
    run_cfg = _load_json(args.run_config) if args.run_config is not None and args.run_config.exists() else {}
    selected_latents = [int(x) for x in payload.get("selected_latents", [])]

    encoder = str(args.encoder or pass1_cfg.get("encoder", "unknown"))
    dataset = str(args.dataset or pass1_cfg.get("dataset", ""))
    magnification = str(pass1_cfg.get("magnification", "20x"))
    stage = str(args.stage or pass1_cfg.get("stage", ""))
    experiment_name = str(args.experiment_name or args.out_dir.parent.name or args.out_dir.name)
    model_id = str(args.model_id or experiment_name)
    model_name = str(args.model_name or model_id)
    data_split = str(args.data_split or pass2_cfg.get("data_split", ""))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_prototype_rows(
        payload,
        encoder=encoder,
        dataset=dataset,
        data_split=data_split,
        run_name=experiment_name,
    )
    latent_summary_rows = build_latent_summary_rows(payload, rows)
    bundle_summary = build_bundle_summary(
        rows=rows,
        latent_summary_rows=latent_summary_rows,
        selected_latents=selected_latents,
        encoder=encoder,
        dataset=dataset,
        data_split=data_split,
    )

    csv_name = "prototype_tiles.csv"
    csv_path = args.out_dir / csv_name
    fieldnames = [
        "latent_idx",
        "latent_group",
        "prototype_rank",
        "activation",
        "attention",
        "label",
        "pred",
        "prob_pos",
        "case_id",
        "slide_key",
        "tile_index",
        "coord_x",
        "coord_y",
        "h5_path",
        "dataset",
        "encoder",
        "data_split",
        "run_name",
        "score",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    artifact_paths: dict[str, str] = {
        "prototype_tiles_csv": csv_name,
        "pass2_json": _safe_copy(args.pass2_json, args.out_dir),
    }

    latent_summary_name = "latent_summary.csv"
    latent_summary_path = args.out_dir / latent_summary_name
    latent_summary_fields = [
        "latent_idx",
        "latent_group",
        "top_tile_count",
        "unique_slide_count",
        "best_activation",
        "best_slide_key",
        "best_case_id",
        "best_tile_index",
        "best_coord_x",
        "best_coord_y",
        "max_activation_global",
        "variance_global",
        "sparsity_score_global",
    ]
    with latent_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=latent_summary_fields)
        writer.writeheader()
        for row in latent_summary_rows:
            writer.writerow(row)
    artifact_paths["latent_summary_csv"] = latent_summary_name

    bundle_summary_name = "bundle_summary.json"
    _write_json(args.out_dir / bundle_summary_name, bundle_summary)
    artifact_paths["bundle_summary_json"] = bundle_summary_name

    wsi_bench_model_name = "wsi_bench_model.json"
    _write_json(
        args.out_dir / wsi_bench_model_name,
        build_wsi_bench_model_entry(
            model_id=model_id,
            model_name=model_name,
            encoder=encoder,
            dataset=dataset,
            slides_root=args.wsi_bench_slides_root,
            prototype_tiles_csv=csv_name,
        ),
    )
    artifact_paths["wsi_bench_model_json"] = wsi_bench_model_name

    pass1_source = payload.get("pass1_source")
    if pass1_source:
        pass1_path = Path(str(pass1_source))
        if pass1_path.exists():
            artifact_paths["pass1_json"] = _safe_copy(pass1_path, args.out_dir)

    for key, path in {
        "run_config_json": args.run_config,
        "prototypes_npz": args.prototypes_npz,
        "prototypes_json": args.prototypes_json,
        "latent_targets_json": args.latent_targets_json,
        "probe_summary_json": args.probe_summary_json,
    }.items():
        if path is not None and path.exists():
            artifact_paths[key] = _safe_copy(path, args.out_dir)

    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_path": str(repo_root),
            "git_commit": _git_commit(repo_root),
        },
        "experiment": {
            "name": experiment_name,
            "stage": stage,
            "data_split": data_split,
            "model_id": model_id,
            "model_name": model_name,
        },
        "model": {
            "stage": stage,
            "d_in": pass1_cfg.get("d_in"),
            "latent_dim": pass1_cfg.get("latent_dim"),
            "checkpoint": pass1_cfg.get("ckpt", ""),
            "run_name": experiment_name,
        },
        "data": {
            "encoder": encoder,
            "dataset": dataset,
            "magnification": magnification,
            "coordinate_convention": "level0_top_left_px",
            "feature_identity": {
                "manifest": pass1_cfg.get("index_json", "") or run_cfg.get("manifest", ""),
                "split": data_split,
                "path_field": "h5_path",
            },
        },
        "summary": bundle_summary,
        "selection": {
            "strategy": pass2_cfg.get("select_strategy", ""),
            "selected_latents": selected_latents,
            "selection_summary": payload.get("selection_summary"),
        },
        "artifacts": artifact_paths,
    }
    _write_json(args.out_dir / "bundle_manifest.json", manifest)
    print(f"Wrote bundle: {args.out_dir}")
    print(f"Wrote CSV:    {csv_path}")


if __name__ == "__main__":
    main()
