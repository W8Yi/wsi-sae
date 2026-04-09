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
                "score": activation,
            }
            rows.append(row)
    rows.sort(key=lambda row: (int(row["latent_idx"]), int(row["prototype_rank"])))
    return rows


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
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    payload = _load_json(args.pass2_json)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected dict payload at {args.pass2_json}")

    pass1_cfg = payload.get("config_pass1", {}) if isinstance(payload.get("config_pass1"), dict) else {}
    pass2_cfg = payload.get("config_pass2", {}) if isinstance(payload.get("config_pass2"), dict) else {}
    selected_latents = [int(x) for x in payload.get("selected_latents", [])]

    encoder = str(args.encoder or pass1_cfg.get("encoder", "unknown"))
    dataset = str(args.dataset or pass1_cfg.get("dataset", ""))
    magnification = str(pass1_cfg.get("magnification", "20x"))
    stage = str(args.stage or pass1_cfg.get("stage", ""))
    experiment_name = str(args.experiment_name or args.out_dir.parent.name or args.out_dir.name)
    model_id = str(args.model_id or experiment_name)
    model_name = str(args.model_name or model_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_prototype_rows(payload, encoder=encoder, dataset=dataset)

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
                "manifest": pass1_cfg.get("index_json", ""),
                "path_field": "h5_path",
            },
        },
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

