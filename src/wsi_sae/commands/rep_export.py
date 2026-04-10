from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wsi_sae.commands.mine import (
    apply_per_slide_cap_to_top_tiles,
    build_sdf_hierarchy_payload,
    load_sae_from_config,
    pass1_collect_stats,
    pass2_top_tiles,
    select_latents,
    unwrap_base_model,
)
from wsi_sae.commands.mine_bundles import (
    _find_stage_dir,
    _h5_list_from_manifest_split,
    _infer_dataset,
    _infer_encoder,
    _select_ckpt,
)
from wsi_sae.commands.export_viewer import _git_commit
from wsi_sae.data.layout import data_root as canonical_data_root
from wsi_sae.models.sae import SDFSAE2Level
from wsi_sae.representatives import (
    DEFAULT_LATENT_STRATEGIES,
    REPRESENTATIVE_BUNDLE_SCHEMA_VERSION,
    REPRESENTATIVE_METHODS,
    _write_json,
    attach_slide_support_stats,
    build_bundle_summary,
    build_latent_summary_row,
    build_source_support_rows,
    build_wsi_bench_model_entry,
    infer_latent_group,
    rank_support_rows,
)


def _log(message: str) -> None:
    print(f"[rep-export] {message}", flush=True)


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _strategy_list(stage: str, requested: str) -> list[str]:
    strategies = _split_csv(requested) if requested else list(DEFAULT_LATENT_STRATEGIES)
    out: list[str] = []
    for strategy in strategies:
        if strategy == "sdf_parent_balanced" and stage != "sdf2":
            continue
        if strategy not in out:
            out.append(strategy)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(
        description=(
            "Representative-latent-first export: choose latents on train, gather top/support tiles on test, "
            "and write a compact bundle for local materialization and wsi-bench."
        )
    )
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--run-root", type=Path, default=repo_root / "runs")
    ap.add_argument("--export-root", type=Path, default=repo_root / "exports")
    ap.add_argument("--stage-dir", type=str, default="")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--selection-split", type=str, default="train")
    ap.add_argument("--export-split", type=str, default="test")
    ap.add_argument("--latent-strategies", type=str, default="")
    ap.add_argument("--n-latents", type=int, default=128)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--slides-per-project", type=int, default=200)
    ap.add_argument("--require-h5-exists", action="store_true")
    ap.add_argument("--tiles-per-slide", type=int, default=0, help="0 uses the training run value.")
    ap.add_argument("--chunk-tiles", type=int, default=0, help="0 uses the training run value.")
    ap.add_argument("--topn-buffer-factor", type=float, default=4.0)
    ap.add_argument("--max-tiles-per-slide-per-latent", type=int, default=3)
    ap.add_argument("--min-distance-px-same-slide-per-latent", type=int, default=512)
    ap.add_argument("--model-name", type=str, default="")
    ap.add_argument("--wsi-bench-slides-root", type=str, default="")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    server_data_root = canonical_data_root(None)

    _log(f"starting representative export for run='{args.run_name}'")
    stage_dir, run_cfg = _find_stage_dir(args.run_root, args.run_name, stage_dir_name=args.stage_dir)
    stage = str(run_cfg.get("stage", stage_dir.name)).strip()
    manifest_path = Path(str(run_cfg.get("manifest", ""))).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found in run_config: {manifest_path}")

    encoder = _infer_encoder(args.run_name, run_cfg)
    dataset = _infer_dataset(args.run_name, run_cfg)
    ckpt_path = _select_ckpt(stage_dir, run_cfg, override=str(args.ckpt) if args.ckpt else "")
    d_in = int(run_cfg["d_in"])
    latent_dim = int(run_cfg["latent_dim"])
    magnification = str(run_cfg.get("magnification", "20x"))
    tiles_per_slide = int(args.tiles_per_slide or run_cfg.get("tiles_per_slide", 2048))
    chunk_tiles = int(args.chunk_tiles or run_cfg.get("chunk_tiles", 512))
    pool_allow_partial = bool(run_cfg.get("pool_allow_partial", False))
    bundle_dir = args.export_root / args.run_name / f"representatives_{args.export_split}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    strategies = _strategy_list(stage, args.latent_strategies)
    if not strategies:
        raise SystemExit("No latent-selection strategies enabled for this run.")
    _log(
        "resolved run config: "
        f"stage={stage} encoder={encoder} dataset={dataset} "
        f"d_in={d_in} latent_dim={latent_dim} magnification={magnification}"
    )
    _log(f"checkpoint={ckpt_path}")
    _log(f"manifest={manifest_path}")
    _log(
        f"split policy: selection_split={args.selection_split} export_split={args.export_split} "
        f"strategies={','.join(strategies)}"
    )

    selection_h5_files = [Path(p) for p in _h5_list_from_manifest_split(
        manifest_path,
        split=args.selection_split,
        slides_per_project=int(args.slides_per_project),
        seed=int(args.seed),
        require_h5_exists=bool(args.require_h5_exists),
    )]
    export_h5_files = [Path(p) for p in _h5_list_from_manifest_split(
        manifest_path,
        split=args.export_split,
        slides_per_project=int(args.slides_per_project),
        seed=int(args.seed) + 1,
        require_h5_exists=bool(args.require_h5_exists),
    )]
    _log(
        f"resolved H5 files: selection={len(selection_h5_files)} export={len(export_h5_files)} "
        f"(slides_per_project={int(args.slides_per_project)})"
    )
    if selection_h5_files:
        _log(f"selection sample h5={selection_h5_files[0]}")
    if export_h5_files:
        _log(f"export sample h5={export_h5_files[0]}")

    model_cfg = dict(run_cfg)
    model_cfg.update(
        {
            "ckpt": str(ckpt_path),
            "stage": stage,
            "d_in": d_in,
            "latent_dim": latent_dim,
        }
    )
    device = args.device
    if device != "cpu":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
    _log(f"loading model on device={device}")
    model = load_sae_from_config(model_cfg, device=device)
    _log("model loaded")

    pool_cache_dir: Path | None = None
    if magnification == "10x_pool2x2":
        pool_cache_dir = Path(tempfile.mkdtemp(prefix=f"rep_export_{args.run_name}_", dir=str(bundle_dir.parent)))
        _log(f"enabled temporary 10x pool cache at {pool_cache_dir}")

    try:
        _log(
            f"pass1 start: collecting latent stats on split='{args.selection_split}' "
            f"for {len(selection_h5_files)} slides"
        )
        global_stats = pass1_collect_stats(
            h5_files=selection_h5_files,
            model=model,
            stage=stage,
            d_latent=latent_dim,
            tiles_per_slide=tiles_per_slide,
            chunk_tiles=chunk_tiles,
            device=device,
            seed=int(args.seed),
            magnification=magnification,
            pool2x2_require_complete=(not pool_allow_partial),
            pool2x2_temp_cache_dir=pool_cache_dir,
        )
        _log("pass1 done")

        parent_assignment = None
        if stage == "sdf2":
            base_model = unwrap_base_model(model)
            if isinstance(base_model, SDFSAE2Level):
                parent_assignment = base_model.parent_assignment().detach().cpu().numpy()
                _log("loaded sdf2 parent assignments for parent-balanced selection")

        representative_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        latent_summary_rows: list[dict[str, Any]] = []
        selection_details: dict[str, Any] = {}

        diversity_enabled = (
            int(args.max_tiles_per_slide_per_latent) > 0
            or int(args.min_distance_px_same_slide_per_latent) > 0
        )
        candidate_topn = int(args.topn)
        if diversity_enabled and float(args.topn_buffer_factor) > 1.0:
            candidate_topn = max(int(args.topn), int(float(args.topn) * float(args.topn_buffer_factor) + 0.9999))
        _log(
            f"support selection settings: topn={int(args.topn)} candidate_topn={candidate_topn} "
            f"per_slide_cap={int(args.max_tiles_per_slide_per_latent)} min_distance_px={int(args.min_distance_px_same_slide_per_latent)}"
        )

        for latent_strategy in strategies:
            _log(f"selecting latents with strategy='{latent_strategy}'")
            selected_latents, selection_summary = select_latents(
                global_stats=global_stats,
                strategy=latent_strategy,
                n_latents=int(args.n_latents),
                manual=None,
                parent_assignment_all_level1=parent_assignment,
            )
            preview = ",".join(str(int(x)) for x in selected_latents[: min(8, len(selected_latents))])
            _log(
                f"strategy='{latent_strategy}' selected {len(selected_latents)} latents"
                + (f" preview=[{preview}]" if preview else "")
            )
            if selection_summary:
                _log(f"strategy='{latent_strategy}' summary={json.dumps(selection_summary, sort_keys=True)}")
            sdf_hierarchy = build_sdf_hierarchy_payload(model=model, selected_latents=selected_latents)
            _log(
                f"pass2 start for strategy='{latent_strategy}' on split='{args.export_split}' "
                f"for {len(export_h5_files)} slides"
            )
            raw_top_tiles = pass2_top_tiles(
                h5_files=export_h5_files,
                model=model,
                stage=stage,
                selected_latents=selected_latents,
                topn=int(args.topn),
                heap_topn=int(candidate_topn),
                tiles_per_slide=tiles_per_slide,
                chunk_tiles=chunk_tiles,
                device=device,
                seed=int(args.seed) + 17,
                magnification=magnification,
                pool2x2_require_complete=(not pool_allow_partial),
                pool2x2_temp_cache_dir=pool_cache_dir,
            )
            _log(f"pass2 done for strategy='{latent_strategy}'")
            raw_top_tiles = apply_per_slide_cap_to_top_tiles(
                raw_top_tiles,
                topn=int(args.topn),
                max_tiles_per_slide_per_latent=int(args.max_tiles_per_slide_per_latent),
                min_distance_px_same_slide_per_latent=int(args.min_distance_px_same_slide_per_latent),
            )
            support_count = int(sum(len(v) for v in raw_top_tiles.values()))
            _log(
                f"strategy='{latent_strategy}' support rows after diversity filtering={support_count}"
            )
            selection_details[latent_strategy] = {
                "selected_latents": [int(x) for x in selected_latents],
                "selection_summary": selection_summary,
            }

            for latent_idx in selected_latents:
                latent_group = infer_latent_group(int(latent_idx), sdf_hierarchy)
                base_rows = attach_slide_support_stats(
                    build_source_support_rows(
                        raw_top_tiles.get(int(latent_idx), []),
                        data_root=server_data_root,
                        encoder=encoder,
                        dataset=dataset,
                        run_name=args.run_name,
                        stage=stage,
                        data_split=args.export_split,
                        latent_strategy=latent_strategy,
                        latent_idx=int(latent_idx),
                        latent_group=latent_group,
                    )
                )
                if not base_rows:
                    continue

                latent_summary_rows.append(
                    build_latent_summary_row(
                        base_rows,
                        run_name=args.run_name,
                        stage=stage,
                        dataset=dataset,
                        encoder=encoder,
                        data_split=args.export_split,
                        latent_strategy=latent_strategy,
                        latent_idx=int(latent_idx),
                        latent_group=latent_group,
                        global_stats_selected={
                            "max_activation": float(global_stats["max"][int(latent_idx)]),
                            "variance": float(global_stats["var"][int(latent_idx)]),
                            "sparsity_score": float(global_stats["sparsity"][int(latent_idx)]),
                        },
                    )
                )

                for method in REPRESENTATIVE_METHODS:
                    ranked = rank_support_rows(base_rows, method)
                    for row in ranked:
                        enriched = dict(row)
                        enriched["max_activation_global"] = float(global_stats["max"][int(latent_idx)])
                        enriched["variance_global"] = float(global_stats["var"][int(latent_idx)])
                        enriched["sparsity_score_global"] = float(global_stats["sparsity"][int(latent_idx)])
                        support_rows.append(enriched)
                    representative = dict(ranked[0])
                    representative["row_kind"] = "representative"
                    representative_rows.append(representative)

        representative_fieldnames = [
            "run_name",
            "stage",
            "dataset",
            "encoder",
            "data_split",
            "latent_strategy",
            "latent_idx",
            "latent_group",
            "representative_method",
            "row_kind",
            "method_rank",
            "source_rank",
            "case_id",
            "slide_key",
            "cohort",
            "tile_index",
            "coord_x",
            "coord_y",
            "feature_relpath",
            "feature_h5_name",
            "legacy_h5_path",
            "activation",
            "method_score",
            "slide_support_count",
            "slide_max_activation",
            "slide_mean_activation",
            "max_activation_global",
            "variance_global",
            "sparsity_score_global",
        ]
        support_fieldnames = list(representative_fieldnames)
        latent_summary_fieldnames = [
            "run_name",
            "stage",
            "dataset",
            "encoder",
            "data_split",
            "latent_strategy",
            "latent_idx",
            "latent_group",
            "support_tile_count",
            "unique_slide_count",
            "unique_case_count",
            "activation_max",
            "activation_mean",
            "activation_p50",
            "activation_p90",
            "max_activation_global",
            "variance_global",
            "sparsity_score_global",
        ]

        representative_csv = bundle_dir / "representative_latents.csv"
        support_csv = bundle_dir / "representative_support_tiles.csv"
        latent_summary_csv = bundle_dir / "latent_summary.csv"
        _write_csv(representative_csv, representative_rows, representative_fieldnames)
        _write_csv(support_csv, support_rows, support_fieldnames)
        _write_csv(latent_summary_csv, latent_summary_rows, latent_summary_fieldnames)
        _log(
            f"wrote CSV artifacts: representatives={len(representative_rows)} support={len(support_rows)} "
            f"latent_summary={len(latent_summary_rows)}"
        )

        bundle_summary = build_bundle_summary(
            run_name=args.run_name,
            stage=stage,
            dataset=dataset,
            encoder=encoder,
            selection_split=args.selection_split,
            export_split=args.export_split,
            representative_rows=representative_rows,
            support_rows=support_rows,
            latent_summary_rows=latent_summary_rows,
            latent_strategies=strategies,
        )
        _write_json(bundle_dir / "bundle_summary.json", bundle_summary)
        _write_json(
            bundle_dir / "wsi_bench_model.json",
            build_wsi_bench_model_entry(
                model_id=f"{args.run_name}_representatives_{args.export_split}",
                model_name=args.model_name or f"{args.run_name} representatives {args.export_split}",
                encoder=encoder,
                dataset=dataset,
                slides_root=args.wsi_bench_slides_root,
            ),
        )

        manifest = {
            "schema_version": REPRESENTATIVE_BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repo_path": str(repo_root),
                "git_commit": _git_commit(repo_root),
            },
            "experiment": {
                "name": args.run_name,
                "stage": stage,
                "model_id": f"{args.run_name}_representatives_{args.export_split}",
                "model_name": args.model_name or f"{args.run_name} representatives {args.export_split}",
            },
            "model": {
                "stage": stage,
                "d_in": d_in,
                "latent_dim": latent_dim,
                "checkpoint": str(ckpt_path),
                "run_name": args.run_name,
            },
            "data": {
                "encoder": encoder,
                "dataset": dataset,
                "magnification": magnification,
                "coordinate_convention": "level0_top_left_px",
                "feature_identity": {
                    "manifest": str(manifest_path),
                    "selection_split": args.selection_split,
                    "export_split": args.export_split,
                    "relpath_field": "feature_relpath",
                    "legacy_path_field": "legacy_h5_path",
                },
            },
            "selection": {
                "latent_strategies": strategies,
                "representative_methods": list(REPRESENTATIVE_METHODS),
                "n_latents": int(args.n_latents),
                "topn_support": int(args.topn),
                "details_by_strategy": selection_details,
            },
            "artifacts": {
                "representative_latents_csv": representative_csv.name,
                "representative_support_tiles_csv": support_csv.name,
                "latent_summary_csv": latent_summary_csv.name,
                "bundle_summary_json": "bundle_summary.json",
                "wsi_bench_model_json": "wsi_bench_model.json",
            },
            "summary": bundle_summary,
        }
        _write_json(bundle_dir / "bundle_manifest.json", manifest)
        _log(f"bundle complete at {bundle_dir}")
        print(json.dumps({"bundle_dir": str(bundle_dir), **bundle_summary}, indent=2))
    finally:
        if pool_cache_dir is not None:
            shutil.rmtree(pool_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
