from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from wsi_sae.commands.export_viewer import _git_commit
from wsi_sae.commands.mine import load_sae_from_config, read_h5_subset, unwrap_base_model
from wsi_sae.commands.mine_bundles import (
    _find_stage_dir,
    _h5_list_from_manifest_split,
    _infer_dataset,
    _infer_encoder,
    _select_ckpt,
)
from wsi_sae.data.layout import extract_case_id, extract_slide_id, infer_cohort_from_path


PLOT_SCHEMA_VERSION = "1.0"
DEFAULT_HIST_BINS = 32
PROGRESS_EVERY_CASES = 10


@dataclass
class SlideEntry:
    h5_path: Path
    case_id: str
    slide_key: str
    cohort: str


def _log(message: str) -> None:
    print(f"[rep-analytics] {message}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _resolve_bundle_dir(bundle_dir: Path | None, *, export_root: Path, run_name: str, split: str) -> Path:
    if bundle_dir is not None:
        path = bundle_dir
    else:
        path = export_root / run_name / f"representatives_{split}"
    if not path.exists():
        raise FileNotFoundError(f"Representative bundle directory not found: {path}")
    manifest = path / "bundle_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Representative bundle manifest not found: {manifest}")
    return path


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _load_bundle_metadata(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, list[int]], dict[tuple[str, int], str]]:
    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    selection = manifest.get("selection", {}) if isinstance(manifest, dict) else {}
    details = selection.get("details_by_strategy", {}) if isinstance(selection, dict) else {}
    selected_by_strategy: dict[str, list[int]] = {}
    for strategy, payload in details.items():
        if not isinstance(payload, dict):
            continue
        selected = payload.get("selected_latents", [])
        selected_by_strategy[str(strategy)] = [int(x) for x in selected]

    group_map: dict[tuple[str, int], str] = {}
    rep_csv = bundle_dir / str(manifest.get("artifacts", {}).get("representative_latents_csv", "representative_latents.csv"))
    if rep_csv.exists():
        with rep_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                strategy = str(row.get("latent_strategy", "")).strip()
                if not strategy:
                    continue
                latent_idx = int(float(row.get("latent_idx", 0)))
                if strategy not in selected_by_strategy:
                    selected_by_strategy[strategy] = []
                if latent_idx not in selected_by_strategy[strategy]:
                    selected_by_strategy[strategy].append(latent_idx)
                group_map[(strategy, latent_idx)] = str(row.get("latent_group", "selected") or "selected")
    for strategy in selected_by_strategy:
        selected_by_strategy[strategy] = sorted(set(int(x) for x in selected_by_strategy[strategy]))
    return manifest, selected_by_strategy, group_map


def _decoder_dictionary_numpy(model: torch.nn.Module) -> np.ndarray:
    base = unwrap_base_model(model)
    if hasattr(base, "decoder_dictionary"):
        W = base.decoder_dictionary().detach().cpu().numpy()
    elif getattr(base, "tied", False):
        W = base.enc.weight.detach().cpu().numpy().T
    elif getattr(base, "dec", None) is not None:
        W = base.dec.weight.detach().cpu().numpy()
    else:
        raise RuntimeError(f"Unable to derive decoder dictionary from model type {type(base).__name__}")
    if W.ndim != 2:
        raise RuntimeError(f"Decoder dictionary must be 2D, got shape {W.shape}")
    return np.asarray(W, dtype=np.float32)


def _compute_latent_umap(
    decoder_dictionary: np.ndarray,
    *,
    alive_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, str]:
    alive_idx = np.flatnonzero(np.asarray(alive_mask, dtype=bool))
    if alive_idx.size == 0:
        return np.empty((0, 2), dtype=np.float32), "none"
    X = decoder_dictionary[:, alive_idx].T.astype(np.float32, copy=False)
    if X.shape[0] == 1:
        return np.asarray([[0.0, 0.0]], dtype=np.float32), "degenerate_single_point"
    n_neighbors = int(max(2, min(15, X.shape[0] - 1)))
    try:
        from umap import UMAP  # type: ignore

        coords = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            random_state=int(seed),
        ).fit_transform(X)
        return np.asarray(coords, dtype=np.float32), "umap"
    except Exception:
        pca = PCA(n_components=2, random_state=int(seed))
        coords = pca.fit_transform(X)
        return np.asarray(coords, dtype=np.float32), "pca_fallback"


def _histogram_payload(
    values: list[float],
    *,
    bins: int,
) -> tuple[list[float], list[int]]:
    arr = np.asarray(values, dtype=np.float32) if values else np.asarray([0.0], dtype=np.float32)
    vmax = float(np.max(arr)) if arr.size else 0.0
    upper = vmax if vmax > 0.0 else 1.0
    counts, edges = np.histogram(arr, bins=int(bins), range=(0.0, upper))
    return edges.astype(np.float32).tolist(), counts.astype(np.int64).tolist()


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float64, copy=False) / total
    probs = probs[probs > 0]
    if probs.size <= 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(
        description=(
            "Build plot-ready analytics artifacts from an existing representative bundle and SAE checkpoint."
        )
    )
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--run-root", type=Path, default=repo_root / "runs")
    ap.add_argument("--export-root", type=Path, default=repo_root / "exports")
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--split", type=str, default="", help="Override export split; defaults to the representative bundle split.")
    ap.add_argument("--stage-dir", type=str, default="")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--chunk-tiles", type=int, default=0, help="0 uses the training run value.")
    ap.add_argument("--magnification", type=str, default="", help="Override magnification; default uses training run value.")
    ap.add_argument("--labels-csv", type=Path, default=None)
    ap.add_argument("--label-columns", type=str, default="")
    ap.add_argument("--umap-source", type=str, default="decoder", choices=["decoder"])
    ap.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    _log(f"starting analytics export for run='{args.run_name}'")
    stage_dir, run_cfg = _find_stage_dir(args.run_root, args.run_name, stage_dir_name=args.stage_dir)
    stage = str(run_cfg.get("stage", stage_dir.name)).strip()
    encoder = _infer_encoder(args.run_name, run_cfg)
    dataset = _infer_dataset(args.run_name, run_cfg)
    d_in = int(run_cfg["d_in"])
    latent_dim = int(run_cfg["latent_dim"])
    manifest_path = Path(str(run_cfg.get("manifest", ""))).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found in run_config: {manifest_path}")
    split = str(args.split or "test").strip()
    bundle_dir = _resolve_bundle_dir(args.bundle_dir, export_root=args.export_root, run_name=args.run_name, split=split)
    bundle_manifest, selected_by_strategy, latent_group_map = _load_bundle_metadata(bundle_dir)
    inferred_split = str(bundle_manifest.get("data", {}).get("feature_identity", {}).get("export_split", split)).strip() or split
    if not args.split:
        split = inferred_split
    analytics_dir = args.export_root / args.run_name / f"analytics_{split}"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _select_ckpt(stage_dir, run_cfg, override=str(args.ckpt) if args.ckpt else "")
    magnification = str(args.magnification or run_cfg.get("magnification", "20x"))
    chunk_tiles = int(args.chunk_tiles or run_cfg.get("chunk_tiles", 512))
    if chunk_tiles <= 0:
        raise ValueError("chunk_tiles must be > 0")

    _log(
        "resolved run config: "
        f"stage={stage} encoder={encoder} dataset={dataset} d_in={d_in} latent_dim={latent_dim} "
        f"split={split} magnification={magnification}"
    )
    _log(f"checkpoint={ckpt_path}")
    _log(f"manifest={manifest_path}")
    _log(f"representative bundle={bundle_dir}")

    export_h5_files = [
        Path(p)
        for p in _h5_list_from_manifest_split(
            manifest_path,
            split=split,
            slides_per_project=-1,
            seed=int(args.seed),
            require_h5_exists=True,
        )
    ]
    if not export_h5_files:
        raise ValueError(f"No readable H5 files found for split '{split}' in manifest {manifest_path}")
    _log(f"resolved full export split with {len(export_h5_files)} slides")

    selected_latent_map: dict[int, list[str]] = defaultdict(list)
    for strategy, latent_list in selected_by_strategy.items():
        for latent_idx in latent_list:
            if strategy not in selected_latent_map[int(latent_idx)]:
                selected_latent_map[int(latent_idx)].append(strategy)
    for latent_idx in selected_latent_map:
        selected_latent_map[latent_idx] = sorted(selected_latent_map[latent_idx])
    selected_union = sorted(selected_latent_map.keys())
    _log(
        f"selected representative latents: union={len(selected_union)} "
        f"strategies={','.join(sorted(selected_by_strategy.keys()))}"
    )

    slide_entries: list[SlideEntry] = []
    cases_to_slides: dict[str, list[SlideEntry]] = defaultdict(list)
    cohort_names: set[str] = set()
    for h5_path in export_h5_files:
        slide_key = extract_slide_id(h5_path.name)
        case_id = extract_case_id(slide_key)
        cohort = infer_cohort_from_path(h5_path) or ""
        entry = SlideEntry(h5_path=Path(h5_path), case_id=case_id, slide_key=slide_key, cohort=cohort)
        slide_entries.append(entry)
        cases_to_slides[case_id].append(entry)
        cohort_names.add(cohort)

    cohort_list = sorted(c for c in cohort_names if c)
    cohort_to_idx = {cohort: idx for idx, cohort in enumerate(cohort_list)}
    total_slides = len(slide_entries)
    total_cases = len(cases_to_slides)
    total_slides_by_cohort = np.zeros((len(cohort_list),), dtype=np.int64)
    for entry in slide_entries:
        if entry.cohort in cohort_to_idx:
            total_slides_by_cohort[cohort_to_idx[entry.cohort]] += 1

    model_cfg = dict(run_cfg)
    model_cfg.update({"ckpt": str(ckpt_path), "stage": stage, "d_in": d_in, "latent_dim": latent_dim})
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    _log(f"loading model on device={device}")
    model = load_sae_from_config(model_cfg, device=device)
    _log("model loaded")

    slides_with_activation = np.zeros((latent_dim,), dtype=np.int64)
    cases_with_activation = np.zeros((latent_dim,), dtype=np.int64)
    num_positive_tiles = np.zeros((latent_dim,), dtype=np.int64)
    sum_positive_activation = np.zeros((latent_dim,), dtype=np.float64)
    max_activation_seen = np.zeros((latent_dim,), dtype=np.float32)
    max_activation_global = np.full((latent_dim,), -np.inf, dtype=np.float32)
    sum_activation = np.zeros((latent_dim,), dtype=np.float64)
    sum_sq_activation = np.zeros((latent_dim,), dtype=np.float64)
    nnz_count = np.zeros((latent_dim,), dtype=np.int64)
    firing_slides_by_cohort = np.zeros((latent_dim, len(cohort_list)), dtype=np.int64) if cohort_list else np.zeros((latent_dim, 0), dtype=np.int64)
    total_tiles_seen = 0

    hist_values: dict[tuple[str, int], list[float]] = {(strategy, int(latent_idx)): [] for strategy, latent_list in selected_by_strategy.items() for latent_idx in latent_list}
    slide_stats_csv = analytics_dir / "selected_latent_slide_stats.csv"
    slide_stats_fields = [
        "latent_strategy",
        "latent_idx",
        "latent_group",
        "case_id",
        "slide_key",
        "cohort",
        "slide_max_activation",
        "slide_mean_positive_activation",
        "positive_tile_count",
        "total_tiles_seen",
        "fires",
    ]
    slide_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    slide_stats_f = slide_stats_csv.open("w", newline="")
    slide_writer = csv.DictWriter(slide_stats_f, fieldnames=slide_stats_fields)
    slide_writer.writeheader()

    try:
        with torch.no_grad():
            for case_idx, (case_id, entries) in enumerate(sorted(cases_to_slides.items()), start=1):
                case_max = np.zeros((latent_dim,), dtype=np.float32)
                for entry in entries:
                    X, _C = read_h5_subset(
                        entry.h5_path,
                        tiles_per_slide=0,
                        rng=np.random.default_rng(int(args.seed)),
                        magnification=magnification,
                    )
                    if X.shape[0] == 0:
                        continue
                    slide_max = np.zeros((latent_dim,), dtype=np.float32)
                    slide_sum_positive = np.zeros((latent_dim,), dtype=np.float64)
                    slide_positive_tiles = np.zeros((latent_dim,), dtype=np.int64)
                    total_tiles_seen += int(X.shape[0])

                    for s in range(0, X.shape[0], int(chunk_tiles)):
                        xb = torch.from_numpy(X[s : s + int(chunk_tiles)]).to(device, non_blocking=True)
                        _xhat, z, _a = model(xb)
                        z_np = z.detach().cpu().numpy().astype(np.float32, copy=False)
                        if z_np.size == 0:
                            continue
                        positive_mask = z_np > 0
                        z_max = np.max(z_np, axis=0).astype(np.float32, copy=False)
                        z_sum = np.sum(z_np, axis=0, dtype=np.float64)
                        z_sum_sq = np.sum(np.square(z_np, dtype=np.float32), axis=0, dtype=np.float64)
                        z_nnz = np.sum(positive_mask, axis=0, dtype=np.int64)
                        z_pos_sum = np.sum(np.where(positive_mask, z_np, 0.0), axis=0, dtype=np.float64)

                        max_activation_global = np.maximum(max_activation_global, z_max)
                        max_activation_seen = np.maximum(max_activation_seen, z_max)
                        sum_activation += z_sum
                        sum_sq_activation += z_sum_sq
                        nnz_count += z_nnz
                        num_positive_tiles += z_nnz
                        sum_positive_activation += z_pos_sum

                        slide_max = np.maximum(slide_max, z_max)
                        slide_positive_tiles += z_nnz
                        slide_sum_positive += z_pos_sum

                    slide_fires = slide_max > 0
                    slides_with_activation += slide_fires.astype(np.int64, copy=False)
                    case_max = np.maximum(case_max, slide_max)
                    if entry.cohort in cohort_to_idx:
                        firing_slides_by_cohort[:, cohort_to_idx[entry.cohort]] += slide_fires.astype(np.int64, copy=False)

                    for latent_idx in selected_union:
                        latent_idx = int(latent_idx)
                        slide_max_value = float(slide_max[latent_idx])
                        pos_tiles = int(slide_positive_tiles[latent_idx])
                        mean_pos = float(slide_sum_positive[latent_idx] / pos_tiles) if pos_tiles > 0 else 0.0
                        for strategy in selected_latent_map.get(latent_idx, []):
                            group = latent_group_map.get((strategy, latent_idx), "selected")
                            slide_writer.writerow(
                                {
                                    "latent_strategy": strategy,
                                    "latent_idx": latent_idx,
                                    "latent_group": group,
                                    "case_id": case_id,
                                    "slide_key": entry.slide_key,
                                    "cohort": entry.cohort,
                                    "slide_max_activation": slide_max_value,
                                    "slide_mean_positive_activation": mean_pos,
                                    "positive_tile_count": pos_tiles,
                                    "total_tiles_seen": int(X.shape[0]),
                                    "fires": int(slide_max_value > 0.0),
                                }
                            )
                            hist_values[(strategy, latent_idx)].append(slide_max_value)

                cases_with_activation += (case_max > 0).astype(np.int64, copy=False)
                if (case_idx == 1) or (case_idx % PROGRESS_EVERY_CASES == 0) or (case_idx == total_cases):
                    _log(
                        f"progress {case_idx}/{total_cases} cases "
                        f"slides={sum(len(v) for _, v in list(sorted(cases_to_slides.items()))[:case_idx])} "
                        f"tiles={int(total_tiles_seen)}"
                    )
    finally:
        slide_stats_f.close()

    alive_mask = max_activation_seen > 0
    alive_count = int(np.sum(alive_mask))
    count = max(int(total_tiles_seen), 1)
    mean_activation = sum_activation / count
    variance_global = np.maximum((sum_sq_activation / count) - (mean_activation * mean_activation), 0.0).astype(np.float32)
    sparsity_score_global = (1.0 - (nnz_count / count)).astype(np.float32)
    max_activation_global = np.where(np.isfinite(max_activation_global), max_activation_global, 0.0).astype(np.float32)

    _log("computing decoder-vector UMAP")
    decoder_dictionary = _decoder_dictionary_numpy(model)
    umap_coords, umap_backend = _compute_latent_umap(decoder_dictionary, alive_mask=alive_mask, seed=int(args.seed))
    alive_indices = np.flatnonzero(alive_mask)

    cohort_enrichment_rows: list[dict[str, Any]] = []
    all_latent_metric_rows: list[dict[str, Any]] = []
    for latent_idx in range(latent_dim):
        selected_strategies = ",".join(selected_latent_map.get(int(latent_idx), []))
        cohort_counts = firing_slides_by_cohort[int(latent_idx)] if firing_slides_by_cohort.shape[1] > 0 else np.zeros((0,), dtype=np.int64)
        top_cohort = ""
        top_cohort_share = 0.0
        cohort_entropy = 0.0
        if cohort_counts.size > 0 and int(np.sum(cohort_counts)) > 0:
            top_idx = int(np.argmax(cohort_counts))
            top_cohort = cohort_list[top_idx]
            top_cohort_share = float(cohort_counts[top_idx] / max(int(slides_with_activation[int(latent_idx)]), 1))
            cohort_entropy = _entropy_from_counts(cohort_counts)
        all_latent_metric_rows.append(
            {
                "latent_idx": int(latent_idx),
                "is_alive": int(alive_mask[int(latent_idx)]),
                "selected_strategies": selected_strategies,
                "max_activation_global": float(max_activation_global[int(latent_idx)]),
                "variance_global": float(variance_global[int(latent_idx)]),
                "sparsity_score_global": float(sparsity_score_global[int(latent_idx)]),
                "slide_prevalence": float(slides_with_activation[int(latent_idx)] / max(total_slides, 1)),
                "case_prevalence": float(cases_with_activation[int(latent_idx)] / max(total_cases, 1)),
                "num_tiles_positive": int(num_positive_tiles[int(latent_idx)]),
                "mean_positive_activation": float(sum_positive_activation[int(latent_idx)] / max(int(num_positive_tiles[int(latent_idx)]), 1)) if int(num_positive_tiles[int(latent_idx)]) > 0 else 0.0,
                "max_activation_seen": float(max_activation_seen[int(latent_idx)]),
                "cohort_entropy": float(cohort_entropy),
                "top_cohort": top_cohort,
                "top_cohort_share": float(top_cohort_share),
            }
        )

    for strategy, latent_list in sorted(selected_by_strategy.items()):
        for latent_idx in latent_list:
            global_prev = float(slides_with_activation[int(latent_idx)] / max(total_slides, 1))
            for cohort_idx, cohort in enumerate(cohort_list):
                slides_in_cohort = int(total_slides_by_cohort[cohort_idx])
                slides_with_act = int(firing_slides_by_cohort[int(latent_idx), cohort_idx])
                prevalence_in_cohort = float(slides_with_act / slides_in_cohort) if slides_in_cohort > 0 else 0.0
                enrichment_ratio = float(prevalence_in_cohort / global_prev) if global_prev > 0 else 0.0
                cohort_enrichment_rows.append(
                    {
                        "latent_strategy": strategy,
                        "latent_idx": int(latent_idx),
                        "latent_group": latent_group_map.get((strategy, int(latent_idx)), "selected"),
                        "cohort": cohort,
                        "slides_in_cohort": slides_in_cohort,
                        "slides_with_activation": slides_with_act,
                        "prevalence_in_cohort": prevalence_in_cohort,
                        "prevalence_global": global_prev,
                        "enrichment_ratio": enrichment_ratio,
                    }
                )

    hist_rows: list[dict[str, Any]] = []
    for (strategy, latent_idx), values in sorted(hist_values.items(), key=lambda item: (item[0][0], item[0][1])):
        bin_edges, counts = _histogram_payload(values, bins=int(args.hist_bins))
        hist_rows.append(
            {
                "latent_strategy": strategy,
                "latent_idx": int(latent_idx),
                "latent_group": latent_group_map.get((strategy, int(latent_idx)), "selected"),
                "bin_edges": bin_edges,
                "counts": counts,
                "n_slides": len(values),
                "n_firing_slides": int(sum(1 for v in values if float(v) > 0.0)),
                "max_activation": float(max(values)) if values else 0.0,
            }
        )

    latent_umap_rows: list[dict[str, Any]] = []
    for coord_idx, latent_idx in enumerate(alive_indices.tolist()):
        coords = umap_coords[coord_idx] if coord_idx < umap_coords.shape[0] else np.asarray([0.0, 0.0], dtype=np.float32)
        latent_umap_rows.append(
            {
                "latent_idx": int(latent_idx),
                "umap_x": float(coords[0]),
                "umap_y": float(coords[1]),
                "is_alive": 1,
                "selected_strategies": ",".join(selected_latent_map.get(int(latent_idx), [])),
                "max_activation_global": float(max_activation_global[int(latent_idx)]),
                "variance_global": float(variance_global[int(latent_idx)]),
                "sparsity_score_global": float(sparsity_score_global[int(latent_idx)]),
            }
        )

    all_latent_metrics_csv = analytics_dir / "all_latent_metrics.csv"
    with all_latent_metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_idx",
                "is_alive",
                "selected_strategies",
                "max_activation_global",
                "variance_global",
                "sparsity_score_global",
                "slide_prevalence",
                "case_prevalence",
                "num_tiles_positive",
                "mean_positive_activation",
                "max_activation_seen",
                "cohort_entropy",
                "top_cohort",
                "top_cohort_share",
            ],
        )
        writer.writeheader()
        for row in all_latent_metric_rows:
            writer.writerow(row)

    cohort_enrichment_csv = analytics_dir / "cohort_enrichment.csv"
    with cohort_enrichment_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_strategy",
                "latent_idx",
                "latent_group",
                "cohort",
                "slides_in_cohort",
                "slides_with_activation",
                "prevalence_in_cohort",
                "prevalence_global",
                "enrichment_ratio",
            ],
        )
        writer.writeheader()
        for row in cohort_enrichment_rows:
            writer.writerow(row)

    latent_umap_csv = analytics_dir / "latent_umap.csv"
    with latent_umap_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_idx",
                "umap_x",
                "umap_y",
                "is_alive",
                "selected_strategies",
                "max_activation_global",
                "variance_global",
                "sparsity_score_global",
            ],
        )
        writer.writeheader()
        for row in latent_umap_rows:
            writer.writerow(row)

    selected_latent_histograms_json = analytics_dir / "selected_latent_histograms.json"
    _write_json(
        selected_latent_histograms_json,
        {
            "schema_version": PLOT_SCHEMA_VERSION,
            "histogram_unit": "slide_max_activation",
            "bin_count": int(args.hist_bins),
            "rows": hist_rows,
        },
    )

    labels_info: dict[str, Any] = {"labels_csv": "", "label_columns": []}
    case_label_enrichment_csv_name = ""
    if args.labels_csv is not None:
        labels_csv = Path(args.labels_csv)
        if not labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {labels_csv}")
        with labels_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if "case_id" not in fieldnames:
                raise ValueError(f"labels_csv must contain case_id column: {labels_csv}")
            requested = [part.strip() for part in str(args.label_columns).split(",") if part.strip()]
            label_columns = requested if requested else [name for name in fieldnames if name != "case_id"]
            label_rows = list(reader)
        label_map: dict[str, dict[str, str]] = {}
        for row in label_rows:
            case_id = extract_case_id(str(row.get("case_id", "")))
            if case_id:
                label_map[case_id] = {col: str(row.get(col, "")) for col in label_columns}
        case_label_rows: list[dict[str, Any]] = []
        case_fires_by_strategy_latent: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        with slide_stats_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                key = (str(row.get("latent_strategy", "")), int(float(row.get("latent_idx", 0))))
                case_id = extract_case_id(str(row.get("case_id", "")))
                case_fires_by_strategy_latent[key][case_id] = max(
                    float(row.get("slide_max_activation", 0.0)),
                    case_fires_by_strategy_latent[key].get(case_id, 0.0),
                )
        export_case_ids = sorted(cases_to_slides.keys())
        labeled_case_ids = [case_id for case_id in export_case_ids if case_id in label_map]
        for strategy, latent_list in sorted(selected_by_strategy.items()):
            for latent_idx in latent_list:
                case_scores = case_fires_by_strategy_latent.get((strategy, int(latent_idx)), {})
                firing_cases = {case_id for case_id, score in case_scores.items() if float(score) > 0.0}
                for label_col in label_columns:
                    values = sorted({label_map[case_id].get(label_col, "") for case_id in labeled_case_ids if label_map[case_id].get(label_col, "")})
                    labeled_total = len(labeled_case_ids)
                    global_prev = float(sum(1 for case_id in labeled_case_ids if case_id in firing_cases) / labeled_total) if labeled_total > 0 else 0.0
                    for label_value in values:
                        cases_in_label = [case_id for case_id in labeled_case_ids if label_map[case_id].get(label_col, "") == label_value]
                        firing_in_label = sum(1 for case_id in cases_in_label if case_id in firing_cases)
                        prevalence_in_label = float(firing_in_label / len(cases_in_label)) if cases_in_label else 0.0
                        enrichment_ratio = float(prevalence_in_label / global_prev) if global_prev > 0 else 0.0
                        case_label_rows.append(
                            {
                                "latent_strategy": strategy,
                                "latent_idx": int(latent_idx),
                                "latent_group": latent_group_map.get((strategy, int(latent_idx)), "selected"),
                                "label_column": label_col,
                                "label_value": label_value,
                                "cases_in_label": int(len(cases_in_label)),
                                "cases_with_activation": int(firing_in_label),
                                "prevalence_in_label": prevalence_in_label,
                                "prevalence_global": global_prev,
                                "enrichment_ratio": enrichment_ratio,
                            }
                        )
        case_label_enrichment_csv = analytics_dir / "case_label_enrichment.csv"
        with case_label_enrichment_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "latent_strategy",
                    "latent_idx",
                    "latent_group",
                    "label_column",
                    "label_value",
                    "cases_in_label",
                    "cases_with_activation",
                    "prevalence_in_label",
                    "prevalence_global",
                    "enrichment_ratio",
                ],
            )
            writer.writeheader()
            for row in case_label_rows:
                writer.writerow(row)
        labels_info = {"labels_csv": str(labels_csv), "label_columns": label_columns}
        case_label_enrichment_csv_name = case_label_enrichment_csv.name

    analytics_summary = {
        "run_name": args.run_name,
        "stage": stage,
        "dataset": dataset,
        "encoder": encoder,
        "split": split,
        "total_slides": total_slides,
        "total_cases": total_cases,
        "total_tiles_seen": int(total_tiles_seen),
        "alive_latents": alive_count,
        "selected_latent_union": len(selected_union),
        "selected_strategies": sorted(selected_by_strategy.keys()),
        "histogram_unit": "slide_max_activation",
        "hist_bins": int(args.hist_bins),
        "umap_source": str(args.umap_source),
        "umap_backend": umap_backend,
        "labels": labels_info,
    }
    _write_json(analytics_dir / "analytics_summary.json", analytics_summary)

    plot_manifest = {
        "schema_version": PLOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_path": str(repo_root),
            "git_commit": _git_commit(repo_root),
            "checkpoint": str(ckpt_path),
            "bundle_manifest": str(bundle_dir / "bundle_manifest.json"),
            "manifest": str(manifest_path),
        },
        "experiment": {
            "run_name": args.run_name,
            "stage": stage,
            "dataset": dataset,
            "encoder": encoder,
            "split": split,
        },
        "selection": {
            "selected_by_strategy": {strategy: [int(x) for x in latent_list] for strategy, latent_list in sorted(selected_by_strategy.items())},
        },
        "plot_defaults": {
            "global_prevalence_x": "slide_prevalence",
            "global_activation_y": "mean_positive_activation",
            "histogram_unit": "slide_max_activation",
            "umap_source": str(args.umap_source),
            "umap_backend": umap_backend,
        },
        "labels": labels_info,
        "artifacts": {
            "all_latent_metrics_csv": all_latent_metrics_csv.name,
            "selected_latent_slide_stats_csv": slide_stats_csv.name,
            "selected_latent_histograms_json": selected_latent_histograms_json.name,
            "cohort_enrichment_csv": cohort_enrichment_csv.name,
            "latent_umap_csv": latent_umap_csv.name,
            "analytics_summary_json": "analytics_summary.json",
        },
    }
    if case_label_enrichment_csv_name:
        plot_manifest["artifacts"]["case_label_enrichment_csv"] = case_label_enrichment_csv_name
    _write_json(analytics_dir / "plot_manifest.json", plot_manifest)
    _log(f"analytics export complete at {analytics_dir}")
    print(json.dumps({"analytics_dir": str(analytics_dir), **analytics_summary}, indent=2))


if __name__ == "__main__":
    main()
