from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsi_sae.data import get_paths_from_manifest, load_manifest_json
from wsi_sae.data.dataloader import _resolve_h5_path


KNOWN_ENCODERS = ("uni2", "seal", "gigapath", "virchow2")
SLIDE_ID_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}-DX[0-9A-Z]+", re.IGNORECASE)
PROJECT_RE = re.compile(r"^TCGA-[A-Z0-9]+$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _infer_encoder(run_name: str, cfg: dict[str, Any]) -> str:
    tags = [x.lower() for x in _split_csv(str(cfg.get("tags", "")))]
    run_name_l = run_name.lower()
    for encoder in KNOWN_ENCODERS:
        if encoder in tags or encoder in run_name_l:
            return encoder
    manifest_path = str(cfg.get("manifest", ""))
    for encoder in KNOWN_ENCODERS:
        if encoder in manifest_path.lower():
            return encoder
    raise ValueError(
        f"Could not infer encoder from run '{run_name}'. "
        f"Expected one of {', '.join(KNOWN_ENCODERS)} in run_name or tags."
    )


def _infer_dataset(run_name: str, cfg: dict[str, Any]) -> str:
    tags = [x.upper() for x in _split_csv(str(cfg.get("tags", "")))]
    for candidate in tags:
        if candidate.startswith("TCGA"):
            return "TCGA"
    manifest_path = str(cfg.get("manifest", ""))
    if "TCGA" in manifest_path.upper() or run_name.lower().startswith("tcga_"):
        return "TCGA"
    return ""


def _find_stage_dir(run_root: Path, run_name: str, stage_dir_name: str = "") -> tuple[Path, dict[str, Any]]:
    base = run_root / run_name
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")

    if stage_dir_name:
        cfg_path = base / stage_dir_name / "run_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
        return cfg_path.parent, _load_json(cfg_path)

    configs = sorted(base.glob("*/run_config.json"))
    if not configs:
        raise FileNotFoundError(f"No stage run_config.json found under: {base}")
    if len(configs) == 1:
        cfg_path = configs[0]
        return cfg_path.parent, _load_json(cfg_path)

    preferred: list[Path] = []
    for cfg_path in configs:
        cfg = _load_json(cfg_path)
        stage = str(cfg.get("stage", "")).strip()
        if stage and cfg_path.parent.name == stage:
            preferred.append(cfg_path)
    chosen = preferred[0] if preferred else configs[0]
    return chosen.parent, _load_json(chosen)


def _candidate_ckpts(stage_dir: Path, stage: str) -> list[Path]:
    prefix = stage.strip() or stage_dir.name
    candidates = [
        stage_dir / f"{prefix}_ckpt_best.pt",
        stage_dir / f"{prefix}_final.pt",
        stage_dir / f"{prefix}_ckpt_last.pt",
    ]
    if prefix != stage_dir.name:
        candidates.extend(
            [
                stage_dir / f"{stage_dir.name}_ckpt_best.pt",
                stage_dir / f"{stage_dir.name}_final.pt",
                stage_dir / f"{stage_dir.name}_ckpt_last.pt",
            ]
        )
    candidates.extend(sorted(stage_dir.glob("*_ckpt_best.pt")))
    candidates.extend(sorted(stage_dir.glob("*_final.pt")))
    candidates.extend(sorted(stage_dir.glob("*_ckpt_last.pt")))
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _select_ckpt(stage_dir: Path, cfg: dict[str, Any], override: str = "") -> Path:
    if override:
        ckpt = Path(override)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint override not found: {ckpt}")
        return ckpt
    stage = str(cfg.get("stage", "")).strip()
    for path in _candidate_ckpts(stage_dir, stage):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find a checkpoint under {stage_dir}. "
        f"Tried best/final/last variants for stage '{stage or stage_dir.name}'."
    )


def _project_key_from_path(path_str: str) -> str:
    p = Path(path_str)
    for seg in p.parts:
        seg_u = seg.upper()
        if PROJECT_RE.match(seg_u) and not SLIDE_ID_RE.match(seg_u):
            return seg_u
    stem = p.stem.upper()
    if stem.startswith("TCGA-"):
        return "TCGA"
    return "UNKNOWN"


def _h5_list_from_manifest_split(
    manifest_path: Path,
    *,
    split: str,
    slides_per_project: int,
    seed: int,
    require_h5_exists: bool,
) -> list[str]:
    manifest = load_manifest_json(str(manifest_path))
    split_paths = get_paths_from_manifest(manifest, split)
    if not split_paths:
        raise ValueError(f"Manifest split '{split}' is empty: {manifest_path}")

    by_project: dict[str, list[str]] = defaultdict(list)
    for raw_path in split_paths:
        resolved = _resolve_h5_path(str(raw_path))
        candidate = Path(resolved)
        if require_h5_exists and not candidate.exists():
            continue
        by_project[_project_key_from_path(str(raw_path))].append(str(candidate))

    if not by_project:
        raise ValueError(
            f"No readable H5 files found for split '{split}' in manifest {manifest_path}. "
            "Check path remapping and feature availability."
        )

    rng = random.Random(seed)
    out: list[str] = []
    for project in sorted(by_project.keys()):
        files = sorted(set(by_project[project]))
        rng.shuffle(files)
        if slides_per_project >= 0:
            files = files[: min(slides_per_project, len(files))]
        out.extend(files)
    rng.shuffle(out)
    return out


def _run_cli_subcommand(args: list[str], *, env: dict[str, str]) -> None:
    cmd = [sys.executable, "-m", "wsi_sae.cli", *args]
    subprocess.run(cmd, check=True, env=env)


def _latest_pass2_json(mining_run_dir: Path) -> Path:
    matches = sorted(mining_run_dir.glob("pass2_top_tiles_*.json"))
    if not matches:
        raise FileNotFoundError(f"No pass2_top_tiles_*.json found under: {mining_run_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _default_model_name(run_name: str, stage: str, export_split: str) -> str:
    return f"{run_name} {stage} {export_split} export"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Mine concepts and export a viewer bundle from an existing training run by inferring "
            "checkpoint/config/dataset/encoder settings from the run directory."
        )
    )
    repo_root = _repo_root()
    ap.add_argument("--run-name", type=str, required=True, help="Training run name under runs/<run-name>/")
    ap.add_argument("--run-root", type=Path, default=repo_root / "runs")
    ap.add_argument("--mining-root", type=Path, default=repo_root / "mining")
    ap.add_argument("--export-root", type=Path, default=repo_root / "exports")
    ap.add_argument("--stage-dir", type=str, default="", help="Optional subdirectory under the run root (for multi-stage runs).")
    ap.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint override.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--discovery-split", type=str, default="train")
    ap.add_argument("--export-split", type=str, default="test")
    ap.add_argument("--slides-per-project", type=int, default=200)
    ap.add_argument("--require-h5-exists", action="store_true")
    ap.add_argument("--tiles-per-slide", type=int, default=0, help="0 uses the training run value.")
    ap.add_argument("--chunk-tiles", type=int, default=0, help="0 uses the training run value.")

    ap.add_argument(
        "--select-strategy",
        choices=["top_activation", "top_variance", "top_sparsity", "manual", "sdf_parent_balanced"],
        default="top_activation",
    )
    ap.add_argument("--n-latents", type=int, default=128)
    ap.add_argument("--latent-indices", type=str, default="")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--topn-buffer-factor", type=float, default=4.0)
    ap.add_argument("--max-tiles-per-slide-per-latent", type=int, default=3)
    ap.add_argument("--min-distance-px-same-slide-per-latent", type=int, default=512)

    ap.add_argument("--skip-discovery", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-prototypes", action="store_true")
    ap.add_argument("--skip-targets", action="store_true")
    ap.add_argument("--model-name", type=str, default="")
    ap.add_argument("--wsi-bench-slides-root", type=str, default="", help="Optional local-PC slides_root to include in the exported wsi-bench snippet.")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
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

    bundle_stage_name = f"{stage}_{args.export_split}_export"
    discovery_run_name = f"{args.run_name}_{args.discovery_split}_discovery"
    export_run_name = f"{args.run_name}_{args.export_split}_export"
    export_dir = args.export_root / args.run_name / bundle_stage_name

    env = os.environ.copy()
    env["WSI_SAE_PREFERRED_ENCODER"] = encoder
    existing = env.get("PYTHONPATH", "").strip()
    src = str(_repo_root() / "src")
    env["PYTHONPATH"] = f"{src}:{existing}" if existing else src

    summary: dict[str, Any] = {
        "run_name": args.run_name,
        "stage_dir": str(stage_dir),
        "stage": stage,
        "encoder": encoder,
        "dataset": dataset,
        "manifest": str(manifest_path),
        "checkpoint": str(ckpt_path),
        "d_in": d_in,
        "latent_dim": latent_dim,
        "magnification": magnification,
        "tiles_per_slide": tiles_per_slide,
        "chunk_tiles": chunk_tiles,
        "discovery_split": args.discovery_split,
        "export_split": args.export_split,
        "discovery_run_name": discovery_run_name,
        "export_run_name": export_run_name,
        "export_dir": str(export_dir),
        "artifacts": {},
    }

    with tempfile.TemporaryDirectory(prefix=f"wsi_sae_mine_{args.run_name}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        def run_mine_for_split(*, split: str, mining_run_name: str) -> Path:
            h5_list = _h5_list_from_manifest_split(
                manifest_path,
                split=split,
                slides_per_project=int(args.slides_per_project),
                seed=int(args.seed),
                require_h5_exists=bool(args.require_h5_exists),
            )
            h5_list_path = tmp_dir / f"{split}_h5_list.txt"
            h5_list_path.write_text("\n".join(h5_list) + "\n")

            mine_args = [
                "mine",
                "--out_root",
                str(args.mining_root),
                "--run_name",
                mining_run_name,
                "--mode",
                "both",
                "--h5_list",
                str(h5_list_path),
                "--ckpt",
                str(ckpt_path),
                "--stage",
                stage,
                "--d_in",
                str(d_in),
                "--latent_dim",
                str(latent_dim),
                "--device",
                args.device,
                "--tiles_per_slide",
                str(tiles_per_slide),
                "--chunk_tiles",
                str(chunk_tiles),
                "--magnification",
                magnification,
                "--seed",
                str(args.seed),
                "--select_strategy",
                args.select_strategy,
                "--n_latents",
                str(args.n_latents),
                "--topn",
                str(args.topn),
                "--topn_buffer_factor",
                str(args.topn_buffer_factor),
                "--max_tiles_per_slide_per_latent",
                str(args.max_tiles_per_slide_per_latent),
                "--min_distance_px_same_slide_per_latent",
                str(args.min_distance_px_same_slide_per_latent),
            ]
            if args.select_strategy == "manual" and args.latent_indices:
                mine_args.extend(["--latent_indices", args.latent_indices])
            _run_cli_subcommand(mine_args, env=env)
            return _latest_pass2_json(args.mining_root / mining_run_name)

        if not args.skip_discovery:
            discovery_pass2 = run_mine_for_split(split=args.discovery_split, mining_run_name=discovery_run_name)
            summary["artifacts"]["discovery_pass2_json"] = str(discovery_pass2)

        if args.skip_export:
            _write_json(export_dir / "mine_bundle_summary.json", summary)
            print(json.dumps(summary, indent=2))
            return

        export_pass2 = run_mine_for_split(split=args.export_split, mining_run_name=export_run_name)
        summary["artifacts"]["export_pass2_json"] = str(export_pass2)

        prototypes_npz = export_dir / "prototypes.npz"
        prototypes_json = export_dir / "prototypes.json"
        latent_targets_json = export_dir / "latent_targets.json"

        if not args.skip_prototypes:
            _run_cli_subcommand(
                [
                    "build-prototypes",
                    "--pass2-json",
                    str(export_pass2),
                    "--sae-ckpt",
                    str(ckpt_path),
                    "--sae-cfg",
                    str(stage_dir / "run_config.json"),
                    "--out-npz",
                    str(prototypes_npz),
                    "--out-json",
                    str(prototypes_json),
                    "--device",
                    args.device,
                ],
                env=env,
            )
            summary["artifacts"]["prototypes_npz"] = str(prototypes_npz)
            summary["artifacts"]["prototypes_json"] = str(prototypes_json)

        if not args.skip_targets:
            _run_cli_subcommand(
                [
                    "build-targets",
                    "--pass2-json",
                    str(export_pass2),
                    "--out-json",
                    str(latent_targets_json),
                ],
                env=env,
            )
            summary["artifacts"]["latent_targets_json"] = str(latent_targets_json)

        export_args = [
            "export-viewer",
            "--pass2-json",
            str(export_pass2),
            "--out-dir",
            str(export_dir),
            "--run-config",
            str(stage_dir / "run_config.json"),
            "--model-id",
            f"{args.run_name}_{bundle_stage_name}",
            "--model-name",
            (args.model_name or _default_model_name(args.run_name, stage, args.export_split)),
            "--encoder",
            encoder,
            "--dataset",
            dataset,
            "--experiment-name",
            args.run_name,
            "--stage",
            bundle_stage_name,
            "--data-split",
            args.export_split,
        ]
        if args.wsi_bench_slides_root:
            export_args.extend(["--wsi-bench-slides-root", args.wsi_bench_slides_root])
        if not args.skip_prototypes and prototypes_npz.exists():
            export_args.extend(["--prototypes-npz", str(prototypes_npz), "--prototypes-json", str(prototypes_json)])
        if not args.skip_targets and latent_targets_json.exists():
            export_args.extend(["--latent-targets-json", str(latent_targets_json)])
        _run_cli_subcommand(export_args, env=env)

        summary["artifacts"]["bundle_manifest_json"] = str(export_dir / "bundle_manifest.json")
        summary["artifacts"]["prototype_tiles_csv"] = str(export_dir / "prototype_tiles.csv")
        if (export_dir / "latent_summary.csv").exists():
            summary["artifacts"]["latent_summary_csv"] = str(export_dir / "latent_summary.csv")
        if (export_dir / "wsi_bench_model.json").exists():
            summary["artifacts"]["wsi_bench_model_json"] = str(export_dir / "wsi_bench_model.json")
        _write_json(export_dir / "mine_bundle_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
