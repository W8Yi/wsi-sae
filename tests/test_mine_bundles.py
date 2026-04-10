from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._helpers import run_cli, save_relu_checkpoint, write_feature_h5, write_manifest


def test_mine_bundles_infers_run_config_and_exports_bundle(tmp_path):
    train_h5 = write_feature_h5(
        tmp_path / "features" / "TCGA-AB-1234-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    test_h5 = write_feature_h5(
        tmp_path / "features" / "TCGA-CD-5678-01Z-00-DX1.h5",
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
                [0.0, 0.0, 0.0, 1.0],
                [0.2, 0.0, 0.0, 0.8],
            ],
            dtype=np.float32,
        ),
    )
    manifest_path = write_manifest(
        tmp_path / "manifest.json",
        train=[str(train_h5)],
        test=[str(test_h5)],
    )

    run_root = tmp_path / "runs"
    stage_dir = run_root / "tcga_seal_sae_relu_v1" / "relu"
    ckpt_path, _cfg_path = save_relu_checkpoint(stage_dir, d_in=4, latent_dim=6)
    run_cfg = {
        "manifest": str(manifest_path),
        "run_name": "tcga_seal_sae_relu_v1",
        "tags": "tcga,seal,sae,relu,20x",
        "stage": "relu",
        "d_in": 4,
        "latent_dim": 6,
        "tied": False,
        "tiles_per_slide": 4,
        "chunk_tiles": 2,
        "magnification": "20x",
    }
    (stage_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    (stage_dir / "relu_ckpt_best.pt").write_bytes(ckpt_path.read_bytes())

    mining_root = tmp_path / "mining"
    export_root = tmp_path / "exports"
    run_cli(
        [
            "mine-bundles",
            "--run-name",
            "tcga_seal_sae_relu_v1",
            "--run-root",
            str(run_root),
            "--mining-root",
            str(mining_root),
            "--export-root",
            str(export_root),
            "--device",
            "cpu",
            "--slides-per-project",
            "-1",
            "--tiles-per-slide",
            "4",
            "--chunk-tiles",
            "2",
            "--n-latents",
            "3",
            "--topn",
            "2",
            "--require-h5-exists",
        ]
    )

    bundle_dir = export_root / "tcga_seal_sae_relu_v1" / "relu_test_export"
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "prototype_tiles.csv").exists()
    assert (bundle_dir / "latent_summary.csv").exists()
    assert (bundle_dir / "wsi_bench_model.json").exists()
    assert (bundle_dir / "mine_bundle_summary.json").exists()
    assert (bundle_dir / "prototypes.npz").exists()
    assert (bundle_dir / "prototypes.json").exists()
    assert (bundle_dir / "latent_targets.json").exists()

    summary = json.loads((bundle_dir / "mine_bundle_summary.json").read_text())
    assert summary["encoder"] == "seal"
    assert summary["stage"] == "relu"
    assert summary["artifacts"]["bundle_manifest_json"].endswith("bundle_manifest.json")

    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    assert bundle_manifest["experiment"]["data_split"] == "test"
    assert bundle_manifest["data"]["feature_identity"]["manifest"] == str(manifest_path)
