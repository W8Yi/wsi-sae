from __future__ import annotations

import csv
import json

import numpy as np

from ._helpers import run_cli, save_relu_checkpoint, write_feature_h5, write_manifest


def test_rep_export_builds_representative_bundle(tmp_path):
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
    (stage_dir / "relu_ckpt_best.pt").write_bytes(ckpt_path.read_bytes())
    (stage_dir / "run_config.json").write_text(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )

    export_root = tmp_path / "exports"
    result = run_cli(
        [
            "rep-export",
            "--run-name",
            "tcga_seal_sae_relu_v1",
            "--run-root",
            str(run_root),
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
            "2",
            "--topn",
            "2",
            "--require-h5-exists",
        ]
    )
    assert "[rep-export] starting representative export" in result.stdout
    assert "[rep-export] pass1 start:" in result.stdout
    assert "[rep-export] bundle complete at" in result.stdout

    bundle_dir = export_root / "tcga_seal_sae_relu_v1" / "representatives_test"
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "representative_latents.csv").exists()
    assert (bundle_dir / "representative_support_tiles.csv").exists()
    assert (bundle_dir / "latent_summary.csv").exists()
    assert (bundle_dir / "bundle_summary.json").exists()
    assert (bundle_dir / "wsi_bench_model.json").exists()

    with (bundle_dir / "representative_latents.csv").open() as f:
        rep_rows = list(csv.DictReader(f))
    assert rep_rows
    assert {row["representative_method"] for row in rep_rows} == {
        "max_activation",
        "median_activation",
        "diverse_support",
        "slide_spread",
    }
    assert {row["latent_strategy"] for row in rep_rows} == {
        "top_activation",
        "top_variance",
        "top_sparsity",
    }
    assert all(row["row_kind"] == "representative" for row in rep_rows)
    assert all(not row["feature_relpath"].startswith("/") for row in rep_rows)

    with (bundle_dir / "representative_support_tiles.csv").open() as f:
        support_rows = list(csv.DictReader(f))
    assert support_rows
    assert all(row["row_kind"] == "support" for row in support_rows)

    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    assert bundle_manifest["schema_version"] == "2.0"
    assert bundle_manifest["data"]["feature_identity"]["selection_split"] == "train"
    assert bundle_manifest["data"]["feature_identity"]["export_split"] == "test"
