from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._helpers import cli_env, run_cli, save_relu_checkpoint, write_feature_h5, write_manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _setup_relu_run_with_rep_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_h5 = write_feature_h5(
        tmp_path / "TCGA-ACC" / "features" / "TCGA-AB-1234-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    test_h5_a = write_feature_h5(
        tmp_path / "TCGA-ACC" / "features" / "TCGA-CD-5678-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    test_h5_b = write_feature_h5(
        tmp_path / "TCGA-BLCA" / "features" / "TCGA-EF-9999-01Z-00-DX1.h5",
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    manifest_path = write_manifest(
        tmp_path / "manifest.json",
        train=[str(train_h5)],
        test=[str(test_h5_a), str(test_h5_b)],
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
    run_cli(
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
    return run_root, export_root, manifest_path


def test_rep_analytics_exports_plot_ready_artifacts(tmp_path):
    run_root, export_root, _manifest_path = _setup_relu_run_with_rep_bundle(tmp_path)

    result = run_cli(
        [
            "rep-analytics",
            "--run-name",
            "tcga_seal_sae_relu_v1",
            "--run-root",
            str(run_root),
            "--export-root",
            str(export_root),
            "--device",
            "cpu",
            "--chunk-tiles",
            "2",
        ]
    )
    assert "[rep-analytics] starting analytics export" in result.stdout
    assert "[rep-analytics] analytics export complete at" in result.stdout

    analytics_dir = export_root / "tcga_seal_sae_relu_v1" / "analytics_test"
    assert (analytics_dir / "plot_manifest.json").exists()
    assert (analytics_dir / "all_latent_metrics.csv").exists()
    assert (analytics_dir / "selected_latent_slide_stats.csv").exists()
    assert (analytics_dir / "selected_latent_histograms.json").exists()
    assert (analytics_dir / "cohort_enrichment.csv").exists()
    assert (analytics_dir / "latent_umap.csv").exists()
    assert (analytics_dir / "analytics_summary.json").exists()
    assert not (analytics_dir / "case_label_enrichment.csv").exists()

    plot_manifest = json.loads((analytics_dir / "plot_manifest.json").read_text())
    selected_pairs = {
        (strategy, int(latent_idx))
        for strategy, latent_list in plot_manifest["selection"]["selected_by_strategy"].items()
        for latent_idx in latent_list
    }

    all_rows = _read_csv(analytics_dir / "all_latent_metrics.csv")
    assert len(all_rows) == 6
    latent0 = next(row for row in all_rows if row["latent_idx"] == "0")
    assert abs(float(latent0["slide_prevalence"]) - 0.5) < 1e-6
    assert abs(float(latent0["case_prevalence"]) - 0.5) < 1e-6

    slide_rows = _read_csv(analytics_dir / "selected_latent_slide_stats.csv")
    assert slide_rows
    assert all((row["latent_strategy"], int(row["latent_idx"])) in selected_pairs for row in slide_rows)

    hist_payload_1 = json.loads((analytics_dir / "selected_latent_histograms.json").read_text())
    assert hist_payload_1["bin_count"] == 32
    assert hist_payload_1["rows"]

    run_cli(
        [
            "rep-analytics",
            "--run-name",
            "tcga_seal_sae_relu_v1",
            "--run-root",
            str(run_root),
            "--export-root",
            str(export_root),
            "--device",
            "cpu",
            "--chunk-tiles",
            "2",
        ]
    )
    hist_payload_2 = json.loads((analytics_dir / "selected_latent_histograms.json").read_text())
    assert hist_payload_1 == hist_payload_2

    umap_rows = _read_csv(analytics_dir / "latent_umap.csv")
    summary = json.loads((analytics_dir / "analytics_summary.json").read_text())
    assert umap_rows
    assert summary["umap_backend"] in {"umap", "pca_fallback", "degenerate_single_point"}
    assert all(int(row["is_alive"]) == 1 for row in umap_rows)


def test_rep_analytics_supports_optional_case_labels_and_missing_bundle_error(tmp_path):
    run_root, export_root, _manifest_path = _setup_relu_run_with_rep_bundle(tmp_path)
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "case_id,subtype\n"
        "TCGA-CD-5678,ACC_like\n"
        "TCGA-EF-9999,BLCA_like\n"
    )

    run_cli(
        [
            "rep-analytics",
            "--run-name",
            "tcga_seal_sae_relu_v1",
            "--run-root",
            str(run_root),
            "--export-root",
            str(export_root),
            "--device",
            "cpu",
            "--chunk-tiles",
            "2",
            "--labels-csv",
            str(labels_csv),
            "--label-columns",
            "subtype",
        ]
    )
    analytics_dir = export_root / "tcga_seal_sae_relu_v1" / "analytics_test"
    assert (analytics_dir / "case_label_enrichment.csv").exists()
    label_rows = _read_csv(analytics_dir / "case_label_enrichment.csv")
    assert label_rows
    assert all(row["label_column"] == "subtype" for row in label_rows)

    missing_run_root = tmp_path / "runs_missing_bundle"
    missing_stage_dir = missing_run_root / "tcga_missing_bundle_relu_v1" / "relu"
    ckpt_path, _cfg_path = save_relu_checkpoint(missing_stage_dir, d_in=4, latent_dim=6)
    (missing_stage_dir / "relu_ckpt_best.pt").write_bytes(ckpt_path.read_bytes())
    (missing_stage_dir / "run_config.json").write_text(
        json.dumps(
            {
                "manifest": str(_manifest_path),
                "run_name": "tcga_missing_bundle_relu_v1",
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

    missing_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wsi_sae.cli",
            "rep-analytics",
            "--run-name",
            "tcga_missing_bundle_relu_v1",
            "--run-root",
            str(missing_run_root),
            "--export-root",
            str(export_root),
            "--device",
            "cpu",
        ],
        cwd=str(Path("/common/users/wq50/wsi-sae")),
        env=cli_env(),
        text=True,
        capture_output=True,
    )
    assert missing_result.returncode != 0
    assert "Representative bundle directory not found" in missing_result.stderr
