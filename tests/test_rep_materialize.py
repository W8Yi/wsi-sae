from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ._helpers import run_cli, write_feature_h5, write_slide_image


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_rep_materialize_uses_local_features_and_slides(tmp_path):
    data_root = tmp_path / "data_root"
    slide_path = write_slide_image(
        data_root / "wsi_slides" / "TCGA" / "ACC" / "slides" / "TCGA-AB-1234-01Z-00-DX1.png",
        width=512,
        height=512,
    )
    feature_path = write_feature_h5(
        data_root / "wsi_features" / "seal" / "TCGA" / "ACC" / "h5" / "TCGA-AB-1234-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    _write_csv(
        data_root / "registry" / "mapping.csv",
        [
            {
                "slide_id": "TCGA-AB-1234-01Z-00-DX1",
                "case_id": "TCGA-AB-1234",
                "dataset": "TCGA",
                "cohort": "ACC",
                "slide_filename": slide_path.name,
                "wsi_path": str(slide_path),
                "uni_path": "",
                "seal_path": str(feature_path),
            }
        ],
        ["slide_id", "case_id", "dataset", "cohort", "slide_filename", "wsi_path", "uni_path", "seal_path"],
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    fieldnames = [
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
    ]
    common_row = {
        "run_name": "tcga_seal_sae_relu_v1",
        "stage": "relu",
        "dataset": "TCGA",
        "encoder": "seal",
        "data_split": "test",
        "latent_strategy": "top_activation",
        "latent_idx": "7",
        "latent_group": "selected",
        "representative_method": "max_activation",
        "case_id": "TCGA-AB-1234",
        "slide_key": "TCGA-AB-1234-01Z-00-DX1",
        "cohort": "ACC",
        "tile_index": "1",
        "coord_x": "64",
        "coord_y": "64",
        "feature_relpath": "wsi_features/seal/TCGA/ACC/h5/TCGA-AB-1234-01Z-00-DX1.h5",
        "feature_h5_name": feature_path.name,
        "legacy_h5_path": "/legacy/TCGA-AB-1234-01Z-00-DX1.h5",
        "activation": "3.5",
        "method_score": "3.5",
        "slide_support_count": "1",
        "slide_max_activation": "3.5",
        "slide_mean_activation": "3.5",
    }
    _write_csv(
        bundle_dir / "representative_latents.csv",
        [{**common_row, "row_kind": "representative", "method_rank": "1", "source_rank": "1"}],
        fieldnames,
    )
    _write_csv(
        bundle_dir / "representative_support_tiles.csv",
        [{**common_row, "row_kind": "support", "method_rank": "1", "source_rank": "1"}],
        fieldnames,
    )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "artifacts": {
                    "representative_latents_csv": "representative_latents.csv",
                    "representative_support_tiles_csv": "representative_support_tiles.csv",
                },
            },
            indent=2,
        )
    )

    out_dir = tmp_path / "materialized"
    result = run_cli(
        [
            "rep-materialize",
            "--bundle",
            str(bundle_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--tile-size",
            "128",
        ]
    )
    payload = json.loads(result.stdout)
    assert payload["rows_total"] == 2
    assert payload["rows_with_feature_vector"] == 2
    assert payload["rows_with_tile_image"] == 2

    materialized_csv = out_dir / "materialized_rows.csv"
    with materialized_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "ok"
    assert Path(rows[0]["tile_image_path"]).exists()
    assert int(rows[0]["feature_vector_row"]) == 0

    features = np.load(out_dir / "encoder_features.npy")
    assert features.shape == (1, 4)
    assert any((out_dir / "contact_sheets").glob("top_activation__latent_0007__max_activation.*"))
