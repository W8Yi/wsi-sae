from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_wsi_bench_module():
    path = Path("/common/users/wq50/wsi-bench/app.py")
    spec = importlib.util.spec_from_file_location("wsi_bench_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["wsi_bench_app"] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def test_wsi_bench_accepts_representative_only_bundle(tmp_path):
    slides_root = tmp_path / "slides"
    slides_root.mkdir()
    (slides_root / "TCGA-AB-1234-01Z-00-DX1.png").write_bytes(b"placeholder")

    bundle_dir = tmp_path / "bundle"
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
        "max_activation_global",
        "variance_global",
        "sparsity_score_global",
    ]
    common = {
        "run_name": "tcga_seal_sae_relu_v1",
        "stage": "relu",
        "dataset": "TCGA",
        "encoder": "seal",
        "data_split": "test",
        "latent_strategy": "top_activation",
        "latent_idx": "7",
        "latent_group": "selected",
        "case_id": "TCGA-AB-1234",
        "slide_key": "TCGA-AB-1234-01Z-00-DX1",
        "cohort": "ACC",
        "tile_index": "11",
        "coord_x": "256",
        "coord_y": "512",
        "feature_relpath": "wsi_features/seal/TCGA/ACC/h5/TCGA-AB-1234-01Z-00-DX1.h5",
        "feature_h5_name": "TCGA-AB-1234-01Z-00-DX1.h5",
        "legacy_h5_path": "/legacy/TCGA-AB-1234-01Z-00-DX1.h5",
        "activation": "3.5",
        "method_score": "3.5",
        "slide_support_count": "2",
        "slide_max_activation": "3.5",
        "slide_mean_activation": "3.1",
        "max_activation_global": "4.0",
        "variance_global": "1.2",
        "sparsity_score_global": "0.8",
    }
    representative_rows = [
        {**common, "representative_method": "max_activation", "row_kind": "representative", "method_rank": "1", "source_rank": "1"},
        {**common, "representative_method": "diverse_support", "row_kind": "representative", "method_rank": "1", "source_rank": "1"},
    ]
    support_rows = [
        {**common, "representative_method": "max_activation", "row_kind": "support", "method_rank": "1", "source_rank": "1"},
        {**common, "representative_method": "max_activation", "row_kind": "support", "method_rank": "2", "source_rank": "2", "tile_index": "12", "coord_x": "512", "coord_y": "768", "activation": "2.7", "method_score": "2.7"},
        {**common, "representative_method": "diverse_support", "row_kind": "support", "method_rank": "1", "source_rank": "1"},
    ]
    _write_csv(bundle_dir / "representative_latents.csv", representative_rows, fieldnames)
    _write_csv(bundle_dir / "representative_support_tiles.csv", support_rows, fieldnames)
    _write_csv(
        bundle_dir / "latent_summary.csv",
        [
            {
                "run_name": "tcga_seal_sae_relu_v1",
                "stage": "relu",
                "dataset": "TCGA",
                "encoder": "seal",
                "data_split": "test",
                "latent_strategy": "top_activation",
                "latent_idx": "7",
                "latent_group": "selected",
                "support_tile_count": "2",
                "unique_slide_count": "1",
                "unique_case_count": "1",
                "activation_max": "3.5",
                "activation_mean": "3.1",
                "activation_p50": "3.1",
                "activation_p90": "3.42",
                "max_activation_global": "4.0",
                "variance_global": "1.2",
                "sparsity_score_global": "0.8",
            }
        ],
        [
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
        ],
    )
    (bundle_dir / "bundle_summary.json").write_text(
        json.dumps(
            {
                "model_id": "rep_model",
                "model_name": "Representative Model",
                "encoder": "seal",
                "dataset": "TCGA",
                "total_support_rows": 3,
                "available_representative_methods": ["max_activation", "diverse_support"],
                "available_latent_strategies": ["top_activation"],
            },
            indent=2,
        )
    )

    sae_manifest = tmp_path / "sae_models.json"
    sae_manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "rep_model",
                        "model_name": "Representative Model",
                        "encoder": "seal",
                        "dataset": "TCGA",
                        "slides_root": str(slides_root),
                        "representative_latents_csv": str(bundle_dir / "representative_latents.csv"),
                        "representative_support_tiles_csv": str(bundle_dir / "representative_support_tiles.csv"),
                        "latent_summary_csv": str(bundle_dir / "latent_summary.csv"),
                        "bundle_summary_json": str(bundle_dir / "bundle_summary.json"),
                        "tile_size": 256,
                    }
                ]
            },
            indent=2,
        )
    )

    wsi_bench = _load_wsi_bench_module()
    cache = wsi_bench.SaeCache(sae_manifest)
    cache.load()
    assert cache.errors == []

    model = cache.get_model("rep_model")
    assert model is not None
    assert sorted(model["summary"]["available_representative_methods"]) == ["diverse_support", "max_activation"]
    assert model["summary"]["available_latent_strategies"] == ["top_activation"]
    assert len(model["representative_methods"]["max_activation"]) == 1
    assert model["support_by_slide"]["TCGA-AB-1234-01Z-00-DX1"][0]["latent_strategy"] == "top_activation"
