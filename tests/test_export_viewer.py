from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from ._helpers import run_cli


def _load_wsi_bench_module():
    path = Path("/common/users/wq50/wsi-bench/app.py")
    spec = importlib.util.spec_from_file_location("wsi_bench_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["wsi_bench_app"] = module
    spec.loader.exec_module(module)
    return module


def test_export_viewer_bundle_and_wsi_bench_compat(tmp_path):
    pass2_json = tmp_path / "pass2_top_tiles.json"
    pass2_json.write_text(
        json.dumps(
            {
                "pass1_source": str(tmp_path / "pass1_stats.json"),
                "config_pass1": {
                    "stage": "sdf2",
                    "d_in": 4,
                    "latent_dim": 6,
                    "magnification": "20x",
                    "ckpt": "/tmp/model.pt",
                },
                "config_pass2": {
                    "select_strategy": "sdf_parent_balanced",
                },
                "selected_latents": [7],
                "sdf_hierarchy": {
                    "level1_to_level2_parent_selected": {
                        "7": 2,
                    }
                },
                "top_tiles": {
                    "7": [
                        {
                            "score": 3.5,
                            "h5_path": "/tmp/TCGA-AB-1234-01Z-00-DX1.h5",
                            "tile_idx": 11,
                            "x": 256,
                            "y": 512,
                        }
                    ]
                },
            },
            indent=2,
        )
    )
    (tmp_path / "pass1_stats.json").write_text(json.dumps({"ok": True}))

    bundle_dir = tmp_path / "bundle"
    run_cli(
        [
            "export-viewer",
            "--pass2-json",
            str(pass2_json),
            "--out-dir",
            str(bundle_dir),
            "--model-id",
            "toy_model",
            "--model-name",
            "Toy Model",
            "--encoder",
            "uni2h",
            "--dataset",
            "TCGA",
            "--experiment-name",
            "toy_experiment",
            "--stage",
            "sdf2",
        ]
    )

    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    assert bundle_manifest["schema_version"] == "1.0"
    assert bundle_manifest["data"]["coordinate_convention"] == "level0_top_left_px"
    assert bundle_manifest["selection"]["selected_latents"] == [7]

    with (bundle_dir / "prototype_tiles.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["latent_group"] == "parent_2"
    assert rows[0]["slide_key"] == "TCGA-AB-1234-01Z-00-DX1"
    assert rows[0]["tile_index"] == "11"

    slides_root = tmp_path / "slides"
    slides_root.mkdir()
    (slides_root / "TCGA-AB-1234-01Z-00-DX1.png").write_bytes(b"placeholder")
    sae_manifest = tmp_path / "sae_models.json"
    sae_manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "toy_model",
                        "model_name": "Toy Model",
                        "encoder": "uni2h",
                        "dataset": "TCGA",
                        "slides_root": str(slides_root),
                        "prototype_tiles_csv": str(bundle_dir / "prototype_tiles.csv"),
                        "top_attention_tiles_csv": "",
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
    model = cache.get_model("toy_model")
    assert model is not None
    assert model["summary"]["total_prototype_rows"] == 1
    assert model["summary"]["total_latents"] == 1

