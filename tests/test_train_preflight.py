from __future__ import annotations

import numpy as np

from ._helpers import run_cli, write_feature_h5, write_manifest


def test_train_preflight_with_tiny_manifest(tmp_path):
    train_h5 = write_feature_h5(tmp_path / "train.h5", np.arange(24, dtype=np.float32).reshape(6, 4))
    test_h5 = write_feature_h5(tmp_path / "test.h5", np.arange(16, dtype=np.float32).reshape(4, 4))
    manifest = write_manifest(
        tmp_path / "manifest.json",
        train=[str(train_h5)],
        test=[str(test_h5)],
        structured=True,
        meta={"dataset": "toy", "encoder": "toy", "feature_dim": 4},
    )

    result = run_cli(
        [
            "train",
            "--manifest",
            str(manifest),
            "--out_dir",
            str(tmp_path / "run"),
            "--stage",
            "relu",
            "--d_in",
            "4",
            "--latent_dim",
            "6",
            "--tiles_per_slide",
            "4",
            "--slide_batch_tiles",
            "4",
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            "--preflight_only",
        ]
    )

    assert "[Preflight]" in result.stdout

