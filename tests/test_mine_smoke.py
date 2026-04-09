from __future__ import annotations

import numpy as np

from ._helpers import run_cli, save_relu_checkpoint, write_feature_h5


def test_mine_mode_both_on_tiny_features(tmp_path):
    feat_a = write_feature_h5(
        tmp_path / "TCGA-AB-1234-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    feat_b = write_feature_h5(
        tmp_path / "TCGA-CD-5678-01Z-00-DX1.h5",
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    h5_list = tmp_path / "h5_list.txt"
    h5_list.write_text(f"{feat_a}\n{feat_b}\n")
    ckpt_path, _cfg_path = save_relu_checkpoint(tmp_path / "artifacts", d_in=4, latent_dim=6)

    run_cli(
        [
            "mine",
            "--out_root",
            str(tmp_path / "mining"),
            "--run_name",
            "smoke",
            "--mode",
            "both",
            "--h5_list",
            str(h5_list),
            "--ckpt",
            str(ckpt_path),
            "--stage",
            "relu",
            "--d_in",
            "4",
            "--latent_dim",
            "6",
            "--device",
            "cpu",
            "--tiles_per_slide",
            "4",
            "--chunk_tiles",
            "2",
            "--select_strategy",
            "top_activation",
            "--n_latents",
            "3",
            "--topn",
            "2",
        ]
    )

    run_dir = tmp_path / "mining" / "smoke"
    assert (run_dir / "pass1_stats.json").exists()
    assert any(run_dir.glob("pass2_top_tiles_*.json"))

