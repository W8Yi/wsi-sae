import tempfile
import unittest
from pathlib import Path

import torch

from wsi_sae.models.sae import BatchTopKSAE, SDFSAE2Level
from wsi_sae.commands.train import maybe_init_model_from_ckpt


class SDF2InitFromCheckpointTests(unittest.TestCase):
    def test_batch_topk_to_sdf2_copies_shared_level1_params(self):
        torch.manual_seed(0)
        src = BatchTopKSAE(d_in=16, d_latent=32, k=4, tied=False, use_pre_bias=True)
        with torch.no_grad():
            for p in src.parameters():
                p.uniform_(-0.1, 0.1)

        dst = SDFSAE2Level(
            d_in=16,
            d_latent=32,
            d_level2=8,
            tied=False,
            use_pre_bias=True,
            coeff_nonneg=False,
            coeff_simplex=True,
        )
        U_before = dst.U.detach().clone()
        A_before = dst.A_raw.detach().clone()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "batch_topk.pt"
            torch.save({"model": src.state_dict(), "stage_name": "batch_topk(k=4)"}, ckpt)

            maybe_init_model_from_ckpt(
                base_model=dst,
                stage="sdf2",
                init_mode="batch_topk_to_sdf2",
                init_from_ckpt=str(ckpt),
                init_allow_missing=False,
            )

        self.assertTrue(torch.allclose(dst.b_pre, src.b_pre))
        self.assertTrue(torch.allclose(dst.enc.weight, src.enc.weight))
        self.assertTrue(torch.allclose(dst.enc.bias, src.enc.bias))
        self.assertTrue(torch.allclose(dst.dec.weight, src.dec.weight))
        self.assertTrue(torch.allclose(dst.dec.bias, src.dec.bias))
        self.assertTrue(torch.allclose(dst.U, U_before))
        self.assertTrue(torch.allclose(dst.A_raw, A_before))

    def test_shape_mismatch_raises(self):
        src = BatchTopKSAE(d_in=16, d_latent=64, k=4, tied=False, use_pre_bias=True)
        dst = SDFSAE2Level(
            d_in=16,
            d_latent=32,
            d_level2=8,
            tied=False,
            use_pre_bias=True,
            coeff_nonneg=False,
            coeff_simplex=True,
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "bad.pt"
            torch.save({"model": src.state_dict(), "stage_name": "batch_topk(k=4)"}, ckpt)
            with self.assertRaises(ValueError):
                maybe_init_model_from_ckpt(
                    base_model=dst,
                    stage="sdf2",
                    init_mode="batch_topk_to_sdf2",
                    init_from_ckpt=str(ckpt),
                    init_allow_missing=False,
                )


if __name__ == "__main__":
    unittest.main()
