from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from wsi_sae.models.sae import ReLUSparseSAE


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = f"{SRC}:{existing}" if existing else str(SRC)
    env.setdefault("WANDB_MODE", "disabled")
    lib_dirs: list[str] = []
    for site_root in [Path(site.getusersitepackages()), *[Path(p) for p in site.getsitepackages()]]:
        lib_dirs.extend(str(p) for p in site_root.glob("nvidia/*/lib") if p.is_dir())
        lib_dirs.extend(str(p) for p in site_root.glob("cusparselt/lib") if p.is_dir())
    existing_ld = env.get("LD_LIBRARY_PATH", "").strip()
    joined = ":".join(dict.fromkeys(lib_dirs))
    env["LD_LIBRARY_PATH"] = f"{joined}:{existing_ld}" if existing_ld else joined
    return env


def run_cli(
    args: list[str],
    cwd: Path | None = None,
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = cli_env()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-m", "wsi_sae.cli", *args],
        cwd=str(ROOT if cwd is None else cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"CLI failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def write_feature_h5(path: Path, features: np.ndarray, coords: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    if coords is None:
        n = features.shape[1] if features.ndim == 3 else features.shape[0]
        coords = np.stack([np.arange(n, dtype=np.int32) * 256, np.zeros(n, dtype=np.int32)], axis=1)
    coords = np.asarray(coords, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("coords", data=coords)
    return path


def write_slide_image(path: Path, *, width: int = 1024, height: int = 1024) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(64, 96, 128))
    img.save(path)
    return path


def write_manifest(path: Path, train: list[str], test: list[str], *, val: list[str] | None = None, meta: dict | None = None, structured: bool = False) -> Path:
    payload: dict[str, object]
    if structured:
        payload = {
            "meta": meta or {},
            "splits": {
                "train": train,
                "test": test,
            },
        }
        if val is not None:
            payload["splits"]["val"] = val
    else:
        payload = {
            "train": train,
            "test": test,
        }
        if val is not None:
            payload["val"] = val
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_relu_checkpoint(workdir: Path, *, d_in: int = 4, latent_dim: int = 6) -> tuple[Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    model = ReLUSparseSAE(d_in=d_in, d_latent=latent_dim, tied=False, use_pre_bias=True)
    with torch.no_grad():
        model.enc.weight.zero_()
        model.enc.bias.zero_()
        for i in range(min(d_in, latent_dim)):
            model.enc.weight[i, i] = 1.0
        if latent_dim > d_in:
            model.enc.weight[d_in:, :] = 0.25
        if model.dec is not None:
            model.dec.weight.zero_()
            model.dec.bias.zero_()
    ckpt_path = workdir / "relu_ckpt.pt"
    cfg_path = workdir / "relu_cfg.json"
    torch.save({"model": model.state_dict(), "wrapper_layernorm": False}, ckpt_path)
    cfg_path.write_text(
        json.dumps(
            {
                "stage": "relu",
                "d_in": d_in,
                "latent_dim": latent_dim,
                "tied": False,
            },
            indent=2,
        )
    )
    return ckpt_path, cfg_path
