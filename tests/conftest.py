from __future__ import annotations

import ctypes
import site
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _preload_optional_cuda_runtime() -> None:
    lib_paths: list[Path] = []
    for site_root in [Path(site.getusersitepackages()), *[Path(p) for p in site.getsitepackages()]]:
        lib_paths.extend(site_root.glob("nvidia/*/lib/*.so*"))
        lib_paths.extend(site_root.glob("cusparselt/lib/*.so*"))
    for cand in sorted({p.resolve() for p in lib_paths if p.exists()}):
        try:
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


_preload_optional_cuda_runtime()
