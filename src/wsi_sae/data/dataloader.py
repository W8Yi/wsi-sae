# dataloade.py
#
# Drop-in replacement for sae_dataloader.py with I/O and magnification optimizations:
# 1) Per-worker HDF5 file-handle cache (avoid open/close per __getitem__)
# 2) Fast contiguous slice sampling (optional; huge speedup vs random indexed reads)
# 3) Optional multi-chunk contiguous sampling (more diversity, still fast)
# 4) Shape cache JSON (avoid scanning thousands of H5s every run)
# 5) Optional 10x training view from existing 20x features via on-the-fly 2x2 pooling
#
# Assumes H5 format:
#   features: (1, N, D) or (N, D)   where D=1536 for UNI2
#   coords:   (1, N, 2) or (N, 2)   (optional)
#
# Notes:
# - HDF5 handle cache is per-process (DataLoader worker). Safe and fast.
# - 10x_pool2x2 requires coords. Features remain at original D; only tile count changes.
# - No duplicated 10x feature files are created: storage usage stays minimal.
# - Optional on-disk pool-map caching stores only index maps, not feature copies.
# - For shared FS, set:
#     export HDF5_USE_FILE_LOCKING=FALSE
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import json
import math
import os
import random
import re
import atexit
import hashlib
import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------
# HDF5 handle cache (per worker)
# -----------------------------

_H5_CACHE: Dict[str, h5py.File] = {}
_POOL_MAP_CACHE: Dict[str, np.ndarray] = {}
_H5_PATH_RESOLVE_CACHE: Dict[str, str] = {}
_H5_SHAPE_CACHE: Dict[str, Tuple[int, int]] = {}
_H5_SHAPE_MISMATCH_WARNED: set = set()


def _close_h5_cache():
    for p, f in list(_H5_CACHE.items()):
        try:
            f.close()
        except Exception:
            pass
    _H5_CACHE.clear()
    _POOL_MAP_CACHE.clear()
    _H5_SHAPE_CACHE.clear()


atexit.register(_close_h5_cache)


class Pool2x2GeometryError(ValueError):
    """Raised when coords cannot be safely quantized into a regular pooling grid."""


def _extract_tcga_project_from_parts(parts: Tuple[str, ...]) -> Optional[str]:
    for seg in parts:
        if seg.startswith("TCGA-"):
            return seg
    return None


def _candidate_tcga_h5_names(p: Path) -> List[str]:
    names: List[str] = [p.name]
    if p.suffix.lower() != ".h5":
        return names

    stem = p.stem
    m = re.match(r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-\d{2}[A-Z]-\d{2}-DX[0-9A-Z]+)", stem)
    if m:
        canon_stem = m.group(1)
        canon = f"{canon_stem}.h5"
        if canon not in names:
            names.append(canon)
        m_dx = re.match(r"^(.*-DX)([A-Z])$", canon_stem)
        if m_dx:
            dx_alpha = m_dx.group(2)
            dx_num = str(ord(dx_alpha) - ord("A") + 1)
            canon_dx_num = f"{m_dx.group(1)}{dx_num}.h5"
            if canon_dx_num not in names:
                names.append(canon_dx_num)

    if "." in stem:
        head = stem.split(".", 1)[0]
        head_name = f"{head}.h5"
        if head_name not in names:
            names.append(head_name)

    return names


def _tcga_features_bases() -> List[str]:
    env_base = os.environ.get("TCGA_FEATURES_BASE", "").strip()
    out: List[str] = []
    for base in [env_base, "/research/projects/mllab/WSI/TCGA_features", "/common/users/wq50/TCGA_features"]:
        if base and base not in out:
            out.append(base)
    return out


def _tcga_feature_subdirs() -> Tuple[str, ...]:
    # UNI2 features were migrated from `features/` to `features_uni2/`.
    return ("features_uni2", "features", "")


def _iter_h5_remap_candidates(path: str):
    p = Path(path)
    yield p
    parts = p.parts
    if "extracted_features" not in parts:
        return
    project = _extract_tcga_project_from_parts(parts)
    if not project:
        return
    names = _candidate_tcga_h5_names(p)
    bases = _tcga_features_bases()
    seen = set()
    for base in bases:
        root = Path(base) / project
        for subdir in _tcga_feature_subdirs():
            for fname in names:
                cand = root / subdir / fname if subdir else root / fname
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                yield cand


def _resolve_h5_path_by_glob(path: str) -> Optional[str]:
    p = Path(path)
    if "extracted_features" not in p.parts:
        return None
    project = _extract_tcga_project_from_parts(p.parts)
    if not project:
        return None

    names = _candidate_tcga_h5_names(p)
    stems: List[str] = []
    for n in names:
        s = Path(n).stem
        if s and s not in stems:
            stems.append(s)

    for base in _tcga_features_bases():
        root = Path(base) / project
        for subdir in _tcga_feature_subdirs():
            feat_dir = root / subdir if subdir else root
            if not feat_dir.is_dir():
                continue
            for stem in stems:
                exact = feat_dir / f"{stem}.h5"
                if exact.exists():
                    return str(exact)
                matches = sorted(feat_dir.glob(f"{stem}*.h5"))
                if matches:
                    return str(matches[0])
    return None


def _resolve_h5_path(path: str) -> str:
    cached = _H5_PATH_RESOLVE_CACHE.get(path)
    if cached is not None:
        return cached
    chosen = path
    for cand in _iter_h5_remap_candidates(path):
        if cand.exists():
            chosen = str(cand)
            break
    if chosen == path:
        glob_found = _resolve_h5_path_by_glob(path)
        if glob_found is not None:
            chosen = glob_found
    _H5_PATH_RESOLVE_CACHE[path] = chosen
    return chosen


def _get_h5_cached(path: str) -> h5py.File:
    """
    Return an open HDF5 handle from the per-process cache.
    Each DataLoader worker is its own process, so this is safe and prevents repeated open/close.
    """
    resolved = _resolve_h5_path(path)
    f = _H5_CACHE.get(resolved)
    if f is None:
        # swmr=True improves concurrency patterns on some systems; harmless if file not SWMR-written.
        # libver="latest" can also improve metadata performance on some installations.
        f = h5py.File(resolved, "r", libver="latest", swmr=True)
        _H5_CACHE[resolved] = f
    return f


def _get_features_dset(f: h5py.File):
    if "features" not in f:
        raise KeyError("Missing dataset 'features' in H5.")
    return f["features"]


def _get_coords_dset(f: h5py.File):
    if "coords" not in f:
        raise KeyError("Missing dataset 'coords' in H5. Required for 10x_pool2x2.")
    return f["coords"]


def _read_h5_coords_uncached(h5_path: str) -> np.ndarray:
    """Read coords with a local H5 handle (safe during dataset init before worker fork)."""
    resolved = _resolve_h5_path(h5_path)
    with h5py.File(resolved, "r") as f:
        c = _get_coords_dset(f)
        if c.ndim == 3 and c.shape[0] == 1:
            out = c[0]
        elif c.ndim == 2 and c.shape[1] == 2:
            out = c[:]
        else:
            raise ValueError(f"Unsupported coords shape {c.shape} in {h5_path}")
    return np.asarray(out)


def _build_pool2x2_groups_from_coords(
    coords: np.ndarray,
    require_complete: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build non-overlapping 2x2 pooling groups from tile top-left coords.

    Returns:
      groups: (M, 4) int64 rows [top-left, top-right, bottom-left, bottom-right]
      info: metadata for logging/debugging
    """
    info: Dict[str, Any] = {"n_coords": 0, "step_x": None, "step_y": None, "n_groups": 0}

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise Pool2x2GeometryError(f"coords must have shape (N,2), got {tuple(coords.shape)}")

    info["n_coords"] = int(coords.shape[0])
    if coords.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int64), info

    if np.issubdtype(coords.dtype, np.integer):
        coords_i = coords.astype(np.int64, copy=False)
    else:
        coords_r = np.rint(coords)
        if not np.allclose(coords, coords_r):
            raise Pool2x2GeometryError("coords contain non-integer values; cannot infer exact pooling grid")
        coords_i = coords_r.astype(np.int64, copy=False)

    xs = np.unique(coords_i[:, 0])
    ys = np.unique(coords_i[:, 1])
    dx = np.diff(xs)
    dy = np.diff(ys)
    dx = dx[dx > 0]
    dy = dy[dy > 0]
    if dx.size == 0 or dy.size == 0:
        return np.zeros((0, 4), dtype=np.int64), info

    step_x = int(np.min(dx))
    step_y = int(np.min(dy))
    info["step_x"] = step_x
    info["step_y"] = step_y
    if step_x <= 0 or step_y <= 0:
        raise Pool2x2GeometryError(f"non-positive inferred grid step: step_x={step_x}, step_y={step_y}")

    # Sparse grids are fine (gaps can be multiples of the base step). Non-multiples indicate jitter/misalignment.
    if np.any(dx % step_x != 0) or np.any(dy % step_y != 0):
        raise Pool2x2GeometryError(
            f"inconsistent grid spacing; inferred base step=({step_x},{step_y}) but found non-multiple diffs"
        )

    x0 = int(xs.min())
    y0 = int(ys.min())
    rel_x = coords_i[:, 0] - x0
    rel_y = coords_i[:, 1] - y0
    if np.any(rel_x % step_x != 0) or np.any(rel_y % step_y != 0):
        raise Pool2x2GeometryError("coords are not aligned to inferred grid after anchoring")

    qx = (rel_x // step_x).astype(np.int64)
    qy = (rel_y // step_y).astype(np.int64)

    qpos = np.stack([qx, qy], axis=1)
    _, inverse, counts = np.unique(qpos, axis=0, return_inverse=True, return_counts=True)
    dup_mask = counts[inverse] > 1
    if np.any(dup_mask):
        dup_n = int(np.count_nonzero(dup_mask))
        raise Pool2x2GeometryError(f"duplicate quantized grid positions detected (affected coords={dup_n})")

    pos_to_idx = {(int(a), int(b)): int(i) for i, (a, b) in enumerate(zip(qx, qy))}
    groups: List[List[int]] = []

    qx0 = int(qx.min())
    qy0 = int(qy.min())
    keys = sorted(pos_to_idx.keys(), key=lambda t: (t[1], t[0]))
    for gx, gy in keys:
        if ((gx - qx0) % 2 != 0) or ((gy - qy0) % 2 != 0):
            continue
        i00 = pos_to_idx.get((gx, gy))
        i10 = pos_to_idx.get((gx + 1, gy))
        i01 = pos_to_idx.get((gx, gy + 1))
        i11 = pos_to_idx.get((gx + 1, gy + 1))
        if None in {i00, i10, i01, i11}:
            if require_complete:
                continue
            present = [i for i in [i00, i10, i01, i11] if i is not None]
            if len(present) == 0:
                continue
            while len(present) < 4:
                present.append(present[-1])
            groups.append(present[:4])
        else:
            groups.append([i00, i10, i01, i11])

    if not groups:
        return np.zeros((0, 4), dtype=np.int64), info

    out = np.asarray(groups, dtype=np.int64)
    info["n_groups"] = int(out.shape[0])
    return out, info


def read_h5_shape(h5_path: str) -> Tuple[int, int]:
    # Only used when building the shape cache (or fallback)
    resolved = _resolve_h5_path(h5_path)
    with h5py.File(resolved, "r") as f:
        feats = _get_features_dset(f)
        shape = feats.shape
        if len(shape) == 2:
            n, d = shape
            return int(n), int(d)
        if len(shape) == 3 and shape[0] == 1:
            _, n, d = shape
            return int(n), int(d)
        raise ValueError(f"Unsupported features shape {shape} in {h5_path}")


def _read_h5_shape_cached_from_open(path: str) -> Tuple[int, int]:
    resolved = _resolve_h5_path(path)
    cached = _H5_SHAPE_CACHE.get(resolved)
    if cached is not None:
        return cached
    f = _get_h5_cached(path)
    feats = _get_features_dset(f)
    if feats.ndim == 2:
        n, d = int(feats.shape[0]), int(feats.shape[1])
    elif feats.ndim == 3 and feats.shape[0] == 1:
        n, d = int(feats.shape[1]), int(feats.shape[2])
    else:
        raise ValueError(f"Unsupported features shape {feats.shape} in {resolved}")
    _H5_SHAPE_CACHE[resolved] = (n, d)
    return n, d


def read_h5_features_indexed_cached(h5_path: str, idx: np.ndarray) -> np.ndarray:
    """
    Indexed read (still can be expensive vs contiguous slicing).
    h5py requires increasing indices. We sort and do not restore order.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    f = _get_h5_cached(h5_path)
    feats = _get_features_dset(f)
    if feats.ndim == 3 and feats.shape[0] == 1:
        n_rows = int(feats.shape[1])
    elif feats.ndim == 2:
        n_rows = int(feats.shape[0])
    else:
        raise ValueError(f"Unsupported features shape {feats.shape} in {h5_path}")
    if n_rows <= 0:
        return np.empty((0, 0), dtype=np.float32)

    idx = np.sort(idx)
    if idx[0] < 0 or idx[-1] >= n_rows:
        np.clip(idx, 0, n_rows - 1, out=idx)

    if feats.ndim == 3 and feats.shape[0] == 1:
        out = feats[0, idx, :]
    elif feats.ndim == 2:
        out = feats[idx, :]
    else:
        raise ValueError(f"Unsupported features shape {feats.shape} in {h5_path}")

    return np.asarray(out, dtype=np.float32)


def read_h5_features_slice_cached(h5_path: str, start: int, k: int) -> np.ndarray:
    """
    Contiguous slice read (usually much faster than indexed gather).
    """
    if k <= 0:
        return np.empty((0, 0), dtype=np.float32)

    f = _get_h5_cached(h5_path)
    feats = _get_features_dset(f)

    if feats.ndim == 3 and feats.shape[0] == 1:
        out = feats[0, start : start + k, :]
    elif feats.ndim == 2:
        out = feats[start : start + k, :]
    else:
        raise ValueError(f"Unsupported features shape {feats.shape} in {h5_path}")

    return np.asarray(out, dtype=np.float32)


# -----------------------------
# Dataset design
# -----------------------------

@dataclass
class SlideTileConfig:
    """
    Configuration for slide-level feature sampling.

    Key fields for magnification support:
    - magnification:
        "20x"          -> sample stored features directly.
        "10x_pool2x2"  -> construct 10x-like samples by pooling 2x2 neighboring 20x tiles.
    - pool2x2_require_complete:
        True  -> drop incomplete boundary groups (cleaner geometry).
        False -> keep boundary groups by repeating available members.
    - pool_map_cache_dir:
        Optional directory for cached 2x2 index maps. This saves startup time without duplicating features.
    - shapes_cache_json:
        Optional JSON cache for precomputed [num_tiles, feat_dim] per slide and magnification mode.
    """
    tiles_per_slide: int = 1024            # subsample per slide per epoch (for epoch_len only)
    slide_batch_tiles: int = 4096          # returned per __getitem__ (chunk size)
    seed: int = 1337
    shuffle_slides: bool = True
    sample_with_replacement: bool = False  # only relevant for indexed fallback
    normalize: Optional[str] = None        # None | "l2" | "layernorm"
    return_meta: bool = False              # return (x, meta) instead of x

    # I/O optimization controls:
    sampling: str = "slice"                # "slice" | "multislice" | "indexed"
    multislice_chunks: int = 4             # used if sampling="multislice": read this many contiguous chunks
    shapes_cache_json: Optional[str] = None  # path to JSON caching {h5_path: [n, d]}

    # Magnification control:
    # - "20x": use original tile features directly.
    # - "10x_pool2x2": build each sample by averaging non-overlapping 2x2 neighboring 20x tiles
    #   using coords grid adjacency; no extra 10x feature files needed.
    magnification: str = "20x"             # "20x" | "10x_pool2x2"
    pool2x2_reduce: str = "mean"           # currently only "mean"
    pool2x2_require_complete: bool = True  # keep only full 2x2 groups
    pool_map_cache_dir: Optional[str] = None  # optional on-disk cache for 2x2 index maps


class SlideTileDataset(Dataset):
    """
    Produces tile-chunks sampled from slides, optimized for I/O.

    sampling modes:
    - "slice":      one contiguous block of size slide_batch_tiles
    - "multislice": multiple contiguous blocks that sum to slide_batch_tiles (more diversity, still fast)
    - "indexed":    random indices (slowest on shared FS; use only if needed)

    magnification modes:
    - "20x": direct read from features dataset.
    - "10x_pool2x2": on-the-fly 10x features from 2x2 pooling of neighboring 20x tile features.
      This avoids storing duplicate 10x feature files and keeps storage usage low.
    """

    def __init__(self, h5_paths: List[str], cfg: SlideTileConfig):
        self.h5_paths = list(h5_paths)
        self.cfg = cfg
        self._validate_cfg()
        self._pool2x2_invalid: Dict[str, str] = {}
        self._pool2x2_counts: Dict[str, int] = {}

        # Load or build shapes cache
        shape_map: Dict[str, Tuple[int, int]] = {}
        if cfg.shapes_cache_json and Path(cfg.shapes_cache_json).exists():
            with open(cfg.shapes_cache_json, "r") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    shape_map[k] = (int(v[0]), int(v[1]))

        self.slide_n: List[int] = []
        self.slide_d: List[int] = []

        missing = []
        for p in self.h5_paths:
            cache_key = self._shape_cache_key(p)
            if cache_key in shape_map:
                n, d = shape_map[cache_key]
            else:
                missing.append(p)
                n, d = (0, 0)  # placeholder
            self.slide_n.append(n)
            self.slide_d.append(d)

        # If cache missing entries, compute them once and optionally write back
        if missing:
            for p in missing:
                n, d = self._read_slide_shape_with_mag(p)
                shape_map[self._shape_cache_key(p)] = (n, d)
            # update lists
            self.slide_n = [shape_map[self._shape_cache_key(p)][0] for p in self.h5_paths]
            self.slide_d = [shape_map[self._shape_cache_key(p)][1] for p in self.h5_paths]

            if cfg.shapes_cache_json:
                Path(cfg.shapes_cache_json).parent.mkdir(parents=True, exist_ok=True)
                with open(cfg.shapes_cache_json, "w") as f:
                    json.dump({k: [v[0], v[1]] for k, v in shape_map.items()}, f, indent=2)

        # Drop slides that have no usable tiles for the selected magnification mode.
        keep_ids = [i for i, n in enumerate(self.slide_n) if int(n) > 0]
        dropped_paths = [self.h5_paths[i] for i, n in enumerate(self.slide_n) if int(n) <= 0]
        if len(keep_ids) < len(self.h5_paths):
            dropped = len(self.h5_paths) - len(keep_ids)
            examples = ", ".join(Path(p).name for p in dropped_paths[:5])
            extra = "" if dropped <= 5 else f" (+{dropped - 5} more)"
            warnings.warn(
                f"Dropping {dropped} slide(s) with zero tiles for magnification={self.cfg.magnification}. "
                f"Examples: {examples}{extra}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.h5_paths = [self.h5_paths[i] for i in keep_ids]
            self.slide_n = [self.slide_n[i] for i in keep_ids]
            self.slide_d = [self.slide_d[i] for i in keep_ids]

        if len(self.h5_paths) == 0:
            raise ValueError(
                f"No usable slides remain after filtering for magnification={self.cfg.magnification}."
            )

        if self.cfg.magnification.lower() == "10x_pool2x2":
            n_slides = len(self.h5_paths)
            pooled_counts = np.asarray(self.slide_n, dtype=np.int64)
            if pooled_counts.size > 0:
                print(
                    "[SlideTileDataset] 10x_pool2x2 preflight: "
                    f"usable_slides={n_slides}, "
                    f"pooled_tiles_total={int(pooled_counts.sum())}, "
                    f"pooled_tiles_per_slide(min/median/max)="
                    f"{int(pooled_counts.min())}/{int(np.median(pooled_counts))}/{int(pooled_counts.max())}",
                    flush=True,
                )
            if self._pool2x2_invalid:
                bad_items = list(self._pool2x2_invalid.items())
                preview = "; ".join(
                    f"{Path(p).name}: {reason}" for p, reason in bad_items[:3]
                )
                tail = "" if len(bad_items) <= 3 else f" (+{len(bad_items) - 3} more)"
                warnings.warn(
                    "10x_pool2x2 preflight found invalid slide geometry/coords and dropped those slides. "
                    f"Examples: {preview}{tail}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Epoch length: each slide contributes ~tiles_per_slide tiles, chunked by slide_batch_tiles
        self.draws_per_slide = [
            max(1, math.ceil(cfg.tiles_per_slide / cfg.slide_batch_tiles)) for _ in self.h5_paths
        ]
        self.epoch_len = int(sum(self.draws_per_slide))
        self._build_draw_index()

    def _validate_cfg(self):
        mag = self.cfg.magnification.lower()
        if mag not in {"20x", "10x_pool2x2"}:
            raise ValueError("cfg.magnification must be one of: '20x', '10x_pool2x2'")
        if self.cfg.pool2x2_reduce.lower() != "mean":
            raise ValueError("cfg.pool2x2_reduce currently supports only 'mean'")

    def _shape_cache_key(self, h5_path: str) -> str:
        resolved = _resolve_h5_path(h5_path)
        mag = self.cfg.magnification.lower()
        if mag == "10x_pool2x2":
            return (
                f"{resolved}::mag={mag}"
                f"::complete={int(self.cfg.pool2x2_require_complete)}"
                "::pool2x2_gridcheck_v2"
            )
        return f"{resolved}::mag={mag}"

    def _pool_cache_path(self, h5_path: str) -> Optional[Path]:
        if not self.cfg.pool_map_cache_dir:
            return None
        p = Path(_resolve_h5_path(h5_path))
        st = p.stat()
        digest = hashlib.sha1(
            (
                f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|pool2x2_v2|"
                f"complete={int(self.cfg.pool2x2_require_complete)}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        base = f"{p.stem}.{digest}.pool2x2.npy"
        out = Path(self.cfg.pool_map_cache_dir) / base
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _build_pool2x2_map(self, h5_path: str) -> np.ndarray:
        """
        Build non-overlapping 2x2 groups of source tile indices using coordinates.

        Assumptions:
        - coords are tile top-left positions from a regular extraction grid.
        - 10x proxy is formed by averaging each 2x2 block of neighboring 20x tiles.
        - Blocks are non-overlapping (stride=2 in grid units), anchored at min coord.
        - If pool2x2_require_complete=True, partial boundary blocks are dropped.

        Efficiency:
        - This creates only an index map, not new feature tensors on disk.
        - Optional pool_map_cache_dir persists this index map so future runs skip recomputation.

        Returns:
          groups: (M, 4) int64 where each row is [top-left, top-right, bottom-left, bottom-right].
        """
        # Use an uncached H5 handle here so dataset init does not populate the global HDF5 cache
        # before DataLoader workers are forked/spawned.
        coords = _read_h5_coords_uncached(h5_path)
        groups, info = _build_pool2x2_groups_from_coords(
            coords=coords,
            require_complete=self.cfg.pool2x2_require_complete,
        )
        self._pool2x2_counts[h5_path] = int(groups.shape[0])
        return groups

    def _get_pool2x2_map(self, h5_path: str) -> np.ndarray:
        key = f"{h5_path}::pool2x2::complete={int(self.cfg.pool2x2_require_complete)}"
        if key in _POOL_MAP_CACHE:
            return _POOL_MAP_CACHE[key]

        cache_path = self._pool_cache_path(h5_path)
        groups = None
        if cache_path is not None and cache_path.exists():
            arr = np.load(cache_path, mmap_mode="r")
            groups = arr
        else:
            groups = self._build_pool2x2_map(h5_path)
            if cache_path is not None:
                np.save(cache_path, groups.astype(np.uint32, copy=False))
                groups = np.load(cache_path, mmap_mode="r")

        _POOL_MAP_CACHE[key] = groups
        return groups

    def _read_slide_shape_with_mag(self, h5_path: str) -> Tuple[int, int]:
        try:
            n20, d = read_h5_shape(h5_path)
        except Exception as e:
            self._pool2x2_invalid[h5_path] = f"features read failed: {type(e).__name__}: {e}"
            warnings.warn(
                f"Failed to read features shape for {h5_path}; dropping slide. Error: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0, 0
        if self.cfg.magnification.lower() == "20x":
            return n20, d
        try:
            groups = self._get_pool2x2_map(h5_path)
        except (Pool2x2GeometryError, KeyError, ValueError, OSError) as e:
            self._pool2x2_invalid[h5_path] = f"{type(e).__name__}: {e}"
            warnings.warn(
                f"10x_pool2x2 preflight failed for {h5_path}; dropping slide. "
                f"Reason: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0, d
        return int(groups.shape[0]), d

    def _build_draw_index(self):
        self.draw_to_slide = []
        for slide_id, k in enumerate(self.draws_per_slide):
            self.draw_to_slide.extend([slide_id] * k)

        if self.cfg.shuffle_slides:
            rng = random.Random(self.cfg.seed)
            rng.shuffle(self.draw_to_slide)

    def set_epoch(self, epoch: int):
        if self.cfg.shuffle_slides:
            rng = random.Random(self.cfg.seed + epoch)
            rng.shuffle(self.draw_to_slide)

    def __len__(self) -> int:
        return self.epoch_len

    def _normalize_feats(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.normalize is None:
            return x
        if self.cfg.normalize == "l2":
            denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            return x / denom
        if self.cfg.normalize == "layernorm":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            return (x - mu) / sd
        raise ValueError(f"Unknown normalize={self.cfg.normalize}")

    def _sample_indexed(self, n_tiles: int, k: int, rng: np.random.Generator) -> np.ndarray:
        if n_tiles >= k:
            return rng.choice(n_tiles, size=k, replace=False)
        if self.cfg.sample_with_replacement and n_tiles > 0:
            return rng.choice(n_tiles, size=k, replace=True)
        return np.arange(n_tiles, dtype=np.int64)

    def _pool_groups_to_features(self, h5_path: str, group_ids: np.ndarray) -> np.ndarray:
        """
        Efficient pooled read:
        1) Select pooled-group rows (k,4) of source indices.
        2) Read only unique source features once from H5.
        3) Re-index and reduce (mean) in memory.
        """
        groups = self._get_pool2x2_map(h5_path)
        if groups.shape[0] == 0 or group_ids.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        chosen = groups[np.asarray(group_ids, dtype=np.int64)]  # (k,4)
        flat = chosen.reshape(-1)
        unique_idx, inv = np.unique(flat, return_inverse=True)
        src = read_h5_features_indexed_cached(h5_path, unique_idx)  # (u,D)
        expanded = src[inv].reshape(chosen.shape[0], 4, src.shape[1])  # (k,4,D)
        if self.cfg.pool2x2_reduce.lower() == "mean":
            return expanded.mean(axis=1, dtype=np.float32)
        raise ValueError(f"Unsupported pool2x2_reduce={self.cfg.pool2x2_reduce}")

    def _read_chunk_pooled(self, h5_path: str, n_pooled_tiles: int, rng: np.random.Generator) -> np.ndarray:
        k = self.cfg.slide_batch_tiles
        mode = self.cfg.sampling.lower()

        if mode == "slice":
            if n_pooled_tiles >= k:
                start = int(rng.integers(0, n_pooled_tiles - k + 1))
                gid = np.arange(start, start + k, dtype=np.int64)
            else:
                gid = self._sample_indexed(n_pooled_tiles, k, rng)
            return self._pool_groups_to_features(h5_path, gid)

        if mode == "multislice":
            chunks = max(1, int(self.cfg.multislice_chunks))
            base = k // chunks
            rem = k % chunks
            sizes = [base + (1 if i < rem else 0) for i in range(chunks)]
            all_gid = []
            for sz in sizes:
                if sz <= 0:
                    continue
                if n_pooled_tiles >= sz:
                    start = int(rng.integers(0, n_pooled_tiles - sz + 1))
                    all_gid.append(np.arange(start, start + sz, dtype=np.int64))
                else:
                    all_gid.append(self._sample_indexed(n_pooled_tiles, sz, rng))
            if not all_gid:
                return np.empty((0, 0), dtype=np.float32)
            gid = np.concatenate(all_gid, axis=0)
            return self._pool_groups_to_features(h5_path, gid)

        if mode == "indexed":
            gid = self._sample_indexed(n_pooled_tiles, k, rng)
            return self._pool_groups_to_features(h5_path, gid)

        raise ValueError(f"Unknown cfg.sampling={self.cfg.sampling} (expected slice|multislice|indexed)")

    def _read_chunk(self, h5_path: str, n_tiles: int, rng: np.random.Generator) -> np.ndarray:
        k = self.cfg.slide_batch_tiles

        if n_tiles <= 0:
            return np.empty((0, 0), dtype=np.float32)

        if self.cfg.magnification.lower() == "10x_pool2x2":
            return self._read_chunk_pooled(h5_path, n_tiles, rng)

        mode = self.cfg.sampling.lower()

        if mode == "slice":
            if n_tiles >= k:
                start = int(rng.integers(0, n_tiles - k + 1))
                return read_h5_features_slice_cached(h5_path, start, k)
            # fallback for small slides
            idx = self._sample_indexed(n_tiles, k, rng)
            return read_h5_features_indexed_cached(h5_path, idx)

        if mode == "multislice":
            # read multiple contiguous chunks that sum to k
            chunks = max(1, int(self.cfg.multislice_chunks))
            # chunk sizes: distribute k approximately evenly
            base = k // chunks
            rem = k % chunks
            sizes = [base + (1 if i < rem else 0) for i in range(chunks)]

            xs = []
            for sz in sizes:
                if sz <= 0:
                    continue
                if n_tiles >= sz:
                    start = int(rng.integers(0, n_tiles - sz + 1))
                    xs.append(read_h5_features_slice_cached(h5_path, start, sz))
                else:
                    # small slide fallback
                    idx = self._sample_indexed(n_tiles, sz, rng)
                    xs.append(read_h5_features_indexed_cached(h5_path, idx))

            if not xs:
                return np.empty((0, 0), dtype=np.float32)
            return np.concatenate(xs, axis=0)

        if mode == "indexed":
            idx = self._sample_indexed(n_tiles, k, rng)
            return read_h5_features_indexed_cached(h5_path, idx)

        raise ValueError(f"Unknown cfg.sampling={self.cfg.sampling} (expected slice|multislice|indexed)")

    def __getitem__(self, idx: int):
        slide_id = self.draw_to_slide[idx]
        h5_path = self.h5_paths[slide_id]
        n_tiles = self.slide_n[slide_id]
        d = self.slide_d[slide_id]
        if self.cfg.magnification.lower() == "20x":
            n_actual, d_actual = _read_h5_shape_cached_from_open(h5_path)
            if int(n_actual) != int(n_tiles) or int(d_actual) != int(d):
                self.slide_n[slide_id] = int(n_actual)
                self.slide_d[slide_id] = int(d_actual)
                n_tiles = int(n_actual)
                d = int(d_actual)
                warn_key = f"{_resolve_h5_path(h5_path)}::{int(n_tiles)}::{int(d)}"
                if warn_key not in _H5_SHAPE_MISMATCH_WARNED:
                    warnings.warn(
                        "Shape-cache mismatch detected and corrected at runtime for "
                        f"{h5_path}: using actual shape (n={n_tiles}, d={d}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _H5_SHAPE_MISMATCH_WARNED.add(warn_key)

        # worker-safe deterministic RNG per draw index
        rng = np.random.default_rng(self.cfg.seed + 1000003 * idx)

        x = self._read_chunk(h5_path, n_tiles, rng)  # (k', D) float32
        x = self._normalize_feats(x)
        xt = torch.from_numpy(x)  # float32 tensor (DataLoader pin_memory can speed H2D)

        if self.cfg.return_meta:
            meta = {
                "h5_path": h5_path,
                "slide_id": Path(h5_path).stem,
                "num_tiles_in_slide": int(n_tiles),
                "sampled_tiles": int(xt.shape[0]),
                "feat_dim": int(d),
                "sampling": self.cfg.sampling,
                "magnification": self.cfg.magnification,
            }
            return xt, meta

        return xt


def flatten_collate(batch):
    """
    Collate for SlideTileDataset:
    each item is (k_i, D) -> returns (sum_i k_i, D)
    """
    if isinstance(batch[0], tuple):
        xs, metas = zip(*batch)
        x = torch.cat(xs, dim=0)
        return x, list(metas)
    return torch.cat(batch, dim=0)


def make_sae_loader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers if num_workers > 0 else False),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        collate_fn=flatten_collate,
        drop_last=False,
    )


def load_manifest_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def get_paths_from_manifest(manifest: Dict, key: str) -> List[str]:
    split_table = manifest.get("splits") if isinstance(manifest, dict) and isinstance(manifest.get("splits"), dict) else manifest
    if key not in split_table:
        raise KeyError(f"Manifest missing key '{key}'. Available keys: {list(split_table.keys())}")
    return list(split_table[key])
