#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WSI_DATA_ROOT="${WSI_DATA_ROOT:-/research/projects/mllab/WSI}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

RUN_NAME="${RUN_NAME:?set RUN_NAME to the training run name under /common/users/wq50/wsi-sae/runs}"
EXPORT_ROOT="${EXPORT_ROOT:-/common/users/wq50/wsi-sae/exports}"
SLIDES_PER_PROJECT="${SLIDES_PER_PROJECT:-200}"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
CHUNK_TILES="${CHUNK_TILES:-512}"
N_LATENTS="${N_LATENTS:-128}"
TOPN="${TOPN:-50}"
DEVICE="${DEVICE:-cuda}"
WSI_BENCH_SLIDES_ROOT="${WSI_BENCH_SLIDES_ROOT:-}"

ARGS=(
  rep-export
  --run-name "${RUN_NAME}"
  --export-root "${EXPORT_ROOT}"
  --slides-per-project "${SLIDES_PER_PROJECT}"
  --tiles-per-slide "${TILES_PER_SLIDE}"
  --chunk-tiles "${CHUNK_TILES}"
  --n-latents "${N_LATENTS}"
  --topn "${TOPN}"
  --device "${DEVICE}"
)
if [[ -n "${WSI_BENCH_SLIDES_ROOT}" ]]; then
  ARGS+=(--wsi-bench-slides-root "${WSI_BENCH_SLIDES_ROOT}")
fi

python -m wsi_sae.cli "${ARGS[@]}"
