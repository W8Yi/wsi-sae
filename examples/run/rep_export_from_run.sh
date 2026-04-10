#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WSI_DATA_ROOT="${WSI_DATA_ROOT:-/research/projects/mllab/WSI}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

RUN_NAME="${RUN_NAME:?set RUN_NAME to the training run name under /common/users/wq50/wsi-sae/runs}"
EXPORT_ROOT="${EXPORT_ROOT:-/common/users/wq50/wsi-sae/exports}"
RUN_ROOT="${RUN_ROOT:-/common/users/wq50/wsi-sae/runs}"
STAGE_DIR="${STAGE_DIR:-}"
CKPT="${CKPT:-}"
SLIDES_PER_PROJECT="${SLIDES_PER_PROJECT:-200}"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
CHUNK_TILES="${CHUNK_TILES:-512}"
SEED="${SEED:-1337}"
SELECTION_SPLIT="${SELECTION_SPLIT:-train}"
EXPORT_SPLIT="${EXPORT_SPLIT:-test}"
LATENT_STRATEGIES="${LATENT_STRATEGIES:-}"
N_LATENTS="${N_LATENTS:-128}"
TOPN="${TOPN:-50}"
TOPN_BUFFER_FACTOR="${TOPN_BUFFER_FACTOR:-4.0}"
MAX_TILES_PER_SLIDE_PER_LATENT="${MAX_TILES_PER_SLIDE_PER_LATENT:-3}"
MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT="${MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT:-512}"
REQUIRE_H5_EXISTS="${REQUIRE_H5_EXISTS:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-}"
WSI_BENCH_SLIDES_ROOT="${WSI_BENCH_SLIDES_ROOT:-}"

ARGS=(
  rep-export
  --run-name "${RUN_NAME}"
  --run-root "${RUN_ROOT}"
  --export-root "${EXPORT_ROOT}"
  --seed "${SEED}"
  --selection-split "${SELECTION_SPLIT}"
  --export-split "${EXPORT_SPLIT}"
  --slides-per-project "${SLIDES_PER_PROJECT}"
  --tiles-per-slide "${TILES_PER_SLIDE}"
  --chunk-tiles "${CHUNK_TILES}"
  --topn-buffer-factor "${TOPN_BUFFER_FACTOR}"
  --max-tiles-per-slide-per-latent "${MAX_TILES_PER_SLIDE_PER_LATENT}"
  --min-distance-px-same-slide-per-latent "${MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT}"
  --n-latents "${N_LATENTS}"
  --topn "${TOPN}"
  --device "${DEVICE}"
)
if [[ -n "${STAGE_DIR}" ]]; then
  ARGS+=(--stage-dir "${STAGE_DIR}")
fi
if [[ -n "${CKPT}" ]]; then
  ARGS+=(--ckpt "${CKPT}")
fi
if [[ -n "${LATENT_STRATEGIES}" ]]; then
  ARGS+=(--latent-strategies "${LATENT_STRATEGIES}")
fi
if [[ "${REQUIRE_H5_EXISTS}" == "1" ]]; then
  ARGS+=(--require-h5-exists)
fi
if [[ -n "${MODEL_NAME}" ]]; then
  ARGS+=(--model-name "${MODEL_NAME}")
fi
if [[ -n "${WSI_BENCH_SLIDES_ROOT}" ]]; then
  ARGS+=(--wsi-bench-slides-root "${WSI_BENCH_SLIDES_ROOT}")
fi

echo "[rep-export-from-run] run_name=${RUN_NAME} export_root=${EXPORT_ROOT}" >&2
echo "[rep-export-from-run] selection_split=${SELECTION_SPLIT} export_split=${EXPORT_SPLIT} slides_per_project=${SLIDES_PER_PROJECT} tiles_per_slide=${TILES_PER_SLIDE} chunk_tiles=${CHUNK_TILES} n_latents=${N_LATENTS} topn=${TOPN} device=${DEVICE}" >&2

python -m wsi_sae.cli "${ARGS[@]}"

echo "[rep-export-from-run] done: ${EXPORT_ROOT}/${RUN_NAME}" >&2
