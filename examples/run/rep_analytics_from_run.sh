#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WSI_DATA_ROOT="${WSI_DATA_ROOT:-/research/projects/mllab/WSI}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

RUN_NAME="${RUN_NAME:?set RUN_NAME to the training run name under /common/users/wq50/wsi-sae/runs}"
EXPORT_ROOT="${EXPORT_ROOT:-/common/users/wq50/wsi-sae/exports}"
RUN_ROOT="${RUN_ROOT:-/common/users/wq50/wsi-sae/runs}"
BUNDLE_DIR="${BUNDLE_DIR:-}"
SPLIT="${SPLIT:-}"
STAGE_DIR="${STAGE_DIR:-}"
CKPT="${CKPT:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"
CHUNK_TILES="${CHUNK_TILES:-512}"
MAGNIFICATION="${MAGNIFICATION:-}"
LABELS_CSV="${LABELS_CSV:-}"
LABEL_COLUMNS="${LABEL_COLUMNS:-}"
UMAP_SOURCE="${UMAP_SOURCE:-decoder}"
HIST_BINS="${HIST_BINS:-32}"

ARGS=(
  rep-analytics
  --run-name "${RUN_NAME}"
  --run-root "${RUN_ROOT}"
  --export-root "${EXPORT_ROOT}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --chunk-tiles "${CHUNK_TILES}"
  --umap-source "${UMAP_SOURCE}"
  --hist-bins "${HIST_BINS}"
)
if [[ -n "${BUNDLE_DIR}" ]]; then
  ARGS+=(--bundle-dir "${BUNDLE_DIR}")
fi
if [[ -n "${SPLIT}" ]]; then
  ARGS+=(--split "${SPLIT}")
fi
if [[ -n "${STAGE_DIR}" ]]; then
  ARGS+=(--stage-dir "${STAGE_DIR}")
fi
if [[ -n "${CKPT}" ]]; then
  ARGS+=(--ckpt "${CKPT}")
fi
if [[ -n "${MAGNIFICATION}" ]]; then
  ARGS+=(--magnification "${MAGNIFICATION}")
fi
if [[ -n "${LABELS_CSV}" ]]; then
  ARGS+=(--labels-csv "${LABELS_CSV}")
fi
if [[ -n "${LABEL_COLUMNS}" ]]; then
  ARGS+=(--label-columns "${LABEL_COLUMNS}")
fi

echo "[rep-analytics-from-run] run_name=${RUN_NAME} export_root=${EXPORT_ROOT} split=${SPLIT:-<bundle-default>} device=${DEVICE} chunk_tiles=${CHUNK_TILES}" >&2
if [[ -n "${LABELS_CSV}" ]]; then
  echo "[rep-analytics-from-run] labels_csv=${LABELS_CSV} label_columns=${LABEL_COLUMNS:-<all>}" >&2
fi

python -m wsi_sae.cli "${ARGS[@]}"

echo "[rep-analytics-from-run] done: ${EXPORT_ROOT}/${RUN_NAME}" >&2
