#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export WSI_DATA_ROOT="${WSI_DATA_ROOT:-/research/projects/mllab/WSI}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

ENCODER="${ENCODER:-uni2}"
export WSI_SAE_PREFERRED_ENCODER="${WSI_SAE_PREFERRED_ENCODER:-${ENCODER}}"
MANIFEST="${MANIFEST:-/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json}"
RUN_FAMILY="${RUN_FAMILY:-tcga}"
RUN_VERSION="${RUN_VERSION:-v1}"
RUN_ROOT="${RUN_ROOT:-/common/users/wq50/wsi-sae/runs}"
PROJECT="${PROJECT:-wsi-sae}"
MODE="${MODE:-online}"

TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
SLIDE_BATCH_TILES="${SLIDE_BATCH_TILES:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-50000}"

case "${ENCODER}" in
  uni2|seal|gigapath)
    D_IN=1536
    ;;
  virchow2)
    D_IN=2560
    ;;
  *)
    echo "[error] unsupported ENCODER='${ENCODER}'. Use one of: uni2, seal, gigapath, virchow2" >&2
    exit 1
    ;;
esac

LATENT_DIM="${LATENT_DIM:-$((8 * D_IN))}"
RUN_NAME="${RUN_NAME:-${RUN_FAMILY}_${ENCODER}_sae_relu_${RUN_VERSION}}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/${RUN_NAME}/relu}"
SHAPES_CACHE_JSON="${SHAPES_CACHE_JSON:-/common/users/wq50/wsi-sae/cache/${ENCODER}_sae_shapes_20x.json}"
COMMON_TAGS="${COMMON_TAGS:-tcga,${ENCODER},sae,relu,20x,baseline}"

if [[ "${DO_PREFLIGHT:-1}" == "1" ]]; then
  python -m wsi_sae.cli train \
    --manifest "${MANIFEST}" \
    --out_dir "${OUT_DIR}" \
    --magnification 20x \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --shapes_cache_json "${SHAPES_CACHE_JSON}" \
    --preflight_only
fi

python -m wsi_sae.cli train \
  --manifest "${MANIFEST}" \
  --out_dir "${OUT_DIR}" \
  --project "${PROJECT}" \
  --run_name "${RUN_NAME}" \
  --tags "${COMMON_TAGS}" \
  --mode "${MODE}" \
  --stage relu \
  --d_in "${D_IN}" \
  --latent_dim "${LATENT_DIM}" \
  --magnification 20x \
  --tiles_per_slide "${TILES_PER_SLIDE}" \
  --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --shapes_cache_json "${SHAPES_CACHE_JSON}" \
  --lr "${LR}" \
  --max_steps "${MAX_STEPS}" \
  --grad_clip 1.0 \
  --log_every 200 \
  --eval_every 5000 \
  --eval_batches 10 \
  --print_every 100 \
  --amp

echo "[done] relu SAE outputs at ${OUT_DIR}"
