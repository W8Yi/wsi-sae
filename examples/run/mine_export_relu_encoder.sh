#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WSI_DATA_ROOT="${WSI_DATA_ROOT:-/research/projects/mllab/WSI}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

ENCODER="${ENCODER:-uni2}"
export WSI_SAE_PREFERRED_ENCODER="${WSI_SAE_PREFERRED_ENCODER:-${ENCODER}}"
RUN_FAMILY="${RUN_FAMILY:-tcga}"
RUN_VERSION="${RUN_VERSION:-v1}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${RUN_FAMILY}_${ENCODER}_sae_relu_${RUN_VERSION}}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-/common/users/wq50/wsi-sae/runs/${TRAIN_RUN_NAME}/relu}"

case "${ENCODER}" in
  uni2|seal|gigapath)
    D_IN=1536
    LATENT_DIM_DEFAULT=12288
    ;;
  virchow2)
    D_IN=2560
    LATENT_DIM_DEFAULT=20480
    ;;
  *)
    echo "[error] unsupported ENCODER='${ENCODER}'. Use one of: uni2, seal, gigapath, virchow2" >&2
    exit 1
    ;;
esac

LATENT_DIM="${LATENT_DIM:-${LATENT_DIM_DEFAULT}}"
CKPT="${CKPT:-${TRAIN_RUN_ROOT}/relu_final.pt}"

MINING_ROOT_BASE="${MINING_ROOT_BASE:-/common/users/wq50/wsi-sae/mining}"
EXPORT_ROOT="${EXPORT_ROOT:-/common/users/wq50/wsi-sae/exports}"

# Default policy:
# - train/discovery mining on train split
# - final exported representative tiles on test split
TRAIN_INDEX_JSON="${TRAIN_INDEX_JSON:-metadata/indexes/manifest_index_train.json}"
TEST_INDEX_JSON="${TEST_INDEX_JSON:-metadata/indexes/manifest_index_test.json}"

DISCOVERY_RUN_NAME="${DISCOVERY_RUN_NAME:-${TRAIN_RUN_NAME}_train_discovery}"
EXPORT_RUN_NAME="${EXPORT_RUN_NAME:-${TRAIN_RUN_NAME}_test_export}"

SLIDES_PER_PROJECT="${SLIDES_PER_PROJECT:-200}"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
CHUNK_TILES="${CHUNK_TILES:-512}"
N_LATENTS="${N_LATENTS:-128}"
TOPN="${TOPN:-50}"
DEVICE="${DEVICE:-cuda}"

DO_TRAIN_DISCOVERY="${DO_TRAIN_DISCOVERY:-1}"
DO_TEST_EXPORT="${DO_TEST_EXPORT:-1}"

mine_one () {
  local index_json="$1"
  local run_name="$2"
  python -m wsi_sae.cli mine \
    --out_root "${MINING_ROOT_BASE}" \
    --run_name "${run_name}" \
    --mode both \
    --ckpt "${CKPT}" \
    --stage relu \
    --d_in "${D_IN}" \
    --latent_dim "${LATENT_DIM}" \
    --index_json "${index_json}" \
    --slides_per_project "${SLIDES_PER_PROJECT}" \
    --require_h5_exists \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --chunk_tiles "${CHUNK_TILES}" \
    --select_strategy top_activation \
    --n_latents "${N_LATENTS}" \
    --topn "${TOPN}" \
    --device "${DEVICE}"
}

export_one () {
  local run_name="$1"
  local export_stage="$2"
  local pass2_json
  pass2_json="$(ls -t "${MINING_ROOT_BASE}/${run_name}"/pass2_top_tiles_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${pass2_json}" ]]; then
    echo "[error] no pass2 json found for ${run_name}" >&2
    exit 1
  fi
  python -m wsi_sae.cli export-viewer \
    --pass2-json "${pass2_json}" \
    --out-dir "${EXPORT_ROOT}/${TRAIN_RUN_NAME}/${export_stage}" \
    --run-config "${TRAIN_RUN_ROOT}/run_config.json" \
    --model-id "${TRAIN_RUN_NAME}_${export_stage}" \
    --model-name "${TRAIN_RUN_NAME} ${export_stage}" \
    --encoder "${ENCODER}" \
    --dataset "TCGA" \
    --experiment-name "${TRAIN_RUN_NAME}" \
    --stage "${export_stage}"
}

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing relu checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ "${DO_TRAIN_DISCOVERY}" == "1" ]]; then
  mine_one "${TRAIN_INDEX_JSON}" "${DISCOVERY_RUN_NAME}"
fi

if [[ "${DO_TEST_EXPORT}" == "1" ]]; then
  mine_one "${TEST_INDEX_JSON}" "${EXPORT_RUN_NAME}"
  export_one "${EXPORT_RUN_NAME}" "relu_test_export"
fi

echo "[done] relu mining outputs at ${MINING_ROOT_BASE}"
echo "[done] exported bundles at ${EXPORT_ROOT}/${TRAIN_RUN_NAME}"
