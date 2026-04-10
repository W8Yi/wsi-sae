#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/mnt/data}"
BUNDLE_DIR="${BUNDLE_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
TILE_SIZE="${TILE_SIZE:-256}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"
LIMIT="${LIMIT:-0}"
SKIP_CONTACT_SHEETS="${SKIP_CONTACT_SHEETS:-0}"

if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "[error] set BUNDLE_DIR to the synced representative bundle directory" >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  echo "[error] set OUT_DIR to the materialization output directory" >&2
  exit 1
fi

echo "[rep-materialize-local] bundle=${BUNDLE_DIR}" >&2
echo "[rep-materialize-local] data_root=${DATA_ROOT}" >&2
echo "[rep-materialize-local] out_dir=${OUT_DIR} tile_size=${TILE_SIZE} image_format=${IMAGE_FORMAT} limit=${LIMIT} skip_contact_sheets=${SKIP_CONTACT_SHEETS}" >&2

ARGS=(
  rep-materialize
  --bundle "${BUNDLE_DIR}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --tile-size "${TILE_SIZE}" \
  --image-format "${IMAGE_FORMAT}" \
  --limit "${LIMIT}"
)
if [[ "${SKIP_CONTACT_SHEETS}" == "1" ]]; then
  ARGS+=(--skip-contact-sheets)
fi

python -m wsi_sae.cli "${ARGS[@]}"

echo "[rep-materialize-local] done: ${OUT_DIR}" >&2
