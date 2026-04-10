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

if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "[error] set BUNDLE_DIR to the synced representative bundle directory" >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  echo "[error] set OUT_DIR to the materialization output directory" >&2
  exit 1
fi

python -m wsi_sae.cli rep-materialize \
  --bundle "${BUNDLE_DIR}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --tile-size "${TILE_SIZE}" \
  --image-format "${IMAGE_FORMAT}" \
  --limit "${LIMIT}"
