#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROOT="${ROOT:-/research/projects/mllab/WSI}"
LEGACY_ROOT="${LEGACY_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
PROJECT="${PROJECT:-TCGA}"
ENCODERS="${ENCODERS:-uni2,seal}"

python -m wsi_sae.cli data init-layout \
  --root "${ROOT}" \
  --project "${PROJECT}" \
  --encoders "${ENCODERS}"

python -m wsi_sae.cli data ingest-tcga-features \
  --root "${ROOT}" \
  --legacy-root "${LEGACY_ROOT}" \
  --project "${PROJECT}" \
  --encoders "${ENCODERS}" \
  --link-mode symlink

python -m wsi_sae.cli data build-registry \
  --root "${ROOT}" \
  --project "${PROJECT}" \
  --encoders "${ENCODERS}"

python -m wsi_sae.cli data validate-layout \
  --root "${ROOT}" \
  --project "${PROJECT}" \
  --encoders "${ENCODERS}"
