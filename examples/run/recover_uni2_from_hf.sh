#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SLIDE_ID="${SLIDE_ID:-}"
PROJECT_DIR="${PROJECT_DIR:-}"
OUT_ROOT="${OUT_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
TMP_ROOT="${TMP_ROOT:-/tmp/wsi_sae_hf_recover}"
HF_REPO="${HF_REPO:-W8Yi/tcga-wsi-uni2h-features}"

if [[ -z "${SLIDE_ID}" ]]; then
  echo "[error] set SLIDE_ID, for example TCGA-VM-A8CH-01Z-00-DX1" >&2
  exit 1
fi

if [[ -z "${PROJECT_DIR}" ]]; then
  echo "[error] set PROJECT_DIR, for example TCGA-LGG" >&2
  exit 1
fi

HF_PATH="${PROJECT_DIR}/features/${SLIDE_ID}.h5"
TMP_DIR="${TMP_ROOT}/${PROJECT_DIR}"
RAW_PATH="${TMP_DIR}/raw/${PROJECT_DIR}/features/${SLIDE_ID}.h5"
UNCOMP_PATH="${TMP_DIR}/uncompressed/${SLIDE_ID}.h5"
DEST_PATH="${OUT_ROOT}/${PROJECT_DIR}/features_uni2/${SLIDE_ID}.h5"
BACKUP_PATH="${DEST_PATH}.broken.$(date +%Y%m%d-%H%M%S)"

mkdir -p "${TMP_DIR}/raw" "${TMP_DIR}/uncompressed" "$(dirname "${DEST_PATH}")"

hf download --repo-type dataset "${HF_REPO}" "${HF_PATH}" --local-dir "${TMP_DIR}/raw"

/common/users/wq50/envs/pace/bin/python - <<'PY' "${RAW_PATH}" "${UNCOMP_PATH}"
from pathlib import Path
import sys
import h5py

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

def copy_item(src_group, dst_group, name):
    obj = src_group[name]
    if isinstance(obj, h5py.Dataset):
        dst_ds = dst_group.create_dataset(name, data=obj[...])
        for k, v in obj.attrs.items():
            dst_ds.attrs[k] = v
    elif isinstance(obj, h5py.Group):
        child = dst_group.create_group(name)
        for k, v in obj.attrs.items():
            child.attrs[k] = v
        for child_name in obj.keys():
            copy_item(obj, child, child_name)

with h5py.File(src, "r") as fin:
    with h5py.File(dst, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for name in fin.keys():
            copy_item(fin, fout, name)

with h5py.File(dst, "r") as fcheck:
    ds = fcheck["features"]
    print("validated_uncompressed_shape", ds.shape, "compression", ds.compression)
PY

if [[ -f "${DEST_PATH}" ]]; then
  mv "${DEST_PATH}" "${BACKUP_PATH}"
  echo "[backup] ${BACKUP_PATH}"
fi

mv "${UNCOMP_PATH}" "${DEST_PATH}"
echo "[recovered] ${DEST_PATH}"
