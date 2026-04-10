# ReLU SAE Tutorial

This tutorial is the current default workflow for `wsi-sae`.

It assumes:

- server root: `/research/projects/mllab/WSI`
- local-PC root: `/mnt/data`
- first-pass encoders: `uni2`, `seal`, `gigapath`, `virchow2`
- default SAE stage: `relu`
- default latent width: `8 x d_in`
- default split policy:
  - train the SAE on the training split
  - use training data for latent discovery and quick mining
  - use test data for final concept exports and representative tile extraction

The provided launcher scripts automatically export `WSI_SAE_PREFERRED_ENCODER=<encoder>` so stale generic manifest paths resolve to the intended encoder family.

## 1. Server Setup

If you have not already reorganized the server feature store:

```bash
bash /common/users/wq50/wsi-sae/examples/run/reorganize_tcga_layout.sh
```

This creates the canonical layout under `/research/projects/mllab/WSI` and links current TCGA feature files into:

```text
/research/projects/mllab/WSI/wsi_features/<encoder>/TCGA/<COHORT>/h5/*.h5
```

## 2. Encoder Dimensions

Current verified input dimensions:

- `uni2`: `1536`
- `seal`: `1536`
- `gigapath`: `1536`
- `virchow2`: `2560`

Current default latent widths:

- `uni2`: `12288`
- `seal`: `12288`
- `gigapath`: `12288`
- `virchow2`: `20480`

Optional health scan before training:

```bash
PYTHONPATH=/common/users/wq50/wsi-sae/src \
python -m wsi_sae.cli data scan-h5-health \
  --root /research/projects/mllab/WSI \
  --project TCGA \
  --encoders uni2,seal,gigapath,virchow2 \
  --source legacy \
  --legacy-root /research/projects/mllab/WSI/TCGA_features \
  --out-dir /common/users/wq50/wsi-sae/reports/h5_health
```

If a UNI2 file is unreadable and the Hugging Face source copy is valid, use:

```bash
SLIDE_ID=TCGA-VM-A8CH-01Z-00-DX1 \
PROJECT_DIR=TCGA-LGG \
bash /common/users/wq50/wsi-sae/examples/run/recover_uni2_from_hf.sh
```

This recovery script:

- downloads the H5 from `W8Yi/tcga-wsi-uni2h-features`
- rewrites it as an uncompressed H5
- backs up the existing local file if present
- installs the uncompressed replacement into `features_uni2`

## 3. Train ReLU SAE On The Server

Default launcher:

```bash
/common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

Default manifest used below:

```bash
/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json
```

### `uni2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=uni2 \
MANIFEST=/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json \
bash /common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

### `seal`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=seal \
MANIFEST=/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json \
bash /common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

### `gigapath`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=gigapath \
MANIFEST=/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json \
bash /common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

### `virchow2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=virchow2 \
MANIFEST=/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json \
bash /common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

## 4. Mine Concepts And Export Bundles On The Server

Default launcher:

```bash
/common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

Recommended policy:

- `TRAIN_INDEX_JSON`: training split index for discovery
- `TEST_INDEX_JSON`: test split index for final exported bundle

If your filenames differ, point these env vars at the real index JSON paths.

### `uni2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=uni2 \
TRAIN_INDEX_JSON=metadata/indexes/manifest_index_train.json \
TEST_INDEX_JSON=metadata/indexes/manifest_index_test.json \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `seal`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=seal \
TRAIN_INDEX_JSON=metadata/indexes/manifest_index_train.json \
TEST_INDEX_JSON=metadata/indexes/manifest_index_test.json \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `gigapath`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=gigapath \
TRAIN_INDEX_JSON=metadata/indexes/manifest_index_train.json \
TEST_INDEX_JSON=metadata/indexes/manifest_index_test.json \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `virchow2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=virchow2 \
TRAIN_INDEX_JSON=metadata/indexes/manifest_index_train.json \
TEST_INDEX_JSON=metadata/indexes/manifest_index_test.json \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

The final exported viewer bundle will be written under:

```text
/common/users/wq50/wsi-sae/exports/<train_run_name>/relu_test_export
```

## 5. Sync Bundle To The Local PC

Example:

```bash
rsync -av \
  /common/users/wq50/wsi-sae/exports/tcga_uni2_sae_relu_v1/relu_test_export/ \
  /local/path/wsi-sae-exports/tcga_uni2_sae_relu_v1/relu_test_export/
```

## 6. Extract Tiles On The Local PC

Default local-PC launcher:

```bash
/common/users/wq50/wsi-sae/examples/run/extract_tiles_local.sh
```

### Example: `uni2`

```bash
DATA_ROOT=/mnt/data \
BUNDLE_DIR=/local/path/wsi-sae-exports/tcga_uni2_sae_relu_v1/relu_test_export \
OUT_DIR=/mnt/data/derived/sae_tiles/tcga_uni2_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/extract_tiles_local.sh
```

### Example: `virchow2`

```bash
DATA_ROOT=/mnt/data \
BUNDLE_DIR=/local/path/wsi-sae-exports/tcga_virchow2_sae_relu_v1/relu_test_export \
OUT_DIR=/mnt/data/derived/sae_tiles/tcga_virchow2_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/extract_tiles_local.sh
```

This writes:

- `extracted_tiles.csv`
- `extract_summary.json`
- extracted tile images
- latent-level contact sheets

## 7. Use With `wsi-bench`

After syncing the bundle, `wsi-bench` can point directly at the synced `prototype_tiles.csv` for interactive browsing.

Use `wsi-sae extract-tiles` when you want a saved batch of representative tiles or contact sheets on the local PC.
