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
  - use training data for latent selection and quick mining
  - use test data for final representative exports and local tile materialization

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

## 4. Export Representative Latents And Support Tiles On The Server

Recommended one-command workflow:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_export_from_run.sh
```

This is the preferred path now. You only provide the training run name, and `wsi-sae` infers:

- the stage directory under `runs/<run_name>/`
- the checkpoint to mine from
- the encoder
- the manifest path
- `d_in`, `latent_dim`, and magnification
- the default split policy:
  - `train` for latent selection
  - `test` for final representative/support bundle export

The underlying CLI is:

```bash
wsi-sae rep-export --run-name tcga_seal_sae_relu_v1
```

The helper script is just a thin wrapper around that CLI and now exposes the most useful controls through environment variables.

For each selected latent, the export writes one representative row for every method:

- `max_activation`
- `median_activation`
- `diverse_support`
- `slide_spread`

and keeps the full ordered support rows for each method in the bundle.

### Common controls for `rep-export`

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
SELECTION_SPLIT=train \
EXPORT_SPLIT=test \
SLIDES_PER_PROJECT=200 \
TILES_PER_SLIDE=2048 \
CHUNK_TILES=512 \
N_LATENTS=128 \
TOPN=50 \
DEVICE=cuda \
bash /common/users/wq50/wsi-sae/examples/run/rep_export_from_run.sh
```

What these controls do:

- `RUN_NAME`
  The training run to mine from under `runs/<run_name>/`.
- `RUN_ROOT`
  Override the run directory root. Default: `/common/users/wq50/wsi-sae/runs`.
- `EXPORT_ROOT`
  Override where representative bundles are written. Default: `/common/users/wq50/wsi-sae/exports`.
- `STAGE_DIR`
  Force a specific stage subdirectory under the run if auto-detection is ambiguous.
- `CKPT`
  Override the checkpoint file used for export.
- `SEED`
  Sampling/random seed for split resolution and mining.
- `SELECTION_SPLIT`
  Split used to choose representative latents. Default: `train`.
- `EXPORT_SPLIT`
  Split used to mine support/top tiles and build the exported bundle. Default: `test`.
- `LATENT_STRATEGIES`
  Comma-separated latent-selection strategies such as `top_activation,top_variance,top_sparsity`.
  For `sdf2` runs, `sdf_parent_balanced` is also allowed.
- `SLIDES_PER_PROJECT`
  Cap how many slides per cohort/project are sampled from the manifest for selection/export.
  Use `-1` in the raw CLI if you want all readable slides.
- `TILES_PER_SLIDE`
  Number of tiles sampled per slide during pass1 latent-stat collection.
- `CHUNK_TILES`
  Chunk size for forward passes over H5 features.
- `N_LATENTS`
  Number of latents to export per latent-selection strategy.
- `TOPN`
  Final number of support rows kept per latent per representative method.
- `TOPN_BUFFER_FACTOR`
  Internal oversampling factor before diversity filtering. Higher values help when many rows are removed by per-slide caps.
- `MAX_TILES_PER_SLIDE_PER_LATENT`
  Maximum exported support rows from one slide for the same latent. Default: `3`.
- `MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT`
  Minimum coordinate spacing between support rows from the same slide for one latent.
- `REQUIRE_H5_EXISTS`
  Set to `1` to require resolved H5 files to exist before mining.
- `DEVICE`
  Usually `cuda` or `cpu`.
- `MODEL_NAME`
  Optional override for the emitted `wsi_bench_model.json` entry.
- `WSI_BENCH_SLIDES_ROOT`
  Optional placeholder path written into `wsi_bench_model.json`.

### Changing split policy

The standard policy is:

- `SELECTION_SPLIT=train`
- `EXPORT_SPLIT=test`

If you want a train-export debug bundle instead:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
SELECTION_SPLIT=train \
EXPORT_SPLIT=train \
bash /common/users/wq50/wsi-sae/examples/run/rep_export_from_run.sh
```

If you want full CLI control instead of the wrapper:

```bash
wsi-sae rep-export \
  --run-name tcga_seal_sae_relu_v1 \
  --selection-split train \
  --export-split train \
  --n-latents 128 \
  --topn 50 \
  --tiles-per-slide 2048 \
  --chunk-tiles 512
```

### Encoder-specific convenience launchers

If you still prefer the encoder-based wrapper, this remains available:

```bash
/common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `uni2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=uni2 \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `seal`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=seal \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `gigapath`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=gigapath \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

### `virchow2`

```bash
CUDA_VISIBLE_DEVICES=0 ENCODER=virchow2 \
bash /common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
```

The final exported viewer bundle will be written under:

```text
/common/users/wq50/wsi-sae/exports/<train_run_name>/representatives_test
```

The representative bundle now includes:

- `bundle_manifest.json`
- `representative_latents.csv`
- `representative_support_tiles.csv`
- `latent_summary.csv`
- `bundle_summary.json`
- `wsi_bench_model.json`

## 5. Build Plot-Ready Analytics On The Server

Default analytics launcher:

```bash
/common/users/wq50/wsi-sae/examples/run/rep_analytics_from_run.sh
```

Example:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_analytics_from_run.sh
```

The underlying CLI is:

```bash
wsi-sae rep-analytics --run-name tcga_seal_sae_relu_v1
```

The wrapper also exposes the common analytics controls through environment variables.

This writes plot-ready analytics under:

```text
/common/users/wq50/wsi-sae/exports/<train_run_name>/analytics_test
```

Main analytics files:

- `plot_manifest.json`
- `all_latent_metrics.csv`
- `selected_latent_slide_stats.csv`
- `selected_latent_histograms.json`
- `cohort_enrichment.csv`
- `latent_umap.csv`
- `analytics_summary.json`
- optional `case_label_enrichment.csv`

Use `LABELS_CSV` and optional `LABEL_COLUMNS` when you want case-level enrichment output:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
LABELS_CSV=/path/to/case_labels.csv \
LABEL_COLUMNS=subtype,site \
bash /common/users/wq50/wsi-sae/examples/run/rep_analytics_from_run.sh
```

### Common controls for `rep-analytics`

- `RUN_NAME`
  Training run name used to find the checkpoint and default representative bundle.
- `RUN_ROOT`
  Override the run directory root.
- `EXPORT_ROOT`
  Override the root that contains representative and analytics bundles.
- `BUNDLE_DIR`
  Explicit representative bundle directory. Useful if you want analytics from a non-default location.
- `SPLIT`
  Override the split suffix. If omitted, analytics uses the split recorded in the representative bundle, usually `test`.
- `STAGE_DIR`
  Force a specific stage directory under the run.
- `CKPT`
  Override the SAE checkpoint used for analytics.
- `DEVICE`
  Usually `cuda` or `cpu`.
- `SEED`
  Random seed for stable analytics/UMAP behavior.
- `CHUNK_TILES`
  Chunk size for dense feature scanning during analytics.
- `MAGNIFICATION`
  Override the run magnification if needed.
- `LABELS_CSV`
  Optional case-level labels table keyed by `case_id`.
- `LABEL_COLUMNS`
  Optional comma-separated subset of label columns to analyze.
- `UMAP_SOURCE`
  Currently only `decoder` is supported in v1.
- `HIST_BINS`
  Number of bins in the selected-latent histogram export.

Example with a train representative bundle and custom labels:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
SPLIT=train \
LABELS_CSV=/path/to/case_labels.csv \
LABEL_COLUMNS=subtype,site \
HIST_BINS=48 \
bash /common/users/wq50/wsi-sae/examples/run/rep_analytics_from_run.sh
```

## 6. Sync Bundle To The Local PC

Example:

```bash
rsync -av \
  /common/users/wq50/wsi-sae/exports/tcga_uni2_sae_relu_v1/representatives_test/ \
  /local/path/wsi-sae-exports/tcga_uni2_sae_relu_v1/representatives_test/
```

## 7. Materialize Encoder Features And Tiles On The Local PC

Default local-PC launcher:

```bash
/common/users/wq50/wsi-sae/examples/run/rep_materialize_local.sh
```

### Example: `uni2`

```bash
DATA_ROOT=/mnt/data \
BUNDLE_DIR=/local/path/wsi-sae-exports/tcga_uni2_sae_relu_v1/representatives_test \
OUT_DIR=/mnt/data/derived/sae_tiles/tcga_uni2_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_materialize_local.sh
```

### Example: `virchow2`

```bash
DATA_ROOT=/mnt/data \
BUNDLE_DIR=/local/path/wsi-sae-exports/tcga_virchow2_sae_relu_v1/representatives_test \
OUT_DIR=/mnt/data/derived/sae_tiles/tcga_virchow2_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_materialize_local.sh
```

This writes:

- `materialized_rows.csv`
- `materialize_summary.json`
- `encoder_features.npy`
- `encoder_feature_index.csv`
- extracted tile images
- latent strategy / representative method contact sheets

### Common controls for `rep-materialize`

- `DATA_ROOT`
  Canonical local data root. Must contain `registry/`, `wsi_features/`, and `wsi_slides/`.
- `BUNDLE_DIR`
  Synced representative bundle directory or a directory that contains `bundle_manifest.json`.
- `OUT_DIR`
  Local output directory for materialized feature vectors, tile images, and summaries.
- `TILE_SIZE`
  Tile crop size in level-0 pixels.
- `IMAGE_FORMAT`
  Output image format. Supported: `png`, `jpg`.
- `LIMIT`
  Optional cap on how many rows are materialized. Useful for debugging.
- `SKIP_CONTACT_SHEETS`
  Set to `1` to skip contact-sheet generation.

Example debug run on a small subset:

```bash
DATA_ROOT=/mnt/data \
BUNDLE_DIR=/local/path/wsi-sae-exports/tcga_uni2_sae_relu_v1/representatives_test \
OUT_DIR=/mnt/data/derived/sae_tiles/debug_uni2 \
TILE_SIZE=256 \
IMAGE_FORMAT=png \
LIMIT=20 \
SKIP_CONTACT_SHEETS=1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_materialize_local.sh
```

## 8. Use With `wsi-bench`

After syncing the bundle, `wsi-bench` can point directly at the synced `representative_latents.csv` and `representative_support_tiles.csv` for interactive browsing.

The viewer now understands:

- `latent_strategy`
- `representative_method`

so you can compare the same latent under different selection strategies and representative-tile methods without recomputing those rankings locally.

You can also copy the generated `wsi_bench_model.json` snippet from the bundle and merge it into your local `wsi-bench/config/sae_models.json`.

Use `wsi-sae rep-materialize` when you want saved representative tile images, encoder feature vectors, and contact sheets on the local PC.
