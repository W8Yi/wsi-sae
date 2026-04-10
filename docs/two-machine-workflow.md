# Two-Machine Workflow

## Roles

1. Server
   Owns GPU compute, feature files, labels, manifests, checkpoints, mining outputs, and exported bundles.
2. Local PC
   Owns the raw slides, tile rendering, and `wsi-bench` for interactive inspection.

## Recommended Operating Model

1. Reorganize the server into the same top-level shape as the local PC, but link features first:
   `wsi-sae data init-layout`, `wsi-sae data ingest-tcga-features`, `wsi-sae data build-registry`, `wsi-sae data validate-layout`
2. Keep active encoding on the old cohort-first tree while canonical `wsi_features/...` paths are symlinks.
3. Start with a `relu` SAE baseline for each encoder, using `latent_dim = 8 x d_in`.
4. Train and mine on the server with immutable run IDs such as `tcga_sae_advanced_v11`.
5. Store model checkpoints under `runs/<experiment>/<stage>/`.
6. Store mining outputs under `mining/<experiment>/<stage>/`.
7. Export representative-latent bundles under `exports/<experiment>/representatives_<split>/`.
8. Sync only `exports/...` to the local PC by `rsync` or `rclone`.
9. Use `wsi-sae rep-materialize` on the local PC to load local feature vectors and extract actual WSI tiles from local slides.
10. Point local `wsi-bench` at the synced representative CSVs plus the local `slides_root` for interactive showcase.

## Default Training And Mining Policy

Current default encoder dimensions observed in the feature store:

- `uni2`: `1536`
- `seal`: `1536`
- `gigapath`: `1536`
- `virchow2`: `2560`

Current default relu latent widths:

- `uni2`: `12288`
- `seal`: `12288`
- `gigapath`: `12288`
- `virchow2`: `20480`

Current default split policy:

- train the SAE on the training split
- do latent selection and quick mining on training data
- do final representative/support export on test data

Example helper scripts:

- server relu training:
  `/common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh`
- server representative export:
  `/common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh`
- local-PC materialization:
  `/common/users/wq50/wsi-sae/examples/run/rep_materialize_local.sh`

## Canonical Shared Layout

Server root:

```text
/research/projects/mllab/WSI/
├── wsi_slides/
├── wsi_features/
├── registry/
└── metadata/
```

Local-PC root follows the same structure, typically under `/mnt/data/`.

The server mirrors the layout exactly, but `wsi_slides/` may be mostly manifest-only there until slide files are actually needed.

## Data Management Commands

Server bootstrap:

```bash
wsi-sae data init-layout \
  --root /research/projects/mllab/WSI \
  --project TCGA

wsi-sae data ingest-tcga-features \
  --root /research/projects/mllab/WSI \
  --legacy-root /research/projects/mllab/WSI/TCGA_features \
  --project TCGA \
  --encoders uni2,seal \
  --link-mode symlink

wsi-sae data build-registry \
  --root /research/projects/mllab/WSI \
  --project TCGA

wsi-sae data validate-layout \
  --root /research/projects/mllab/WSI \
  --project TCGA
```

After current encoding is done:

```bash
wsi-sae data promote-links \
  --root /research/projects/mllab/WSI \
  --project TCGA \
  --encoders uni2,seal
```

## Naming Conventions

- Experiment ID
  Use immutable names like `tcga_sae_advanced_v11`.
- Stage
  Use stable stage keys such as `relu`, `topk`, `batch_topk`, or `sdf2`.
- Slide identity
  Use `slide_key` as the cross-machine slide identifier.
- Feature identity
  Preserve the original `legacy_h5_path` in exports for provenance, but treat `feature_relpath` plus `slide_key` as the cross-machine lookup key.

## Artifact Contract

- `bundle_manifest.json`
  Canonical machine-to-machine metadata.
- `representative_latents.csv`
  One representative row per `latent_strategy x latent_idx x representative_method`.
- `representative_support_tiles.csv`
  Full ordered support rows per `latent_strategy x latent_idx x representative_method`.
- `latent_summary.csv`
  Per-latent summary table for fast UI ingestion and filtering in `wsi-bench`.
- `bundle_summary.json`
  Compact aggregate stats about the bundle.
- `wsi_bench_model.json`
  Ready-to-merge `sae_models.json` snippet for the local viewer.
- Local materialization output
  `wsi-sae rep-materialize` writes `materialized_rows.csv`, `materialize_summary.json`, `encoder_features.npy`, `encoder_feature_index.csv`, extracted tile images, and method-level contact sheets.

The coordinate convention is fixed to `level0_top_left_px`.

## Cache And Versioning

- Never overwrite an old experiment/stage bundle in place when semantics change.
- Use the bundle schema version plus repo commit hash for cache invalidation.
- If you need to regenerate the same experiment with changed code, write a new run ID or stage-specific suffix.

## Sync Pattern

Example:

```bash
RUN_NAME=tcga_seal_sae_relu_v1 \
bash /common/users/wq50/wsi-sae/examples/run/rep_export_from_run.sh
```

Sync the resulting export directory:

```bash
rsync -av \
  /common/users/wq50/wsi-sae/exports/tcga_seal_sae_relu_v1/ \
  /local/path/wsi-sae-exports/tcga_seal_sae_relu_v1/
```

Then on the local PC:

```bash
wsi-sae rep-materialize \
  --bundle /local/path/wsi-sae-exports/tcga_seal_sae_relu_v1/representatives_test \
  --data-root /mnt/data \
  --out-dir /mnt/data/derived/sae_tiles/tcga_seal_sae_relu_v1
```

## Local PC `wsi-bench` Setup

Create a local manifest entry pointing to the synced CSV and the local slide root, or start from the generated `wsi_bench_model.json` in the bundle:

```json
{
  "model_id": "tcga_seal_sae_relu_v1_representatives_test",
  "model_name": "tcga_seal_sae_relu_v1 representatives test",
  "encoder": "seal",
  "dataset": "TCGA",
  "slides_root": "/path/to/local/slides",
  "representative_latents_csv": "/path/to/synced/bundle/representative_latents.csv",
  "representative_support_tiles_csv": "/path/to/synced/bundle/representative_support_tiles.csv",
  "latent_summary_csv": "/path/to/synced/bundle/latent_summary.csv",
  "bundle_summary_json": "/path/to/synced/bundle/bundle_summary.json",
  "tile_size": 256
}
```

## Failure Handling

- Missing local slide
  Keep the row in `representative_support_tiles.csv`; `wsi-bench` will show unresolved slide paths, which is better than losing provenance.
- Missing local slide during `rep-materialize`
  The row is preserved in `materialized_rows.csv` with status `missing_slide`.
- Slide naming mismatch
  Fix the local lookup rule or rename local slide files so `slide_key` resolves consistently.
- Feature path mismatch
  This is expected across machines. `legacy_h5_path` is provenance, while `feature_relpath` is the local lookup path.
