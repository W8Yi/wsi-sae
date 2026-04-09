# Two-Machine Workflow

## Roles

1. Server
   Owns GPU compute, feature files, labels, manifests, checkpoints, mining outputs, and exported bundles.
2. Local PC
   Owns the raw slides, tile rendering, and `wsi-bench` for interactive inspection.

## Recommended Operating Model

1. Train and mine on the server with immutable run IDs such as `tcga_sae_advanced_v11`.
2. Store model checkpoints under `runs/<experiment>/<stage>/`.
3. Store mining outputs under `mining/<experiment>/<stage>/`.
4. Export sync-ready bundles under `exports/<experiment>/<stage>/`.
5. Sync only `exports/...` to the local PC by `rsync` or `rclone`.
6. Point local `wsi-bench` at the synced `prototype_tiles.csv` plus the local `slides_root`.

## Naming Conventions

- Experiment ID
  Use immutable names like `tcga_sae_advanced_v11`.
- Stage
  Use stable stage keys such as `relu`, `topk`, `batch_topk`, or `sdf2`.
- Slide identity
  Use `slide_key` as the cross-machine slide identifier.
- Feature identity
  Preserve the original `h5_path` in exports for provenance, but treat `slide_key` as the cross-machine lookup key.

## Artifact Contract

- `bundle_manifest.json`
  Canonical machine-to-machine metadata.
- `prototype_tiles.csv`
  Viewer-compatible export for `wsi-bench`.
- Optional copied artifacts
  `pass1_stats.json`, `pass2_top_tiles*.json`, `run_config.json`, `prototypes.npz`, `prototypes.json`, `latent_targets.json`, `probe_summary.json`.

The coordinate convention is fixed to `level0_top_left_px`.

## Cache And Versioning

- Never overwrite an old experiment/stage bundle in place when semantics change.
- Use the bundle schema version plus repo commit hash for cache invalidation.
- If you need to regenerate the same experiment with changed code, write a new run ID or stage-specific suffix.

## Sync Pattern

Example:

```bash
rsync -av \
  /common/users/wq50/wsi-sae/exports/tcga_sae_advanced_v11/ \
  /local/path/wsi-sae-exports/tcga_sae_advanced_v11/
```

## Local PC `wsi-bench` Setup

Create a local manifest entry pointing to the synced CSV and the local slide root:

```json
{
  "model_id": "tcga_sae_advanced_v11_sdf2",
  "model_name": "TCGA SAE Advanced v11 SDF2",
  "encoder": "uni2h",
  "dataset": "TCGA",
  "slides_root": "/path/to/local/slides",
  "prototype_tiles_csv": "/path/to/synced/bundle/prototype_tiles.csv",
  "top_attention_tiles_csv": "",
  "tile_size": 256
}
```

## Failure Handling

- Missing local slide
  Keep the row in `prototype_tiles.csv`; `wsi-bench` will show unresolved slide paths, which is better than losing provenance.
- Slide naming mismatch
  Fix the local lookup rule or rename local slide files so `slide_key` resolves consistently.
- Feature path mismatch
  This is expected across machines. `h5_path` is provenance, not the local rendering lookup path.

