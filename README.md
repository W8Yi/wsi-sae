# wsi-sae

SAE training, concept mining, data-layout management, viewer export, and local tile extraction for WSI embeddings.

This repo is designed to run on both of your machines:

- Server: owns GPUs, encoder features, labels, manifests, checkpoints, mining outputs, and exported bundles.
- Local PC: owns the real slides and uses the same `wsi-sae` bundle contract to extract representative tiles or other slide-backed outputs.

Primary CLI:

```bash
wsi-sae train
wsi-sae mine
wsi-sae build-prototypes
wsi-sae build-targets
wsi-sae compute-percentiles
wsi-sae probe
wsi-sae export-viewer
wsi-sae data ...
wsi-sae extract-tiles
```

Key workflows:

```bash
wsi-sae data init-layout --root /research/projects/mllab/WSI --project TCGA
wsi-sae data ingest-tcga-features --root /research/projects/mllab/WSI --legacy-root /research/projects/mllab/WSI/TCGA_features --encoders uni2,seal --link-mode symlink
wsi-sae data build-registry --root /research/projects/mllab/WSI --project TCGA
wsi-sae data validate-layout --root /research/projects/mllab/WSI --project TCGA
wsi-sae data scan-h5-health --root /research/projects/mllab/WSI --project TCGA --encoders uni2,seal,gigapath,virchow2 --source legacy --legacy-root /research/projects/mllab/WSI/TCGA_features --out-dir /common/users/wq50/wsi-sae/reports/h5_health
```

Later, after encoding is finished:

```bash
wsi-sae data promote-links --root /research/projects/mllab/WSI --project TCGA --encoders uni2,seal
```

For the local PC:

```bash
wsi-sae extract-tiles --bundle /path/to/exported_bundle --data-root /mnt/data --out-dir /path/to/output
```

See [two-machine-workflow.md](/common/users/wq50/wsi-sae/docs/two-machine-workflow.md) for the server/local-PC handoff design, [relu-tutorial.md](/common/users/wq50/wsi-sae/docs/relu-tutorial.md) for the current step-by-step relu workflow, and [data_structure.md](/common/users/wq50/wsi-sae/data_structure.md) for the canonical shared layout contract.

## Current Defaults

Current default policy for this repo:

- Start with a `relu` SAE for every encoder as the first baseline.
- Use `latent_dim = 8 x d_in` per encoder.
- Current verified encoder input dimensions:
  - `uni2`: `1536`
  - `seal`: `1536`
  - `gigapath`: `1536`
  - `virchow2`: `2560`
- That gives default latent widths:
  - `uni2`: `12288`
  - `seal`: `12288`
  - `gigapath`: `12288`
  - `virchow2`: `20480`

Default interpretation workflow:

- Train the SAE on the training split.
- Use training data for latent discovery and quick concept debugging.
- Use test data for final concept mining outputs, representative tile extraction, and showcase bundles.

A simple relu baseline launcher is available at:

```bash
/common/users/wq50/wsi-sae/examples/run/train_relu_sae_encoder.sh
```

Default relu mining/export and local extraction launchers are also available:

```bash
/common/users/wq50/wsi-sae/examples/run/mine_export_relu_encoder.sh
/common/users/wq50/wsi-sae/examples/run/extract_tiles_local.sh
/common/users/wq50/wsi-sae/examples/run/recover_uni2_from_hf.sh
```
