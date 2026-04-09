# wsi-sae

Feature-only SAE training, concept mining, and viewer export tooling for WSI embeddings.

This repo is designed for the GPU server that stores encoder features and labels, but not the raw slides.

Primary CLI:

```bash
wsi-sae train
wsi-sae mine
wsi-sae build-prototypes
wsi-sae build-targets
wsi-sae compute-percentiles
wsi-sae probe
wsi-sae export-viewer
```

See [docs/two-machine-workflow.md](/common/users/wq50/wsi-sae/docs/two-machine-workflow.md) for the server/local-PC handoff design.

