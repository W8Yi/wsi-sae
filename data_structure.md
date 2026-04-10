# WSI + Feature Data Structure

## Overview

This document is the source-of-truth layout for your local PC and the target mirrored structure for the server.

- Local PC: the full layout lives under `/mnt/data` and contains both slides and features.
- Server: the same top-level structure lives under `/research/projects/mllab/WSI`, but `wsi_slides/` may be mostly manifest-only there.
- During the active UNI2/SEAL encoding phase, canonical server feature paths are created as links first; later they can be promoted into real moved files.

Design goals:
- Keep WSI slides and extracted features separated.
- Use stable registry/manifests as source of truth.
- Enforce consistent naming conventions for automation.

## Naming Conventions

- Root folders:
  - `wsi_slides` for WSIs
  - `wsi_features` for extracted feature files
- Project folder name is uppercase: `TCGA`
- TCGA cohort folders are uppercase (for example `ACC`, `BLCA`, `HNSC`)
- Feature encoder folder names are lowercase:
  - `seal`
  - `uni2`
  - `virchow2`
  - `gigapath`

## Directory Structure

```text
/mnt/data/
├── wsi_slides/
│   └── TCGA/
│       ├── ACC/
│       │   ├── slides/
│       │   │   └── *.svs
│       │   └── metadata/
│       │       └── slide_list.csv
│       ├── BLCA/
│       └── ...
├── wsi_features/
│   ├── seal/
│   │   └── TCGA/
│   │       ├── ACC/
│   │       │   ├── h5/
│   │       │   │   └── *.h5
│   │       │   └── index.csv
│   │       ├── BLCA/
│   │       └── ...
│   ├── uni2/
│   │   └── TCGA/
│   ├── virchow2/
│   │   └── TCGA/
│   └── gigapath/
│       └── TCGA/
├── registry/
│   ├── slides.csv
│   ├── features.csv
│   ├── mapping.csv
│   ├── missing_slides.csv
│   ├── missing_features.csv
│   └── ambiguous_slides.csv
└── metadata/
    ├── indexes/
    │   ├── tcga_wsi_index.json
    │   └── manifest_index.json
    ├── manifests/
    │   ├── gdc_manifest.json
    │   └── uuid_mapping.csv
    ├── splits/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── pan_cancer_feature_details.csv
```

## File Contracts

### Per-cohort slide list
`/mnt/data/wsi_slides/TCGA/<COHORT>/metadata/slide_list.csv`

Columns:
- `slide_id`
- `slide_filename`
- `case_id`
- `dataset`
- `cohort`
- `wsi_path`

### Per-encoder per-cohort index
`/mnt/data/wsi_features/<encoder>/TCGA/<COHORT>/index.csv`

Columns:
- `slide_id`
- `path`
- `num_tiles`
- `feature_dim`

`num_tiles` and `feature_dim` are extracted from `.h5` payloads.

### Global registries

`/mnt/data/registry/slides.csv`  
Columns: `slide_id, slide_filename, case_id, dataset, cohort, wsi_path, wsi_ext`

`/mnt/data/registry/features.csv`  
Columns: `slide_id, case_id, dataset, cohort, encoder, feature_path, feature_ext`

`/mnt/data/registry/mapping.csv`  
Canonical columns:
- `slide_id, case_id, dataset, cohort, slide_filename, wsi_path`
- `uni_path, seal_path, gene_path`
- Optional extra encoder columns when present (for example `virchow2_path`, `gigapath_path`)

## Reconciliation Snapshot (Example Local-PC State)

- Total slides: `11,682`
- Total feature files: `11,593`
- Feature records with missing slide: `1`
- Slide records with no feature: `90`
- Ambiguous slide ID matches: `0`

Detailed reports:
- `/mnt/data/registry/missing_slides.csv`
- `/mnt/data/registry/missing_features.csv`

## Notes

- Legacy roots (`/mnt/data/TCGA_slides`, `/mnt/data/TCGA_features`) were retired.
- Registry and metadata files were regenerated after each naming/layout migration.
- `wsi-sae data ...` is the canonical way to initialize, ingest, validate, and reconcile this structure.
