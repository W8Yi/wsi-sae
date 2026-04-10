from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "Pool2x2GeometryError": "wsi_sae.data.dataloader",
    "SlideTileConfig": "wsi_sae.data.dataloader",
    "SlideTileDataset": "wsi_sae.data.dataloader",
    "_build_pool2x2_groups_from_coords": "wsi_sae.data.dataloader",
    "_resolve_h5_path": "wsi_sae.data.dataloader",
    "get_paths_from_manifest": "wsi_sae.data.dataloader",
    "load_manifest_json": "wsi_sae.data.dataloader",
    "make_sae_loader": "wsi_sae.data.dataloader",
    "build_registry": "wsi_sae.data.layout",
    "canonical_feature_h5_dir": "wsi_sae.data.layout",
    "canonical_slide_dir": "wsi_sae.data.layout",
    "data_root": "wsi_sae.data.layout",
    "ensure_feature_indexes": "wsi_sae.data.layout",
    "extract_case_id": "wsi_sae.data.layout",
    "extract_slide_id": "wsi_sae.data.layout",
    "ingest_tcga_features": "wsi_sae.data.layout",
    "init_layout": "wsi_sae.data.layout",
    "normalize_cohort_name": "wsi_sae.data.layout",
    "parse_encoder_list": "wsi_sae.data.layout",
    "promote_links": "wsi_sae.data.layout",
    "resolve_slide_path_from_mapping": "wsi_sae.data.layout",
    "resolve_feature_path": "wsi_sae.data.layout",
    "scan_h5_health": "wsi_sae.data.layout",
    "validate_layout": "wsi_sae.data.layout",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
