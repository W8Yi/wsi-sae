from __future__ import annotations

from wsi_sae.data.dataloader import get_paths_from_manifest, load_manifest_json

from ._helpers import write_manifest


def test_legacy_manifest_schema(tmp_path):
    manifest_path = write_manifest(
        tmp_path / "legacy.json",
        train=["train_a.h5"],
        test=["test_a.h5"],
    )
    manifest = load_manifest_json(str(manifest_path))
    assert get_paths_from_manifest(manifest, "train") == ["train_a.h5"]
    assert get_paths_from_manifest(manifest, "test") == ["test_a.h5"]


def test_structured_manifest_schema(tmp_path):
    manifest_path = write_manifest(
        tmp_path / "structured.json",
        train=["train_a.h5"],
        test=["test_a.h5"],
        structured=True,
        meta={"dataset": "TCGA", "encoder": "uni2h", "feature_dim": 1536},
    )
    manifest = load_manifest_json(str(manifest_path))
    assert get_paths_from_manifest(manifest, "train") == ["train_a.h5"]
    assert get_paths_from_manifest(manifest, "test") == ["test_a.h5"]
    assert manifest["meta"]["dataset"] == "TCGA"

