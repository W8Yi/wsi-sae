from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wsi_sae.data.dataloader import _resolve_h5_path

from ._helpers import run_cli, write_feature_h5


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_data_ingest_registry_validate_and_promote(tmp_path):
    canonical_root = tmp_path / "data_root"
    legacy_root = tmp_path / "legacy"

    acc_uni2 = write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_uni2" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_seal" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 0, 0, 0], [0, 1, 0, 0]],
    )
    write_feature_h5(
        legacy_root / "TCGA-BLCA" / "features_uni2" / "TCGA-CD-5678-01Z-00-DX1.h5",
        [[2, 2, 2, 2], [3, 3, 3, 3]],
    )

    run_cli(["data", "init-layout", "--root", str(canonical_root), "--project", "TCGA"])
    run_cli(
        [
            "data",
            "ingest-tcga-features",
            "--root",
            str(canonical_root),
            "--legacy-root",
            str(legacy_root),
            "--project",
            "TCGA",
            "--encoders",
            "uni2,seal",
            "--link-mode",
            "symlink",
        ]
    )

    canonical_uni2 = canonical_root / "wsi_features" / "uni2" / "TCGA" / "ACC" / "h5" / acc_uni2.name
    assert canonical_uni2.is_symlink()
    assert canonical_uni2.resolve() == acc_uni2.resolve()
    index_rows = _read_csv(canonical_root / "wsi_features" / "uni2" / "TCGA" / "ACC" / "index.csv")
    assert index_rows[0]["feature_dim"] == "4"

    blca_slide_list = canonical_root / "wsi_slides" / "TCGA" / "BLCA" / "metadata" / "slide_list.csv"
    blca_rows = _read_csv(blca_slide_list)
    _write_csv(blca_slide_list, [], list(blca_rows[0].keys()))

    hnsc_slide_list = canonical_root / "wsi_slides" / "TCGA" / "HNSC" / "metadata" / "slide_list.csv"
    _write_csv(
        hnsc_slide_list,
        [
            {
                "slide_id": "TCGA-EF-9999-01Z-00-DX1",
                "slide_filename": "",
                "case_id": "TCGA-EF-9999",
                "dataset": "TCGA",
                "cohort": "HNSC",
                "wsi_path": "",
            }
        ],
        ["slide_id", "slide_filename", "case_id", "dataset", "cohort", "wsi_path"],
    )

    acc_slide_list = canonical_root / "wsi_slides" / "TCGA" / "ACC" / "metadata" / "slide_list.csv"
    acc_rows = _read_csv(acc_slide_list)
    acc_rows.append(dict(acc_rows[0]))
    _write_csv(acc_slide_list, acc_rows, list(acc_rows[0].keys()))

    result = run_cli(["data", "build-registry", "--root", str(canonical_root), "--project", "TCGA"])
    payload = json.loads(result.stdout)
    assert payload["missing_slides"] == 1
    assert payload["missing_features"] == 1
    assert payload["ambiguous_slides"] == 1

    mapping_rows = _read_csv(canonical_root / "registry" / "mapping.csv")
    acc_mapping = next(row for row in mapping_rows if row["slide_id"] == "TCGA-CD-5678-01Z-00-DX1")
    assert acc_mapping["uni_path"].endswith("TCGA-CD-5678-01Z-00-DX1.h5")
    assert acc_mapping["seal_path"] == ""

    validate = run_cli(["data", "validate-layout", "--root", str(canonical_root), "--project", "TCGA"])
    validate_payload = json.loads(validate.stdout)
    assert validate_payload["validated_feature_files"] == 3

    promoted = run_cli(["data", "promote-links", "--root", str(canonical_root), "--project", "TCGA"])
    promoted_payload = json.loads(promoted.stdout)
    assert promoted_payload["moved_files"] == 3
    assert not canonical_uni2.is_symlink()
    assert canonical_uni2.exists()
    assert not acc_uni2.exists()


def test_resolve_h5_prefers_canonical_layout(tmp_path, monkeypatch):
    canonical_root = tmp_path / "data_root"
    legacy_root = tmp_path / "legacy"
    feature_path = write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_uni2" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 2, 3, 4]],
    )
    run_cli(["data", "init-layout", "--root", str(canonical_root), "--project", "TCGA"])
    run_cli(
        [
            "data",
            "ingest-tcga-features",
            "--root",
            str(canonical_root),
            "--legacy-root",
            str(legacy_root),
            "--project",
            "TCGA",
            "--encoders",
            "uni2",
        ]
    )

    monkeypatch.setenv("WSI_DATA_ROOT", str(canonical_root))
    unresolved = str(tmp_path / "extracted_features" / "TCGA-ACC" / "TCGA-AB-1234-01Z-00-DX1.h5")
    resolved = _resolve_h5_path(unresolved)
    assert resolved == str(canonical_root / "wsi_features" / "uni2" / "TCGA" / "ACC" / "h5" / feature_path.name)


def test_resolve_h5_prefers_requested_encoder_for_ambiguous_legacy_path(tmp_path, monkeypatch):
    canonical_root = tmp_path / "data_root"
    legacy_root = tmp_path / "legacy"
    uni2_path = write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_uni2" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 2, 3, 4]],
    )
    seal_path = write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_seal" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[9, 8, 7, 6]],
    )
    run_cli(
        [
            "data",
            "ingest-tcga-features",
            "--root",
            str(canonical_root),
            "--legacy-root",
            str(legacy_root),
            "--project",
            "TCGA",
            "--encoders",
            "uni2,seal",
        ]
    )

    monkeypatch.setenv("WSI_DATA_ROOT", str(canonical_root))
    monkeypatch.setenv("WSI_SAE_PREFERRED_ENCODER", "seal")
    unresolved = str(tmp_path / "legacy_manifest" / "TCGA-ACC" / "features" / "TCGA-AB-1234-01Z-00-DX1.h5")
    resolved = _resolve_h5_path(unresolved)
    assert resolved == str(canonical_root / "wsi_features" / "seal" / "TCGA" / "ACC" / "h5" / seal_path.name)
    assert resolved != str(canonical_root / "wsi_features" / "uni2" / "TCGA" / "ACC" / "h5" / uni2_path.name)


@pytest.mark.parametrize("encoder", ["uni2", "seal"])
def test_ingest_creates_encoder_specific_index(tmp_path, encoder):
    canonical_root = tmp_path / "data_root"
    legacy_root = tmp_path / "legacy"
    write_feature_h5(
        legacy_root / "TCGA-ACC" / f"features_{encoder}" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 2, 3, 4]],
    )
    run_cli(
        [
            "data",
            "ingest-tcga-features",
            "--root",
            str(canonical_root),
            "--legacy-root",
            str(legacy_root),
            "--project",
            "TCGA",
            "--encoders",
            encoder,
        ]
    )
    index_path = canonical_root / "wsi_features" / encoder / "TCGA" / "ACC" / "index.csv"
    assert index_path.exists()
    rows = _read_csv(index_path)
    assert rows[0]["slide_id"] == "TCGA-AB-1234-01Z-00-DX1"


def test_scan_h5_health_reports_bad_legacy_file(tmp_path):
    legacy_root = tmp_path / "legacy"
    write_feature_h5(
        legacy_root / "TCGA-ACC" / "features_uni2" / "TCGA-AB-1234-01Z-00-DX1.h5",
        [[1, 2, 3, 4]],
    )
    bad_path = legacy_root / "TCGA-ACC" / "features_uni2" / "TCGA-AB-5678-01Z-00-DX1.h5"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"\x89HDF\r\n\x1a\nbad-h5")

    out_dir = tmp_path / "scan_out"
    result = run_cli(
        [
            "data",
            "scan-h5-health",
            "--root",
            str(tmp_path / "data_root"),
            "--project",
            "TCGA",
            "--encoders",
            "uni2",
            "--source",
            "legacy",
            "--legacy-root",
            str(legacy_root),
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(result.stdout)
    assert payload["files_scanned"] == 2
    assert payload["files_error"] == 1
    rows = _read_csv(out_dir / "h5_health_report.csv")
    assert any(row["status"] == "error" for row in rows)
