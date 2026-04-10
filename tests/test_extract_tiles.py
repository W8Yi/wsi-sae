from __future__ import annotations

import csv
import json
from pathlib import Path

from ._helpers import run_cli, write_slide_image


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_extract_tiles_uses_mapping_registry_and_writes_outputs(tmp_path):
    data_root = tmp_path / "data_root"
    slide_path = write_slide_image(
        data_root / "wsi_slides" / "TCGA" / "ACC" / "slides" / "TCGA-AB-1234-01Z-00-DX1.png",
        width=512,
        height=512,
    )

    _write_csv(
        data_root / "registry" / "mapping.csv",
        [
            {
                "slide_id": "TCGA-AB-1234-01Z-00-DX1",
                "case_id": "TCGA-AB-1234",
                "dataset": "TCGA",
                "cohort": "ACC",
                "slide_filename": slide_path.name,
                "wsi_path": str(slide_path),
                "uni_path": "",
                "seal_path": "",
            }
        ],
        ["slide_id", "case_id", "dataset", "cohort", "slide_filename", "wsi_path", "uni_path", "seal_path"],
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_csv(
        bundle_dir / "prototype_tiles.csv",
        [
            {
                "latent_idx": "7",
                "latent_group": "parent_2",
                "prototype_rank": "1",
                "activation": "3.5",
                "case_id": "TCGA-AB-1234",
                "slide_key": "TCGA-AB-1234-01Z-00-DX1",
                "tile_index": "11",
                "coord_x": "64",
                "coord_y": "96",
                "h5_path": "/tmp/TCGA-AB-1234-01Z-00-DX1.h5",
                "dataset": "TCGA",
                "encoder": "uni2",
            },
            {
                "latent_idx": "8",
                "latent_group": "selected",
                "prototype_rank": "1",
                "activation": "2.0",
                "case_id": "TCGA-ZZ-9999",
                "slide_key": "TCGA-ZZ-9999-01Z-00-DX1",
                "tile_index": "2",
                "coord_x": "0",
                "coord_y": "0",
                "h5_path": "/tmp/TCGA-ZZ-9999-01Z-00-DX1.h5",
                "dataset": "TCGA",
                "encoder": "uni2",
            },
        ],
        [
            "latent_idx",
            "latent_group",
            "prototype_rank",
            "activation",
            "case_id",
            "slide_key",
            "tile_index",
            "coord_x",
            "coord_y",
            "h5_path",
            "dataset",
            "encoder",
        ],
    )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifacts": {
                    "prototype_tiles_csv": "prototype_tiles.csv",
                },
            },
            indent=2,
        )
    )

    out_dir = tmp_path / "extract_out"
    result = run_cli(
        [
            "extract-tiles",
            "--bundle",
            str(bundle_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--tile-size",
            "128",
        ]
    )
    payload = json.loads(result.stdout)
    assert payload["rows_total"] == 2
    assert payload["rows_extracted"] == 1
    assert payload["rows_missing_slide"] == 1

    extracted_csv = out_dir / "extracted_tiles.csv"
    with extracted_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "ok"
    assert Path(rows[0]["tile_path"]).exists()
    assert rows[1]["status"] == "missing_slide"

    assert any((out_dir / "contact_sheets").glob("latent_0007.*"))
    summary = json.loads((out_dir / "extract_summary.json").read_text())
    assert summary["tile_size"] == 128
