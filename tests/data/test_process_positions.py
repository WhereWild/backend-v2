from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import process_positions as pp


def test_resolve_context_label_normalizes_underscores() -> None:
    assert (
        pp._resolve_context_label(
            {
                "taxon_key": "10",
                "scientific_name": "Quercus_robur",
                "common_name": "English_oak",
                "path": Path("/tmp/taxon"),
                "rank": "SPECIES",
            }
        )
        == "Quercus robur"
    )
    assert (
        pp._resolve_context_label(
            {
                "taxon_key": "10",
                "scientific_name": "",
                "common_name": "English_oak",
                "path": Path("/tmp/taxon"),
                "rank": "SPECIES",
            }
        )
        == "English oak"
    )


def test_upsert_rows_rewrites_existing_context_labels_normalized(
    monkeypatch,
    tmp_path: Path,
) -> None:
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir()
    positions_path = taxon_dir / pp.POSITION_FILENAME
    pq.write_table(
        pa.table(
            {
                "variable": ["bio_1"],
                "metric": ["mean"],
                "position": [0],
                "count": [1],
                "sampleCount": [5],
                "contextTaxonId": ["10"],
                "contextLabel": ["Quercus_robur"],
            }
        ),
        positions_path,
    )

    monkeypatch.setattr(
        pp.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "path": taxon_dir} if str(key) == "1" else None,
    )
    monkeypatch.setattr(pp.PARQUET, "read_table", lambda path, columns: pq.read_table(path, columns=columns))
    monkeypatch.setattr(pp.PARQUET, "exists", lambda path: Path(path).exists())

    wrote, inserted = pp._upsert_rows_for_taxon(
        "1",
        [
            {
                "variable": "bio_1",
                "metric": "median",
                "position": 1,
                "count": 2,
                "sampleCount": 8,
                "contextTaxonId": "10",
                "contextLabel": "Pinus_ponderosa",
            }
        ],
    )

    assert wrote is True
    assert inserted == 1

    written = pq.read_table(positions_path).to_pylist()
    assert written == [
        {
            "variable": "bio_1",
            "metric": "mean",
            "position": 0,
            "count": 1,
            "sampleCount": 5,
            "contextTaxonId": "10",
            "contextLabel": "Quercus robur",
        },
        {
            "variable": "bio_1",
            "metric": "median",
            "position": 1,
            "count": 2,
            "sampleCount": 8,
            "contextTaxonId": "10",
            "contextLabel": "Pinus ponderosa",
        },
    ]
