# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

import scripts.populate_tree as pt

CATALOG = {
    "2923970": {"taxon_key": "2923970", "path": "Plantae_6/Cactaceae/Opuntia_humifusa_2923970", "rank": "SPECIES"},
    "9999001": {"taxon_key": "9999001", "path": "Plantae_6/Cactaceae/Opuntia_fragilis_2923971/Opuntia_fragilis_subsp_9999001", "rank": "SUBSPECIES"},
}

COLUMNS = [
    "gbifID", "taxonRank", "taxonKey", "speciesKey",
    "decimalLatitude", "decimalLongitude", "catalogNumber",
    "coordinateUncertaintyInMeters", "eventDate", "eventTime",
    "informationWithheld", "dynamicProperties", "reproductiveCondition",
    "vitality", "gbifRegion", "level0Gid", "level1Gid", "level2Gid",
]

BASE_ROW = {
    "gbifID": "1",
    "taxonRank": "SPECIES",
    "taxonKey": "2923970",
    "speciesKey": "2923970",
    "decimalLatitude": "40.0",
    "decimalLongitude": "-105.0",
    "catalogNumber": "obs123",
    "coordinateUncertaintyInMeters": "10.0",
    "eventDate": "2023-06-15",
    "eventTime": "10:30:00",
    "informationWithheld": "",
    "dynamicProperties": '{"evidenceOfPresence":"organism"}',
    "reproductiveCondition": "flowers",
    "vitality": "Alive",
    "gbifRegion": "NORTH_AMERICA",
    "level0Gid": "USA",
    "level1Gid": "USA.5",
    "level2Gid": "USA.5.12",
}


def _make_tsv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({**BASE_ROW, **row})
    return buf.getvalue()


def _run_main(tsv: str, tmp_path: Path) -> Path:
    """Run pt.main() against a tmp occurrence.txt and return the output parquet path."""
    occurrences_file = tmp_path / "taxonomy" / "occurrences.parquet"
    with patch.object(pt, "OCCURRENCE_PATH", tmp_path / "occurrence.txt"), \
         patch.object(pt, "OCCURRENCES_FILE", occurrences_file), \
         patch.object(pt, "load_catalog", return_value=CATALOG):
        (tmp_path / "occurrence.txt").write_text(tsv)
        pt.main()
    return occurrences_file


def _read_rows(occurrences_file: Path) -> list[dict]:
    if not occurrences_file.exists():
        return []
    return pq.read_table(occurrences_file).to_pylist()


# --- _parse_timestamp ---

def test_parse_timestamp_date_and_time():
    ts = pt._parse_timestamp("2023-06-15", "10:30:00")
    assert isinstance(ts, int)
    assert ts > 0


def test_parse_timestamp_date_only():
    ts = pt._parse_timestamp("2023-06-15", "")
    assert isinstance(ts, int)


def test_parse_timestamp_na_time():
    ts = pt._parse_timestamp("2023-06-15", "NA")
    assert isinstance(ts, int)


def test_parse_timestamp_empty_date():
    assert pt._parse_timestamp("", "10:30:00") is None


def test_parse_timestamp_invalid():
    assert pt._parse_timestamp("not-a-date", "") is None


def test_parse_timestamp_no_timezone_adds_utc():
    ts1 = pt._parse_timestamp("2023-06-15", "10:30:00")
    ts2 = pt._parse_timestamp("2023-06-15", "10:30:00+00:00")
    assert ts1 == ts2


# --- _parse_dp ---

def test_parse_dp_string_value():
    raw = json.dumps({"evidenceOfPresence": "organism"})
    assert pt._parse_dp(raw) == "organism"


def test_parse_dp_list_value():
    raw = json.dumps({"evidenceOfPresence": ["organism", "track"]})
    assert pt._parse_dp(raw) == "organism|track"


def test_parse_dp_empty_json():
    assert pt._parse_dp(json.dumps({})) == ""


def test_parse_dp_empty_string():
    assert pt._parse_dp("") == ""


# --- _parse_obscured ---

def test_parse_obscured_no():
    assert pt._parse_obscured("") == "No"


def test_parse_obscured_hidden_taxon():
    assert pt._parse_obscured("Location obscured for taxon") == "Hidden"


def test_parse_obscured_user():
    assert pt._parse_obscured("Location obscured by user") == "Obscured"


# --- _flush ---

def _sample_row(**overrides) -> dict:
    row = {
        "decimalLatitude": 40.0, "decimalLongitude": -105.0,
        "catalogNumber": "obs1", "hilbertIdx": 12345,
        "eventTimestamp": None, "coordinateUncertaintyInMeters": 10.0,
        "obscured": "No", "gbifRegion": None, "level0Gid": None,
        "level1Gid": None, "level2Gid": None,
        "dp": "organism", "vitality": "alive", "rcs": "flowers",
        "taxon_key": "2923970",
        "_seq": 0,
    }
    row.update(overrides)
    return row


def test_flush_writes_temp_parquet(tmp_path):
    tmp_file = tmp_path / ".tmp.parquet"
    rows = [_sample_row()]
    writer_holder: dict = {}
    pt._flush(writer_holder, tmp_file, rows)
    writer_holder["writer"].close()
    assert tmp_file.exists()
    assert rows == []
    table = pq.read_table(tmp_file)
    assert table.num_rows == 1


def test_flush_empty_rows_is_noop(tmp_path):
    tmp_file = tmp_path / ".tmp.parquet"
    writer_holder: dict = {}
    pt._flush(writer_holder, tmp_file, [])
    assert "writer" not in writer_holder
    assert not tmp_file.exists()


def test_flush_multiple_batches_append(tmp_path):
    tmp_file = tmp_path / ".tmp.parquet"
    writer_holder: dict = {}
    batch1 = [_sample_row(catalogNumber="obs1", _seq=0)]
    batch2 = [_sample_row(catalogNumber="obs2", _seq=1)]
    pt._flush(writer_holder, tmp_file, batch1)
    pt._flush(writer_holder, tmp_file, batch2)
    writer_holder["writer"].close()
    table = pq.read_table(tmp_file)
    assert table.num_rows == 2


# --- _consolidate ---

def test_consolidate_dedupes_and_sorts_by_taxon_key(tmp_path):
    tmp_file = tmp_path / ".tmp.parquet"
    dest = tmp_path / "occurrences.parquet"
    writer_holder: dict = {}
    rows = [
        _sample_row(catalogNumber="dup", taxon_key="333", _seq=0),
        _sample_row(catalogNumber="dup", taxon_key="111", _seq=1),  # same catalogNumber, later _seq
        _sample_row(catalogNumber="unique", taxon_key="222", _seq=2),
    ]
    pt._flush(writer_holder, tmp_file, rows)
    writer_holder["writer"].close()

    with patch.object(pt, "OCCURRENCES_FILE", dest):
        pt._consolidate(tmp_file)

    table = pq.read_table(dest)
    assert table.num_rows == 2  # "dup" deduped down to one row
    out = table.to_pylist()
    assert [r["taxon_key"] for r in out] == sorted(r["taxon_key"] for r in out)  # sorted by taxon_key
    dup_row = next(r for r in out if r["catalogNumber"] == "dup")
    assert dup_row["taxon_key"] == "333"  # first-seen (_seq=0) wins
    assert "_seq" not in table.schema.names


# --- main ---

def test_main_writes_consolidated_parquet(tmp_path):
    occurrences_file = _run_main(_make_tsv([{}]), tmp_path)
    rows = _read_rows(occurrences_file)
    assert len(rows) == 1
    assert rows[0]["catalogNumber"] == "obs123"
    assert rows[0]["taxon_key"] == "2923970"


def test_main_skips_non_leaf_rank(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"taxonRank": "GENUS"}]), tmp_path)
    assert _read_rows(occurrences_file) == []


def test_main_skips_missing_coords(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"decimalLatitude": ""}]), tmp_path)
    assert _read_rows(occurrences_file) == []


def test_main_skips_invalid_coords(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"decimalLatitude": "not_a_number"}]), tmp_path)
    assert _read_rows(occurrences_file) == []


def test_main_skips_unknown_taxon(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"taxonKey": "9999999", "speciesKey": "9999999"}]), tmp_path)
    assert _read_rows(occurrences_file) == []


def test_main_subspecies_routing(tmp_path):
    row = {"taxonRank": "SUBSPECIES", "taxonKey": "9999001", "speciesKey": "2923971"}
    occurrences_file = _run_main(_make_tsv([row]), tmp_path)
    rows = _read_rows(occurrences_file)
    assert len(rows) == 1
    assert rows[0]["taxon_key"] == "9999001"


def test_main_species_uses_taxon_key(tmp_path):
    row = {"taxonRank": "SPECIES", "taxonKey": "2923970", "speciesKey": "2923970"}
    occurrences_file = _run_main(_make_tsv([row]), tmp_path)
    rows = _read_rows(occurrences_file)
    assert rows[0]["taxon_key"] == "2923970"


def test_main_multi_batch_flush(tmp_path):
    # Force multiple small batches (instead of one 500k-row batch) to exercise
    # the multi-flush + final consolidation path end-to-end.
    n = 7
    rows = [{"catalogNumber": f"obs{i}"} for i in range(n)]
    with patch.object(pt, "BATCH_ROWS", 2):
        occurrences_file = _run_main(_make_tsv(rows), tmp_path)
    out_rows = _read_rows(occurrences_file)
    assert len(out_rows) == n
    assert {r["catalogNumber"] for r in out_rows} == {f"obs{i}" for i in range(n)}


def test_main_uncertainty_invalid_falls_back_to_none(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"coordinateUncertaintyInMeters": "bad"}]), tmp_path)
    rows = _read_rows(occurrences_file)
    assert rows[0]["coordinateUncertaintyInMeters"] is None


def test_main_skips_empty_lookup_key(tmp_path):
    occurrences_file = _run_main(_make_tsv([{"taxonKey": "", "speciesKey": ""}]), tmp_path)
    assert _read_rows(occurrences_file) == []


def test_main_no_rows_written_skips_consolidate(tmp_path):
    # All rows filtered out (non-leaf rank) — occurrences.parquet should never be created.
    occurrences_file = _run_main(_make_tsv([{"taxonRank": "GENUS"}]), tmp_path)
    assert not occurrences_file.exists()


def test_main_dedupes_duplicate_catalog_number_across_taxa(tmp_path):
    rows = [
        {"catalogNumber": "shared", "taxonKey": "2923970", "speciesKey": "2923970"},
        {
            "catalogNumber": "shared", "taxonRank": "SUBSPECIES",
            "taxonKey": "9999001", "speciesKey": "2923971",
        },
    ]
    occurrences_file = _run_main(_make_tsv(rows), tmp_path)
    out_rows = _read_rows(occurrences_file)
    assert len(out_rows) == 1  # cross-taxon catalogNumber collision deduped, first-seen wins
    assert out_rows[0]["taxon_key"] == "2923970"
