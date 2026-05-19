import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.populate_tree as pt

CATALOG = {
    "2923970": {"path": "Plantae_6/Cactaceae/Opuntia_humifusa_2923970", "rank": "SPECIES"},
    "9999001": {"path": "Plantae_6/Cactaceae/Opuntia_fragilis_2923971/Opuntia_fragilis_subsp_9999001", "rank": "SUBSPECIES"},
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


def _run_main(tsv: str, tmp_path: Path) -> None:
    tree_root = tmp_path / "tree"
    with patch.object(pt, "OCCURRENCE_PATH", tmp_path / "occurrence.txt"), \
         patch.object(pt, "TREE_ROOT", tree_root), \
         patch.object(pt, "load_catalog", return_value=CATALOG):
        (tmp_path / "occurrence.txt").write_text(tsv)
        pt.main()


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

def test_flush_writes_parquet(tmp_path):
    buffers = defaultdict(list)
    buffers["taxon/path"].append({
        "decimalLatitude": 40.0, "decimalLongitude": -105.0,
        "catalogNumber": "obs1", "hilbertIdx": 12345,
        "eventTimestamp": None, "coordinateUncertaintyInMeters": 10.0,
        "obscured": "No", "gbifRegion": None, "level0Gid": None,
        "level1Gid": None, "level2Gid": None,
        "dp": "organism", "vitality": "alive", "rcs": "flowers",
    })
    with patch.object(pt, "TREE_ROOT", tmp_path):
        pt._flush(buffers, "taxon/path")
    assert (tmp_path / "taxon/path/occurrence.parquet").exists()
    assert len(buffers["taxon/path"]) == 0


def test_flush_appends_to_existing(tmp_path):
    buffers = defaultdict(list)
    row = {
        "decimalLatitude": 40.0, "decimalLongitude": -105.0,
        "catalogNumber": "obs1", "hilbertIdx": 12345,
        "eventTimestamp": None, "coordinateUncertaintyInMeters": None,
        "obscured": "No", "gbifRegion": None, "level0Gid": None,
        "level1Gid": None, "level2Gid": None,
        "dp": "", "vitality": "", "rcs": "",
    }
    with patch.object(pt, "TREE_ROOT", tmp_path):
        buffers["taxon/path"].append(row)
        pt._flush(buffers, "taxon/path")
        buffers["taxon/path"].append({**row, "catalogNumber": "obs2"})
        pt._flush(buffers, "taxon/path")

    table = pq.read_table(tmp_path / "taxon/path/occurrence.parquet")
    assert table.num_rows == 2


def test_flush_schema_mismatch_casts(tmp_path):
    # Write an existing parquet with hilbertIdx as int64 (legacy) instead of
    # int32 — triggers the schema cast branch (line 108).
    folder = tmp_path / "taxon/path"
    folder.mkdir(parents=True)
    legacy_schema = pt.SCHEMA.set(
        pt.SCHEMA.get_field_index("hilbertIdx"),
        pa.field("hilbertIdx", pa.int64()),
    )
    legacy = pa.table(
        {name: pa.array([], type=legacy_schema.field(name).type) for name in legacy_schema.names},
        schema=legacy_schema,
    )
    pq.write_table(legacy, folder / "occurrence.parquet")

    buffers = defaultdict(list)
    buffers["taxon/path"].append({
        "decimalLatitude": 40.0, "decimalLongitude": -105.0,
        "catalogNumber": "obs1", "hilbertIdx": 12345,
        "eventTimestamp": None, "coordinateUncertaintyInMeters": None,
        "obscured": "No", "gbifRegion": None, "level0Gid": None,
        "level1Gid": None, "level2Gid": None,
        "dp": "", "vitality": "", "rcs": "",
    })
    with patch.object(pt, "TREE_ROOT", tmp_path):
        pt._flush(buffers, "taxon/path")

    table = pq.read_table(folder / "occurrence.parquet")
    assert table.num_rows >= 1


def test_flush_empty_buffer_is_noop(tmp_path):
    buffers = defaultdict(list)
    with patch.object(pt, "TREE_ROOT", tmp_path):
        pt._flush(buffers, "taxon/path")
    assert not (tmp_path / "taxon/path/occurrence.parquet").exists()


# --- main ---

def test_main_writes_parquet(tmp_path):
    _run_main(_make_tsv([{}]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 1
    assert table["catalogNumber"][0].as_py() == "obs123"


def test_main_skips_non_leaf_rank(tmp_path):
    _run_main(_make_tsv([{"taxonRank": "GENUS"}]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    assert not out.exists()


def test_main_skips_missing_coords(tmp_path):
    _run_main(_make_tsv([{"decimalLatitude": ""}]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    assert not out.exists()


def test_main_skips_invalid_coords(tmp_path):
    _run_main(_make_tsv([{"decimalLatitude": "not_a_number"}]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    assert not out.exists()


def test_main_skips_unknown_taxon(tmp_path):
    _run_main(_make_tsv([{"taxonKey": "9999999", "speciesKey": "9999999"}]), tmp_path)
    assert not any((tmp_path / "tree").rglob("occurrence.parquet")) if (tmp_path / "tree").exists() else True


def test_main_subspecies_routing(tmp_path):
    row = {"taxonRank": "SUBSPECIES", "taxonKey": "9999001", "speciesKey": "2923971"}
    _run_main(_make_tsv([row]), tmp_path)
    out = tmp_path / "tree" / CATALOG["9999001"]["path"] / "occurrence.parquet"
    assert out.exists()


def test_main_species_uses_species_key(tmp_path):
    row = {"taxonRank": "SPECIES", "taxonKey": "99999", "speciesKey": "2923970"}
    _run_main(_make_tsv([row]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    assert out.exists()


def test_main_buffer_flush_on_limit(tmp_path):
    rows = [{"catalogNumber": f"obs{i}"} for i in range(pt.BUFFER_LIMIT + 1)]
    _run_main(_make_tsv(rows), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    table = pq.read_table(out)
    assert table.num_rows == pt.BUFFER_LIMIT + 1


def test_main_uncertainty_invalid_falls_back_to_none(tmp_path):
    _run_main(_make_tsv([{"coordinateUncertaintyInMeters": "bad"}]), tmp_path)
    out = tmp_path / "tree" / CATALOG["2923970"]["path"] / "occurrence.parquet"
    table = pq.read_table(out)
    assert table["coordinateUncertaintyInMeters"][0].as_py() is None


def test_main_skips_empty_lookup_key(tmp_path):
    _run_main(_make_tsv([{"taxonKey": "", "speciesKey": ""}]), tmp_path)
    assert not any((tmp_path / "tree").rglob("occurrence.parquet")) if (tmp_path / "tree").exists() else True
