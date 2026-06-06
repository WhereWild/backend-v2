"""Tests for util/rankings.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import util.rankings as rk
from config.config import ValueType

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_RATIO_LAYER = {"id": "bio1", "value_type": "ratio"}
_NOMINAL_LAYER = {"id": "kg0", "value_type": "nominal"}
_CIRCULAR_LAYER = {"id": "aspect_deg", "value_type": "circular"}
_ORDINAL_LAYER = {"id": "foo", "value_type": "ordinal"}
_ALL_LAYERS = [_RATIO_LAYER, _NOMINAL_LAYER]

_ANCESTOR: dict = {
    "taxon_key": "1",
    "path": "Root_1",
    "scientific_name": "Plantae",
    "common_name": "",
    "rank": "KINGDOM",
}

_GENUS: dict = {
    "taxon_key": "100",
    "path": "Root_1/Order_10/Family_50/Genus_100",
    "scientific_name": "Testus",
    "common_name": "",
    "rank": "GENUS",
}

_SPECIES_A: dict = {
    "taxon_key": "200",
    "path": "Root_1/Order_10/Family_50/Genus_100/Species_200",
    "scientific_name": "Testus alpha",
    "common_name": "alpha plant",
    "rank": "SPECIES",
}

_SPECIES_B: dict = {
    "taxon_key": "201",
    "path": "Root_1/Order_10/Family_50/Genus_100/Species_201",
    "scientific_name": "Testus beta",
    "common_name": "",
    "rank": "SPECIES",
}

_SUBSPECIES_A: dict = {
    "taxon_key": "300",
    "path": "Root_1/Order_10/Family_50/Genus_100/Species_200/Subspecies_300",
    "scientific_name": "Testus alpha subsp.",
    "common_name": "",
    "rank": "SUBSPECIES",
}


def _write_numerical_stats(taxon_dir: Path, variable: str, **metrics) -> None:
    """Write a minimal numerical_stats.parquet for one variable."""
    taxon_dir.mkdir(parents=True, exist_ok=True)
    row = {"variable": variable, **metrics}
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([row]), preserve_index=False),
                   taxon_dir / rk.NUMERICAL_STATS_FILE)


def _write_nominal_stats(taxon_dir: Path, variable: str, entries: list[tuple[str, float]]) -> None:
    """Write a minimal nominal_stats.parquet for one variable."""
    taxon_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"variable": variable, "metric": m, "value": v} for m, v in entries]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
                   taxon_dir / rk.NOMINAL_STATS_FILE)


def _write_rank_index(
    index_path: Path,
    entries: dict[str, list[tuple[str, float, int]]],  # col_name → [(taxon_key, value, sample_count)]
) -> None:
    """Write a minimal rank index parquet with column_lengths metadata."""
    struct_type = pa.struct([
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int64()),
    ])
    max_len = max(len(v) for v in entries.values())
    arrays: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    for col_name, rows in entries.items():
        column_lengths[col_name] = len(rows)
        arr = pa.StructArray.from_arrays(
            [
                pa.array([r[0] for r in rows], type=pa.string()),
                pa.array([r[1] for r in rows], type=pa.float64()),
                pa.array([r[2] for r in rows], type=pa.int64()),
            ],
            fields=[pa.field("taxonKey", pa.string()),
                    pa.field("value", pa.float64()),
                    pa.field("sampleCount", pa.int64())],
        )
        if len(arr) < max_len:
            arr = pa.concat_arrays([arr, pa.nulls(max_len - len(arr), type=struct_type)])
        arrays[col_name] = arr
    table = pa.table(arrays)
    metadata = {b"column_lengths": json.dumps(column_lengths).encode("utf-8")}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), index_path)


# ---------------------------------------------------------------------------
# _descendant_rank_targets
# ---------------------------------------------------------------------------

def test_descendant_rank_targets_kingdom():
    targets = rk._descendant_rank_targets("KINGDOM")
    assert targets == ["PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES", "SUBSPECIES"]


def test_descendant_rank_targets_genus():
    targets = rk._descendant_rank_targets("GENUS")
    assert targets == ["SPECIES", "SUBSPECIES"]


def test_descendant_rank_targets_species():
    targets = rk._descendant_rank_targets("SPECIES")
    assert targets == ["SUBSPECIES"]


def test_descendant_rank_targets_subspecies():
    assert rk._descendant_rank_targets("SUBSPECIES") == []


def test_descendant_rank_targets_unknown_rank():
    assert rk._descendant_rank_targets("DOMAIN") == []


# ---------------------------------------------------------------------------
# _metrics_for_vtype
# ---------------------------------------------------------------------------

def test_metrics_for_vtype_ratio():
    metrics = rk._metrics_for_vtype(_RATIO_LAYER, ValueType.RATIO)
    assert "mean" in metrics
    assert "median" in metrics
    assert "count" in metrics


def test_metrics_for_vtype_interval():
    metrics = rk._metrics_for_vtype({"id": "x", "value_type": "interval"}, ValueType.INTERVAL)
    assert "mean" in metrics


def test_metrics_for_vtype_nominal():
    metrics = rk._metrics_for_vtype(_NOMINAL_LAYER, ValueType.NOMINAL)
    assert "entropy" in metrics
    assert "unique_classes" in metrics
    assert "mean" not in metrics


def test_metrics_for_vtype_circular_returns_full_tuple():
    metrics = rk._metrics_for_vtype(_CIRCULAR_LAYER, ValueType.CIRCULAR)
    assert "rbar" in metrics
    assert "circular_mean" in metrics
    assert "circular_std" in metrics
    assert "count" in metrics


def test_metrics_for_vtype_ordinal_empty():
    assert rk._metrics_for_vtype(_ORDINAL_LAYER, ValueType.ORDINAL) == ()


# ---------------------------------------------------------------------------
# _resolve_context_label
# ---------------------------------------------------------------------------

def test_resolve_context_label_scientific():
    assert rk._resolve_context_label(_ANCESTOR) == "Plantae"


def test_resolve_context_label_falls_back_to_common():
    taxon = {**_ANCESTOR, "scientific_name": "", "common_name": "Flowering plants"}
    assert rk._resolve_context_label(taxon) == "Flowering plants"


def test_resolve_context_label_falls_back_to_key():
    taxon = {**_ANCESTOR, "scientific_name": "", "common_name": ""}
    assert rk._resolve_context_label(taxon) == "1"


# ---------------------------------------------------------------------------
# _infer_sample_count
# ---------------------------------------------------------------------------

def test_infer_sample_count_from_numerical_stats(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", count=42, mean=5.0)
    assert rk._infer_sample_count(tmp_path) == 42


def test_infer_sample_count_from_nominal_stats(tmp_path):
    _write_nominal_stats(tmp_path, "kg0", [("total_samples", 17.0), ("entropy", 1.5)])
    assert rk._infer_sample_count(tmp_path) == 17
def test_infer_sample_count_no_files(tmp_path):
    assert rk._infer_sample_count(tmp_path) == 0


# ---------------------------------------------------------------------------
# _descendants_for_rank
# ---------------------------------------------------------------------------

def test_descendants_for_rank_subspecies_skipped_for_genus(monkeypatch):
    """_descendants_for_rank returns [] for SUBSPECIES when ancestor is not SPECIES."""
    with patch("util.rankings.iter_descendants", return_value=[_SUBSPECIES_A]):
        result = rk._descendants_for_rank(_GENUS, "SUBSPECIES")
    assert result == []


def test_descendants_for_rank_species_combines_subspecies_for_genus(monkeypatch):
    """GENUS→SPECIES includes both SPECIES and SUBSPECIES descendants."""
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SUBSPECIES_A]):
        result = rk._descendants_for_rank(_GENUS, "SPECIES")
    keys = {t["taxon_key"] for t in result}
    assert "200" in keys
    assert "300" in keys


# ---------------------------------------------------------------------------
# _collect_entries_from_numerical_stats
# ---------------------------------------------------------------------------

def test_collect_entries_numerical_stats_ratio(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", count=20, mean=5.0, median=4.5)
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 20, [_RATIO_LAYER])
    assert "bio1::mean" in entries
    assert entries["bio1::mean"]["value"] == pytest.approx(5.0)
    assert entries["bio1::mean"]["taxon_key"] == "100"
    assert entries["bio1::mean"]["sample_count"] == 20


def test_collect_entries_numerical_stats_skips_unknown_layer(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", count=5, mean=1.0)
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 5, [_NOMINAL_LAYER])
    assert len(entries) == 0


def test_collect_entries_numerical_stats_skips_circular(tmp_path):
    _write_numerical_stats(tmp_path, "aspect_deg", count=5, mean=90.0)
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 5, [_CIRCULAR_LAYER])
    assert len(entries) == 0


def test_collect_entries_numerical_stats_no_file(tmp_path):
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 0, [_RATIO_LAYER])
    assert entries == {}


# ---------------------------------------------------------------------------
# _collect_entries_from_nominal_stats
# ---------------------------------------------------------------------------

def test_collect_entries_nominal_stats(tmp_path):
    _write_nominal_stats(tmp_path, "kg0", [
        ("entropy", 1.5),
        ("unique_classes", 3.0),
        ("total_samples", 20.0),
        ("mode", 5.0),
        ("unique_samples", 18.0),
    ])
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 20, [_NOMINAL_LAYER])
    assert "kg0::entropy" in entries
    assert entries["kg0::entropy"]["value"] == pytest.approx(1.5)
    assert entries["kg0::entropy"]["taxon_key"] == "50"
    assert "kg0::total_samples" in entries


def test_collect_entries_nominal_stats_skips_non_nominal_layers(tmp_path):
    _write_nominal_stats(tmp_path, "kg0", [("entropy", 1.5)])
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 10, [_RATIO_LAYER])
    assert len(entries) == 0


def test_collect_entries_nominal_stats_no_file(tmp_path):
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 0, [_NOMINAL_LAYER])
    assert entries == {}


# ---------------------------------------------------------------------------
# _build_rank_index
# ---------------------------------------------------------------------------

def test_build_rank_index_sorts_by_value(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    index_path = tmp_path / "species_index.parquet"

    sp_a_dir = tmp_path / _SPECIES_A["path"]
    sp_b_dir = tmp_path / _SPECIES_B["path"]
    _write_numerical_stats(sp_a_dir, "bio1", count=10, mean=3.0)
    _write_numerical_stats(sp_b_dir, "bio1", count=8, mean=7.0)

    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A, _SPECIES_B]):
        rk._build_rank_index(_GENUS, "SPECIES", index_path, [_RATIO_LAYER])

    assert index_path.exists()
    assert "bio1::mean" in pq.read_schema(index_path).names
    tbl = pq.read_table(index_path)
    col = tbl.column("bio1::mean").combine_chunks()
    keys = [col[i].as_py()["taxonKey"] for i in range(2)]
    assert keys == ["200", "201"]  # A (3.0) before B (7.0)


def test_build_rank_index_no_descendants_removes_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    index_path = tmp_path / "i.parquet"
    index_path.touch()
    with patch("util.rankings._descendants_for_rank", return_value=[]):
        rk._build_rank_index(_GENUS, "SPECIES", index_path, [_RATIO_LAYER])
    assert not index_path.exists()


def test_build_rank_index_no_stats_removes_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    index_path = tmp_path / "i.parquet"
    index_path.touch()
    # SPECIES_A has no stats file → no entries → index removed
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A]):
        rk._build_rank_index(_GENUS, "SPECIES", index_path, [_RATIO_LAYER])
    assert not index_path.exists()


def test_build_rank_index_stores_column_lengths_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    index_path = tmp_path / "si.parquet"
    sp_dir = tmp_path / _SPECIES_A["path"]
    _write_numerical_stats(sp_dir, "bio1", count=5, mean=1.0)
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A]):
        rk._build_rank_index(_GENUS, "SPECIES", index_path, [_RATIO_LAYER])
    lengths = rk._load_column_lengths(index_path)
    assert "bio1::mean" in lengths
    assert lengths["bio1::mean"] == 1


# ---------------------------------------------------------------------------
# Coverage-gap tests
# ---------------------------------------------------------------------------

_FAMILY: dict = {
    "taxon_key": "50",
    "path": "Root_1/Order_10/Family_50",
    "scientific_name": "Testaceae",
    "common_name": "",
    "rank": "FAMILY",
}


# _atomic_write — finally cleanup on write failure (line 77)
def test_atomic_write_cleanup_on_write_failure(tmp_path):
    with patch("util.rankings.pq.write_table", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            rk._atomic_write(tmp_path / "out.parquet", pa.table({"x": [1]}))
    # no stray temp files remain
    assert list(tmp_path.glob("*.parquet")) == []


# _infer_sample_count — bad int in count column (inner except, lines 131-132)
def test_infer_sample_count_bad_count_value(tmp_path):
    # Write numerical_stats with count = "bad" (not castable to int)
    pq.write_table(
        pa.table({"variable": ["bio1"], "count": ["bad"]}),
        tmp_path / rk.NUMERICAL_STATS_FILE,
    )
    # Falls through to 0 since none of the count values are usable
    assert rk._infer_sample_count(tmp_path) == 0


# _infer_sample_count — pq.read_table raises (outer except, lines 133-134)
def test_infer_sample_count_numerical_stats_read_fails(tmp_path):
    # Write a parquet with no "count" column → read_table(columns=["count"]) raises
    pq.write_table(pa.table({"variable": ["bio1"]}), tmp_path / rk.NUMERICAL_STATS_FILE)
    assert rk._infer_sample_count(tmp_path) == 0


# _infer_sample_count — corrupt nominal stats (lines 142-143)
def test_infer_sample_count_corrupt_nominal_stats(tmp_path):
    (tmp_path / rk.NOMINAL_STATS_FILE).write_bytes(b"not a parquet")
    assert rk._infer_sample_count(tmp_path) == 0


# _infer_sample_count — corrupt occurrence index (lines 148-149)
def test_infer_sample_count_corrupt_occurrence_index(tmp_path):
    (tmp_path / "occurrence_index.parquet").write_bytes(b"not a parquet")
    assert rk._infer_sample_count(tmp_path) == 0


# _collect_entries_from_numerical_stats — corrupt stats file (lines 223-224)
def test_collect_entries_numerical_stats_corrupt_file(tmp_path):
    (tmp_path / rk.NUMERICAL_STATS_FILE).write_bytes(b"garbage")
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 0, [_RATIO_LAYER])
    assert entries == {}


# _collect_entries_from_numerical_stats — bad value_type in layer (lines 236-237)
def test_collect_entries_numerical_stats_bad_vtype(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", count=5, mean=1.0)
    bad_layer = {"id": "bio1", "value_type": "not_a_real_type"}
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 5, [bad_layer])
    assert entries == {}


# _collect_entries_from_numerical_stats — non-castable float (lines 248-249)
def test_collect_entries_numerical_stats_nonfloat_metric(tmp_path):
    # Write stats where mean is a string that can't be cast to float
    pq.write_table(
        pa.table({"variable": ["bio1"], "count": [5], "mean": ["N/A"]}),
        tmp_path / rk.NUMERICAL_STATS_FILE,
    )
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 5, [_RATIO_LAYER])
    assert "bio1::mean" not in entries


# _collect_entries_from_numerical_stats — non-finite metric value (line 251)
def test_collect_entries_numerical_stats_nonfinite_metric(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", count=5, mean=float("inf"))
    entries = rk._collect_entries_from_numerical_stats("100", tmp_path, 5, [_RATIO_LAYER])
    assert "bio1::mean" not in entries


# _collect_entries_from_nominal_stats — corrupt stats file (lines 272-273)
def test_collect_entries_nominal_stats_corrupt_file(tmp_path):
    (tmp_path / rk.NOMINAL_STATS_FILE).write_bytes(b"garbage")
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 0, [_NOMINAL_LAYER])
    assert entries == {}


# _collect_entries_from_nominal_stats — non-castable value (lines 287-288)
def test_collect_entries_nominal_stats_bad_value(tmp_path):
    pq.write_table(
        pa.table({"variable": ["kg0"], "metric": ["entropy"], "value": ["bad"]}),
        tmp_path / rk.NOMINAL_STATS_FILE,
    )
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 0, [_NOMINAL_LAYER])
    assert entries == {}


# _collect_entries_from_nominal_stats — non-finite value (line 290)
def test_collect_entries_nominal_stats_nonfinite_value(tmp_path):
    pq.write_table(
        pa.table({"variable": ["kg0"], "metric": ["entropy"], "value": [float("inf")]}),
        tmp_path / rk.NOMINAL_STATS_FILE,
    )
    entries = rk._collect_entries_from_nominal_stats("50", tmp_path, 0, [_NOMINAL_LAYER])
    assert "kg0::entropy" not in entries



# build_rank_indexes — no-targets branch (line 378)
def test_build_rank_indexes_no_targets(tmp_path, monkeypatch):
    """SUBSPECIES has no ranks below it → returns immediately."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    rk.build_rank_indexes(_SUBSPECIES_A, [_RATIO_LAYER])
    # Nothing created
    assert not (tmp_path / _SUBSPECIES_A["path"]).exists()


# _load_column_lengths — no metadata (line 399)
def test_load_column_lengths_no_metadata(tmp_path):
    idx_path = tmp_path / "idx.parquet"
    pq.write_table(pa.table({"x": [1]}), idx_path)
    assert rk._load_column_lengths(idx_path) == {}


# _load_column_lengths — corrupt schema (lines 401-402)
def test_load_column_lengths_corrupt_file(tmp_path):
    idx_path = tmp_path / "idx.parquet"
    idx_path.write_bytes(b"garbage")
    assert rk._load_column_lengths(idx_path) == {}


# _load_existing_positions — corrupt file (lines 411-412)
def test_load_gid_levels_reads_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "hierarchy.csv"
    csv_path.write_text("level,gid,name,parent_gid\n0,USA,United States,\n1,USA.1,Alabama,USA\n")
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", csv_path)
    rk._load_gid_levels.cache_clear()
    levels = rk._load_gid_levels()
    assert levels["USA"] == 0
    assert levels["USA.1"] == 1
    rk._load_gid_levels.cache_clear()


def test_load_gid_levels_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", tmp_path / "nonexistent.csv")
    rk._load_gid_levels.cache_clear()
    assert rk._load_gid_levels() == {}
    rk._load_gid_levels.cache_clear()


def test_gid_to_scope_known_level(tmp_path, monkeypatch):
    csv_path = tmp_path / "hierarchy.csv"
    csv_path.write_text("level,gid,name,parent_gid\n0,USA,United States,\n")
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", csv_path)
    rk._load_gid_levels.cache_clear()
    assert rk._gid_to_scope("USA") == "gadm_level0"
    rk._load_gid_levels.cache_clear()


def test_gid_to_scope_unknown_gid(tmp_path, monkeypatch):
    csv_path = tmp_path / "hierarchy.csv"
    csv_path.write_text("level,gid,name,parent_gid\n")
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", csv_path)
    rk._load_gid_levels.cache_clear()
    assert rk._gid_to_scope("UNKNOWN") == "gbif_region"
    rk._load_gid_levels.cache_clear()


def test_location_taxon_keys_reads_parquet(tmp_path, monkeypatch):
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(
        pa.table({
            "scope": pa.array(["gadm_level0", "gadm_level0"]),
            "gid": pa.array(["USA", "USA"]),
            "taxon_key": pa.array(["100", "200"]),
            "count": pa.array([10, 20], type=pa.int64()),
        }),
        loc_path,
    )
    csv_path = tmp_path / "hierarchy.csv"
    csv_path.write_text("level,gid,name,parent_gid\n0,USA,United States,\n")
    monkeypatch.setattr(rk, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", csv_path)
    rk._load_gid_levels.cache_clear()
    rk._location_taxon_keys.cache_clear()
    keys, counts = rk._location_taxon_keys("USA")
    assert keys == frozenset({"100", "200"})
    assert counts["100"] == 10
    assert counts["200"] == 20
    rk._load_gid_levels.cache_clear()
    rk._location_taxon_keys.cache_clear()


def test_location_taxon_keys_bad_parquet(tmp_path, monkeypatch):
    bad_path = tmp_path / "bad.parquet"
    bad_path.write_bytes(b"garbage")
    monkeypatch.setattr(rk, "_LOC_TAXA_PATH", bad_path)
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", tmp_path / "none.csv")
    rk._load_gid_levels.cache_clear()
    rk._location_taxon_keys.cache_clear()
    keys, counts = rk._location_taxon_keys("USA")
    assert keys == frozenset()
    assert counts == {}
    rk._load_gid_levels.cache_clear()
    rk._location_taxon_keys.cache_clear()


def test_read_index_entries_bad_file(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"garbage")
    assert rk._read_index_entries(bad, "bio1::mean", 5) == []


def test_taxon_metric_value_from_numerical(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", mean=12.5, count=100)
    result = rk._taxon_metric_value(tmp_path, "bio1", "mean")
    assert result == pytest.approx(12.5)


def test_taxon_metric_value_from_nominal(tmp_path):
    _write_nominal_stats(tmp_path, "kg0", [("total_samples", 50.0), ("unique_classes", 3.0)])
    result = rk._taxon_metric_value(tmp_path, "kg0", "total_samples")
    assert result == pytest.approx(50.0)


def test_taxon_metric_value_missing_variable(tmp_path):
    _write_numerical_stats(tmp_path, "bio1", mean=5.0)
    assert rk._taxon_metric_value(tmp_path, "bio99", "mean") is None


def test_taxon_metric_value_no_files(tmp_path):
    assert rk._taxon_metric_value(tmp_path, "bio1", "mean") is None


def test_accepted_ranks_non_species():
    assert rk._accepted_ranks("GENUS", False) is None
    assert rk._accepted_ranks("FAMILY", True) is None


def test_accepted_ranks_species_no_flag():
    result = rk._accepted_ranks("SPECIES", False)
    assert result == frozenset({"SPECIES"})


def test_accepted_ranks_species_with_flag():
    result = rk._accepted_ranks("SPECIES", True)
    assert "SPECIES" in result
    assert "SUBSPECIES" in result


def test_query_ranked_scoped_no_column(tmp_path, monkeypatch):
    """Index exists but requested column is absent → no_column."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    ancestor_dir = tmp_path / _GENUS["path"]
    ancestor_dir.mkdir(parents=True)
    # Write index with a different column
    _write_rank_index(ancestor_dir / "species_index.parquet", {"other::mean": [("200", 1.0, 10)]})
    result = rk._query_ranked_scoped(
        q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
        sort_variable="bio1", sort_metric="mean", sort_order="asc",
        limit=10, offset=0, min_samples=0, include_species_like=False,
        loc_keys=None, loc_counts={},
    )
    assert result["empty_reason"] == "no_column"


def test_query_ranked_scoped_taxon_none_in_accepted_ranks(tmp_path, monkeypatch):
    """Entries whose get_taxon_by_id returns None are skipped in accepted_ranks filter."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    ancestor_dir = tmp_path / _GENUS["path"]
    ancestor_dir.mkdir(parents=True)
    _write_rank_index(ancestor_dir / "species_index.parquet", {
        "bio1::mean": [("200", 10.0, 100), ("999", 20.0, 50)]  # 999 is unknown
    })
    with patch("util.rankings.get_taxon_by_id", side_effect=lambda k: _SPECIES_A if k == "200" else None):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert len(result["results"]) == 1
    assert result["results"][0]["taxon"]["taxon_key"] == "200"


def test_query_ranked_scoped_taxon_none_in_results(tmp_path, monkeypatch):
    """get_taxon_by_id returning None during result building skips the entry."""
    family: dict = {
        "taxon_key": "50", "path": "Root_1/Order_10/Family_50",
        "scientific_name": "Testaceae", "common_name": "", "rank": "FAMILY",
    }
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    ancestor_dir = tmp_path / family["path"]
    ancestor_dir.mkdir(parents=True)
    # Use genus_index.parquet (descendant_rank=GENUS has no accepted_ranks filter)
    _write_rank_index(ancestor_dir / "genus_index.parquet", {
        "bio1::mean": [("100", 10.0, 100), ("999", 20.0, 50)]
    })

    def _resolve(k):
        return _GENUS if k == "100" else None

    with patch("util.rankings.get_taxon_by_id", side_effect=_resolve):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=family, descendant_rank="GENUS",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    valid_ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert "100" in valid_ids
    assert "999" not in valid_ids


def test_query_ranked_text_loc_keys_filter(tmp_path, monkeypatch):
    """location filter in ranked-text mode skips taxa not in loc_keys."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    (tmp_path / _SPECIES_A["path"]).mkdir(parents=True)
    (tmp_path / _SPECIES_B["path"]).mkdir(parents=True)

    with patch("util.rankings._taxon_metric_value", return_value=5.0), \
         patch("util.rankings._infer_sample_count", return_value=100), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (_SPECIES_B, 80.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=frozenset({"200"}), loc_counts={},
        )
    assert len(result["results"]) == 1
    assert result["results"][0]["taxon"]["taxon_key"] == "200"


def test_query_ranked_text_no_metric_value(tmp_path, monkeypatch):
    """Candidates with no metric value are excluded."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    with patch("util.rankings._taxon_metric_value", return_value=None), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"


def test_query_ranked_text_min_samples_filter(tmp_path, monkeypatch):
    """Candidates with too few samples are excluded."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    with patch("util.rankings._taxon_metric_value", return_value=5.0), \
         patch("util.rankings._infer_sample_count", return_value=3), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=10, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"



def test_query_text_loc_keys_filter(tmp_path, monkeypatch):
    """Location filter excludes candidates not in loc_keys."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    with patch("util.rankings._infer_sample_count", return_value=50), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (_SPECIES_B, 80.0, "")]):
        result = rk._query_text(
            q="testus", within_taxon=None, descendant_rank=None,
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=frozenset({"201"}), loc_counts={},
        )
    ids = [r["taxon"]["taxon_key"] for r in result["results"]]
    assert "201" in ids
    assert "200" not in ids


def test_query_text_accepted_ranks_filter(tmp_path, monkeypatch):
    """Rank filter excludes non-matching ranks."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    subsp = {**_SUBSPECIES_A, "rank": "SUBSPECIES"}
    with patch("util.rankings._infer_sample_count", return_value=50), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (subsp, 85.0, "")]):
        result = rk._query_text(
            q="testus", within_taxon=None, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    ids = [r["taxon"]["taxon_key"] for r in result["results"]]
    assert "200" in ids
    assert "300" not in ids


def test_query_text_min_samples_filter(tmp_path, monkeypatch):
    """min_samples filter excludes candidates with too few samples."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    with patch("util.rankings._infer_sample_count", return_value=2), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_text(
            q="testus", within_taxon=None, descendant_rank=None,
            limit=10, offset=0, min_samples=10, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"


def test_load_scope_keys_dfs_fallback(tmp_path, monkeypatch):
    """Falls back to DFS iteration when catalog parquet is absent."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    (tmp_path / _GENUS["path"]).mkdir(parents=True)
    with patch("util.rankings.iter_descendants",
               return_value=[_SPECIES_A, _SUBSPECIES_A]):
        keys = rk._load_scope_keys(_GENUS, "SPECIES", False)
    assert "200" in keys
    assert "300" not in keys  # SUBSPECIES excluded when include_species_like=False


def test_load_scope_keys_dfs_fallback_include_species_like(tmp_path, monkeypatch):
    """DFS fallback with include_species_like includes subspecies equivalents."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    (tmp_path / _GENUS["path"]).mkdir(parents=True)
    with patch("util.rankings.iter_descendants",
               return_value=[_SPECIES_A, _SUBSPECIES_A]):
        keys = rk._load_scope_keys(_GENUS, "SPECIES", True)
    assert "200" in keys
    assert "300" in keys


def test_query_catalog_corrupt_parquet(tmp_path, monkeypatch):
    """Corrupt catalog parquet returns no_catalog."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    catalog_dir = tmp_path / _GENUS["path"]
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "species.parquet").write_bytes(b"garbage")
    result = rk._query_catalog(
        within_taxon=_GENUS, descendant_rank="SPECIES",
        limit=10, offset=0, min_samples=0, include_species_like=False,
        loc_keys=None, loc_counts={},
    )
    assert result["empty_reason"] == "no_catalog"



def test_load_gid_levels_bad_level_value(tmp_path, monkeypatch):
    """Rows with non-integer level values are skipped."""
    csv_path = tmp_path / "hierarchy.csv"
    csv_path.write_text("level,gid,name,parent_gid\nbad,USA,United States,\n1,USA.1,Alabama,USA\n")
    monkeypatch.setattr(rk, "_HIERARCHY_CSV", csv_path)
    rk._load_gid_levels.cache_clear()
    levels = rk._load_gid_levels()
    assert "USA" not in levels  # bad level skipped
    assert levels["USA.1"] == 1
    rk._load_gid_levels.cache_clear()


def test_taxon_metric_value_corrupt_numerical_stats(tmp_path):
    """Corrupt numerical stats parquet falls through to nominal stats check."""
    from util.stats import NUMERICAL_STATS_FILE
    (tmp_path / NUMERICAL_STATS_FILE).write_bytes(b"garbage")
    _write_nominal_stats(tmp_path, "kg0", [("total_samples", 25.0)])
    result = rk._taxon_metric_value(tmp_path, "kg0", "total_samples")
    assert result == pytest.approx(25.0)


def test_taxon_metric_value_corrupt_nominal_stats(tmp_path):
    """Corrupt nominal stats parquet returns None."""
    from util.stats import NOMINAL_STATS_FILE
    (tmp_path / NOMINAL_STATS_FILE).write_bytes(b"garbage")
    assert rk._taxon_metric_value(tmp_path, "kg0", "total_samples") is None


def test_query_ranked_scoped_all_null_entries(tmp_path, monkeypatch):
    """Index column where all entries are null → no_column."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    ancestor_dir = tmp_path / _GENUS["path"]
    ancestor_dir.mkdir(parents=True)
    # Write an index with a null-only column
    struct_type = pa.struct([
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int64()),
    ])
    null_arr = pa.nulls(2, type=struct_type)
    import json
    table = pa.table({"bio1::mean": null_arr}).replace_schema_metadata(
        {b"column_lengths": json.dumps({"bio1::mean": 2}).encode()}
    )
    pq.write_table(table, ancestor_dir / "species_index.parquet")
    result = rk._query_ranked_scoped(
        q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
        sort_variable="bio1", sort_metric="mean", sort_order="asc",
        limit=10, offset=0, min_samples=0, include_species_like=False,
        loc_keys=None, loc_counts={},
    )
    assert result["empty_reason"] == "no_column"


def test_load_scope_keys_corrupt_catalog(tmp_path, monkeypatch):
    """Corrupt catalog parquet in _load_scope_keys falls through to DFS."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    catalog_dir = tmp_path / _GENUS["path"]
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "species.parquet").write_bytes(b"garbage")
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A]):
        keys = rk._load_scope_keys(_GENUS, "SPECIES", False)
    assert "200" in keys


def test_query_catalog_taxon_none_skipped(tmp_path, monkeypatch):
    """Entries whose get_taxon_by_id returns None are skipped."""
    monkeypatch.setattr(rk, "TREE_ROOT", tmp_path)
    catalog_dir = tmp_path / _GENUS["path"]
    catalog_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"taxon_key": "999", "path": "x/999", "scientific_name": "",
             "common_name": "", "rank": "SPECIES", "sample_count": 50},
        ]),
        catalog_dir / "species.parquet",
    )
    with patch("util.rankings.get_taxon_by_id", return_value=None):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["total"] == 0
