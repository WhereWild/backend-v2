# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for util/rankings.py."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
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
_NOMINAL_LAYER = {"id": "kg2", "value_type": "nominal"}
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

_FAMILY: dict = {
    "taxon_key": "50",
    "path": "Root_1/Order_10/Family_50",
    "scientific_name": "Testaceae",
    "common_name": "",
    "rank": "FAMILY",
}


def _seed_stats_cache(monkeypatch, tmp_path, layers, *, numerical=None, nominal=None, circular=None):
    """Populate rk._stats_cache (+ vocab) via the real preload_stats_cache(),
    same global-stats-file format scripts/process_tree.py::run_stats() writes.
    _write_rank_positions/build_rank_indexes/compute_relative_ranks always
    require _stats_cache to be populated (no per-taxon-directory disk fallback
    exists any more), so this is the standard setup for testing them."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    monkeypatch.setattr(rk, "_CACHE_FILE", tmp_path / "stats_cache.pkl.gz")
    if numerical is not None:
        pq.write_table(pa.Table.from_pylist(numerical), tmp_path / rk.NUMERICAL_STATS_FILE)
    if nominal is not None:
        pq.write_table(pa.Table.from_pylist(nominal), tmp_path / rk.NOMINAL_STATS_FILE)
    if circular is not None:
        pq.write_table(pa.Table.from_pylist(circular), tmp_path / rk.CIRCULAR_STATS_FILE)
    rk.preload_stats_cache(layers)


def _fake_rank_positions(table: dict[tuple[str, str, str, str], list[dict]]):
    """side_effect for patching rk._read_rank_positions with canned rows,
    keyed by (context_id, rank, variable, metric)."""
    def _read(context_id, rank, variable, metric):
        return table.get((context_id, rank, variable, metric), [])
    return _read


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


def test_metrics_for_vtype_ordinal():
    metrics = rk._metrics_for_vtype(_ORDINAL_LAYER, ValueType.ORDINAL)
    assert "median" in metrics
    assert "entropy" in metrics
    assert "unique_classes" in metrics
    assert "mean" not in metrics
    assert "min" not in metrics
    assert "max" not in metrics


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
# _batch_sample_counts
# ---------------------------------------------------------------------------

def test_batch_sample_counts_from_numerical(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "200"],
        "count": [42, 0],
    }), tmp_path / rk.NUMERICAL_STATS_FILE)
    result = rk._batch_sample_counts(["100", "200"])
    assert result == {"100": 42}  # 200's count of 0 isn't usable, falls to nominal fallback (no file → absent)


def test_batch_sample_counts_falls_back_to_nominal_total_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["200"],
        "metric": ["total_samples"],
        "value": [17.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    result = rk._batch_sample_counts(["200"])
    assert result == {"200": 17}


def test_batch_sample_counts_empty_input():
    assert rk._batch_sample_counts([]) == {}


def test_batch_sample_counts_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    assert rk._batch_sample_counts(["100"]) == {}


# ---------------------------------------------------------------------------
# _batch_metric_values
# ---------------------------------------------------------------------------

def test_batch_metric_values_from_numerical(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "200"],
        "variable": ["bio1", "bio1"],
        "mean": [5.0, 7.0],
    }), tmp_path / rk.NUMERICAL_STATS_FILE)
    values, rbars = rk._batch_metric_values(["100", "200"], "bio1", "mean")
    assert values == {"100": pytest.approx(5.0), "200": pytest.approx(7.0)}
    assert rbars == {}


def test_batch_metric_values_nominal_class_implicit_zero(tmp_path, monkeypatch):
    """A taxon with total_samples but no class_1 row gets an implicit 0.0."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "100", "200"],
        "variable": ["kg2", "kg2", "kg2"],
        "metric": ["class_1", "total_samples", "total_samples"],
        "value": [0.6, 50.0, 30.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    values, _ = rk._batch_metric_values(["100", "200"], "kg2", "class_1")
    assert values["100"] == pytest.approx(0.6)
    assert values["200"] == pytest.approx(0.0)


def test_batch_metric_values_nominal_no_total_samples_excluded(tmp_path, monkeypatch):
    """A taxon with no class_1 row and no total_samples row has no data for
    the variable at all — genuinely absent, not implicit zero."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["kg2"],
        "metric": ["class_1"],
        "value": [0.6],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    values, _ = rk._batch_metric_values(["100", "999"], "kg2", "class_1")
    assert "999" not in values


def test_batch_metric_values_circular_with_rbar(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["aspect_deg"],
        "circular_mean": [180.0],
        "rbar": [0.9],
    }), tmp_path / rk.CIRCULAR_STATS_FILE)
    values, rbars = rk._batch_metric_values(["100"], "aspect_deg", "circular_mean", need_rbar=True)
    assert values["100"] == pytest.approx(180.0)
    assert rbars["100"] == pytest.approx(0.9)


def test_batch_metric_values_empty_input():
    values, rbars = rk._batch_metric_values([], "bio1", "mean")
    assert values == {}
    assert rbars == {}


def test_batch_metric_values_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    values, rbars = rk._batch_metric_values(["100"], "bio1", "mean")
    assert values == {}
    assert rbars == {}


# ---------------------------------------------------------------------------
# RankingsSink
# ---------------------------------------------------------------------------

def test_rankings_sink_writes_rows(tmp_path):
    sink = rk.RankingsSink(tmp_path, "level_0001")
    sink.write(pa.table({"taxon_key": ["100"], "value": [1.0]}))
    sink.write(pa.table({"taxon_key": ["200"], "value": [2.0]}))
    sink.close()
    chunk = tmp_path / "level_0001.parquet"
    assert chunk.exists()
    df = pq.read_table(chunk).to_pandas()
    assert set(df["taxon_key"]) == {"100", "200"}


def test_rankings_sink_skips_empty_write(tmp_path):
    sink = rk.RankingsSink(tmp_path, "level_0001")
    sink.write(pa.table({"taxon_key": pa.array([], type=pa.string())}))
    sink.close()
    assert not (tmp_path / "level_0001.parquet").exists()


# ---------------------------------------------------------------------------
# _write_rank_positions / build_rank_indexes / compute_relative_ranks
# ---------------------------------------------------------------------------

def test_write_rank_positions_sorts_by_value(tmp_path, monkeypatch):
    _seed_stats_cache(monkeypatch, tmp_path, [_RATIO_LAYER], numerical=[
        {"taxon_key": "200", "variable": "bio1", "count": 10, "mean": 3.0},
        {"taxon_key": "201", "variable": "bio1", "count": 10, "mean": 7.0},
    ])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A, _SPECIES_B]):
        rk._write_rank_positions(_GENUS, "SPECIES", [_RATIO_LAYER], sink)
    sink.close()
    df = pq.read_table(tmp_path / "level_0000.parquet").to_pandas()
    row = df[(df["variable"] == "bio1") & (df["metric"] == "mean")].sort_values("position")
    assert list(row["taxon_key"]) == ["200", "201"]  # A (3.0) before B (7.0)
    assert row.iloc[0]["contextTaxonId"] == _GENUS["taxon_key"]
    assert row.iloc[0]["rank"] == "SPECIES"


def test_write_rank_positions_no_descendants_writes_nothing(tmp_path, monkeypatch):
    _seed_stats_cache(monkeypatch, tmp_path, [_RATIO_LAYER])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[]):
        rk._write_rank_positions(_GENUS, "SPECIES", [_RATIO_LAYER], sink)
    sink.close()
    assert not (tmp_path / "level_0000.parquet").exists()


def test_write_rank_positions_no_stats_cache_is_noop(tmp_path, monkeypatch):
    """No _stats_cache populated (e.g. never called preload_stats_cache) →
    no per-taxon-directory disk fallback exists any more, just a no-op."""
    monkeypatch.setattr(rk, "_stats_cache", None)
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A]):
        rk._write_rank_positions(_GENUS, "SPECIES", [_RATIO_LAYER], sink)
    sink.close()
    assert not (tmp_path / "level_0000.parquet").exists()


def test_write_rank_positions_no_matching_stats_writes_nothing(tmp_path, monkeypatch):
    """Descendant has no entry in _stats_cache → no rows produced."""
    _seed_stats_cache(monkeypatch, tmp_path, [_RATIO_LAYER])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A]):
        rk._write_rank_positions(_GENUS, "SPECIES", [_RATIO_LAYER], sink)
    sink.close()
    assert not (tmp_path / "level_0000.parquet").exists()


def test_write_rank_positions_class_metric_offset_by_implicit_zeros(tmp_path, monkeypatch):
    """A class_ metric's position is offset by the count of taxa with
    total_samples but no real (nonzero) entry for that class."""
    _seed_stats_cache(monkeypatch, tmp_path, [_NOMINAL_LAYER], nominal=[
        {"taxon_key": "200", "variable": "kg2", "metric": "class_1", "value": 0.6},
        {"taxon_key": "200", "variable": "kg2", "metric": "total_samples", "value": 50.0},
        {"taxon_key": "201", "variable": "kg2", "metric": "total_samples", "value": 30.0},
    ])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A, _SPECIES_B]):
        rk._write_rank_positions(_GENUS, "SPECIES", [_NOMINAL_LAYER], sink)
    sink.close()
    df = pq.read_table(tmp_path / "level_0000.parquet").to_pandas()
    row = df[(df["variable"] == "kg2") & (df["metric"] == "class_1")].iloc[0]
    assert row["taxon_key"] == "200"
    assert row["position"] == 1  # one implicit-zero taxon (201) occupies position 0
    assert row["count"] == 2  # full population = total_samples population


def test_build_rank_indexes_no_targets_writes_nothing(tmp_path, monkeypatch):
    """SUBSPECIES has no ranks below it → returns immediately, nothing written."""
    _seed_stats_cache(monkeypatch, tmp_path, [_RATIO_LAYER])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    rk.build_rank_indexes(_SUBSPECIES_A, [_RATIO_LAYER], sink)
    sink.close()
    assert not (tmp_path / "level_0000.parquet").exists()


def test_compute_relative_ranks_streams_into_sink(tmp_path, monkeypatch):
    _seed_stats_cache(monkeypatch, tmp_path, [_RATIO_LAYER], numerical=[
        {"taxon_key": "200", "variable": "bio1", "count": 10, "mean": 3.0},
    ])
    sink = rk.RankingsSink(tmp_path, "level_0000")
    with patch("util.rankings._descendants_for_rank", return_value=[_SPECIES_A]):
        rk.compute_relative_ranks(_GENUS, [_RATIO_LAYER], sink)
    sink.close()
    assert (tmp_path / "level_0000.parquet").exists()


# ---------------------------------------------------------------------------
# _read_rank_positions
# ---------------------------------------------------------------------------

def test_read_rank_positions_filters_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["200", "999"],
        "variable": ["bio1", "bio1"],
        "metric": ["mean", "mean"],
        "value": [3.0, 9.0],
        "position": pa.array([0, 0], type=pa.int32()),
        "count": pa.array([1, 1], type=pa.int32()),
        "sampleCount": pa.array([10, 10], type=pa.int32()),
        "contextTaxonId": ["100", "other"],
        "rank": ["SPECIES", "SPECIES"],
        "contextLabel": ["Testus", "Other"],
    }), tmp_path / rk.RANKINGS_FILE)
    rows = rk._read_rank_positions("100", "SPECIES", "bio1", "mean")
    assert len(rows) == 1
    assert rows[0]["taxon_key"] == "200"


def test_read_rank_positions_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    assert rk._read_rank_positions("100", "SPECIES", "bio1", "mean") == []


# ---------------------------------------------------------------------------
# _accepted_ranks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _query_ranked_scoped
# ---------------------------------------------------------------------------

def test_query_ranked_scoped_no_column():
    """No rows for this (context, rank, variable, metric) → no_column."""
    with patch("util.rankings._read_rank_positions", return_value=[]):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_column"


def test_query_ranked_scoped_taxon_none_in_accepted_ranks():
    """Entries whose get_taxon_by_id returns None are skipped in accepted_ranks filter."""
    fake = _fake_rank_positions({
        ("100", "SPECIES", "bio1", "mean"): [
            {"taxon_key": "200", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
            {"taxon_key": "999", "value": 20.0, "position": 1, "count": 2, "sampleCount": 50},
        ],
    })
    with patch("util.rankings._read_rank_positions", side_effect=fake), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: _SPECIES_A if k == "200" else None):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert len(result["results"]) == 1
    assert result["results"][0]["taxon"]["taxon_key"] == "200"


def test_query_ranked_scoped_taxon_none_in_results():
    """get_taxon_by_id returning None during result building skips the entry."""
    fake = _fake_rank_positions({
        ("50", "GENUS", "bio1", "mean"): [
            {"taxon_key": "100", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
            {"taxon_key": "999", "value": 20.0, "position": 1, "count": 2, "sampleCount": 50},
        ],
    })

    def _resolve(k):
        return _GENUS if k == "100" else None

    with patch("util.rankings._read_rank_positions", side_effect=fake), \
         patch("util.rankings.get_taxon_by_id", side_effect=_resolve):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_FAMILY, descendant_rank="GENUS",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    valid_ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert "100" in valid_ids
    assert "999" not in valid_ids


def test_query_ranked_scoped_class_metric_implicit_zero():
    """class_ metric: implicit-zero taxa (present in total_samples but not
    the class) are synthesized as zero-value entries."""
    fake = _fake_rank_positions({
        ("100", "SPECIES", "kg2", "class_1"): [
            {"taxon_key": "200", "value": 0.6, "position": 1, "count": 2, "sampleCount": 50},
        ],
        ("100", "SPECIES", "kg2", "total_samples"): [
            {"taxon_key": "200", "value": 50.0, "position": 0, "count": 2, "sampleCount": 50},
            {"taxon_key": "201", "value": 30.0, "position": 0, "count": 2, "sampleCount": 30},
        ],
    })
    with patch("util.rankings._read_rank_positions", side_effect=fake), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: {"200": _SPECIES_A, "201": _SPECIES_B}.get(k)):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
            sort_variable="kg2", sort_metric="class_1", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    ids = [r["taxon"]["taxon_key"] for r in result["results"]]
    assert ids == ["201", "200"]  # 201 implicit-zero sorts before 200's 0.6
    assert result["eligible_total"] == 2


# ---------------------------------------------------------------------------
# _query_ranked_text
# ---------------------------------------------------------------------------

def test_query_ranked_text_loc_keys_filter():
    """location filter in ranked-text mode skips taxa not in loc_keys."""
    with patch("util.rankings._batch_metric_values", return_value=({"200": 5.0, "201": 5.0}, {})), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 100, "201": 100}), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (_SPECIES_B, 80.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=frozenset({"200"}), loc_counts={},
        )
    assert len(result["results"]) == 1
    assert result["results"][0]["taxon"]["taxon_key"] == "200"


def test_query_ranked_text_no_metric_value():
    """Candidates with no metric value are excluded."""
    with patch("util.rankings._batch_metric_values", return_value=({}, {})), \
         patch("util.rankings._batch_sample_counts", return_value={}), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"


def test_query_ranked_text_min_samples_filter():
    """Candidates with too few samples are excluded."""
    with patch("util.rankings._batch_metric_values", return_value=({"200": 5.0}, {})), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 3}), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=10, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"


def test_query_ranked_text_min_rbar_filter():
    """Circular bearing sort with min_rbar excludes low-concentration taxa."""
    with patch("util.rankings._batch_metric_values",
               return_value=({"200": 90.0, "201": 90.0}, {"200": 0.8, "201": 0.1})), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 50, "201": 50}), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (_SPECIES_B, 80.0, "")]):
        result = rk._query_ranked_text(
            q="testus", sort_variable="aspect_deg", sort_metric="circular_mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={}, reference_value=0.0, min_rbar=0.5,
        )
    ids = [r["taxon"]["taxon_key"] for r in result["results"]]
    assert ids == ["200"]


# ---------------------------------------------------------------------------
# _query_text
# ---------------------------------------------------------------------------

def test_query_text_loc_keys_filter():
    """Location filter excludes candidates not in loc_keys."""
    with patch("util.rankings._batch_sample_counts", return_value={"200": 50, "201": 50}), \
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


def test_query_text_accepted_ranks_filter():
    """Rank filter excludes non-matching ranks."""
    subsp = {**_SUBSPECIES_A, "rank": "SUBSPECIES"}
    with patch("util.rankings._batch_sample_counts", return_value={"200": 50, "300": 50}), \
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


def test_query_text_min_samples_filter():
    """min_samples filter excludes candidates with too few samples."""
    with patch("util.rankings._batch_sample_counts", return_value={"200": 2}), \
         patch("util.rankings.search_taxa_by_name", return_value=[(_SPECIES_A, 90.0, "")]):
        result = rk._query_text(
            q="testus", within_taxon=None, descendant_rank=None,
            limit=10, offset=0, min_samples=10, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_results"


def test_query_text_scope_filter():
    """within_taxon + descendant_rank restricts candidates to the scope."""
    with patch("util.rankings._batch_sample_counts", return_value={"200": 50, "201": 50}), \
         patch("util.rankings.search_taxa_by_name",
               return_value=[(_SPECIES_A, 90.0, ""), (_SPECIES_B, 80.0, "")]), \
         patch("util.rankings.iter_descendants", return_value=[_SPECIES_A]):
        result = rk._query_text(
            q="testus", within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    ids = [r["taxon"]["taxon_key"] for r in result["results"]]
    assert ids == ["200"]


# ---------------------------------------------------------------------------
# _load_scope_keys
# ---------------------------------------------------------------------------

def test_load_scope_keys_excludes_subspecies_by_default():
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SUBSPECIES_A]):
        keys = rk._load_scope_keys(_GENUS, "SPECIES", False)
    assert "200" in keys
    assert "300" not in keys  # SUBSPECIES excluded when include_species_like=False


def test_load_scope_keys_include_species_like():
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SUBSPECIES_A]):
        keys = rk._load_scope_keys(_GENUS, "SPECIES", True)
    assert "200" in keys
    assert "300" in keys


def test_load_scope_keys_non_species_rank():
    family = {**_FAMILY}
    with patch("util.rankings.iter_descendants", return_value=[_GENUS, _SPECIES_A]):
        keys = rk._load_scope_keys(family, "GENUS", False)
    assert keys == frozenset({"100"})


# ---------------------------------------------------------------------------
# _query_catalog
# ---------------------------------------------------------------------------

def test_query_catalog_no_scope_members():
    with patch("util.rankings.iter_descendants", return_value=[]):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["empty_reason"] == "no_catalog"


def test_query_catalog_taxon_none_skipped():
    """Entries whose get_taxon_by_id returns None are skipped."""
    unknown = {**_SPECIES_A, "taxon_key": "999"}
    with patch("util.rankings.iter_descendants", return_value=[unknown]), \
         patch("util.rankings._batch_sample_counts", return_value={"999": 50}), \
         patch("util.rankings.get_taxon_by_id", return_value=None):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["total"] == 0


def test_query_catalog_returns_scope_members_sorted():
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SPECIES_B]), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 10, "201": 20}), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: {"200": _SPECIES_A, "201": _SPECIES_B}.get(k)):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    assert result["total"] == 2
    assert result["eligible_total"] == 2
    ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert ids == {"200", "201"}


def test_query_catalog_min_samples_filter():
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SPECIES_B]), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 2, "201": 20}), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: {"200": _SPECIES_A, "201": _SPECIES_B}.get(k)):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=10, include_species_like=False,
            loc_keys=None, loc_counts={},
        )
    ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert ids == {"201"}


# ---------------------------------------------------------------------------
# _load_gid_levels / _gid_to_scope / _location_taxon_keys
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# preload_stats_cache — reads the consolidated global stats files (not
# per-taxon files) since scripts/process_tree.py::run_stats() now writes
# stats straight into one sorted file per type.
# ---------------------------------------------------------------------------

def test_wide_seen_and_counts_numerical(tmp_path):
    path = tmp_path / "numerical_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100", "200"],
        "variable": ["bio1", "bio1"],
        "count": [10, 20],
        "mean": [5.0, 7.5],
    }), path)
    seen, sample_counts = rk._wide_seen_and_counts(path, "numerical", {"bio1"})
    assert seen == {"100", "200"}
    assert sample_counts["100"] == 10
    assert sample_counts["200"] == 20


def test_wide_seen_and_counts_ignores_unknown_variable(tmp_path):
    path = tmp_path / "numerical_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["not_a_configured_layer"],
        "count": [10],
        "mean": [5.0],
    }), path)
    seen, sample_counts = rk._wide_seen_and_counts(path, "numerical", {"bio1"})
    assert seen == set()
    assert sample_counts is None


def test_wide_seen_and_counts_missing_file_is_noop(tmp_path):
    seen, sample_counts = rk._wide_seen_and_counts(tmp_path / "nope.parquet", "numerical", {"bio1"})
    assert seen == set()
    assert sample_counts is None


def test_wide_fill_numerical(tmp_path):
    path = tmp_path / "numerical_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100", "200"],
        "variable": ["bio1", "bio1"],
        "count": [10, 20],
        "mean": [5.0, 7.5],
    }), path)
    row_idx = pd.Index(["100", "200"])
    taxon_pos = pd.Series([0, 1], index=row_idx)
    metric_pos = pd.Series([0], index=["bio1::mean"])
    values = np.full((2, 1), np.nan, dtype=np.float32)
    rk._wide_fill(values, taxon_pos, metric_pos, path, "numerical", {"bio1"}, {"bio1": ("mean",)})
    assert values[0, 0] == pytest.approx(5.0)
    assert values[1, 0] == pytest.approx(7.5)


def test_wide_fill_missing_file_is_noop(tmp_path):
    row_idx = pd.Index(["100"])
    taxon_pos = pd.Series([0], index=row_idx)
    metric_pos = pd.Series([0], index=["bio1::mean"])
    values = np.full((1, 1), np.nan, dtype=np.float32)
    rk._wide_fill(values, taxon_pos, metric_pos, tmp_path / "nope.parquet", "numerical", {"bio1"}, {})
    assert np.isnan(values[0, 0])


def test_wide_fill_circular(tmp_path):
    path = tmp_path / "circular_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["aspect_deg"],
        "circular_mean": [180.0],
        "rbar": [0.9],
    }), path)
    row_idx = pd.Index(["100"])
    taxon_pos = pd.Series([0], index=row_idx)
    metric_pos = pd.Series([0, 1], index=["aspect_deg::circular_mean", "aspect_deg::rbar"])
    values = np.full((1, 2), np.nan, dtype=np.float32)
    rk._wide_fill(values, taxon_pos, metric_pos, path, "circular", {"aspect_deg"}, None,
                  circ_metrics=("circular_mean", "rbar"))
    assert values[0, 0] == pytest.approx(180.0)
    assert values[0, 1] == pytest.approx(0.9)


def test_tall_seen_counts_and_vocab_nominal(tmp_path):
    path = tmp_path / "nominal_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100", "100"],
        "variable": ["kg2", "kg2"],
        "metric": ["total_samples", "class_1"],
        "value": [50.0, 0.6],
    }), path)
    seen, sample_counts, vocab = rk._tall_seen_counts_and_vocab(path, {"kg2"}, {"total_samples"})
    assert seen == {"100"}
    assert sample_counts["100"] == 50
    assert vocab == {"kg2::total_samples", "kg2::class_1"}


def test_tall_seen_counts_and_vocab_ignores_unknown_variable(tmp_path):
    path = tmp_path / "nominal_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["not_configured"],
        "metric": ["total_samples"],
        "value": [50.0],
    }), path)
    seen, sample_counts, vocab = rk._tall_seen_counts_and_vocab(path, {"kg2"}, {"total_samples"})
    assert seen == set()
    assert sample_counts is None
    assert vocab == set()


def test_tall_fill_nominal(tmp_path):
    path = tmp_path / "nominal_stats.parquet"
    pq.write_table(pa.table({
        "taxon_key": ["100", "100"],
        "variable": ["kg2", "kg2"],
        "metric": ["total_samples", "class_1"],
        "value": [50.0, 0.6],
    }), path)
    row_idx = pd.Index(["100"])
    taxon_pos = pd.Series([0], index=row_idx)
    metric_pos = pd.Series([0, 1], index=["kg2::total_samples", "kg2::class_1"])
    values = np.full((1, 2), np.nan, dtype=np.float32)
    rk._tall_fill(values, taxon_pos, metric_pos, path, {"kg2"}, {"total_samples"})
    assert values[0, 1] == pytest.approx(0.6)


def test_preload_stats_cache_end_to_end(tmp_path, monkeypatch):
    """Reads the four global stats files and builds the compact
    (sample_count, float32 array) cache format."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    monkeypatch.setattr(rk, "_CACHE_FILE", tmp_path / "stats_cache.pkl.gz")
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["bio1"],
        "count": [10],
        "mean": [5.0],
    }), tmp_path / rk.NUMERICAL_STATS_FILE)
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["kg2"],
        "metric": ["class_1"],
        "value": [0.6],
    }), tmp_path / rk.NOMINAL_STATS_FILE)

    rk.preload_stats_cache([_RATIO_LAYER, _NOMINAL_LAYER])

    assert "100" in rk._stats_cache
    sample_count, arr = rk._stats_cache["100"]
    assert sample_count == 10
    mean_idx = rk._metric_to_idx["bio1::mean"]
    assert arr[mean_idx] == pytest.approx(5.0, abs=1e-4)
    class_idx = rk._metric_to_idx["kg2::class_1"]
    assert arr[class_idx] == pytest.approx(0.6, abs=1e-4)


def test_preload_stats_cache_missing_files_yields_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    monkeypatch.setattr(rk, "_CACHE_FILE", tmp_path / "stats_cache.pkl.gz")
    rk.preload_stats_cache([_RATIO_LAYER])
    assert rk._stats_cache == {}


# ---------------------------------------------------------------------------
# parse_stat_filter
# ---------------------------------------------------------------------------

def test_parse_stat_filter_basic():
    f = rk.parse_stat_filter("bio1:mean:gte:10")
    assert f == rk.StatFilter(variable="bio1", metric="mean", op="gte", value=10.0, as_count=False)


def test_parse_stat_filter_count_modifier():
    f = rk.parse_stat_filter("ecoregions:class_356:gte:10:count")
    assert f.as_count is True
    assert f.value == pytest.approx(10.0)


def test_parse_stat_filter_bad_operator():
    with pytest.raises(ValueError, match="unknown filter operator"):
        rk.parse_stat_filter("bio1:mean:xyz:10")


def test_parse_stat_filter_bad_value():
    with pytest.raises(ValueError, match="non-numeric"):
        rk.parse_stat_filter("bio1:mean:gte:abc")


def test_parse_stat_filter_wrong_part_count():
    with pytest.raises(ValueError, match="malformed filter"):
        rk.parse_stat_filter("bio1:mean:gte")


def test_parse_stat_filter_unknown_modifier():
    with pytest.raises(ValueError, match="unknown filter modifier"):
        rk.parse_stat_filter("bio1:mean:gte:10:bogus")


# ---------------------------------------------------------------------------
# _filter_wide / _filter_tall / _apply_stat_filters
# ---------------------------------------------------------------------------

def test_filter_wide_numerical(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "200", "300"],
        "variable": ["bio1", "bio1", "bio1"],
        "mean": [5.0, 15.0, 25.0],
    }), tmp_path / rk.NUMERICAL_STATS_FILE)
    f = rk.StatFilter(variable="bio1", metric="mean", op="lt", value=20.0)
    result = rk._filter_wide(tmp_path / rk.NUMERICAL_STATS_FILE, frozenset({"100", "200", "300"}), f)
    assert result == frozenset({"100", "200"})


def test_filter_wide_missing_file(tmp_path):
    f = rk.StatFilter(variable="bio1", metric="mean", op="lt", value=20.0)
    assert rk._filter_wide(tmp_path / "nope.parquet", frozenset({"100"}), f) == frozenset()


def test_filter_wide_empty_keys(tmp_path):
    f = rk.StatFilter(variable="bio1", metric="mean", op="lt", value=20.0)
    assert rk._filter_wide(tmp_path / "nope.parquet", frozenset(), f) == frozenset()


def test_filter_tall_nominal_value_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "200"],
        "variable": ["kg2", "kg2"],
        "metric": ["entropy", "entropy"],
        "value": [1.0, 3.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    f = rk.StatFilter(variable="kg2", metric="entropy", op="lt", value=2.0)
    result = rk._filter_tall(tmp_path / rk.NOMINAL_STATS_FILE, frozenset({"100", "200"}), f)
    assert result == frozenset({"100"})


def test_filter_tall_class_count_reconstruction(tmp_path, monkeypatch):
    """class_356 fraction * total_samples reconstructs the raw observation count."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "100", "200"],
        "variable": ["ecoregions", "ecoregions", "ecoregions"],
        "metric": ["class_356", "total_samples", "total_samples"],
        "value": [0.5, 20.0, 30.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    # taxon 100: 0.5 * 20 = 10 observations in class_356 -> passes gte 10
    # taxon 200: no class_356 row (implicit 0.0) * 30 = 0 -> fails gte 10
    f = rk.StatFilter(variable="ecoregions", metric="class_356", op="gte", value=10.0, as_count=True)
    result = rk._filter_tall(tmp_path / rk.NOMINAL_STATS_FILE, frozenset({"100", "200"}), f)
    assert result == frozenset({"100"})


def test_filter_tall_implicit_zero_passes_lt_filter(tmp_path, monkeypatch):
    """A taxon with total_samples but no class row has an implicit 0.0 —
    that should satisfy a 'less than' filter, not be silently excluded."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["ecoregions"],
        "metric": ["total_samples"],
        "value": [30.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    f = rk.StatFilter(variable="ecoregions", metric="class_356", op="lt", value=0.5)
    result = rk._filter_tall(tmp_path / rk.NOMINAL_STATS_FILE, frozenset({"100"}), f)
    assert result == frozenset({"100"})


def test_filter_tall_no_data_excluded(tmp_path, monkeypatch):
    """A taxon with no total_samples and no class row for the variable at all
    has no data for it — excluded regardless of operator."""
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100"],
        "variable": ["ecoregions"],
        "metric": ["total_samples"],
        "value": [30.0],
    }), tmp_path / rk.NOMINAL_STATS_FILE)
    f = rk.StatFilter(variable="ecoregions", metric="class_356", op="lt", value=0.5)
    result = rk._filter_tall(tmp_path / rk.NOMINAL_STATS_FILE, frozenset({"100", "999"}), f)
    assert "999" not in result


def test_apply_stat_filters_chains_multiple(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "taxon_key": ["100", "200", "300"],
        "variable": ["bio1", "bio1", "bio1"],
        "mean": [5.0, 15.0, 25.0],
        "std": [1.0, 2.0, 3.0],
    }), tmp_path / rk.NUMERICAL_STATS_FILE)
    filters = [
        rk.StatFilter(variable="bio1", metric="mean", op="lt", value=20.0),
        rk.StatFilter(variable="bio1", metric="std", op="gte", value=2.0),
    ]
    result = rk._apply_stat_filters(frozenset({"100", "200", "300"}), filters, [_RATIO_LAYER])
    assert result == frozenset({"200"})


def test_apply_stat_filters_unknown_variable_excludes_all():
    filters = [rk.StatFilter(variable="not_a_layer", metric="mean", op="gte", value=0.0)]
    result = rk._apply_stat_filters(frozenset({"100"}), filters, [_RATIO_LAYER])
    assert result == frozenset()


def test_apply_stat_filters_empty_list_returns_input():
    result = rk._apply_stat_filters(frozenset({"100", "200"}), [], [_RATIO_LAYER])
    assert result == frozenset({"100", "200"})


# ---------------------------------------------------------------------------
# stat_filters integration with query modes
# ---------------------------------------------------------------------------

def test_query_catalog_stat_filter_narrows_results():
    with patch("util.rankings.iter_descendants", return_value=[_SPECIES_A, _SPECIES_B]), \
         patch("util.rankings._batch_sample_counts", return_value={"200": 10, "201": 20}), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: {"200": _SPECIES_A, "201": _SPECIES_B}.get(k)), \
         patch("util.rankings._apply_stat_filters", return_value=frozenset({"201"})):
        result = rk._query_catalog(
            within_taxon=_GENUS, descendant_rank="SPECIES",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
            stat_filters=[rk.StatFilter(variable="bio1", metric="mean", op="gte", value=0.0)],
            layers=[_RATIO_LAYER],
        )
    ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert ids == {"201"}


def test_query_ranked_scoped_stat_filter_narrows_results():
    rows = [
        {"taxon_key": "200", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
        {"taxon_key": "201", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]
    fake = _fake_rank_positions({("100", "SPECIES", "bio1", "mean"): rows})
    with patch("util.rankings._read_rank_positions", side_effect=fake), \
         patch("util.rankings.get_taxon_by_id", side_effect=lambda k: {"200": _SPECIES_A, "201": _SPECIES_B}.get(k)), \
         patch("util.rankings._apply_stat_filters", return_value=frozenset({"201"})):
        result = rk._query_ranked_scoped(
            q=None, within_taxon=_GENUS, descendant_rank="SPECIES",
            sort_variable="bio1", sort_metric="mean", sort_order="asc",
            limit=10, offset=0, min_samples=0, include_species_like=False,
            loc_keys=None, loc_counts={},
            stat_filters=[rk.StatFilter(variable="kg2", metric="entropy", op="lt", value=2.0)],
            layers=[_RATIO_LAYER],
        )
    ids = {r["taxon"]["taxon_key"] for r in result["results"]}
    assert ids == {"201"}
