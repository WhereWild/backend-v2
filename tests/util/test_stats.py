# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely

import util.stats as st
from config.config import ValueType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_occ_rows(
    occurrences_file: Path, taxon: dict, extra_cols: dict | None = None,
    n: int = 20, offset: int = 0,
) -> None:
    """Append n rows for one taxon to the shared consolidated occurrences file."""
    occurrences_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": [f"obs{offset + i}" for i in range(n)],
        "decimalLatitude": [40.0 + i * 0.01 for i in range(n)],
        "decimalLongitude": [-105.0 + i * 0.01 for i in range(n)],
        "hilbertIdx": list(range(n)),
        "obscured": ["No"] * n,
        "coordinateUncertaintyInMeters": [100.0] * n,
        "taxon_key": [taxon["taxon_key"]] * n,
    }
    if extra_cols:
        data.update(extra_cols)
    new_table = pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False)
    if occurrences_file.exists():
        existing = pq.read_table(occurrences_file)
        new_table = pa.concat_tables([existing, new_table], promote_options="default")
    pq.write_table(new_table, occurrences_file)


_CONTINUOUS_LAYER = {"id": "bio1", "value_type": "ratio", "scale_factor": 0.1, "add_offset": -273.15}
_NOMINAL_LAYER    = {"id": "kg2",  "value_type": "nominal", "scale_factor": None, "add_offset": None}
_DISCRETE_LAYER   = {"id": "gsl",  "value_type": "ratio",   "scale_factor": None, "add_offset": None, "domain": "discrete"}

_LEAF_TAXON = {"taxon_key": "1", "path": "Root_1/Leaf_1", "scientific_name": "Leafus", "common_name": "", "rank": "SPECIES"}

FAKE_TAXON: dict = {
    "taxon_key": "9999",
    "path": "Root_1/Parent_9999",
    "scientific_name": "Parentus testus",
    "common_name": "",
    "rank": "GENUS",
}

CHILD_TAXON: dict = {
    "taxon_key": "10000",
    "path": "Root_1/Parent_9999/Child_10000",
    "scientific_name": "Parentus testus subsp. child",
    "common_name": "",
    "rank": "SPECIES",
}

SPECIES_TAXON: dict = {
    "taxon_key": "2923970",
    "path": "Root_1/Species_2923970",
    "scientific_name": "Testus specius",
    "common_name": "",
    "rank": "SPECIES",
}

SUBSPECIES_TAXON: dict = {
    "taxon_key": "2923971",
    "path": "Root_1/Species_2923970/Subspecies_2923971",
    "scientific_name": "Testus specius subsp. test",
    "common_name": "",
    "rank": "SUBSPECIES",
}

_ALL_TAXA_CATALOG = {
    t["taxon_key"]: t for t in (_LEAF_TAXON, FAKE_TAXON, CHILD_TAXON, SPECIES_TAXON, SUBSPECIES_TAXON)
}


@pytest.fixture(autouse=True)
def _patch_catalog(monkeypatch):
    """Subtree-scoped reads (_read_subtree_rows) resolve descendant taxon_keys
    from the catalog, not a stored path column — every test needs a catalog
    covering whichever taxa it writes occurrence rows for."""
    monkeypatch.setattr(st, "load_catalog", lambda: _ALL_TAXA_CATALOG)


# ---------------------------------------------------------------------------
# _layer_value_type
# ---------------------------------------------------------------------------

def test_layer_value_type_known():
    assert st._layer_value_type({"value_type": "ratio"}) == ValueType.RATIO
    assert st._layer_value_type({"value_type": "nominal"}) == ValueType.NOMINAL


def test_layer_value_type_unknown():
    assert st._layer_value_type({"value_type": "bogus"}) is None
    assert st._layer_value_type({}) is None


# ---------------------------------------------------------------------------
# _filter_df
# ---------------------------------------------------------------------------

def test_filter_df_removes_obscured():
    df = pd.DataFrame({"obscured": ["No", "Yes", "No"], "x": [1, 2, 3]})
    result = st._filter_df(df)
    assert list(result["x"]) == [1, 3]


def test_filter_df_removes_high_uncertainty():
    df = pd.DataFrame({"coordinateUncertaintyInMeters": [100.0, 600.0, 500.0], "x": [1, 2, 3]})
    result = st._filter_df(df)
    assert list(result["x"]) == [1, 3]


def test_filter_df_missing_columns_ok():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = st._filter_df(df)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# _reservoir_update
# ---------------------------------------------------------------------------

def test_reservoir_fills_up_to_max():
    reservoir, n = [], 0
    vals = np.arange(100.0)
    n = st._reservoir_update(reservoir, n, vals)
    assert n == 100
    assert len(reservoir) == 100


def test_reservoir_caps_at_max_samples(monkeypatch):
    monkeypatch.setattr(st, "_KDE_MAX_SAMPLES", 5)
    reservoir, n = [], 0
    n = st._reservoir_update(reservoir, n, np.arange(10.0))
    assert len(reservoir) == 5
    assert n == 10


# ---------------------------------------------------------------------------
# build_density_curve
# ---------------------------------------------------------------------------

def test_build_density_curve_ratio():
    vals = np.linspace(1, 10, 200)
    curve = st.build_density_curve(vals, ValueType.RATIO)
    assert curve is not None
    assert "points" in curve
    assert "density" in curve
    assert "mode" in curve
    assert len(curve["points"]) == st._KDE_N_POINTS
    assert len(curve["points"]) == len(curve["density"])


def test_build_density_curve_interval():
    vals = np.linspace(0, 100, 200)
    curve = st.build_density_curve(vals, ValueType.INTERVAL)
    assert curve is not None
    assert math.isfinite(curve["mode"])


def test_build_density_curve_too_few_values():
    curve = st.build_density_curve(np.array([5.0]), ValueType.RATIO)
    assert curve is None


def test_build_density_curve_constant_values():
    # All same value — should still return a curve (with expanded range)
    vals = np.full(50, 3.14)
    curve = st.build_density_curve(vals, ValueType.RATIO)
    assert curve is not None
    assert math.isfinite(curve["mode"])


def test_build_density_curve_circular_returns_curve():
    curve = st.build_density_curve(np.array([0.0, 90.0, 180.0, 270.0]), ValueType.CIRCULAR)
    assert curve is not None
    assert len(curve["points"]) == len(curve["density"])
    assert curve["min"] == 0.0
    assert curve["max"] == 360.0
    assert math.isfinite(curve["mode"])


def test_build_density_curve_nominal_returns_none():
    assert st.build_density_curve(np.array([1.0, 2.0, 3.0]), ValueType.NOMINAL) is None


def test_build_density_curve_ordinal_returns_none():
    assert st.build_density_curve(np.array([1.0, 2.0, 3.0]), ValueType.ORDINAL) is None


# ---------------------------------------------------------------------------
# _continuous_stats_exact
# ---------------------------------------------------------------------------

def test_continuous_stats_exact_keys():
    series = pd.Series(np.linspace(1, 100, 200))
    kde = st.build_density_curve(series.to_numpy(), ValueType.RATIO)
    stats = st._continuous_stats_exact(series, 200, kde)
    expected = {"count", "unique_samples", "min", "10th_percentile", "25th_percentile",
                "median", "75th_percentile", "90th_percentile", "max",
                "mean", "std", "variance", "iqr", "10_90_range", "range", "mode"}
    assert expected.issubset(set(stats.keys()))


def test_continuous_stats_exact_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10)
    stats = st._continuous_stats_exact(series, 100, None)
    assert stats["count"] == 100
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(10.0)
    assert stats["mode"] is None
    assert stats["iqr"] == pytest.approx(stats["75th_percentile"] - stats["25th_percentile"])
    assert stats["10_90_range"] == pytest.approx(stats["90th_percentile"] - stats["10th_percentile"])
    assert stats["range"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# _continuous_stats_streaming
# ---------------------------------------------------------------------------

def test_continuous_stats_streaming_keys():
    from fastdigest import TDigest
    digest = TDigest()
    digest.batch_update(np.linspace(1, 100, 1000).tolist())
    kde = st.build_density_curve(np.linspace(1, 100, 1000), ValueType.RATIO)
    stats = st._continuous_stats_streaming(digest, 1000, kde)
    expected = {"count", "unique_samples", "min", "10th_percentile", "25th_percentile",
                "median", "75th_percentile", "90th_percentile", "max",
                "mean", "std", "variance", "iqr", "10_90_range", "range", "mode"}
    assert expected.issubset(set(stats.keys()))


def test_continuous_stats_streaming_accuracy():
    from fastdigest import TDigest
    rng = np.random.default_rng(0)
    vals = rng.normal(50, 10, 5000)
    digest = TDigest()
    digest.batch_update(vals.tolist())
    stats = st._continuous_stats_streaming(digest, 5000, None)
    assert stats["mean"] == pytest.approx(vals.mean(), abs=0.5)
    assert stats["min"] == pytest.approx(vals.min(), abs=0.01)
    assert stats["max"] == pytest.approx(vals.max(), abs=0.01)
    assert stats["mode"] is None


# ---------------------------------------------------------------------------
# _nominal_stats
# ---------------------------------------------------------------------------

def test_nominal_stats_basic():
    counts = Counter({1: 50, 2: 30, 3: 20})
    summary, distribution = st._nominal_stats(counts, 100)
    assert summary["unique_samples"] == 100
    assert summary["total_samples"] == 100
    assert summary["unique_classes"] == 3
    assert summary["mode"] == 1
    assert summary["entropy"] == pytest.approx(-0.5 * math.log(0.5) - 0.3 * math.log(0.3) - 0.2 * math.log(0.2), abs=1e-6)
    assert distribution[0]["class_id"] == 1
    assert distribution[0]["fraction"] == pytest.approx(0.5)


def test_nominal_stats_empty():
    summary, distribution = st._nominal_stats(Counter(), 0)
    assert summary == {}
    assert distribution == []


def test_nominal_stats_uniform_max_entropy():
    counts = Counter({k: 10 for k in range(4)})
    summary, _ = st._nominal_stats(counts, 40)
    assert summary["entropy"] == pytest.approx(math.log(4), abs=1e-6)


# ---------------------------------------------------------------------------
# _nominal_cat_entries
# ---------------------------------------------------------------------------

def test_nominal_cat_entries_structure():
    counts = Counter({1: 80, 2: 20})
    summary = {"unique_samples": 50, "total_samples": 100, "unique_classes": 2, "entropy": 0.5, "mode": 1}
    layer = {"id": "kg2", "display_name": "Köppen-Geiger", "source": "chelsa_v2_1"}
    entries = st._nominal_cat_entries("kg2", layer, counts, summary)
    metrics = {e["metric"] for e in entries}
    assert "unique_samples" in metrics
    assert "total_samples" in metrics
    assert "unique_classes" in metrics
    assert "entropy" in metrics
    assert "mode" in metrics
    assert "class_1" in metrics
    assert "class_2" in metrics
    fracs = {e["metric"]: e["value"] for e in entries if e["metric"].startswith("class_")}
    assert fracs["class_1"] == pytest.approx(0.8)
    assert fracs["class_2"] == pytest.approx(0.2)
    assert all("variableName" not in e for e in entries)
    assert all("variableCategory" not in e for e in entries)


def test_process_observations_df_delegates_to_leaf(tmp_path):
    df = pd.DataFrame({
        "catalogNumber": ["A"],
        "decimalLatitude": [45.0],
        "decimalLongitude": [-120.0],
    })
    with patch("util.stats._process_leaf_df") as mock_leaf:
        st.process_observations_df(tmp_path, df, {})
    mock_leaf.assert_called_once()
    call_args = mock_leaf.call_args[0]
    assert isinstance(call_args[0], st._DirStatsTarget)
    assert call_args[0].directory == tmp_path
    assert call_args[1] == ""
    assert call_args[2] is df
    assert call_args[3] == {}


# ---------------------------------------------------------------------------
# _write_* helpers (round-trip)
# ---------------------------------------------------------------------------

def test_write_read_stats_frame(tmp_path):
    stats = {"bio1": {"count": 100, "mean": 20.0, "mode": 19.5}}
    st._write_stats_frame(tmp_path / st.NUMERICAL_STATS_FILE, stats)
    assert (tmp_path / st.NUMERICAL_STATS_FILE).exists()
    df = pd.read_parquet(tmp_path / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "bio1"].iloc[0]
    assert row["count"] == pytest.approx(100)
    assert row["mean"] == pytest.approx(20.0)


def test_write_stats_frame_empty(tmp_path):
    st._write_stats_frame(tmp_path / st.NUMERICAL_STATS_FILE, {})
    assert not (tmp_path / st.NUMERICAL_STATS_FILE).exists()


def test_write_read_nominal_stats(tmp_path):
    entries = [
        {"variable": "kg2", "metric": "total_samples", "value": 100.0},
        {"variable": "kg2", "metric": "class_1", "value": 0.6},
    ]
    st._write_nominal_stats(tmp_path, entries)
    df = pd.read_parquet(tmp_path / st.NOMINAL_STATS_FILE)
    assert len(df) == 2


def test_write_nominal_stats_empty(tmp_path):
    st._write_nominal_stats(tmp_path, [])
    assert not (tmp_path / st.NOMINAL_STATS_FILE).exists()


def test_write_read_density(tmp_path):
    rows = [{"variable": "bio1", "count": 50, "sampleCount": 50, "pointCount": 8,
             "points": [1.0, 2.0], "density": [0.3, 0.7], "min": 1.0, "max": 2.0,
             "bandwidth": 0.5}]
    st._write_density(tmp_path, rows)
    df = pd.read_parquet(tmp_path / st.DENSITY_FILE)
    assert df["variable"].iloc[0] == "bio1"


def test_write_density_empty(tmp_path):
    st._write_density(tmp_path, [])
    assert not (tmp_path / st.DENSITY_FILE).exists()


# ---------------------------------------------------------------------------
# StatsSink
# ---------------------------------------------------------------------------

def test_stats_sink_writes_numerical_chunk_with_taxon_key(tmp_path):
    sink = st.StatsSink(tmp_path, "level_0001")
    sink.write_numerical("111", {"bio1": {"mean": 5.0, "count": 10}})
    sink.write_numerical("222", {"bio1": {"mean": 7.0, "count": 20}})
    sink.close()
    chunk = tmp_path / "numerical_stats" / "level_0001.parquet"
    assert chunk.exists()
    df = pq.read_table(chunk).to_pandas()
    assert set(df["taxon_key"]) == {"111", "222"}
    assert "variable" in df.columns
    assert "mean" in df.columns


def test_stats_sink_writes_phenology_alongside_numerical(tmp_path):
    sink = st.StatsSink(tmp_path, "level_0001")
    sink.write_numerical("111", {"bio1": {"mean": 5.0}}, pheno_meta={"phenology_counts": '{"flowers": 3}'})
    sink.close()
    pheno_chunk = tmp_path / "phenology_counts" / "level_0001.parquet"
    assert pheno_chunk.exists()
    rows = pq.read_table(pheno_chunk).to_pylist()
    assert rows == [{"taxon_key": "111", "phenology_value": "flowers", "count": 3}]


def test_stats_sink_skips_empty_writes(tmp_path):
    sink = st.StatsSink(tmp_path, "level_0001")
    sink.write_numerical("111", {})
    sink.write_nominal("111", [])
    sink.write_ordinal("111", [])
    sink.write_circular("111", {})
    sink.write_density("111", [])
    sink.close()
    assert not any(tmp_path.iterdir())


def test_stats_sink_nominal_and_density_include_taxon_key(tmp_path):
    sink = st.StatsSink(tmp_path, "level_0001")
    sink.write_nominal("111", [{"variable": "kg2", "metric": "class_1", "value": 0.5}])
    sink.write_density("111", [{"variable": "bio1", "points": [1.0], "density": [1.0]}])
    sink.close()
    nom = pq.read_table(tmp_path / "nominal_stats" / "level_0001.parquet").to_pylist()
    assert nom == [{"taxon_key": "111", "variable": "kg2", "metric": "class_1", "value": 0.5}]
    den = pq.read_table(tmp_path / "density" / "level_0001.parquet").to_pylist()
    assert den[0]["taxon_key"] == "111"


def test_stats_sink_multiple_taxa_append_to_same_level_chunk(tmp_path):
    sink = st.StatsSink(tmp_path, "level_0001")
    for i in range(5):
        sink.write_nominal(str(i), [{"variable": "kg2", "metric": "class_1", "value": float(i)}])
    sink.close()
    df = pq.read_table(tmp_path / "nominal_stats" / "level_0001.parquet").to_pandas()
    assert len(df) == 5
    assert set(df["taxon_key"]) == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# _process_leaf
# ---------------------------------------------------------------------------

def test_process_leaf_continuous(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    bio1_vals = list(np.linspace(10.0, 30.0, 20))
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"bio1": bio1_vals})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.DENSITY_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "bio1"].iloc[0]
    assert row["count"] == 20
    assert row["unique_samples"] == 20


def test_process_leaf_discrete(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    vals = [42] * 10 + [43] * 5 + [44] * 5
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"gsl": [float(v) for v in vals]})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"gsl": _DISCRETE_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.DENSITY_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "gsl"].iloc[0]
    assert row["mode"] == 42
    assert isinstance(row["mode"], (int, np.integer))
    den = pd.read_parquet(taxon_dir / st.DENSITY_FILE)
    hist_row = den[den["variable"] == "gsl"].iloc[0]
    assert hist_row["pointCount"] == 3
    assert list(hist_row["points"]) == [42.0, 43.0, 44.0]
    assert abs(sum(hist_row["density"]) - 1.0) < 1e-9


def test_process_leaf_nominal(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"kg2": [1.0] * 15 + [2.0] * 5})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"kg2": _NOMINAL_LAYER})
    assert (taxon_dir / st.NOMINAL_STATS_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NOMINAL_STATS_FILE)
    metrics = dict(zip(df["metric"], df["value"]))
    assert metrics["unique_classes"] == 2
    assert metrics["total_samples"] == 20
    assert metrics["mode"] == pytest.approx(1.0)
    assert "class_1" in metrics


def test_process_leaf_no_occurrences_file(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet")
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_empty_parquet(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    pq.write_table(pa.table({
        "catalogNumber": pa.array([], type=pa.string()),
        "taxon_key": pa.array([], type=pa.string()),
    }), occurrences_file)
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_filtered_out(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"bio1": [5.0] * 20})
    df = pq.read_table(occurrences_file).to_pandas()
    df["obscured"] = "Yes"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), occurrences_file)
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_circular_produces_stats(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"circ": [45.0] * 20})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"circ": {"id": "circ", "value_type": "circular"}})
    assert (taxon_dir / st.CIRCULAR_STATS_FILE).exists()
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_no_gis_cols(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON)
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_null_continuous(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"bio1": [None] * 20})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


# ---------------------------------------------------------------------------
# _process_nonleaf / compute_taxon_stats / _collect_species_df / _process_species
# ---------------------------------------------------------------------------

def test_compute_taxon_stats_dispatches_species(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    species = {**CHILD_TAXON, "rank": "SPECIES"}
    species_dir = tmp_path / species["path"]
    _write_occ_rows(occurrences_file, species, extra_cols={"bio1": list(np.linspace(1, 10, 20))})
    st.compute_taxon_stats(species, [_CONTINUOUS_LAYER], st._DirStatsTarget(species_dir))
    assert (species_dir / st.NUMERICAL_STATS_FILE).exists()


def test_collect_species_df_own_only(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, SPECIES_TAXON, extra_cols={"bio1": [10.0] * 20})
    df = st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {})
    assert df is not None
    assert len(df) == 20


def test_collect_species_df_combines_subspecies(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, SPECIES_TAXON, extra_cols={"bio1": [10.0] * 20})
    _write_occ_rows(occurrences_file, SUBSPECIES_TAXON, extra_cols={"bio1": [20.0] * 20}, offset=100)
    df = st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {})
    assert df is not None
    assert len(df) == 40


def test_collect_species_df_no_own_obs_has_subspecies(tmp_path, monkeypatch):
    """Species with no direct rows but subspecies have data → still works."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, SUBSPECIES_TAXON, extra_cols={"bio1": [5.0] * 20})
    df = st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {})
    assert df is not None
    assert len(df) == 20


def test_collect_species_df_deduplicates_shared_obs(tmp_path, monkeypatch):
    """Observations shared between species and subspecies are deduplicated."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    # obs0-obs9 in species, obs5-obs14 in subspecies → 15 unique catalogNumbers
    _write_occ_rows(occurrences_file, SPECIES_TAXON, extra_cols={"bio1": [1.0] * 10}, n=10, offset=0)
    _write_occ_rows(occurrences_file, SUBSPECIES_TAXON, extra_cols={"bio1": [2.0] * 10}, n=10, offset=5)
    df = st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {})
    assert df is not None
    assert len(df) == 15
    assert df["catalogNumber"].nunique() == 15


def test_collect_species_df_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet")
    assert st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {}) is None


def test_collect_species_df_skips_empty_parquet(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    pq.write_table(pa.table({
        "catalogNumber": pa.array([], type=pa.string()),
        "taxon_key": pa.array([], type=pa.string()),
    }), occurrences_file)
    assert st._collect_species_df(SPECIES_TAXON, tmp_path / "unused", {}) is None


def test_process_species_builds_stats_from_subspecies(tmp_path, monkeypatch):
    """Stats for a species reflect combined own + subspecies observations."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, SPECIES_TAXON, extra_cols={"bio1": [10.0] * 20})
    _write_occ_rows(occurrences_file, SUBSPECIES_TAXON, extra_cols={"bio1": [20.0] * 20}, offset=100)
    species_dir = tmp_path / "species_dir"
    st._process_species(SPECIES_TAXON, species_dir, st._DirStatsTarget(species_dir), {"bio1": _CONTINUOUS_LAYER})
    df = pd.read_parquet(species_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "bio1"].iloc[0]
    assert row["count"] == 40


def test_process_species_no_data_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet")
    species_dir = tmp_path / "species_dir"
    st._process_species(SPECIES_TAXON, species_dir, st._DirStatsTarget(species_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (species_dir / st.NUMERICAL_STATS_FILE).exists()


# ---------------------------------------------------------------------------
# Coverage gap tests — _process_leaf edge cases
# ---------------------------------------------------------------------------

def test_process_leaf_unknown_value_type_skipped(tmp_path, monkeypatch):
    """Column with unresolvable value_type is silently skipped (vtype is None)."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"bio1": [1.0] * 20})
    taxon_dir = tmp_path / "taxon_dir"
    # value_type "bogus" → _layer_value_type returns None → continue
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": {"id": "bio1", "value_type": "bogus"}})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_nan_after_isfinite(tmp_path, monkeypatch):
    """values.size == 0 after isfinite filter."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"bio1": [float("inf")] * 20})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_nominal_series_empty_after_dropna(tmp_path, monkeypatch):
    """Nominal series empty after dropna."""
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, _LEAF_TAXON, extra_cols={"kg2": [None] * 20})
    taxon_dir = tmp_path / "taxon_dir"
    st._process_leaf(_LEAF_TAXON, st._DirStatsTarget(taxon_dir), {"kg2": _NOMINAL_LAYER})
    assert not (taxon_dir / st.NOMINAL_STATS_FILE).exists()


# ---------------------------------------------------------------------------
# collect_taxon_df
# ---------------------------------------------------------------------------

def test_collect_taxon_df_species_deduplicates(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    shared_cols = {"bio1": [1.0] * 10}
    _write_occ_rows(occurrences_file, SPECIES_TAXON, extra_cols=shared_cols, n=10)
    _write_occ_rows(occurrences_file, SUBSPECIES_TAXON, extra_cols=shared_cols, n=10)  # same catalogNumbers
    df = st.collect_taxon_df(SPECIES_TAXON)
    assert df is not None
    assert len(df) == 10  # duplicate catalogNumbers across species+subspecies collapse to one


def test_collect_taxon_df_nonleaf_excludes_self_but_reads_descendants(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, FAKE_TAXON, extra_cols={"bio1": [1.0] * 20})       # own rows, excluded
    _write_occ_rows(occurrences_file, CHILD_TAXON, extra_cols={"bio1": [5.0] * 20}, offset=100)
    # FAKE_TAXON is GENUS (non-leaf, non-species) — include_self=False
    df = st.collect_taxon_df(FAKE_TAXON)
    assert df is not None
    assert len(df) == 20


def test_collect_taxon_df_nonleaf_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet")
    assert st.collect_taxon_df(FAKE_TAXON) is None


def test_collect_taxon_df_nonleaf_skips_zero_row_parquet(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    pq.write_table(pa.table({
        "catalogNumber": pa.array([], type=pa.string()),
        "taxon_key": pa.array([], type=pa.string()),
    }), occurrences_file)
    assert st.collect_taxon_df(FAKE_TAXON) is None


def test_collect_taxon_df_nonleaf_skips_empty_and_filtered(tmp_path, monkeypatch):
    occurrences_file = tmp_path / "occurrences.parquet"
    monkeypatch.setattr(st, "OCCURRENCES_FILE", occurrences_file)
    _write_occ_rows(occurrences_file, CHILD_TAXON, extra_cols={"bio1": [5.0] * 20})
    df = pq.read_table(occurrences_file).to_pandas()
    df["obscured"] = "Yes"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), occurrences_file)
    assert st.collect_taxon_df(FAKE_TAXON) is None


# ---------------------------------------------------------------------------
# compute_location_filtered_stats
# ---------------------------------------------------------------------------

def test_compute_loc_stats_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet")
    result = st.compute_location_filtered_stats(SPECIES_TAXON, "bio1", "level0Gid", "USA", _CONTINUOUS_LAYER)
    assert result is None


# ---------------------------------------------------------------------------
# decode_polyline / parse_polygon_param / apply_polygon_filter
# ---------------------------------------------------------------------------

def _encode_polyline(points: list[tuple[float, float]], precision: int = 5) -> str:
    """Test-only mirror of the frontend's encoder (speciesOccurrenceMapHelpers.ts
    encodePolyline) — production code only ever needs to decode, since the
    frontend is the one building the `polygon` query param."""
    factor = 10**precision
    out: list[str] = []
    prev_lat = prev_lon = 0
    for lat, lon in points:
        lat_i = round(lat * factor)
        lon_i = round(lon * factor)
        for delta in (lat_i - prev_lat, lon_i - prev_lon):
            v = ~(delta << 1) if delta < 0 else (delta << 1)
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        prev_lat, prev_lon = lat_i, lon_i
    return "".join(out)


def test_decode_polyline_roundtrip():
    points = [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453), (-1.234, 179.9999)]
    decoded = st.decode_polyline(_encode_polyline(points))
    assert len(decoded) == len(points)
    for (lat, lon), (dlat, dlon) in zip(points, decoded):
        assert abs(lat - dlat) < 1e-4
        assert abs(lon - dlon) < 1e-4


def test_decode_polyline_truncated_raises():
    with pytest.raises(ValueError):
        st.decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq")


def test_parse_polygon_param_none_and_empty():
    assert st.parse_polygon_param(None) is None
    assert st.parse_polygon_param("") is None


def test_parse_polygon_param_fewer_than_three_points_dropped():
    # A 2-point "ring" can't form a polygon — dropped, leaving nothing.
    two_points = _encode_polyline([(0.0, 0.0), (1.0, 1.0)])
    assert st.parse_polygon_param(two_points) is None


def test_parse_polygon_param_single_ring():
    ring = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    geom = st.parse_polygon_param(_encode_polyline(ring))
    assert geom is not None
    assert geom.contains(st.ShapelyPolygon([(4, 4), (4, 5), (5, 5), (5, 4)]).centroid)


def test_parse_polygon_param_unions_multiple_rings():
    ring1 = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    ring2 = [(20.0, 20.0), (20.0, 30.0), (30.0, 30.0), (30.0, 20.0)]
    param = _encode_polyline(ring1) + ";" + _encode_polyline(ring2)
    geom = st.parse_polygon_param(param)
    assert geom is not None
    assert shapely.contains_xy(geom, np.array([5.0, 25.0, 100.0]), np.array([5.0, 25.0, 100.0])).tolist() == [
        True,
        True,
        False,
    ]


def test_parse_polygon_param_too_long_raises():
    huge = "a" * (st._MAX_POLYGON_PARAM_LENGTH + 1)
    with pytest.raises(ValueError):
        st.parse_polygon_param(huge)


def test_parse_polygon_param_too_many_rings_raises():
    ring = _encode_polyline([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    param = ";".join([ring] * (st._MAX_POLYGON_RINGS + 1))
    with pytest.raises(ValueError):
        st.parse_polygon_param(param)


def test_parse_polygon_param_invalid_encoding_raises():
    with pytest.raises(ValueError):
        st.parse_polygon_param("!!!not-a-valid-polyline")


def test_apply_polygon_filter_basic():
    ring = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    geom = st.parse_polygon_param(_encode_polyline(ring))
    df = pd.DataFrame({
        "catalogNumber": ["inside", "outside", "nan_coord"],
        "decimalLatitude": [5.0, 50.0, np.nan],
        "decimalLongitude": [5.0, 50.0, 5.0],
    })
    result = st.apply_polygon_filter(df, geom)
    assert list(result["catalogNumber"]) == ["inside"]


def test_apply_polygon_filter_missing_columns_returns_empty():
    df = pd.DataFrame({"catalogNumber": ["a", "b"]})
    ring = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    geom = st.parse_polygon_param(_encode_polyline(ring))
    result = st.apply_polygon_filter(df, geom)
    assert result.empty
