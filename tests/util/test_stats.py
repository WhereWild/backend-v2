import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import util.stats as st
from config.config import ValueType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_occ_parquet(path: Path, extra_cols: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": [f"obs{i}" for i in range(20)],
        "decimalLatitude": [40.0 + i * 0.01 for i in range(20)],
        "decimalLongitude": [-105.0 + i * 0.01 for i in range(20)],
        "hilbertIdx": list(range(20)),
        "obscured": ["No"] * 20,
        "coordinateUncertaintyInMeters": [100.0] * 20,
    }
    if extra_cols:
        data.update(extra_cols)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False), path)


_CONTINUOUS_LAYER = {"id": "bio1", "value_type": "ratio", "scale_factor": 0.1, "add_offset": -273.15}
_NOMINAL_LAYER    = {"id": "kg0",  "value_type": "nominal", "scale_factor": None, "add_offset": None}
_INTERVAL_LAYER   = {"id": "bio12","value_type": "interval","scale_factor": 1.0, "add_offset": 0.0}
_DISCRETE_LAYER   = {"id": "gsl",  "value_type": "ratio",   "scale_factor": None, "add_offset": None, "domain": "discrete"}


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


def test_build_density_curve_circular_raises():
    with pytest.raises(NotImplementedError):
        st.build_density_curve(np.array([0.0, 90.0, 180.0]), ValueType.CIRCULAR)


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
                "mean", "std", "iqr", "10_90_range", "range", "mode"}
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
                "mean", "std", "iqr", "10_90_range", "range", "mode"}
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
    entries = st._nominal_cat_entries("kg0", counts, summary)
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
        {"variable": "kg0", "metric": "total_samples", "value": 100.0},
        {"variable": "kg0", "metric": "class_1", "value": 0.6},
    ]
    st._write_nominal_stats(tmp_path, entries)
    df = pd.read_parquet(tmp_path / st.NOMINAL_STATS_FILE)
    assert len(df) == 2


def test_write_nominal_stats_empty(tmp_path):
    st._write_nominal_stats(tmp_path, [])
    assert not (tmp_path / st.NOMINAL_STATS_FILE).exists()


def test_write_read_numerical_density(tmp_path):
    rows = [{"variable": "bio1", "count": 50, "sampleCount": 50, "pointCount": 8,
             "points": [1.0, 2.0], "density": [0.3, 0.7], "min": 1.0, "max": 2.0,
             "bandwidth": 0.5}]
    st._write_numerical_density(tmp_path, rows)
    df = pd.read_parquet(tmp_path / st.NUMERICAL_DENSITY_FILE)
    assert df["variable"].iloc[0] == "bio1"


def test_write_numerical_density_empty(tmp_path):
    st._write_numerical_density(tmp_path, [])
    assert not (tmp_path / st.NUMERICAL_DENSITY_FILE).exists()


# ---------------------------------------------------------------------------
# _process_leaf
# ---------------------------------------------------------------------------

def test_process_leaf_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    occ_path = tmp_path / "bio1.tif" / st.OCCURRENCE_FILE
    bio1_vals = list(np.linspace(10.0, 30.0, 20))
    _make_occ_parquet(occ_path.parent / st.OCCURRENCE_FILE,
                      extra_cols={"bio1": bio1_vals})
    taxon_dir = occ_path.parent
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.NUMERICAL_DENSITY_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "bio1"].iloc[0]
    assert row["count"] == 20
    assert row["unique_samples"] == 20


def test_process_leaf_discrete(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    taxon_dir = tmp_path / "taxon_disc"
    vals = [42] * 10 + [43] * 5 + [44] * 5
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"gsl": [float(v) for v in vals]})
    st._process_leaf(taxon_dir, {"gsl": _DISCRETE_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.NUMERICAL_DENSITY_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "gsl"].iloc[0]
    assert row["mode"] == 42
    assert isinstance(row["mode"], (int, np.integer))
    den = pd.read_parquet(taxon_dir / st.NUMERICAL_DENSITY_FILE)
    hist_row = den[den["variable"] == "gsl"].iloc[0]
    assert hist_row["pointCount"] == 3
    assert list(hist_row["points"]) == [42.0, 43.0, 44.0]
    assert abs(sum(hist_row["density"]) - 1.0) < 1e-9


def test_process_nonleaf_discrete(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    vals = [10] * 12 + [20] * 8
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"gsl": [float(v) for v in vals]})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"gsl": _DISCRETE_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.NUMERICAL_DENSITY_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "gsl"].iloc[0]
    assert row["mode"] == 10
    den = pd.read_parquet(taxon_dir / st.NUMERICAL_DENSITY_FILE)
    hist_row = den[den["variable"] == "gsl"].iloc[0]
    # integers 10..20 inclusive, zeros filled between observed values
    assert hist_row["pointCount"] == 11
    assert hist_row["points"][0] == 10.0
    assert hist_row["points"][-1] == 20.0
    assert hist_row["density"][5] == 0.0   # value 15 was never observed
    assert abs(sum(hist_row["density"]) - 1.0) < 1e-9


def test_process_leaf_nominal(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    taxon_dir = tmp_path / "taxon"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"kg0": [1.0] * 15 + [2.0] * 5})
    st._process_leaf(taxon_dir, {"kg0": _NOMINAL_LAYER})
    assert (taxon_dir / st.NOMINAL_STATS_FILE).exists()
    df = pd.read_parquet(taxon_dir / st.NOMINAL_STATS_FILE)
    metrics = dict(zip(df["metric"], df["value"]))
    assert metrics["unique_classes"] == 2
    assert metrics["total_samples"] == 20
    assert metrics["mode"] == pytest.approx(1.0)
    assert "class_1" in metrics


def test_process_leaf_no_parquet(tmp_path):
    taxon_dir = tmp_path / "empty"
    taxon_dir.mkdir()
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_empty_parquet(tmp_path):
    taxon_dir = tmp_path / "empty_pq"
    taxon_dir.mkdir()
    pq.write_table(pa.table({"catalogNumber": pa.array([], type=pa.string())}),
                   taxon_dir / st.OCCURRENCE_FILE)
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_filtered_out(tmp_path):
    taxon_dir = tmp_path / "filtered"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [5.0] * 20})
    df = pd.read_parquet(taxon_dir / st.OCCURRENCE_FILE)
    df["obscured"] = "Yes"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), taxon_dir / st.OCCURRENCE_FILE)
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_unknown_value_type_raises(tmp_path):
    taxon_dir = tmp_path / "unk"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"circ": [45.0] * 20})
    with pytest.raises(NotImplementedError):
        st._process_leaf(taxon_dir, {"circ": {"id": "circ", "value_type": "circular"}})


def test_process_leaf_no_gis_cols(tmp_path):
    taxon_dir = tmp_path / "nogis"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE)
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_null_continuous(tmp_path):
    taxon_dir = tmp_path / "nulls"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [None] * 20})
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


# ---------------------------------------------------------------------------
# _process_nonleaf
# ---------------------------------------------------------------------------

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


def _make_fake_descendants(taxon, children):
    """Patch iter_descendants to yield taxon + children."""
    def _fake_iter(t, *, include_self=True):
        if include_self:
            yield t
        yield from children
    return _fake_iter


def test_process_nonleaf_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE,
                      extra_cols={"bio1": list(np.linspace(5.0, 25.0, 20))})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert (taxon_dir / st.NUMERICAL_STATS_FILE).exists()
    assert (taxon_dir / st.NUMERICAL_DENSITY_FILE).exists()


def test_process_nonleaf_nominal(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE,
                      extra_cols={"kg0": [1.0] * 12 + [2.0] * 8})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"kg0": _NOMINAL_LAYER})
    assert (taxon_dir / st.NOMINAL_STATS_FILE).exists()


def test_process_nonleaf_no_descendants(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, []))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_skips_unimplemented_type(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"circ": [45.0] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    # should NOT raise — streaming silently skips unimplemented types
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"circ": {"id": "circ", "value_type": "circular"}})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_aggregates_multiple_children(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child2 = {**CHILD_TAXON, "taxon_key": "10001", "path": "Root_1/Parent_9999/Child_10001"}
    for child in [CHILD_TAXON, child2]:
        child_dir = tmp_path / child["path"]
        _make_occ_parquet(child_dir / st.OCCURRENCE_FILE,
                          extra_cols={"bio1": [10.0] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants",
                        _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON, child2]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    df = pd.read_parquet(taxon_dir / st.NUMERICAL_STATS_FILE)
    row = df[df["variable"] == "bio1"].iloc[0]
    assert row["count"] == 40
    assert row["unique_samples"] == 40


# ---------------------------------------------------------------------------
# compute_taxon_stats (dispatch)
# ---------------------------------------------------------------------------

def test_compute_taxon_stats_dispatches_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    leaf = {**CHILD_TAXON, "rank": "SPECIES"}
    leaf_dir = tmp_path / leaf["path"]
    _make_occ_parquet(leaf_dir / st.OCCURRENCE_FILE,
                      extra_cols={"bio1": list(np.linspace(1, 10, 20))})
    st.compute_taxon_stats(leaf, [_CONTINUOUS_LAYER])
    assert (leaf_dir / st.NUMERICAL_STATS_FILE).exists()


def test_compute_taxon_stats_dispatches_nonleaf(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, []))
    parent_dir = tmp_path / FAKE_TAXON["path"]
    st.compute_taxon_stats(FAKE_TAXON, [_CONTINUOUS_LAYER])
    assert not (parent_dir / st.NUMERICAL_STATS_FILE).exists()


# ---------------------------------------------------------------------------
# Coverage gap tests — _process_leaf edge cases
# ---------------------------------------------------------------------------

def test_process_leaf_unknown_value_type_skipped(tmp_path):
    """Column with unresolvable value_type is silently skipped (vtype is None)."""
    taxon_dir = tmp_path / "t"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [1.0] * 20})
    # value_type "bogus" → _layer_value_type returns None → continue
    st._process_leaf(taxon_dir, {"bio1": {"id": "bio1", "value_type": "bogus"}})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_all_nan_after_isfinite(tmp_path):
    """values.size == 0 after isfinite filter (line 303)."""
    taxon_dir = tmp_path / "inf"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE,
                      extra_cols={"bio1": [float("inf")] * 20})
    st._process_leaf(taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_leaf_nominal_series_empty_after_dropna(tmp_path):
    """Nominal series empty after dropna (line 325)."""
    taxon_dir = tmp_path / "nominal_null"
    _make_occ_parquet(taxon_dir / st.OCCURRENCE_FILE, extra_cols={"kg0": [None] * 20})
    st._process_leaf(taxon_dir, {"kg0": _NOMINAL_LAYER})
    assert not (taxon_dir / st.NOMINAL_STATS_FILE).exists()




# ---------------------------------------------------------------------------
# Coverage gap tests — _process_nonleaf edge cases
# ---------------------------------------------------------------------------

def test_process_nonleaf_empty_table_skipped(tmp_path, monkeypatch):
    """Descendant with 0-row table is skipped (line 364)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    child_dir.mkdir(parents=True)
    pq.write_table(pa.table({"catalogNumber": pa.array([], type=pa.string())}),
                   child_dir / st.OCCURRENCE_FILE)
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_all_filtered_skipped(tmp_path, monkeypatch):
    """Descendant that filters to empty df is skipped (line 367)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [5.0] * 20})
    df = pd.read_parquet(child_dir / st.OCCURRENCE_FILE)
    df["obscured"] = "Yes"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), child_dir / st.OCCURRENCE_FILE)
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_unknown_vtype_column_skipped(tmp_path, monkeypatch):
    """Column with unresolvable value_type is skipped (line 374)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [5.0] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    # layer with no value_type key → _layer_value_type returns None
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": {"id": "bio1"}})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_continuous_series_empty(tmp_path, monkeypatch):
    """All-null continuous column in streaming is skipped (lines 380, 384)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"bio1": [None] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_continuous_all_inf(tmp_path, monkeypatch):
    """All-inf values after isfinite filter in streaming (line 384)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE,
                      extra_cols={"bio1": [float("inf")] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"bio1": _CONTINUOUS_LAYER})
    assert not (taxon_dir / st.NUMERICAL_STATS_FILE).exists()


def test_process_nonleaf_nominal_series_empty_streaming(tmp_path, monkeypatch):
    """All-null nominal column in streaming is skipped (line 395)."""
    monkeypatch.setattr(st, "TREE_ROOT", tmp_path)
    child_dir = tmp_path / CHILD_TAXON["path"]
    _make_occ_parquet(child_dir / st.OCCURRENCE_FILE, extra_cols={"kg0": [None] * 20})
    taxon_dir = tmp_path / FAKE_TAXON["path"]
    monkeypatch.setattr(st, "iter_descendants", _make_fake_descendants(FAKE_TAXON, [CHILD_TAXON]))
    st._process_nonleaf(FAKE_TAXON, taxon_dir, {"kg0": _NOMINAL_LAYER})
    assert not (taxon_dir / st.NOMINAL_STATS_FILE).exists()




