"""Unit tests for util.summary_stats helper behavior."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from util import summary_stats as ss


class _StubParquet:
    def __init__(self):
        self._exists = {}
        self._tables = {}
        self._schemas = {}
        self._metadata = {}
        self._files = {}
        self.is_remote = False

    def exists(self, path):
        return self._exists.get(Path(path), False)

    def read_table(self, path, **_kwargs):
        value = self._tables.get(Path(path))
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(path, **_kwargs)
        return value

    def read_schema(self, path):
        return self._schemas[Path(path)]

    def read_metadata(self, path):
        return self._metadata[Path(path)]

    def open_input_file(self, path):
        return io.BytesIO(self._files[Path(path)])


@pytest.fixture(autouse=True)
def _clear_caches():
    ss._load_summary_stats.cache_clear()
    ss._load_categorical_stats.cache_clear()
    yield


@pytest.fixture
def stub_env(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        gis_catalog_path=tmp_path / "catalog.json",
        occurrence_parquet_filename="occurrence.parquet",
        project_root=tmp_path,
    )
    stub = _StubParquet()
    monkeypatch.setattr(ss, "CONFIG", cfg)
    monkeypatch.setattr(ss, "PARQUET", stub)
    return cfg, stub


def _make_index_table():
    struct_type = pa.struct(
        [("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())]
    )
    arr = pa.array(
        [
            {"catalogNumber": "a", "originId": 1, "value": 1.0},
            {"catalogNumber": "b", "originId": 1, "value": 2.0},
            {"catalogNumber": "c", "originId": 2, "value": 3.0},
        ],
        type=struct_type,
    )
    meta = {
        b"origin_map": json.dumps([{"id": 1, "relative_path": "one"}, {"id": 2, "relative_path": "two"}]).encode("utf-8"),
        b"catalog_column": b"catalogNumber",
        b"category_offsets": json.dumps({"bio_1": {"1": {"start": 0, "count": 2}}}).encode("utf-8"),
    }
    schema = pa.schema([pa.field("bio_1", struct_type)]).with_metadata(meta)
    return pa.Table.from_arrays([arr], schema=schema)


def test_small_helpers_and_metadata_access(stub_env, monkeypatch):
    cfg, stub = stub_env
    monkeypatch.setattr(ss.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "Numeric"}})
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"1": {"id": 1, "name": "Forest"}, "x": "bad", "2": {"id": "bad", "name": "Skip"}})
    assert ss._layer_value_type("bio_1") == "numeric"
    assert ss._layer_value_type("missing") is None
    assert ss._legend_for_layer("bio_1") == {1: "Forest"}
    assert ss._slugify_metric(" Total Samples ", "fallback") == "total_samples"
    assert ss._slugify_metric("", "fallback") == "fallback"
    assert ss._format_category_label("class__one::two") == "Class One Two"

    assert ss.categorical_value_key(2.0) == ("2", 2)
    assert ss.categorical_value_key("2.5") == ("2.5", 2.5)
    assert ss.categorical_value_key("x") == ("x", "x")
    assert ss._legend_key(2.0) == "2"
    assert ss._legend_key("  a ") == "a"

    catalog = {"categories": [{"layers": [{"id": "bio_1", "display_name": "Temp"}]}]}
    stub._files[cfg.gis_catalog_path] = json.dumps(catalog).encode("utf-8")
    assert ss.code_to_name("bio_1") == "Temp"
    assert ss.code_to_name("missing") is None


def test_prepare_slice_and_sorted_queries(stub_env):
    cfg, stub = stub_env
    index_path = Path("/tmp/index.parquet")
    stub._exists[index_path] = True
    idx_table = _make_index_table()
    stub._tables[index_path] = idx_table

    one = Path("/tmp/one") / cfg.occurrence_parquet_filename
    two = Path("/tmp/two") / cfg.occurrence_parquet_filename
    stub._exists[one] = True
    stub._exists[two] = True
    stub._tables[one] = pa.table(
        {
            "catalogNumber": ["a", "b"],
            "decimalLatitude": [1.0, 2.0],
            "decimalLongitude": [3.0, 4.0],
            "bio_1": [1.0, 2.0],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [10, 10],
        }
    )
    stub._tables[two] = pa.table(
        {
            "catalogNumber": ["c"],
            "decimalLatitude": [5.0],
            "decimalLongitude": [6.0],
            "bio_1": [3.0],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [10],
        }
    )

    prepared = ss._prepare_index_column(index_path, "bio_1")
    assert prepared["catalog_column"] == "catalogNumber"
    rows = ss.get_sorted_layer_records(index_path, "bio_1", start=0, stop=3)
    assert [r[0] for r in rows] == ["a", "b", "c"]
    assert ss.get_sorted_layer_records(index_path, "bio_1", start=99, stop=100) == []
    assert ss.get_sorted_layer_records_in_value_range(index_path, "bio_1", 1.5, 3.0, limit=2)[0][0] == "b"
    assert ss.get_sorted_layer_records_in_value_range(index_path, "bio_1", 9.0, 10.0) == []

    class_rows = ss.get_layer_records_for_class(index_path, "bio_1", 1)
    assert [r[0] for r in class_rows] == ["a", "b"]


def test_prepare_index_errors_and_null_counts(stub_env):
    _cfg, stub = stub_env
    path = Path("/tmp/missing.parquet")
    with pytest.raises(FileNotFoundError):
        ss._prepare_index_column(path, "x")

    stub._exists[path] = True
    empty = pa.table({"x": pa.array([], type=pa.int64())})
    stub._tables[path] = empty
    with pytest.raises(ValueError):
        ss._prepare_index_column(path, "x")

    tbl = pa.table({"a": [1, None], "b": [None, "x"]})
    stub._tables[path] = tbl
    assert ss.column_null_counts(path) == {"a": 1, "b": 1}


def test_distribution_and_categorical_builders(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    data_dir = tmp_path
    stats_path = data_dir / "categorical_stats.parquet"
    stub._exists[stats_path] = True
    stub._tables[stats_path] = pa.table(
        {
            "variable": ["koppen", "koppen", "koppen", "koppen"],
            "metric": ["class_1", "total_samples", "unique_classes", "bad"],
            "value": ["0.7", "10.0", "1.0", "x"],
        }
    )
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"1": {"id": 1, "name": "Forest", "description": "desc"}, "class 1": {"id": 1, "name": "Forest"}})
    dist = ss.load_categorical_distribution(data_dir, "koppen")
    assert dist["distribution"][0]["class_name"] == "Forest"
    assert dist["distribution"][0]["count"] == 7
    assert ss.load_categorical_distribution(data_dir, "missing") is None

    table = pa.table({"catalogNumber": ["a", "b"], "koppen": [1, 1]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [table])
    built = ss.build_categorical_stats_for_location(1, "koppen", "USA", sample_limit=1)
    assert built["totals"]["total_samples"] == 2
    assert len(built["samples"][0]["observationIds"]) == 1

    monkeypatch.setattr(ss, "get_layer_records_for_class", lambda *_a, **_k: [("a", 1, 2, 1)])
    stub._exists[data_dir / "occurrence_index.parquet"] = True
    samples = ss.build_categorical_samples(data_dir, "koppen", [{"value": 1}, {"value": object()}])
    assert samples == [{"value": 1, "observationIds": ["a"]}]


def test_numeric_summary_and_loading_helpers(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    summary = ss.summarize_values([1.0, 2.0, 3.0, float("nan")])
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert ss.summarize_values([])["count"] == 0

    out = ss._density_point_count(0), ss._density_point_count(10), ss._density_point_count(100), ss._density_point_count(1000)
    assert out == (0, 10, 64, 128)
    assert ss._build_density_curve([], 10) is None
    assert ss._build_density_curve([1.0, 1.0], 8)["points"]

    p = tmp_path / "occ.parquet"
    stub._tables[p] = pa.table(
        {
            "catalogNumber": ["a", "b"],
            "decimalLatitude": [1.0, 2.0],
            "decimalLongitude": [3.0, 4.0],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [1, 1],
            "bio_1": [1.5, np.nan],
        }
    )
    monkeypatch.setattr(ss.taxa_navigation, "base_observation_mask", lambda t: pa.array([True] * t.num_rows))
    samples = ss.read_numeric_from_parquet(p, "bio_1")
    assert samples == [{"catalog_id": "a", "value": 1.5, "latitude": 1.0, "longitude": 3.0}]

    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "bio_1": [4.0]})])
    gathered = ss.gather_numeric_records_from_tables(1, "bio_1", None)
    assert gathered[0]["value"] == 4.0

    monkeypatch.setattr(ss, "get_sorted_layer_records", lambda *_a, **_k: [("a", 1.0, 2.0, 3.0), ("b", 1.0, 2.0, "x")])
    d = tmp_path / "data"
    d.mkdir()
    idx = d / "occurrence_index.parquet"
    stub._exists[idx] = True
    gathered2 = ss.gather_numeric_records(1, d, "bio_1")
    assert [r["catalog_id"] for r in gathered2] == ["a"]

    monkeypatch.setattr(ss, "gather_numeric_records_from_tables", lambda *_a, **_k: [{"catalog_id": "z"}])
    stub._exists[idx] = False
    assert ss.gather_numeric_records(1, d, "bio_1") == [{"catalog_id": "z"}]


def test_cache_loading_and_dataframe_converters(stub_env, tmp_path):
    _cfg, stub = stub_env
    d = tmp_path / "x"
    d.mkdir()
    summary_path = d / "summary_stats.parquet"
    cat_path = d / "categorical_stats.parquet"
    density_path = d / ss.density_graph_filename
    stub._exists[summary_path] = True
    stub._exists[cat_path] = True
    stub._exists[density_path] = True
    stub._tables[summary_path] = pa.table({"variable": ["bio_1"], "mean": [2.0]})
    stub._tables[cat_path] = pa.table({"variable": ["koppen"], "metric": ["class_1"], "value": [0.5]})
    stub._tables[density_path] = pa.table(
        {
            "variable": ["bio_1"],
            "points": [[1.0, 2.0]],
            "density": [[0.1, 0.2]],
            "min": [1.0],
            "max": [2.0],
            "bandwidth": [0.5],
            "count": [2],
            "sampleCount": [2],
            "pointCount": [2],
        }
    )

    assert ss.load_numeric_summary(str(d), "bio_1")["mean"] == 2.0
    assert ss._load_categorical_stats(str(d))["koppen"]["class_1"] == 0.5
    assert ss.load_density_graph(str(d), "bio_1")["count"] == 2
    assert ss.load_density_graph(str(d), "missing") is None

    wide = pd.DataFrame({"variable": ["bio_1"], "mean": [np.float64(2.0)], "nan_col": [np.nan]})
    tall = pd.DataFrame({"variable": ["koppen"], "metric": ["class_1"], "value": [np.float64(0.5)]})
    assert ss._dataframe_to_stats(wide)["bio_1"]["mean"] == 2.0
    assert ss._tall_dataframe_to_stats(tall)["koppen"]["class_1"] == 0.5
    assert ss._dataframe_to_stats(pd.DataFrame()) == {}
    assert ss._tall_dataframe_to_stats(pd.DataFrame()) == {}


def test_range_and_class_location_helpers(monkeypatch):
    table = pa.table(
        {
            "catalogNumber": ["a", "b", "c"],
            "decimalLatitude": [1.0, 2.0, 3.0],
            "decimalLongitude": [4.0, 5.0, 6.0],
            "bio_1": [1.0, 2.0, 3.0],
            "koppen": [1, 2, None],
        }
    )
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [table])
    monkeypatch.setattr(ss, "resolve_categorical_class_value", lambda *_a, **_k: 2)
    rows = ss.numeric_range_samples_for_location(1, "bio_1", 1.5, 3.0, location_gid="USA", limit=1)
    assert rows == [("b", 2.0, 5.0, 2.0)]

    class_rows = ss.categorical_class_samples_for_location(1, "koppen", 2, location_gid="USA", limit=1)
    assert class_rows == [{"catalogNumber": "b", "latitude": 2.0, "longitude": 5.0, "value": 2}]


def test_iter_descendant_tables_and_digest_helpers(stub_env, monkeypatch):
    _cfg, stub = stub_env
    base = Path("/tmp/taxa/1")
    p = base / "occurrence.parquet"
    stub._exists[p] = True
    stub._tables[p] = pa.table({"x": [1]})
    monkeypatch.setattr(ss.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(ss.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1"})
    monkeypatch.setattr(ss.taxa_navigation, "iter_descendants", lambda *_a, **_k: [{"path": base}])
    yielded = list(ss._iter_descendant_tables(p))
    assert len(yielded) == 1

    monkeypatch.setattr(ss.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert list(ss._iter_descendant_tables(p)) == []

    class _Q:
        def quantile(self, q):
            return q

    class _P:
        def percentile(self, p):
            return p

    assert ss._digest_quantile(_Q(), 0.25) == 0.25
    assert ss._digest_quantile(_P(), 0.25) == 25.0
    assert ss._digest_quantile(object(), 0.25) is None


def test_streaming_stats_and_numeric_column_stats(monkeypatch, tmp_path):
    class _Digest:
        def __init__(self):
            self.values = []

        def batch_update(self, vals):
            self.values.extend(vals)

        def quantile(self, q):
            if not self.values:
                return 0.0
            return sorted(self.values)[int((len(self.values) - 1) * q)]

    monkeypatch.setattr(ss, "_FastTDigest", _Digest)
    stats = ss._init_streaming_stats()
    ss._update_streaming_stats(stats, pd.Series([1, 2, np.nan]))
    assert stats["count"] == 2
    assert stats["min_value"] == 1.0
    ss._update_streaming_stats(stats, pd.Series(["bad"]))
    assert stats["count"] == 2

    node = tmp_path / "node"
    node.mkdir()
    monkeypatch.setattr(ss, "_load_summary_stats", lambda _p: {"done": {}})
    monkeypatch.setattr(ss, "_load_categorical_stats", lambda _p: {"cat": {}})
    monkeypatch.setattr(ss, "_numeric_column_stats_streaming", lambda *_a, **_k: {"stream": {"count": 1}})
    monkeypatch.setattr(ss, "_numeric_column_stats_exact", lambda *_a, **_k: {"exact": {"count": 1}})
    with ss.stats_context(node):
        assert ss.numeric_column_stats(streaming=True) == {"stream": {"count": 1}}
    with ss.stats_context(node):
        assert ss.numeric_column_stats(streaming=False) == {"exact": {"count": 1}}
    with pytest.raises(RuntimeError):
        ss.numeric_column_stats()


def test_exact_and_streaming_collectors(monkeypatch, tmp_path):
    p = tmp_path / "occurrence.parquet"
    df = pd.DataFrame(
        {
            "obscured": ["No", "Yes"],
            "coordinateUncertaintyInMeters": [1, 1000],
            "num": [1.0, 2.0],
            "cat": [1, 1],
        }
    )
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [pa.Table.from_pandas(df)])
    monkeypatch.setattr(ss, "_layer_value_type", lambda c: "categorical" if c == "cat" else "numeric")
    writes = {"num": None, "cat": None}
    monkeypatch.setattr(ss, "_write_summary_stats", lambda _d, st, **_k: writes.__setitem__("num", st))
    monkeypatch.setattr(ss, "_write_categorical_stats", lambda _d, st, **_k: writes.__setitem__("cat", st))
    out_exact = ss._numeric_column_stats_exact(p, existing_numeric=set(), existing_categorical=set())
    assert out_exact["num"]["count"] == 1
    out_stream = ss._numeric_column_stats_streaming(p, existing_numeric=set(), existing_categorical=set())
    assert out_stream["num"]["count"] == 1
    assert writes["cat"] is not None


def test_collectors_and_write_helpers(tmp_path, monkeypatch):
    df = pd.DataFrame({"cat": [1, 1, 2]})
    monkeypatch.setattr(ss, "_legend_for_layer", lambda _c: {1: "Forest"})
    entries = ss._collect_categorical_stats(df, ["cat"])
    assert any(e["metric"] == "class_2" for e in entries)
    by_counts = ss._collect_categorical_stats_from_counts({"cat": {1: 2, 2: 1}}, {"cat": 3})
    assert any(e["metric"] == "total_samples" for e in by_counts)

    d = tmp_path / "out"
    d.mkdir()
    ss._write_summary_stats(d, {"bio_1": {"mean": 1.0}})
    ss._write_summary_stats(d, {"bio_1": {"mean": 2.0}}, merge_existing=True)
    assert (d / "summary_stats.parquet").exists()
    ss._write_categorical_stats(d, [{"variable": "cat", "metric": "m", "value": 1.0}])
    ss._write_categorical_stats(d, [], merge_existing=False)
    assert not (d / "categorical_stats.parquet").exists()


def test_density_graph_and_loaders_edge_cases(stub_env, monkeypatch, tmp_path, capsys):
    _cfg, stub = stub_env
    d = tmp_path / "d"
    d.mkdir()
    occ = d / "occurrence.parquet"
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [pa.table({"obscured": ["No"], "coordinateUncertaintyInMeters": [1], "x": [1.0]})])
    monkeypatch.setattr(ss, "_layer_value_type", lambda _c: "numeric")
    ss.write_density_graph(d)
    assert (d / ss.density_graph_filename).exists()

    # no rows path removes output
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [])
    ss.write_density_graph(d)
    assert not (d / ss.density_graph_filename).exists()

    # load_density_graph missing/read-empty/error branches
    missing = ss.load_density_graph(str(d), "x")
    assert missing is None
    assert "[density] missing file" in capsys.readouterr().out

    density_path = d / ss.density_graph_filename
    stub._exists[density_path] = True
    stub._tables[density_path] = pa.table({"variable": [], "points": [], "density": [], "min": [], "max": [], "bandwidth": [], "count": [], "sampleCount": [], "pointCount": []})
    assert ss.load_density_graph(str(d), "x") is None

    stub._tables[density_path] = OSError("boom")
    assert ss.load_density_graph(str(d), "x") is None


def test_slice_and_range_edge_branches(stub_env, monkeypatch):
    _cfg, stub = stub_env
    index_path = Path("/tmp/idx2.parquet")
    stub._exists[index_path] = True
    struct_type = pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64())])
    arr = pa.array([{"catalogNumber": "a", "originId": 1}], type=struct_type)
    meta = {b"origin_map": json.dumps([{"id": 1, "relative_path": "one"}]).encode("utf-8")}
    stub._tables[index_path] = pa.Table.from_arrays([arr], schema=pa.schema([pa.field("bio_1", struct_type)]).with_metadata(meta))

    one = Path("/tmp/one") / "occurrence.parquet"
    stub._exists[one] = True
    stub._tables[one] = pa.table(
        {
            "catalogNumber": ["a"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [2.0],
            "bio_1": [None],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [1],
        }
    )
    rows = ss.get_sorted_layer_records(index_path, "bio_1")
    assert rows[0][3] is None
    assert ss.get_sorted_layer_records_in_value_range(index_path, "bio_1", 0.0, 1.0) == []
    assert ss.get_layer_records_for_class(index_path, "bio_1", 999) == []

    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: None)
    assert ss.get_sorted_layer_records(index_path, "bio_1") == []
    assert ss.get_sorted_layer_records_in_value_range(index_path, "bio_1", None, None) == []


def test_more_distribution_and_resolution_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    data_dir = tmp_path
    stats_path = data_dir / "categorical_stats.parquet"
    stub._exists[stats_path] = True
    stub._tables[stats_path] = pa.table({"variable": ["koppen"], "metric": ["class_a"], "value": ["0.5"]})
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"class a": {"id": 7, "name": "Name"}})
    assert ss.load_categorical_distribution(data_dir, "koppen")["distribution"][0]["value"] == 7

    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"forest": {"id": 5}})
    assert ss.resolve_categorical_class_value("koppen", "forest") == 5
    assert ss.resolve_categorical_class_value("koppen", "class_5") == 5
    assert ss.resolve_categorical_class_value("koppen", "class_word") == "word"

    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [pa.table({"catalogNumber": ["a"], "koppen": [None]})])
    assert ss.build_categorical_stats_for_location(1, "koppen", "USA", sample_limit=3) is None

    monkeypatch.setattr(ss, "get_layer_records_for_class", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    stub._exists[data_dir / "occurrence_index.parquet"] = True
    assert ss.build_categorical_samples(data_dir, "koppen", [{"value": 1}]) == []


def test_more_numeric_and_cache_error_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    d = tmp_path / "d"
    d.mkdir()
    idx = d / "occurrence_index.parquet"
    occ = d / "occurrence.parquet"
    comb = d / "combined.parquet"
    stub._exists[idx] = True
    monkeypatch.setattr(ss, "get_sorted_layer_records", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    stub._exists[occ] = True
    stub._tables[occ] = pa.table(
        {
            "catalogNumber": ["a"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [2.0],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [1],
            "bio_1": [1.0],
        }
    )
    monkeypatch.setattr(ss.taxa_navigation, "base_observation_mask", lambda t: pa.array([True] * t.num_rows))
    assert ss.gather_numeric_records(1, d, "bio_1")[0]["catalog_id"] == "a"
    stub._exists[occ] = False
    stub._exists[comb] = True
    stub._tables[comb] = pa.table(
        {
            "catalogNumber": ["x"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [1.0],
            "obscured": ["Yes"],
            "coordinateUncertaintyInMeters": [1],
            "bio_1": [None],
        }
    )
    monkeypatch.setattr(ss, "gather_numeric_records_from_tables", lambda *_a, **_k: [{"catalog_id": "z"}])
    assert ss.gather_numeric_records(1, d, "bio_1")[0]["catalog_id"] == "z"
    assert ss.gather_numeric_records(1, d, "bio_1", location_gid="USA")[0]["catalog_id"] == "z"

    # summary/categorical loader error paths
    summary_path = d / "summary_stats.parquet"
    cat_path = d / "categorical_stats.parquet"
    stub._exists[summary_path] = True
    stub._exists[cat_path] = True
    stub._tables[summary_path] = OSError("boom")
    stub._tables[cat_path] = OSError("boom")
    assert ss._load_summary_stats(str(d)) is None
    assert ss._load_categorical_stats(str(d)) == {}
    assert ss.load_numeric_summary(str(d), "bio_1") is None


def test_dataframe_item_valueerror_branches():
    class _BadItem:
        def item(self):
            raise ValueError("bad item")

    wide = pd.DataFrame({"variable": ["bio_1"], "x": [_BadItem()]})
    tall = pd.DataFrame({"variable": ["bio_1"], "metric": ["m"], "value": [_BadItem()]})
    assert "x" in ss._dataframe_to_stats(wide)["bio_1"]
    assert "m" in ss._tall_dataframe_to_stats(tall)["bio_1"]
    assert ss._tall_dataframe_to_stats(pd.DataFrame({"x": [1]})) == {}


def test_prepare_and_loader_additional_branches(stub_env, monkeypatch):
    _cfg, stub = stub_env
    path = Path("/tmp/prep.parquet")
    stub._exists[path] = True
    tbl = pa.table({"bio_1": [1.0]})
    stub._tables[path] = tbl
    with pytest.raises(ValueError):
        ss._prepare_index_column(path, "bio_1")

    meta = {
        b"origin_map": json.dumps([{"id": 1, "relative_path": "a"}]).encode("utf-8"),
        b"category_offsets": json.dumps({"bio_1": {"1": {"start": 0, "count": 1}}}).encode("utf-8"),
    }
    schema = pa.schema([pa.field("bio_1", pa.float64())]).with_metadata(meta)
    stub._tables[path] = pa.Table.from_arrays([pa.array([1.0])], schema=schema)
    out = ss._prepare_index_column(path, "bio_1")
    assert out["catalog_column"] == "catalogNumber"

    # dataset loader edge branches
    load = ss._make_dataset_loader({}, Path("/tmp"), catalog_column="catalogNumber", layer_id="bio_1", data_filename="occ.parquet", lat_col="decimalLatitude", lon_col="decimalLongitude")
    assert load(1) is None

    monkeypatch.setattr(ss.PARQUET, "exists", lambda _p: False)
    load2 = ss._make_dataset_loader({1: {"relative_path": "a"}}, Path("/tmp"), catalog_column="catalogNumber", layer_id="bio_1", data_filename="occ.parquet", lat_col="decimalLatitude", lon_col="decimalLongitude")
    assert load2(1) is None


def test_slice_records_dataset_miss_branches(monkeypatch):
    prepared = {
        "column": pa.array(
            [{"catalogNumber": "a", "originId": 1, "value": 1.0}],
            type=pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())]),
        ),
        "origin_lookup": {1: {"relative_path": "a"}},
        "index_dir": Path("/tmp"),
        "catalog_column": "catalogNumber",
    }
    monkeypatch.setattr(ss, "_make_dataset_loader", lambda *_a, **_k: (lambda _o: None))
    assert ss._slice_records(prepared, "bio_1", start=0, stop=1, data_filename="x", lat_col="lat", lon_col="lon") == []
    monkeypatch.setattr(ss, "_make_dataset_loader", lambda *_a, **_k: (lambda _o: {"index": {}, "latitudes": pa.array([]), "longitudes": pa.array([]), "layer_values": pa.array([])}))
    assert ss._slice_records(prepared, "bio_1", start=0, stop=1, data_filename="x", lat_col="lat", lon_col="lon") == []


def test_categorical_distribution_additional_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    p = tmp_path / "categorical_stats.parquet"
    stub._exists[p] = True
    stub._tables[p] = pa.table(
        {
            "variable": ["v", "v", "v", "v"],
            "metric": ["significant_unique_classes", "class_2", "class_bad", "name slug"],
            "value": ["2", "0.5", "0.2", "0.3"],
        }
    )
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"2": {"id": 2, "name": "Two"}, "name slug": {"id": 5, "name": "Five"}})
    dist = ss.load_categorical_distribution(tmp_path, "v")
    assert dist["totals"]["significant_unique_classes"] == 2.0
    values = [d["value"] for d in dist["distribution"]]
    assert 2 in values and 5 in values

    stub._tables[p] = Exception("read-fail")
    assert ss.load_categorical_distribution(tmp_path, "v") is None


def test_resolve_and_samples_more_branches(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {})
    assert ss.resolve_categorical_class_value("v", "notfound") == "notfound"

    t = pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "koppen": ["x"]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [t])
    assert ss.categorical_class_samples_for_location(1, "koppen", "x", location_gid="USA", limit=1)[0]["value"] == "x"

    t2 = pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "bio_1": [5.0]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [t2])
    assert ss.numeric_range_samples_for_location(1, "bio_1", 0, 10, location_gid="USA", limit=5) == [("a", 1.0, 2.0, 5.0)]


def test_numeric_read_and_gather_more_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    p = tmp_path / "p.parquet"
    stub._tables[p] = pa.table(
        {
            "catalogNumber": ["a", "b"],
            "decimalLatitude": [1.0, 2.0],
            "decimalLongitude": [3.0, 4.0],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [1, 1],
            "bio_1": ["bad", "2"],
        }
    )
    monkeypatch.setattr(ss.taxa_navigation, "base_observation_mask", lambda t: pa.array([True] * t.num_rows))
    out = ss.read_numeric_from_parquet(p, "bio_1")
    assert out == [{"catalog_id": "b", "value": 2.0, "latitude": 2.0, "longitude": 4.0}]

    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "bio_1": [None]})])
    assert ss.gather_numeric_records_from_tables(1, "bio_1", None) == []

    d = tmp_path / "d2"
    d.mkdir()
    idx_path = d / "occurrence_index.parquet"
    occ = d / "occurrence.parquet"
    stub._exists[idx_path] = True
    monkeypatch.setattr(ss, "get_sorted_layer_records", lambda *_a, **_k: [("a", 1, 2, None)])
    stub._exists[occ] = True
    monkeypatch.setattr(ss, "read_numeric_from_parquet", lambda *_a, **_k: (_ for _ in ()).throw(KeyError("x")))
    monkeypatch.setattr(ss.taxa_navigation, "combined_parquet_filename", "combined.parquet")
    assert ss.gather_numeric_records(1, d, "bio_1") == ss.gather_numeric_records_from_tables(1, "bio_1", None)


def test_streaming_and_exact_more_branches(monkeypatch, tmp_path):
    class _D2:
        def __init__(self):
            self.values = []

        def update(self, v):
            self.values.append(v)

        def quantile(self, q):
            return 0.0

    monkeypatch.setattr(ss, "_FastTDigest", _D2)
    st = ss._init_streaming_stats()
    ss._update_streaming_stats(st, pd.Series(["1", "2"]))
    ss._update_streaming_stats(st, pd.Series(["3", "4"]))
    assert st["count"] == 4

    p = tmp_path / "occurrence.parquet"
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [])
    assert ss._numeric_column_stats_exact(p) == {}
    assert ss._numeric_column_stats_streaming(p) == {}

    df1 = pd.DataFrame({"obscured": ["No"], "coordinateUncertaintyInMeters": [1], "num": [1.0], "cat": [1]})
    df2 = pd.DataFrame({"obscured": ["No"], "coordinateUncertaintyInMeters": [1], "num": [2.0], "cat": [2]})
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [pa.Table.from_pandas(df1), pa.Table.from_pandas(df2)])
    monkeypatch.setattr(ss, "_layer_value_type", lambda c: "categorical" if c == "cat" else "numeric")
    monkeypatch.setattr(ss, "_write_summary_stats", lambda *_a, **_k: None)
    monkeypatch.setattr(ss, "_write_categorical_stats", lambda *_a, **_k: None)
    assert ss._numeric_column_stats_exact(p, existing_numeric={"num"}) == {}
    assert ss._numeric_column_stats_streaming(p, existing_categorical={"cat"})["num"]["count"] == 2


def test_write_helpers_and_loaders_more_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    d = tmp_path / "o"
    d.mkdir()
    # _write_summary_stats empty and exception paths
    ss._write_summary_stats(d, {})
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    ss._write_summary_stats(d, {"v": {"mean": 1.0}})

    # _write_categorical_stats merge/empty paths
    ss._write_categorical_stats(d, [], merge_existing=True)
    ss._write_categorical_stats(d, pd.DataFrame().to_dict("records"), merge_existing=False)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    ss._write_categorical_stats(d, [{"variable": "v", "metric": "m", "value": 1.0}], merge_existing=False)

    # summary/category cache missing paths
    assert ss._load_summary_stats(str(d)) is None
    assert ss._load_categorical_stats(str(d)) == {}

    # load_density_graph cast fallback path
    density = d / ss.density_graph_filename
    stub._exists[density] = True
    stub._tables[density] = pa.table(
        {
            "variable": [1],
            "points": [[1.0]],
            "density": [[0.1]],
            "min": [1.0],
            "max": [1.0],
            "bandwidth": [0.1],
            "count": [1],
            "sampleCount": [1],
            "pointCount": [1],
        }
    )
    assert ss.load_density_graph(str(d), "1")["count"] == 1

    # remote+local branch
    density.touch(exist_ok=True)
    stub.is_remote = True
    assert ss.load_density_graph(str(d), "missing") is None
    stub.is_remote = False


def test_summary_stats_remaining_branches(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env

    # _layer_value_type / _legend_for_layer edge returns
    monkeypatch.setattr(ss.gis_lookup, "load_layer_metadata", lambda: {"x": {"value_type": 1}})
    assert ss._layer_value_type("x") is None
    monkeypatch.setattr(ss.gis_lookup, "load_layer_legend", lambda _id: {"1": {"id": 1, "name": ""}})
    assert ss._legend_for_layer("x") == {}

    # _build_density_curve filtered-empty + bandwidth fallback
    assert ss._build_density_curve([float("nan")], 10) is None
    original_isfinite = ss.math.isfinite
    monkeypatch.setattr(ss.math, "isfinite", lambda _x: False)
    assert ss._build_density_curve([1.0, 2.0], 8) is not None
    monkeypatch.setattr(ss.math, "isfinite", original_isfinite)

    # write_density_graph reservoir/curve-skip branches
    d = tmp_path / "r"
    d.mkdir()
    monkeypatch.setattr(ss, "density_max_samples", 1)
    monkeypatch.setattr(ss.random, "randrange", lambda _n: 0)
    monkeypatch.setattr(
        ss,
        "_iter_descendant_tables",
        lambda _p: [pa.table({"obscured": ["No"], "coordinateUncertaintyInMeters": [1], "x": [1.0], "y": [float("nan")]})],
    )
    monkeypatch.setattr(ss, "_layer_value_type", lambda c: "numeric")
    monkeypatch.setattr(ss, "_build_density_curve", lambda vals, _pc: None if vals and vals[0] == 1.0 else {"points": [0.0], "density": [1.0], "min": 0.0, "max": 0.0, "bandwidth": 1.0})
    ss.write_density_graph(d)

    # _write_categorical_stats merge-existing path
    cat_path = d / "categorical_stats.parquet"
    pd.DataFrame([{"variable": "v", "metric": "m", "value": 1.0}]).to_parquet(cat_path, index=False)
    ss._write_categorical_stats(d, [{"variable": "v", "metric": "m2", "value": 2.0}], merge_existing=True)
    merged = pd.read_parquet(cat_path)
    assert set(merged["metric"]) == {"m", "m2"}

    # dataframe conversion specific lines
    class _GoodItem:
        def item(self):
            return 5

    wide = pd.DataFrame({"variable": ["a"], "x": [_GoodItem()]})
    assert ss._dataframe_to_stats(wide)["a"]["x"] == 5
    tall = pd.DataFrame({"variable": ["a", "a"], "metric": [None, "m"], "value": [1.0, np.nan]})
    out = ss._tall_dataframe_to_stats(tall)
    assert "a" in out


def test_summary_stats_branch_sweep(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # _prepare_index_column returns None when table has zero columns
    p0 = tmp_path / "z.parquet"
    stub._exists[p0] = True
    stub._tables[p0] = pa.table({})
    assert ss._prepare_index_column(p0, "x") is None

    # range query branches: total==0 and stop<=start
    struct_type = pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())])
    p1 = tmp_path / "r.parquet"
    stub._exists[p1] = True
    meta = {b"origin_map": json.dumps([{"id": 1, "relative_path": "a"}]).encode("utf-8")}
    stub._tables[p1] = pa.Table.from_arrays([pa.array([], type=struct_type)], schema=pa.schema([pa.field("bio_1", struct_type)]).with_metadata(meta))
    assert ss.get_sorted_layer_records_in_value_range(p1, "bio_1", 0, 1) == []

    arr = pa.array([{"catalogNumber": "a", "originId": 1, "value": None}], type=struct_type)
    stub._tables[p1] = pa.Table.from_arrays([arr], schema=pa.schema([pa.field("bio_1", struct_type)]).with_metadata(meta))
    data = (tmp_path / "a" / "occurrence.parquet")
    data.parent.mkdir(parents=True, exist_ok=True)
    stub._exists[data] = True
    stub._tables[data] = pa.table(
        {
            "catalogNumber": ["a"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [2.0],
            "bio_1": ["bad"],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [1],
        }
    )
    assert ss.get_sorted_layer_records_in_value_range(p1, "bio_1", 0, 1) == []

    # class offsets fallback/zero-count branches
    meta2 = {
        b"origin_map": json.dumps([{"id": 1, "relative_path": "a"}]).encode("utf-8"),
        b"category_offsets": json.dumps({"bio_1": {"1.0": {"start": 0, "count": 0}}}).encode("utf-8"),
    }
    stub._tables[p1] = pa.Table.from_arrays([pa.array([{"catalogNumber": "a", "originId": 1, "value": 1.0}], type=struct_type)], schema=pa.schema([pa.field("bio_1", struct_type)]).with_metadata(meta2))
    monkeypatch.setattr(ss, "resolve_categorical_class_value", lambda *_a, **_k: 1)
    assert ss.get_layer_records_for_class(p1, "bio_1", 1) == []

    # schema wrappers
    schema = pa.schema([pa.field("x", pa.int64())])
    stub._schemas[p1] = schema
    stub._metadata[p1] = SimpleNamespace(num_rows=3)
    assert ss.get_schema(p1).names == ["x"]
    assert ss.get_num_rows(p1) == 3
    assert ss.get_column_names(p1) == ["x"]
    assert ss.get_column_types(p1)["x"] == "int64"

    # legend key float non-integer
    assert ss._legend_key(1.25) == "1.25"

    # load_categorical_distribution missing file and filter exception
    miss_dir = tmp_path / "miss"
    miss_dir.mkdir()
    assert ss.load_categorical_distribution(miss_dir, "x") is None
    stats_path = miss_dir / "categorical_stats.parquet"
    stub._exists[stats_path] = True
    stub._tables[stats_path] = pa.table({"variable": [1], "metric": ["m"], "value": [1.0]})
    assert ss.load_categorical_distribution(miss_dir, "x") is None

    # samples early return when index missing
    assert ss.build_categorical_samples(miss_dir, "x", [{"value": 1}]) == []

    # numeric gather parse-exception lines
    tbl = pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "bio_1": ["bad"]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [tbl])
    assert ss.gather_numeric_records_from_tables(1, "bio_1", None) == []

    # categorical sample filtered empty line
    tcat = pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "koppen": [None]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [tcat])
    assert ss.categorical_class_samples_for_location(1, "koppen", 1, location_gid="USA", limit=1) == []

    # numeric range filtered empty and parse-exception
    tnum = pa.table({"catalogNumber": ["a"], "decimalLatitude": [1.0], "decimalLongitude": [2.0], "bio_1": [None]})
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [tnum])
    assert ss.numeric_range_samples_for_location(1, "bio_1", 0, 2, location_gid="USA", limit=5) == []


def test_summary_stats_deep_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # _slice_records stop<=start branch via wrapper
    prepared = {
        "column": pa.array(
            [{"catalogNumber": "a", "originId": 1, "value": 1.0}],
            type=pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())]),
        ),
        "origin_lookup": {1: {"relative_path": "a"}},
        "index_dir": tmp_path,
        "catalog_column": "catalogNumber",
    }
    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: prepared)
    assert ss.get_sorted_layer_records(tmp_path / "x.parquet", "bio_1", start=2, stop=1) == []

    # range query deep fallback branches (no value field -> dataset lookup path)
    col = pa.array(
        [
            {"catalogNumber": None, "originId": 1},
            {"catalogNumber": "a", "originId": 1},
            {"catalogNumber": "b", "originId": 2},
            {"catalogNumber": "c", "originId": 3},
            {"catalogNumber": "d", "originId": 4},
            {"catalogNumber": "e", "originId": 5},
            {"catalogNumber": "f", "originId": 6},
        ],
        type=pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64())]),
    )
    prepared2 = dict(prepared)
    prepared2["column"] = col
    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: prepared2)

    data = {
        1: None,
        2: {"index": {}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([1.0])},
        3: {"index": {"c": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([None])},
        4: {"index": {"d": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array(["bad"])},
        5: {"index": {"e": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([2.5])},
        6: {"index": {"f": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([3.5])},
    }
    monkeypatch.setattr(ss, "_make_dataset_loader", lambda *_a, **_k: (lambda oid: data.get(oid)))
    out = ss.get_sorted_layer_records_in_value_range(tmp_path / "x.parquet", "bio_1", 2.0, 3.0)
    assert out and out[0][0] == "e"

    # get_layer_records_for_class prepared-none and empty-distribution branch
    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: None)
    assert ss.get_layer_records_for_class(tmp_path / "x.parquet", "bio_1", 1) == []

    cat_dir = tmp_path / "cat"
    cat_dir.mkdir()
    cat_path = cat_dir / "categorical_stats.parquet"
    stub._exists[cat_path] = True
    stub._tables[cat_path] = pa.table({"variable": ["v"], "metric": ["total_samples"], "value": ["1"]})
    assert ss.load_categorical_distribution(cat_dir, "v") is None

    # update streaming early-return numeric.size==0
    st = ss._init_streaming_stats()
    ss._update_streaming_stats(st, pd.Series([], dtype=float))
    assert st["count"] == 0

    # exact/streaming branch filters
    p = tmp_path / "occurrence.parquet"
    df = pd.DataFrame({"cat": [None], "num": [None]})
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [pa.Table.from_pandas(df)])
    monkeypatch.setattr(ss, "_layer_value_type", lambda c: "categorical" if c == "cat" else "numeric")
    monkeypatch.setattr(ss, "_write_summary_stats", lambda *_a, **_k: None)
    monkeypatch.setattr(ss, "_write_categorical_stats", lambda *_a, **_k: None)
    assert ss._numeric_column_stats_exact(p, existing_categorical={"cat"}) == {}
    assert ss._numeric_column_stats_streaming(p, existing_numeric={"num"}) == {}

    # categorical collectors extra branches
    assert ss._collect_categorical_stats(pd.DataFrame({"x": [1]}), ["missing"]) == []
    assert ss._collect_categorical_stats(pd.DataFrame({"x": [None]}), ["x"]) == []
    assert ss._collect_categorical_stats_from_counts({"x": {"a": 1}}, {"x": 0}) == []

    # _write_summary_stats frame.empty and write_density_graph cleanup-on-failure
    ss._write_summary_stats(tmp_path, {"v": {}})
    monkeypatch.setattr(ss.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    ss.write_density_graph(tmp_path)

    # _write_categorical_stats frame.empty path
    ss._write_categorical_stats(tmp_path, [{}], merge_existing=False)

    # load_density_graph cast exception branch (handled exceptions only)
    density = tmp_path / ss.density_graph_filename
    stub._exists[density] = True
    stub._tables[density] = pa.table(
        {
            "variable": ["v1"],
            "points": [[1.0]],
            "density": [[1.0]],
            "min": [1.0],
            "max": [1.0],
            "bandwidth": [0.1],
            "count": [1],
            "sampleCount": [1],
            "pointCount": [1],
        }
    )
    monkeypatch.setattr(ss.pc, "cast", lambda *_a, **_k: (_ for _ in ()).throw(pa.ArrowInvalid("bad cast")))
    assert ss.load_density_graph(str(tmp_path), "x") is None

    # tall dataframe variable/metric skip line
    out_tall = ss._tall_dataframe_to_stats(pd.DataFrame({"variable": [None], "metric": [None], "value": [1]}))
    assert out_tall == {}


def test_summary_stats_remaining_missing_lines(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env

    # _slice_records stop<=start direct branch
    prepared = {
        "column": pa.array(
            [{"catalogNumber": "a", "originId": 1, "value": 1.0}],
            type=pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())]),
        ),
        "origin_lookup": {1: {"relative_path": "a"}},
        "index_dir": tmp_path,
        "catalog_column": "catalogNumber",
    }
    assert ss._slice_records(prepared, "bio_1", start=0, stop=0, data_filename="occurrence.parquet", lat_col="decimalLatitude", lon_col="decimalLongitude") == []

    # value-range result loop branches: missing dataset/index, parse fail, min-continue, max-break
    col = pa.array(
        [
            {"catalogNumber": "a", "originId": 1, "value": 1.0},
            {"catalogNumber": "b", "originId": 2, "value": 2.0},
            {"catalogNumber": "c", "originId": 3, "value": 2.0},
            {"catalogNumber": "d", "originId": 4, "value": 0.5},
            {"catalogNumber": "e", "originId": 5, "value": 2.5},
        ],
        type=pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())]),
    )
    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: dict(prepared, column=col))
    datasets = {
        1: None,
        2: {"index": {}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([1.0])},
        3: {"index": {"c": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array(["bad"])},
        4: {"index": {"d": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([0.5])},
        5: {"index": {"e": 0}, "latitudes": pa.array([1.0]), "longitudes": pa.array([2.0]), "layer_values": pa.array([5.0])},
    }
    monkeypatch.setattr(ss, "_make_dataset_loader", lambda *_a, **_k: (lambda origin_id: datasets.get(origin_id)))
    out = ss.get_sorted_layer_records_in_value_range(tmp_path / "x.parquet", "bio_1", 1.0, 3.0)
    assert out == [("c", 1.0, 2.0, "bad")]

    # numeric range parse-fail branch
    bad_numeric = pa.table(
        {
            "catalogNumber": ["x"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [2.0],
            "bio_1": ["bad"],
        }
    )
    monkeypatch.setattr(ss.taxa_navigation, "iter_filtered_occurrence_tables", lambda *_a, **_k: [bad_numeric])
    monkeypatch.setattr(ss.pc, "greater_equal", lambda *_a, **_k: pa.array([True]))
    monkeypatch.setattr(ss.pc, "less_equal", lambda *_a, **_k: pa.array([True]))
    monkeypatch.setattr(ss.pc, "and_", lambda left, _right: left)
    assert ss.numeric_range_samples_for_location(1, "bio_1", 0.0, 10.0, location_gid=None, limit=10) == []

    # streaming branches: column removed after discovery (1509) and count==0 skip (1550)
    stream_df = pd.DataFrame({"cat": ["x"], "num": [1.0], "obscured": ["No"], "coordinateUncertaintyInMeters": [1]})

    class _Table:
        def to_pandas(self):
            return stream_df

    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [_Table()])

    def _layer_type(column):
        if column == "cat":
            stream_df.drop(columns=["cat"], inplace=True)
            return "categorical"
        return "numeric"

    monkeypatch.setattr(ss, "_layer_value_type", _layer_type)
    monkeypatch.setattr(ss, "_update_streaming_stats", lambda *_a, **_k: None)
    monkeypatch.setattr(ss, "_write_summary_stats", lambda *_a, **_k: None)
    monkeypatch.setattr(ss, "_write_categorical_stats", lambda *_a, **_k: None)
    assert ss._numeric_column_stats_streaming(tmp_path / "occurrence.parquet") == {}

    # categorical collectors: series.count exception fallback and non-int class ids
    orig_count = pd.Series.count
    monkeypatch.setattr(
        pd.Series,
        "count",
        lambda *_a, **_k: (_ for _ in ()).throw(TypeError("count-bad")),
    )
    entries = ss._collect_categorical_stats(pd.DataFrame({"cat": [1, "x"]}), ["cat"])
    assert any(e["metric"] == "class_x" for e in entries)
    monkeypatch.setattr(pd.Series, "count", orig_count)
    by_counts = ss._collect_categorical_stats_from_counts({"cat": {"x": 1}}, {"cat": 1})
    assert any(e["metric"] == "class_x" for e in by_counts)

    # _write_summary_stats frame.empty branch with truthy-empty mapping
    ss._write_summary_stats(tmp_path, {"v": {}})

    # write_density_graph branches: values empty (1776), reservoir replacement (1784-1786),
    # count/sample skip (1792), and temp cleanup on write failure (1828)
    orig_astype = pd.Series.astype

    class _EmptyList:
        def tolist(self):
            return []

    def _fake_astype(self, *args, **kwargs):
        if getattr(self, "name", "") == "skipcol":
            return _EmptyList()
        return orig_astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "astype", _fake_astype)
    monkeypatch.setattr(ss, "_iter_descendant_tables", lambda _p: [pa.table({"obscured": ["No", "No"], "coordinateUncertaintyInMeters": [1, 1], "skipcol": [1.0, 2.0], "keep": [1.0, 2.0]})])
    monkeypatch.setattr(ss, "_layer_value_type", lambda _c: "numeric")
    monkeypatch.setattr(ss, "_build_density_curve", lambda *_a, **_k: {"points": [0.0], "density": [1.0], "min": 0.0, "max": 0.0, "bandwidth": 1.0})
    monkeypatch.setattr(ss, "density_max_samples", 1)
    monkeypatch.setattr(ss.random, "randrange", lambda _n: 0)
    orig_write_table = ss.pq.write_table
    monkeypatch.setattr(ss.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ss.write_density_graph(tmp_path / "density_fail")

    monkeypatch.setattr(ss.pq, "write_table", orig_write_table)
    monkeypatch.setattr(ss, "density_max_samples", 0)
    ss.write_density_graph(tmp_path / "density_skip")

    # _write_categorical_stats frame.empty branch
    ss._write_categorical_stats(tmp_path / "cat", [{}], merge_existing=False)

    # load_density_graph missing/stat-ok branch
    density_dir = tmp_path / "density_stat"
    density_dir.mkdir()
    (density_dir / ss.density_graph_filename).touch()
    monkeypatch.setattr(Path, "exists", lambda _self: False)
    assert ss.load_density_graph(str(density_dir), "x") is None

    # tall dataframe explicit None branch
    tall = pd.DataFrame(
        {
            "variable": pd.Series([None], dtype=object),
            "metric": pd.Series(["m"], dtype=object),
            "value": [1],
        }
    )
    assert ss._tall_dataframe_to_stats(tall) == {}


def test_summary_stats_final_edge_lines(monkeypatch, tmp_path):
    # categorical writer with truthy entries but empty dict rows
    out_dir = tmp_path / "cat-empty"
    out_dir.mkdir()
    ss._write_categorical_stats(out_dir, [{}], merge_existing=False)
    assert not (out_dir / "categorical_stats.parquet").exists()


def test_summary_stats_upper_bound_binary_else_branch(monkeypatch, tmp_path):
    struct_type = pa.struct([("catalogNumber", pa.string()), ("originId", pa.int64()), ("value", pa.float64())])
    column = pa.array(
        [
            {"catalogNumber": "a", "originId": 1, "value": 1.0},
            {"catalogNumber": "b", "originId": 1, "value": 2.0},
            {"catalogNumber": "c", "originId": 1, "value": 3.0},
        ],
        type=struct_type,
    )
    prepared = {
        "column": column,
        "origin_lookup": {1: {"relative_path": "x"}},
        "index_dir": tmp_path,
        "catalog_column": "catalogNumber",
    }
    monkeypatch.setattr(ss, "_prepare_index_column", lambda *_a, **_k: prepared)
    monkeypatch.setattr(
        ss,
        "_make_dataset_loader",
        lambda *_a, **_k: (
            lambda _oid: {
                "index": {"a": 0, "b": 1, "c": 2},
                "latitudes": pa.array([1.0, 1.0, 1.0]),
                "longitudes": pa.array([2.0, 2.0, 2.0]),
                "layer_values": pa.array([1.0, 2.0, 3.0]),
            }
        ),
    )
    out = ss.get_sorted_layer_records_in_value_range(tmp_path / "x.parquet", "bio_1", None, 1.5)
    assert [row[0] for row in out] == ["a"]
