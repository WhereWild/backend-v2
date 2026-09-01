# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for scripts/enrich_temporal.py — script-level logic only."""
from __future__ import annotations

import builtins
import threading
from pathlib import Path

import pyarrow as pa
import pytest

import scripts.enrich_temporal as et
from util.temporal import ChunkIndex, ChunkRange, TemporalLayer


def _layers() -> list[TemporalLayer]:
    return [
        TemporalLayer(id="temperature_2m", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24]),
        TemporalLayer(id="precipitation", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="sum", windows=[24]),
        TemporalLayer(id="snow_depth", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[1]),
        TemporalLayer(id="vapor_pressure_deficit", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24], derived=True),
        TemporalLayer(id="weather_code_simple", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="snapshot", windows=[1], derived=True),
    ]


class TestFilterLayers:
    def test_none_returns_all(self) -> None:
        layers = _layers()
        assert et._filter_layers(layers, None) == layers

    def test_single_temporal_id(self) -> None:
        result = et._filter_layers(_layers(), ["precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_multiple_temporal_ids(self) -> None:
        result = et._filter_layers(_layers(), ["precipitation", "snow_depth"])
        ids = {layer.id for layer in result}
        assert ids == {"precipitation", "snow_depth"}

    def test_no_temporal_ids_returns_all(self) -> None:
        # All ids are spatial → treat as "do all temporal"
        layers = _layers()
        result = et._filter_layers(layers, ["bio1", "bio12", "gsl"])
        assert result == layers

    def test_mixed_ids_returns_only_temporal_matches(self) -> None:
        result = et._filter_layers(_layers(), ["bio1", "precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_derived_var_included_when_requested(self) -> None:
        result = et._filter_layers(_layers(), ["vapor_pressure_deficit"])
        assert len(result) == 1
        assert result[0].derived is True

    def test_empty_list_returns_all(self) -> None:
        layers = _layers()
        assert et._filter_layers(layers, []) == layers

    def test_order_preserved(self) -> None:
        result = et._filter_layers(_layers(), ["snow_depth", "temperature_2m"])
        assert [layer.id for layer in result] == ["temperature_2m", "snow_depth"]


class TestVarsToEnrichParsing:
    def test_module_level_parsing_none_when_empty(self) -> None:
        # VARS_TO_ENRICH should be None when env var was not set (empty string)
        # This relies on the module being imported without the env var set
        assert et.VARS_TO_ENRICH is None or isinstance(et.VARS_TO_ENRICH, list)


# ---------------------------------------------------------------------------
# _rss_mb
# ---------------------------------------------------------------------------

class TestRssMb:
    def test_returns_float_or_none(self) -> None:
        result = et._rss_mb()
        assert result is None or isinstance(result, float)

    def test_positive_when_present(self) -> None:
        result = et._rss_mb()
        if result is not None:
            assert result > 0

    def test_returns_none_on_open_exception(self, monkeypatch) -> None:
        _orig = builtins.open
        def _raise(path, *a, **kw):
            if "/proc/self/status" in str(path):
                raise OSError("permission denied")
            return _orig(path, *a, **kw)
        monkeypatch.setattr(builtins, "open", _raise)
        assert et._rss_mb() is None

    def test_returns_none_when_no_vmrss_line(self, monkeypatch) -> None:
        _orig = builtins.open
        class _NoVmRSSFile:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self): return iter(["Name:\tpython3\n", "Pid:\t1\n"])
        def _fake(path, *a, **kw):
            if "/proc/self/status" in str(path):
                return _NoVmRSSFile()
            return _orig(path, *a, **kw)
        monkeypatch.setattr(builtins, "open", _fake)
        assert et._rss_mb() is None


# ---------------------------------------------------------------------------
# _cleanup_cache
# ---------------------------------------------------------------------------

class TestCleanupCache:
    def test_deletes_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.om"
        f2 = tmp_path / "sub" / "b.om"
        f2.parent.mkdir()
        f1.write_bytes(b"x")
        f2.write_bytes(b"y")
        et._cleanup_cache(str(tmp_path))
        assert not f1.exists()
        assert not f2.exists()

    def test_nonexistent_dir_ok(self, tmp_path: Path) -> None:
        et._cleanup_cache(str(tmp_path / "nonexistent"))  # no error

    def test_exception_in_unlink_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "locked.om"
        f.write_bytes(b"data")

        def _raise(self, *args, **kwargs):  # noqa: ANN001
            raise PermissionError("locked")

        monkeypatch.setattr(Path, "unlink", _raise)
        et._cleanup_cache(str(tmp_path))  # must not raise


# ---------------------------------------------------------------------------
# _run_layer
# ---------------------------------------------------------------------------

def _make_layer(layer_id: str = "precipitation") -> TemporalLayer:
    return TemporalLayer(
        id=layer_id, model="copernicus_era5",
        grid_mode="lat_asc_lon_pm180", agg="sum", windows=[24],
    )


def _make_chunk_index() -> ChunkIndex:
    entry = ChunkRange(chunk_num=2019, start=0.0, end=8759 * 3600.0, time_len=8760, source="year")
    return ChunkIndex(latest_end_time=8759 * 3600.0, resolution=3600.0, ranges=[entry])


def _occ_table_with_chunk() -> pa.Table:
    return pa.table({
        "taxon_path": pa.array(["/data/occ.parquet"]),
        "row_idx": pa.array([0], type=pa.int64()),
        "chunk_num": pa.array([2019], type=pa.int32()),
        "lat_idx": pa.array([360], type=pa.int32()),
        "lon_idx": pa.array([720], type=pa.int32()),
        "time_idx": pa.array([500], type=pa.int32()),
    })


class _MockCfg:
    taxonomy_roots = (1,)
    data_root = "/data"
    occurrence_parquet_filename = "occurrence.parquet"
    temporal_min_date = "2000-01-01"
    temporal_cache_dir = "/tmp/test_cache"


_DUMMY_OCC_PATH = Path("/dev/null")  # placeholder — always mocked in TestRunLayer


def _mock_batches(batch: pa.Table):
    """Return iter_occ_index_batches mock that yields one batch."""
    return lambda *a, **kw: [batch]


class TestRunLayer:
    def test_skips_when_chunk_index_fails(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no S3")))
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({})))
        et._run_layer(_make_layer(), _DUMMY_OCC_PATH, _MockCfg())
        assert "[skip]" in capsys.readouterr().out

    def test_skips_when_no_worklist_rows(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        empty = pa.table({
            "taxon_path": pa.array([], type=pa.string()),
            "row_idx": pa.array([], type=pa.int64()),
            "chunk_num": pa.array([], type=pa.int32()),
            "lat_idx": pa.array([], type=pa.int32()),
            "lon_idx": pa.array([], type=pa.int32()),
            "time_idx": pa.array([], type=pa.int32()),
        })
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({})))
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: empty)
        et._run_layer(_make_layer(), _DUMMY_OCC_PATH, _MockCfg())
        assert "[skip]" in capsys.readouterr().out

    def test_normal_run(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({})))
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)
        monkeypatch.setattr("scripts.enrich_temporal.process_chunk",
                            lambda *a, **kw: ({}, {}))
        monkeypatch.setattr("scripts.enrich_temporal.write_back", lambda *a, **kw: None)

        et._run_layer(_make_layer(), _DUMMY_OCC_PATH, _MockCfg())
        out = capsys.readouterr().out
        assert "[done]" in out

    def test_run_layer_mode_calls_process_chunk_mode(self, monkeypatch, capsys) -> None:
        mode_layer = TemporalLayer(
            id="weather_code_simple", model="copernicus_era5",
            grid_mode="lat_asc_lon_pm180", agg="mode", windows=[1, 24],
            sources=["cloud_cover", "precipitation", "snowfall_water_equivalent"],
        )
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({})))
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        mode_called = []
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)
        monkeypatch.setattr("scripts.enrich_temporal.process_chunk_mode",
                            lambda *a, **kw: (mode_called.append(1), ({}, {}))[-1])
        monkeypatch.setattr("scripts.enrich_temporal.write_back", lambda *a, **kw: None)
        et._run_layer(mode_layer, _DUMMY_OCC_PATH, _MockCfg())
        assert mode_called

    def test_process_chunk_exception_propagates(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({})))
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)

        def _raise(*a, **kw):
            raise RuntimeError("chunk failed")

        monkeypatch.setattr("scripts.enrich_temporal.process_chunk", _raise)
        with pytest.raises(RuntimeError, match="chunk failed"):
            et._run_layer(_make_layer(), _DUMMY_OCC_PATH, _MockCfg())


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _make_occ_table(n: int = 1) -> pa.Table:
    return pa.table({
        "taxon_path": pa.array(["/data/occ.parquet"] * n),
        "row_idx": pa.array(list(range(n)), type=pa.int64()),
        "latitude": pa.array([52.52] * n, type=pa.float64()),
        "longitude": pa.array([13.40] * n, type=pa.float64()),
        "timestamp": pa.array([1_000_000.0] * n, type=pa.float64()),
    })


def _all_layers() -> list[TemporalLayer]:
    return [
        TemporalLayer(id="precipitation", model="copernicus_era5",
                      grid_mode="lat_asc_lon_pm180", agg="sum", windows=[24]),
        TemporalLayer(id="vapor_pressure_deficit", model="copernicus_era5",
                      grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24],
                      sources=["temperature_2m", "dew_point_2m"]),
        TemporalLayer(id="weather_code_simple", model="copernicus_era5",
                      grid_mode="lat_asc_lon_pm180", agg="mode", windows=[1, 24],
                      sources=["cloud_cover", "precipitation", "snowfall_water_equivalent"]),
    ]


class TestMain:
    def _patch_base(self, monkeypatch, tmp_path: Path, occ_table: pa.Table) -> None:
        class _Cfg:
            taxonomy_roots = (1,)
            data_root = str(tmp_path)
            occurrence_parquet_filename = "occurrence.parquet"
            temporal_min_date = "2000-01-01"
            temporal_cache_dir = str(tmp_path / "cache")

        monkeypatch.setattr("scripts.enrich_temporal.load_config", lambda _: _Cfg())
        monkeypatch.setattr("scripts.enrich_temporal.load_temporal_layers", lambda _: _all_layers())
        monkeypatch.setattr(
            "scripts.enrich_temporal.build_per_layer_occ_indices",
            lambda *a, layers=(), **kw: {lyr.id: occ_table.num_rows for lyr in layers},
        )
        monkeypatch.setattr("scripts.enrich_temporal.VARS_TO_ENRICH", None)
        monkeypatch.setattr("scripts.enrich_temporal._start_prefetcher", lambda *a, **kw: None)

    def test_no_observations_exits_early(self, monkeypatch, tmp_path: Path, capsys) -> None:
        empty = _make_occ_table(0)
        self._patch_base(monkeypatch, tmp_path, empty)
        run_layer_calls: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._run_layer",
                            lambda *a, **kw: run_layer_calls.append(a[0].id) or {})
        et.main()
        assert run_layer_calls == []
        assert "[done] no observations" in capsys.readouterr().out

    def test_full_run_calls_run_layer_for_all(self, monkeypatch, tmp_path: Path) -> None:
        self._patch_base(monkeypatch, tmp_path, _make_occ_table())
        run_layer_calls: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._run_layer",
                            lambda *a, **kw: run_layer_calls.append(a[0].id) or {})
        et.main()
        assert "precipitation" in run_layer_calls
        assert "weather_code_simple" in run_layer_calls
        assert "vapor_pressure_deficit" in run_layer_calls

    def test_interrupt_skips_cleanup(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(et, "CLEAR_CACHE", True)
        self._patch_base(monkeypatch, tmp_path, _make_occ_table())
        cleaned: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._cleanup_cache", lambda d: cleaned.append(d))

        def _interrupt(*a, **kw):
            raise KeyboardInterrupt

        monkeypatch.setattr("scripts.enrich_temporal._run_layer", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            et.main()
        assert cleaned == []

    def test_clear_cache_true_calls_cleanup(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(et, "CLEAR_CACHE", True)
        self._patch_base(monkeypatch, tmp_path, _make_occ_table(0))
        cleaned: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._cleanup_cache", lambda d: cleaned.append(d))
        et.main()
        assert len(cleaned) == 1

    def test_clear_cache_false_preserves_cache(self, monkeypatch, tmp_path: Path, capsys) -> None:
        monkeypatch.setattr(et, "CLEAR_CACHE", False)
        self._patch_base(monkeypatch, tmp_path, _make_occ_table(0))
        cleaned: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._cleanup_cache", lambda d: cleaned.append(d))
        et.main()
        assert cleaned == []
        assert "preserved" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Background prefetch
# ---------------------------------------------------------------------------

def _prefetch_chunk_index() -> ChunkIndex:
    ranges = [
        ChunkRange(chunk_num=10, start=0.0, end=3599.0, time_len=1, source="chunk"),
        ChunkRange(chunk_num=11, start=3600.0, end=7199.0, time_len=1, source="chunk"),
    ]
    return ChunkIndex(latest_end_time=7199.0, resolution=3600.0, ranges=ranges)


class TestPlanLayerDownloads:
    def test_yields_dense_skips_sparse(self, monkeypatch) -> None:
        monkeypatch.setattr(et, "_RANGE_REQUEST_THRESHOLD", 2)
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _prefetch_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({"x": pa.array([1])})))
        # chunk 10 -> 3 rows (dense), chunk 11 -> 1 row (sparse)
        worklist = pa.table({"chunk_num": pa.array([10, 10, 10, 11], type=pa.int32())})
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: worklist)
        out = list(et._plan_layer_downloads(_make_layer(), _DUMMY_OCC_PATH, _MockCfg()))
        assert out == [
            (_prefetch_chunk_index().ranges[0], "copernicus_era5", ["precipitation"]),
        ]

    def test_dedups_across_batches(self, monkeypatch) -> None:
        monkeypatch.setattr(et, "_RANGE_REQUEST_THRESHOLD", 0)
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _prefetch_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            lambda *a, **kw: [pa.table({"x": [1]}), pa.table({"x": [2]})])
        worklist = pa.table({"chunk_num": pa.array([10, 11], type=pa.int32())})
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: worklist)
        out = list(et._plan_layer_downloads(_make_layer(), _DUMMY_OCC_PATH, _MockCfg()))
        assert [e.chunk_num for e, _, _ in out] == [10, 11]

    def test_empty_worklist_yields_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _prefetch_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.iter_occ_index_batches",
                            _mock_batches(pa.table({"x": pa.array([1])})))
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: pa.table({"chunk_num": pa.array([], type=pa.int32())}))
        assert list(et._plan_layer_downloads(_make_layer(), _DUMMY_OCC_PATH, _MockCfg())) == []


class TestStartPrefetcher:
    def test_downloads_planned_chunks(self, monkeypatch, tmp_path: Path) -> None:
        idx = tmp_path / "occ_index_precipitation.parquet"
        idx.write_bytes(b"x")
        layer = _make_layer()
        entry = ChunkRange(chunk_num=10, start=0.0, end=1.0, time_len=1, source="chunk")
        monkeypatch.setattr("scripts.enrich_temporal._plan_layer_downloads",
                            lambda *a, **kw: iter([(entry, "copernicus_era5", ["precipitation"])]))
        got: list = []
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk",
                            lambda e, m, v, d: got.append((e.chunk_num, m, tuple(v))))
        t = et._start_prefetcher([layer], {"precipitation": idx}, {"precipitation": 5},
                                 _MockCfg(), threading.Event())
        t.join(timeout=5)
        assert got == [(10, "copernicus_era5", ("precipitation",))]

    def test_skips_layer_without_index_file(self, monkeypatch, tmp_path: Path) -> None:
        called = threading.Event()
        monkeypatch.setattr("scripts.enrich_temporal._plan_layer_downloads",
                            lambda *a, **kw: called.set() or iter([]))
        t = et._start_prefetcher([_make_layer()], {"precipitation": tmp_path / "missing.parquet"},
                                 {"precipitation": 5}, _MockCfg(), threading.Event())
        t.join(timeout=5)
        assert not called.is_set()

    def test_prefetch_stop_halts_planning(self, monkeypatch, tmp_path: Path) -> None:
        idx = tmp_path / "occ_index_precipitation.parquet"
        idx.write_bytes(b"x")
        stop_evt = threading.Event()
        stop_evt.set()
        planned = threading.Event()
        monkeypatch.setattr("scripts.enrich_temporal._plan_layer_downloads",
                            lambda *a, **kw: planned.set() or iter([]))
        t = et._start_prefetcher([_make_layer()], {"precipitation": idx},
                                 {"precipitation": 5}, _MockCfg(), stop_evt)
        t.join(timeout=5)
        assert not planned.is_set()
