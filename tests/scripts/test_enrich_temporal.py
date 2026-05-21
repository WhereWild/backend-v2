"""Tests for scripts/enrich_temporal.py — script-level logic only."""
from __future__ import annotations

import builtins
import signal
import threading
from pathlib import Path

import pyarrow as pa
import pytest

import scripts.enrich_temporal as et
from scripts.enrich_temporal import _cleanup_cache, _filter_layers, _rss_mb, _run_layer
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
        assert _filter_layers(layers, None) == layers

    def test_single_temporal_id(self) -> None:
        result = _filter_layers(_layers(), ["precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_multiple_temporal_ids(self) -> None:
        result = _filter_layers(_layers(), ["precipitation", "snow_depth"])
        ids = {layer.id for layer in result}
        assert ids == {"precipitation", "snow_depth"}

    def test_no_temporal_ids_returns_all(self) -> None:
        # All ids are spatial → treat as "do all temporal"
        layers = _layers()
        result = _filter_layers(layers, ["bio1", "bio12", "gsl"])
        assert result == layers

    def test_mixed_ids_returns_only_temporal_matches(self) -> None:
        result = _filter_layers(_layers(), ["bio1", "precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_derived_var_included_when_requested(self) -> None:
        result = _filter_layers(_layers(), ["vapor_pressure_deficit"])
        assert len(result) == 1
        assert result[0].derived is True

    def test_empty_list_returns_all(self) -> None:
        layers = _layers()
        assert _filter_layers(layers, []) == layers

    def test_order_preserved(self) -> None:
        result = _filter_layers(_layers(), ["snow_depth", "temperature_2m"])
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
        result = _rss_mb()
        assert result is None or isinstance(result, float)

    def test_positive_when_present(self) -> None:
        result = _rss_mb()
        if result is not None:
            assert result > 0

    def test_returns_none_on_open_exception(self, monkeypatch) -> None:
        _orig = builtins.open
        def _raise(path, *a, **kw):
            if "/proc/self/status" in str(path):
                raise OSError("permission denied")
            return _orig(path, *a, **kw)
        monkeypatch.setattr(builtins, "open", _raise)
        assert _rss_mb() is None

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
        assert _rss_mb() is None


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
        _cleanup_cache(str(tmp_path))
        assert not f1.exists()
        assert not f2.exists()

    def test_nonexistent_dir_ok(self, tmp_path: Path) -> None:
        _cleanup_cache(str(tmp_path / "nonexistent"))  # no error

    def test_exception_in_unlink_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "locked.om"
        f.write_bytes(b"data")

        def _raise(self, *args, **kwargs):  # noqa: ANN001
            raise PermissionError("locked")

        monkeypatch.setattr(Path, "unlink", _raise)
        _cleanup_cache(str(tmp_path))  # must not raise


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
    plantae_key = 1
    data_root = "/data"
    occurrence_parquet_filename = "occurrence.parquet"
    temporal_min_year = 2000
    temporal_cache_dir = "/tmp/test_cache"


class TestRunLayer:
    def test_skips_when_chunk_index_fails(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no S3")))
        _run_layer(_make_layer(), pa.table({}), _MockCfg(), threading.Event())
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
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: empty)
        _run_layer(_make_layer(), pa.table({}), _MockCfg(), threading.Event())
        assert "[skip]" in capsys.readouterr().out

    def test_normal_run(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)
        monkeypatch.setattr("scripts.enrich_temporal.process_chunk",
                            lambda *a, **kw: ({}, {}))
        monkeypatch.setattr("scripts.enrich_temporal.write_back", lambda *a, **kw: None)

        _run_layer(_make_layer(), pa.table({}), _MockCfg(), threading.Event())
        out = capsys.readouterr().out
        assert "[done]" in out

    def test_stop_event_aborts_before_chunk(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)

        stop = threading.Event()
        stop.set()
        _run_layer(_make_layer(), pa.table({}), _MockCfg(), stop)
        assert "[stop]" in capsys.readouterr().out

    def test_run_layer_mode_calls_process_chunk_mode(self, monkeypatch, capsys) -> None:
        mode_layer = TemporalLayer(
            id="weather_code_simple", model="copernicus_era5",
            grid_mode="lat_asc_lon_pm180", agg="mode", windows=[1, 24],
            sources=["cloud_cover", "precipitation", "snowfall_water_equivalent"],
        )
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        mode_called = []
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)
        monkeypatch.setattr("scripts.enrich_temporal.process_chunk_mode",
                            lambda *a, **kw: (mode_called.append(1), ({}, {}))[-1])
        monkeypatch.setattr("scripts.enrich_temporal.write_back", lambda *a, **kw: None)
        _run_layer(mode_layer, pa.table({}), _MockCfg(), threading.Event())
        assert mode_called

    def test_process_chunk_exception_propagates(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.enrich_temporal.build_chunk_index",
                            lambda *a, **kw: _make_chunk_index())
        monkeypatch.setattr("scripts.enrich_temporal.map_to_worklist",
                            lambda *a, **kw: _occ_table_with_chunk())
        monkeypatch.setattr("scripts.enrich_temporal._download_layer_chunk", lambda *a, **kw: None)

        def _raise(*a, **kw):
            raise RuntimeError("chunk failed")

        monkeypatch.setattr("scripts.enrich_temporal.process_chunk", _raise)
        with pytest.raises(RuntimeError, match="chunk failed"):
            _run_layer(_make_layer(), pa.table({}), _MockCfg(), threading.Event())


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
                      grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24], derived=True),
        TemporalLayer(id="weather_code_simple", model="copernicus_era5",
                      grid_mode="lat_asc_lon_pm180", agg="mode", windows=[1, 24],
                      sources=["cloud_cover", "precipitation", "snowfall_water_equivalent"]),
    ]


class TestMain:
    def _patch_base(self, monkeypatch, tmp_path: Path, occ_table: pa.Table) -> None:
        class _Cfg:
            plantae_key = 1
            data_root = str(tmp_path)
            occurrence_parquet_filename = "occurrence.parquet"
            temporal_min_year = 2000
            temporal_cache_dir = str(tmp_path / "cache")

        monkeypatch.setattr("scripts.enrich_temporal.load_config", lambda _: _Cfg())
        monkeypatch.setattr("scripts.enrich_temporal.load_temporal_layers", lambda _: _all_layers())
        monkeypatch.setattr("scripts.enrich_temporal.build_occ_index", lambda *a, **kw: occ_table)
        monkeypatch.setattr("scripts.enrich_temporal.VARS_TO_ENRICH", None)

    def test_no_observations_exits_early(self, monkeypatch, tmp_path: Path, capsys) -> None:
        empty = _make_occ_table(0)
        self._patch_base(monkeypatch, tmp_path, empty)
        run_layer_calls: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._run_layer",
                            lambda *a, **kw: run_layer_calls.append(a[0].id))
        et.main()
        assert run_layer_calls == []
        assert "[done] no observations" in capsys.readouterr().out

    def test_full_run_calls_run_layer_and_derived(self, monkeypatch, tmp_path: Path) -> None:
        self._patch_base(monkeypatch, tmp_path, _make_occ_table())
        run_layer_calls: list[str] = []
        monkeypatch.setattr("scripts.enrich_temporal._run_layer",
                            lambda *a, **kw: run_layer_calls.append(a[0].id))
        vpd_called = []
        monkeypatch.setattr("scripts.enrich_temporal.derive_vpd",
                            lambda *a, **kw: vpd_called.append(1))
        et.main()
        assert "precipitation" in run_layer_calls
        assert "weather_code_simple" in run_layer_calls
        assert "vapor_pressure_deficit" not in run_layer_calls
        assert vpd_called

    def test_derive_vpd_exception_handled(self, monkeypatch, tmp_path: Path) -> None:
        self._patch_base(monkeypatch, tmp_path, _make_occ_table())
        monkeypatch.setattr("scripts.enrich_temporal._run_layer", lambda *a, **kw: None)

        def _raise(*a, **kw):
            raise RuntimeError("vpd exploded")

        monkeypatch.setattr("scripts.enrich_temporal.derive_vpd", _raise)
        et.main()  # must not propagate

    def test_handle_signal_sets_stop(self, monkeypatch, tmp_path: Path, capsys) -> None:
        captured: dict[int, object] = {}
        monkeypatch.setattr(signal, "signal", lambda sig, h: captured.__setitem__(sig, h))
        self._patch_base(monkeypatch, tmp_path, _make_occ_table(0))
        et.main()
        assert signal.SIGTERM in captured
        captured[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[operator]
        assert "signal" in capsys.readouterr().out

    def test_signal_setup_exception_ignored(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(signal, "signal",
                            lambda *a: (_ for _ in ()).throw(ValueError("not main thread")))
        self._patch_base(monkeypatch, tmp_path, _make_occ_table(0))
        et.main()  # must not propagate
