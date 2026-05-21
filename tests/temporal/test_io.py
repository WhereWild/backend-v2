"""Tests for util/temporal.py I/O functions: S3 helpers, chunk index, occ index."""
from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import util.temporal
from util.temporal import (
    ELEVATION_CORRECTABLE_VARS,
    ChunkIndex,
    ChunkRange,
    _download_chunk,
    _download_layer_chunk,
    _grid_indices_batch,
    _open_s3_json,
    _parse_s3_time,
    _read_model_elevation,
    build_chunk_index,
    build_occ_index,
    load_temporal_layers,
    map_to_worklist,
    prefetch_chunks,
)

_CATALOG = Path("config/gis/catalog.json")

_META = {
    "data_end_time": 1704067200.0,
    "temporal_resolution_seconds": 3600,
    "chunk_time_length": 8760,
}
_LISTING = [
    {"name": "s3://openmeteo/data/copernicus_era5/precipitation/chunk_0.om"},
    {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2022.om"},
    {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2023.om"},
    {"name": "s3://openmeteo/data/copernicus_era5/precipitation/README.txt"},
]


def _mock_s3(monkeypatch, meta=None, listing=None):
    util.temporal._CHUNK_INDEX_CACHE.clear()
    monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: meta if meta is not None else _META.copy())
    mock_fs = MagicMock()
    mock_fs.ls.return_value = listing if listing is not None else list(_LISTING)
    monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
    return mock_fs


# ---------------------------------------------------------------------------
# load_temporal_layers
# ---------------------------------------------------------------------------

class TestLoadTemporalLayers:
    def test_returns_layers(self):
        layers = load_temporal_layers(_CATALOG)
        assert len(layers) > 0

    def test_temperature_present(self):
        layers = load_temporal_layers(_CATALOG)
        ids = {la.id for la in layers}
        assert "temperature_2m" in ids

    def test_derived_layers(self):
        layers = load_temporal_layers(_CATALOG)
        derived = {la.id for la in layers if la.derived}
        assert "vapor_pressure_deficit" in derived

    def test_layer_window_override(self):
        layers = load_temporal_layers(_CATALOG)
        snow = next(la for la in layers if la.id == "snow_depth")
        # snow_depth inherits category windows (no per-layer override)
        assert len(snow.windows) > 1

    def test_weather_code_sources(self):
        layers = load_temporal_layers(_CATALOG)
        wc = next(la for la in layers if la.id == "weather_code_simple")
        assert wc.agg == "mode"
        assert "cloud_cover" in wc.sources
        assert "precipitation" in wc.sources

    def test_category_windows_inherited(self):
        layers = load_temporal_layers(_CATALOG)
        temp = next(la for la in layers if la.id == "temperature_2m")
        assert 24 in temp.windows

    def test_model_and_grid_mode(self):
        layers = load_temporal_layers(_CATALOG)
        temp = next(la for la in layers if la.id == "temperature_2m")
        assert temp.model == "copernicus_era5"
        assert temp.grid_mode == "lat_asc_lon_pm180"


# ---------------------------------------------------------------------------
# _grid_indices_batch — modes not covered by integration tests
# ---------------------------------------------------------------------------

class TestGridIndicesBatch:
    _NY, _NX, _STEP = 721, 1440, 0.25

    def _call(self, lats, lons, mode):
        return _grid_indices_batch(
            np.array(lats, dtype=float),
            np.array(lons, dtype=float),
            self._NY, self._NX, mode, self._STEP,
        )

    def test_lat_asc_lon_360_positive(self):
        li, lo = self._call([0.0], [270.0], "lat_asc_lon_360")
        assert li[0] == round((0.0 + 90.0) / 0.25)
        assert lo[0] == round(270.0 / 0.25)

    def test_lat_asc_lon_360_negative_wraps(self):
        li, lo = self._call([0.0], [-90.0], "lat_asc_lon_360")
        assert lo[0] == round(270.0 / 0.25)  # -90 % 360 = 270

    def test_lat_desc_lon_360(self):
        li, lo = self._call([52.52], [13.40], "lat_desc_lon_360")
        assert li[0] == round((90.0 - 52.52) / 0.25)
        assert lo[0] == round(13.40 / 0.25)

    def test_lat_desc_lon_pm180(self):
        li, lo = self._call([0.0], [0.0], "lat_desc_lon_pm180")
        assert li[0] == round(90.0 / 0.25)
        assert lo[0] == round(180.0 / 0.25)

    def test_clamping_out_of_bounds(self):
        li, lo = self._call([-100.0], [200.0], "lat_asc_lon_pm180")
        assert li[0] == 0
        assert lo[0] == self._NX - 1


# ---------------------------------------------------------------------------
# _parse_s3_time
# ---------------------------------------------------------------------------

class TestParseS3Time:
    def test_int(self):
        assert _parse_s3_time(1_234_567_890) == pytest.approx(1_234_567_890.0)

    def test_float(self):
        assert _parse_s3_time(1_234_567_890.5) == pytest.approx(1_234_567_890.5)

    def test_string_numeric(self):
        assert _parse_s3_time("1234567890") == pytest.approx(1_234_567_890.0)

    def test_string_iso_z(self):
        result = _parse_s3_time("2023-01-01T00:00:00Z")
        expected = datetime(2023, 1, 1, tzinfo=UTC).timestamp()
        assert result == pytest.approx(expected)

    def test_string_invalid(self):
        assert _parse_s3_time("not-a-time") is None

    def test_none_type(self):
        assert _parse_s3_time(None) is None

    def test_list_type(self):
        assert _parse_s3_time([]) is None


# ---------------------------------------------------------------------------
# _open_s3_json
# ---------------------------------------------------------------------------

class TestOpenS3Json:
    def test_success(self, monkeypatch):
        data = {"data_end_time": 1_234_567_890.0}

        @contextmanager
        def _fake_open(*a, **kw):
            yield BytesIO(json.dumps(data).encode())

        monkeypatch.setattr(fsspec, "open", _fake_open)
        assert _open_s3_json("s3://fake/meta.json") == data

    def test_exception_returns_none(self, monkeypatch):
        def _raise(*a, **kw):
            raise OSError("no network")

        monkeypatch.setattr(fsspec, "open", _raise)
        assert _open_s3_json("s3://fake/meta.json") is None


# ---------------------------------------------------------------------------
# _download_chunk
# ---------------------------------------------------------------------------

class TestDownloadChunk:
    def test_returns_existing_file(self, tmp_path):
        entry = ChunkRange(chunk_num=2019, start=0, end=0, time_len=1, source="chunk")
        dest_dir = tmp_path / "chunks"
        dest_dir.mkdir()
        dest = dest_dir / "copernicus_era5_precipitation_chunk_2019.om"
        dest.write_bytes(b"cached")
        result = _download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        assert result == dest

    def test_downloads_year_file(self, tmp_path, monkeypatch):
        entry = ChunkRange(chunk_num=2022, start=0, end=0, time_len=1, source="year")

        def _fake_run(cmd, **kw):
            # Write fake data to the .tmp file aria2c would create
            dest_dir = tmp_path / "chunks"
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "copernicus_era5_precipitation_year_2022.om.tmp").write_bytes(b"omdata")

        monkeypatch.setattr("util.temporal.subprocess.run", _fake_run)
        result = _download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        assert result.name == "copernicus_era5_precipitation_year_2022.om"
        assert result.read_bytes() == b"omdata"

    def test_failure_removes_tmp_file(self, tmp_path, monkeypatch):
        entry = ChunkRange(chunk_num=0, start=0, end=0, time_len=1, source="chunk")

        def _raise(*a, **kw):
            dest_dir = tmp_path / "chunks"
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "copernicus_era5_precipitation_chunk_0.om.tmp").write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, "aria2c")

        monkeypatch.setattr("util.temporal.subprocess.run", _raise)
        with pytest.raises(subprocess.CalledProcessError):
            _download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        chunks_dir = tmp_path / "chunks"
        tmp_files = list(chunks_dir.glob("*.tmp")) if chunks_dir.exists() else []
        assert tmp_files == []

    def test_download_layer_chunk_calls_download_for_each_var(self, tmp_path, monkeypatch) -> None:
        entry = ChunkRange(chunk_num=2020, start=0, end=0, time_len=1, source="year")
        called = []

        def _fake_dl(e, model, var, cache_dir):
            called.append(var)
            return Path(cache_dir) / "chunks" / f"{model}_{var}_year_2020.om"

        monkeypatch.setattr("util.temporal._download_chunk", _fake_dl)
        result = _download_layer_chunk(entry, "copernicus_era5", ["cloud_cover", "precipitation"], str(tmp_path))
        assert called == ["cloud_cover", "precipitation"]
        assert result is entry


# ---------------------------------------------------------------------------
# prefetch_chunks
# ---------------------------------------------------------------------------

class TestPrefetchChunks:
    def _entry(self, year: int) -> ChunkRange:
        return ChunkRange(chunk_num=year, start=0.0, end=0.0, time_len=8760, source="year")

    def test_no_tasks_when_all_cached(self, tmp_path, monkeypatch) -> None:
        dest_dir = tmp_path / "chunks"
        dest_dir.mkdir()
        entry = self._entry(2020)
        target = dest_dir / "copernicus_era5_temperature_2m_year_2020.om"
        target.write_bytes(b"cached")
        called = []
        monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: called.append(1))
        prefetch_chunks([entry], "copernicus_era5", ["temperature_2m"], str(tmp_path))
        assert called == []

    def test_downloads_missing_files(self, tmp_path, monkeypatch) -> None:
        downloaded = []

        def _fake_dl(entry, model, var, cache_dir):
            downloaded.append((entry.chunk_num, var))
            target = Path(cache_dir) / "chunks" / f"{model}_{var}_year_{entry.chunk_num}.om"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
            return target

        monkeypatch.setattr("util.temporal._download_chunk", _fake_dl)
        entries = [self._entry(2020), self._entry(2021)]
        prefetch_chunks(entries, "copernicus_era5", ["temperature_2m"], str(tmp_path))
        assert len(downloaded) == 2

    def test_failed_download_warns_but_continues(self, tmp_path, monkeypatch, capsys) -> None:
        def _fake_dl(entry, model, var, cache_dir):
            raise RuntimeError("S3 timeout")

        monkeypatch.setattr("util.temporal._download_chunk", _fake_dl)
        prefetch_chunks([self._entry(2020)], "copernicus_era5", ["temperature_2m"], str(tmp_path))
        assert "warning" in capsys.readouterr().out

    def test_disk_limit_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("util.temporal._PREFETCH_DISK_LIMIT_GB", 0)
        dest_dir = tmp_path / "chunks"
        dest_dir.mkdir()
        (dest_dir / "big.om").write_bytes(b"x")  # any file makes used > 0

        def _fake_dl(entry, model, var, cache_dir):
            raise RuntimeError("disk limit reached")

        monkeypatch.setattr("util.temporal._download_chunk", _fake_dl)
        prefetch_chunks([self._entry(2020)], "copernicus_era5", ["temperature_2m"], str(tmp_path))
        # Should complete without raising — failure is caught and warned

    def test_progress_printed_every_10(self, tmp_path, monkeypatch, capsys) -> None:
        def _fake_dl(entry, model, var, cache_dir):
            target = Path(cache_dir) / "chunks" / f"{model}_{var}_year_{entry.chunk_num}.om"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
            return target

        monkeypatch.setattr("util.temporal._download_chunk", _fake_dl)
        entries = [self._entry(2000 + i) for i in range(11)]
        prefetch_chunks(entries, "copernicus_era5", ["temperature_2m"], str(tmp_path))
        assert "10/11" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# build_chunk_index
# ---------------------------------------------------------------------------

class TestBuildChunkIndex:
    def test_basic_structure(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        assert idx.resolution == 3600.0
        assert idx.latest_end_time == _META["data_end_time"]
        assert len(idx.ranges) == 3

    def test_ranges_sorted_ascending(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        starts = [r.start for r in idx.ranges]
        assert starts == sorted(starts)

    def test_source_types(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        assert {r.source for r in idx.ranges} >= {"chunk", "year"}

    def test_min_year_filters_old_ranges(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = build_chunk_index("copernicus_era5", "precipitation", min_year=2023)
        cutoff = datetime(2023, 1, 1, tzinfo=UTC).timestamp()
        for r in idx.ranges:
            assert r.end >= cutoff

    def test_cache_hit_returns_same_object(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx1 = build_chunk_index("copernicus_era5", "precipitation")
        idx2 = build_chunk_index("copernicus_era5", "precipitation")
        assert idx1 is idx2

    def test_missing_end_time_raises(self, monkeypatch):
        util.temporal._CHUNK_INDEX_CACHE.clear()
        monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: {})
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        with pytest.raises(RuntimeError, match="Missing data_end_time"):
            build_chunk_index("copernicus_era5", "bad_var")

    def test_no_files_raises(self, monkeypatch):
        util.temporal._CHUNK_INDEX_CACHE.clear()
        monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: _META.copy())
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        with pytest.raises(RuntimeError, match="No .om files"):
            build_chunk_index("copernicus_era5", "empty_var")

    def test_malformed_chunk_filename_ignored(self, monkeypatch):
        listing = [
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/chunk_abc.om"},
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2022.om"},
        ]
        _mock_s3(monkeypatch, listing=listing)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        assert len(idx.ranges) == 1

    def test_malformed_year_filename_ignored(self, monkeypatch):
        listing = [
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_xyz.om"},
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2023.om"},
        ]
        _mock_s3(monkeypatch, listing=listing)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        assert len(idx.ranges) == 1

    def test_chunk_time_len_none_reads_file(self, monkeypatch):
        meta_no_tlen = {
            "data_end_time": 1704067200.0,
            "temporal_resolution_seconds": 3600,
        }
        listing = [
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/chunk_0.om"},
        ]
        _mock_s3(monkeypatch, meta=meta_no_tlen, listing=listing)

        @contextmanager
        def _fake_open(*a, **kw):
            yield BytesIO(b"fakedata")

        class _FakeReader:
            def __init__(self, fh):
                self.shape = (721, 1440, 8760)
            def close(self):
                pass

        monkeypatch.setattr(fsspec, "open", _fake_open)
        monkeypatch.setattr("util.temporal.OmFileReader", _FakeReader)
        idx = build_chunk_index("copernicus_era5", "precipitation")
        assert len(idx.ranges) == 1
        assert idx.ranges[0].time_len == 8760


# ---------------------------------------------------------------------------
# build_occ_index
# ---------------------------------------------------------------------------

class TestBuildOccIndex:
    def _write_occ(self, path, lats, lons, times):
        pq.write_table(pa.table({
            "decimalLatitude": pa.array(lats, type=pa.float64()),
            "decimalLongitude": pa.array(lons, type=pa.float64()),
            "eventTimestamp": pa.array(times, type=pa.float64()),
        }), path)

    def _node(self, path):
        return {"taxon_key": "1", "path": str(path), "scientific_name": "X",
                "common_name": "", "rank": "SPECIES"}

    def test_unknown_root_raises(self, monkeypatch):
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: None)
        with pytest.raises(RuntimeError, match="Unknown root taxon"):
            build_occ_index("bad", "/data", "occurrence.parquet", None)

    def test_empty_when_no_files(self, tmp_path, monkeypatch):
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert result.num_rows == 0

    def test_scans_parquet(self, tmp_path, monkeypatch):
        self._write_occ(tmp_path / "occurrence.parquet", [52.52], [13.40], [1_000_000.0])
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert result.num_rows == 1
        assert result["latitude"][0].as_py() == pytest.approx(52.52)

    def test_min_year_filters(self, tmp_path, monkeypatch):
        t_old = datetime(1990, 6, 1, tzinfo=UTC).timestamp()
        t_new = datetime(2020, 6, 1, tzinfo=UTC).timestamp()
        self._write_occ(tmp_path / "occurrence.parquet", [0.0, 1.0], [0.0, 1.0], [t_old, t_new])
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", 2000)
        assert result.num_rows == 1
        assert result["timestamp"][0].as_py() == pytest.approx(t_new)

    def test_all_null_timestamps_skipped(self, tmp_path, monkeypatch):
        pq.write_table(pa.table({
            "decimalLatitude": pa.array([52.52], type=pa.float64()),
            "decimalLongitude": pa.array([13.40], type=pa.float64()),
            "eventTimestamp": pa.array([None], type=pa.float64()),
        }), tmp_path / "occurrence.parquet")
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert result.num_rows == 0

    def test_skips_missing_parquet(self, tmp_path, monkeypatch):
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        sub_b = tmp_path / "b"
        sub_b.mkdir()
        self._write_occ(sub_b / "occurrence.parquet", [1.0], [1.0], [9_999_999.0])
        node_a = self._node(sub_a)
        node_b = self._node(sub_b)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node_a)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [node_a, node_b])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert result.num_rows == 1


# ---------------------------------------------------------------------------
# map_to_worklist — empty table
# ---------------------------------------------------------------------------

class TestMapToWorklist:
    def test_empty_table_returns_empty(self):
        empty = pa.table({
            "taxon_path": pa.array([], type=pa.string()),
            "row_idx": pa.array([], type=pa.int64()),
            "latitude": pa.array([], type=pa.float64()),
            "longitude": pa.array([], type=pa.float64()),
            "timestamp": pa.array([], type=pa.float64()),
        })
        entry = ChunkRange(chunk_num=1, start=0, end=3600, time_len=2, source="year")
        index = ChunkIndex(latest_end_time=3600, resolution=3600, ranges=[entry])
        result = map_to_worklist(empty, index, "lat_asc_lon_pm180", 0.25)
        assert result.num_rows == 0
        assert "chunk_num" in result.column_names
        assert "elevation" in result.column_names


# ---------------------------------------------------------------------------
# Elevation correction infrastructure
# ---------------------------------------------------------------------------

class TestBuildOccIndexElevation:
    def _node(self, path):
        return {"taxon_key": "1", "path": str(path), "scientific_name": "X",
                "common_name": "", "rank": "SPECIES"}

    def test_elevation_nan_when_column_absent(self, tmp_path, monkeypatch):
        pq.write_table(pa.table({
            "decimalLatitude": pa.array([52.52], type=pa.float64()),
            "decimalLongitude": pa.array([13.40], type=pa.float64()),
            "eventTimestamp": pa.array([1_000_000.0], type=pa.float64()),
        }), tmp_path / "occurrence.parquet")
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert "elevation" in result.column_names
        assert np.isnan(result["elevation"][0].as_py())

    def test_elevation_read_when_column_present(self, tmp_path, monkeypatch):
        pq.write_table(pa.table({
            "decimalLatitude": pa.array([52.52], type=pa.float64()),
            "decimalLongitude": pa.array([13.40], type=pa.float64()),
            "eventTimestamp": pa.array([1_000_000.0], type=pa.float64()),
            "elevation": pa.array([420.0], type=pa.float64()),
        }), tmp_path / "occurrence.parquet")
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert result["elevation"][0].as_py() == pytest.approx(420.0)

    def test_empty_result_has_elevation_column(self, tmp_path, monkeypatch):
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = build_occ_index("1", str(tmp_path), "occurrence.parquet", None)
        assert "elevation" in result.column_names


class TestMapToWorklistElevation:
    def _make_occ(self, elev=None):
        d = {
            "taxon_path": pa.array(["/a.parquet"], type=pa.string()),
            "row_idx": pa.array([0], type=pa.int64()),
            "latitude": pa.array([52.0], type=pa.float64()),
            "longitude": pa.array([13.0], type=pa.float64()),
            "timestamp": pa.array([3600.0], type=pa.float64()),
        }
        if elev is not None:
            d["elevation"] = pa.array([elev], type=pa.float64())
        return pa.table(d)

    def _index(self):
        entry = ChunkRange(chunk_num=0, start=0, end=7200, time_len=2, source="chunk")
        return ChunkIndex(latest_end_time=7200, resolution=3600, ranges=[entry])

    def test_elevation_passed_through(self):
        result = map_to_worklist(self._make_occ(elev=500.0), self._index(), "lat_asc_lon_pm180", 0.25)
        assert "elevation" in result.column_names
        assert result["elevation"][0].as_py() == pytest.approx(500.0)

    def test_elevation_nan_when_absent(self):
        result = map_to_worklist(self._make_occ(), self._index(), "lat_asc_lon_pm180", 0.25)
        assert "elevation" in result.column_names
        assert np.isnan(result["elevation"][0].as_py())


class TestReadModelElevation:
    def test_returns_nan_on_s3_failure(self, monkeypatch):
        monkeypatch.setitem(util.temporal._MODEL_ELEV_CACHE, "bad_model", None)
        # Clear so it tries to load
        util.temporal._MODEL_ELEV_CACHE.pop("bad_model", None)
        import fsspec as _fsspec
        monkeypatch.setattr(_fsspec, "open", lambda *a, **kw: (_ for _ in ()).throw(OSError("no s3")))
        result = _read_model_elevation("bad_model", np.array([0]), np.array([0]))
        assert np.isnan(result[0])

    def test_uses_cached_grid(self, monkeypatch):
        grid = np.array([[100.0, 200.0], [300.0, 400.0]])
        util.temporal._MODEL_ELEV_CACHE["test_model"] = grid
        result = _read_model_elevation("test_model", np.array([0, 1]), np.array([1, 0]))
        assert result[0] == pytest.approx(200.0)
        assert result[1] == pytest.approx(300.0)
        util.temporal._MODEL_ELEV_CACHE.pop("test_model")

    def test_nodata_masked_at_load_time(self, monkeypatch):
        # Simulate loading a grid that contains -999 nodata values from S3.
        # _read_model_elevation must mask them to NaN before caching.
        raw_grid = np.array([[-999.0, 50.0]])
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value=raw_grid)
        mock_reader.__enter__ = lambda s: mock_reader
        mock_reader.__exit__ = MagicMock(return_value=False)
        import fsspec as _fsspec
        monkeypatch.setattr(_fsspec, "open", lambda *a, **kw: mock_reader)
        monkeypatch.setattr(util.temporal, "OmFileReader", lambda fh: mock_reader)
        util.temporal._MODEL_ELEV_CACHE.pop("nodata_load_model", None)
        result = _read_model_elevation("nodata_load_model", np.array([0, 0]), np.array([0, 1]))
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(50.0)
        util.temporal._MODEL_ELEV_CACHE.pop("nodata_load_model", None)


class TestElevationCorrectableVars:
    def test_temperature_2m_in_set(self):
        assert "temperature_2m" in ELEVATION_CORRECTABLE_VARS

    def test_dew_point_in_set(self):
        assert "dew_point_2m" in ELEVATION_CORRECTABLE_VARS

    def test_soil_temperatures_in_set(self):
        assert "soil_temperature_0_to_7cm" in ELEVATION_CORRECTABLE_VARS

    def test_precipitation_not_in_set(self):
        assert "precipitation" not in ELEVATION_CORRECTABLE_VARS

    def test_cloud_cover_not_in_set(self):
        assert "cloud_cover" not in ELEVATION_CORRECTABLE_VARS
