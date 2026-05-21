"""Tests for util/temporal.py I/O functions: S3 helpers, chunk index, occ index."""
from __future__ import annotations

import json
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

import util.temporal as tm
from util.temporal import (
    ChunkIndex,
    ChunkRange,
    _download_chunk,
    _grid_indices_batch,
    _open_s3_json,
    _parse_s3_time,
    build_chunk_index,
    build_occ_index,
    load_temporal_layers,
    map_to_worklist,
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
    tm._CHUNK_INDEX_CACHE.clear()
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
        assert "weather_code_simple" in derived

    def test_layer_window_override(self):
        layers = load_temporal_layers(_CATALOG)
        snow = next(la for la in layers if la.id == "snow_depth")
        assert snow.windows == [1]

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

        @contextmanager
        def _fake_open(*a, **kw):
            yield BytesIO(b"omdata")

        monkeypatch.setattr(fsspec, "open", _fake_open)
        result = _download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        assert result.name == "copernicus_era5_precipitation_year_2022.om"
        assert result.read_bytes() == b"omdata"

    def test_failure_removes_tmp_file(self, tmp_path, monkeypatch):
        entry = ChunkRange(chunk_num=0, start=0, end=0, time_len=1, source="chunk")

        def _raise(*a, **kw):
            raise OSError("S3 down")

        monkeypatch.setattr(fsspec, "open", _raise)
        with pytest.raises(OSError, match="S3 down"):
            _download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        chunks_dir = tmp_path / "chunks"
        tmp_files = list(chunks_dir.glob("*.tmp")) if chunks_dir.exists() else []
        assert tmp_files == []


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
        tm._CHUNK_INDEX_CACHE.clear()
        monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: {})
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        with pytest.raises(RuntimeError, match="Missing data_end_time"):
            build_chunk_index("copernicus_era5", "bad_var")

    def test_no_files_raises(self, monkeypatch):
        tm._CHUNK_INDEX_CACHE.clear()
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
