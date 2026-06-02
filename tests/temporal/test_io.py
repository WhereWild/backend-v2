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

import util.temporal

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
        layers = util.temporal.load_temporal_layers(_CATALOG)
        assert len(layers) > 0

    def test_temperature_present(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        ids = {la.id for la in layers}
        assert "temperature_2m" in ids

    def test_vpd_has_sources(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        vpd = next(la for la in layers if la.id == "vapor_pressure_deficit")
        assert vpd.derived is False
        assert "temperature_2m" in vpd.sources
        assert "dew_point_2m" in vpd.sources

    def test_layer_window_override(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        snow = next(la for la in layers if la.id == "snow_depth")
        # snow_depth inherits category windows (no per-layer override)
        assert len(snow.windows) > 1

    def test_weather_code_sources(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        wc = next(la for la in layers if la.id == "weather_code_simple")
        assert wc.agg == "mode"
        assert "cloud_cover" in wc.sources
        assert "precipitation" in wc.sources

    def test_category_windows_inherited(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        temp = next(la for la in layers if la.id == "temperature_2m")
        assert 24 in temp.windows

    def test_model_and_grid_mode(self):
        layers = util.temporal.load_temporal_layers(_CATALOG)
        temp = next(la for la in layers if la.id == "temperature_2m")
        assert temp.model == "copernicus_era5"
        assert temp.grid_mode == "lat_asc_lon_pm180"


# ---------------------------------------------------------------------------
# _grid_indices_batch — modes not covered by integration tests
# ---------------------------------------------------------------------------

class TestGridIndicesBatch:
    _NY, _NX, _STEP = 721, 1440, 0.25

    def _call(self, lats, lons, mode):
        return util.temporal._grid_indices_batch(
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
        assert util.temporal._parse_s3_time(1_234_567_890) == pytest.approx(1_234_567_890.0)

    def test_float(self):
        assert util.temporal._parse_s3_time(1_234_567_890.5) == pytest.approx(1_234_567_890.5)

    def test_string_numeric(self):
        assert util.temporal._parse_s3_time("1234567890") == pytest.approx(1_234_567_890.0)

    def test_string_iso_z(self):
        result = util.temporal._parse_s3_time("2023-01-01T00:00:00Z")
        expected = datetime(2023, 1, 1, tzinfo=UTC).timestamp()
        assert result == pytest.approx(expected)

    def test_string_invalid(self):
        assert util.temporal._parse_s3_time("not-a-time") is None

    def test_none_type(self):
        assert util.temporal._parse_s3_time(None) is None

    def test_list_type(self):
        assert util.temporal._parse_s3_time([]) is None


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
        assert util.temporal._open_s3_json("s3://fake/meta.json") == data

    def test_exception_returns_none(self, monkeypatch):
        def _raise(*a, **kw):
            raise OSError("no network")

        monkeypatch.setattr(fsspec, "open", _raise)
        assert util.temporal._open_s3_json("s3://fake/meta.json") is None


# ---------------------------------------------------------------------------
# _chunk_filename / _open_chunk / _download_chunk
# ---------------------------------------------------------------------------

class TestChunkHelpers:
    def test_chunk_filename_year(self):
        entry = util.temporal.ChunkRange(chunk_num=2022, start=0, end=0, time_len=1, source="year")
        assert util.temporal._chunk_filename(entry) == "year_2022.om"

    def test_chunk_filename_chunk(self):
        entry = util.temporal.ChunkRange(chunk_num=5, start=0, end=0, time_len=1, source="chunk")
        assert util.temporal._chunk_filename(entry) == "chunk_5.om"

    def test_open_chunk_builds_correct_uri(self, monkeypatch):
        captured = {}
        mock_fs = MagicMock()

        def _fake_from_fsspec(fs, path):
            captured["path"] = path
            return MagicMock()

        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        monkeypatch.setattr("util.temporal.OmFileReader.from_fsspec", staticmethod(_fake_from_fsspec))
        monkeypatch.setattr("util.temporal._RASTER_CHUNK_CACHE_DIR", None)
        entry = util.temporal.ChunkRange(chunk_num=2023, start=0, end=0, time_len=1, source="year")
        util.temporal._open_chunk(entry, "copernicus_era5", "precipitation")
        assert captured["path"] == "openmeteo/data/copernicus_era5/precipitation/year_2023.om"

    def test_open_chunk_uses_cache_dir(self, monkeypatch, tmp_path):
        fake_local = tmp_path / "copernicus_era5_precipitation_year_2023.om"
        fake_local.write_bytes(b"")
        monkeypatch.setattr("util.temporal._RASTER_CHUNK_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr("util.temporal._download_chunk", lambda e, m, v, d: fake_local)
        monkeypatch.setattr("util.temporal.OmFileReader", lambda path: path)
        entry = util.temporal.ChunkRange(chunk_num=2023, start=0, end=0, time_len=1, source="year")
        result = util.temporal._open_chunk(entry, "copernicus_era5", "precipitation")
        assert result == str(fake_local)

    def test_download_chunk_returns_cached(self, tmp_path):
        entry = util.temporal.ChunkRange(chunk_num=2019, start=0, end=0, time_len=1, source="chunk")
        dest_dir = tmp_path / "chunks"
        dest_dir.mkdir()
        dest = dest_dir / "copernicus_era5_precipitation_chunk_2019.om"
        dest.write_bytes(b"cached")
        result = util.temporal._download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        assert result == dest

    def test_download_chunk_fetches_missing(self, tmp_path, monkeypatch):
        entry = util.temporal.ChunkRange(chunk_num=2022, start=0, end=0, time_len=1, source="year")
        fetched = {}

        def _fake_get(uri, dest):
            fetched["uri"] = uri
            Path(dest).write_bytes(b"omdata")

        mock_fs = MagicMock()
        mock_fs.get = _fake_get
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        result = util.temporal._download_chunk(entry, "copernicus_era5", "precipitation", str(tmp_path))
        assert result.name == "copernicus_era5_precipitation_year_2022.om"
        assert result.read_bytes() == b"omdata"
        assert "year_2022.om" in fetched["uri"]

    def test_download_layer_chunk_downloads_each_var(self, monkeypatch):
        entry = util.temporal.ChunkRange(chunk_num=2020, start=0, end=0, time_len=1, source="year")
        called = []
        monkeypatch.setattr("util.temporal._download_chunk", lambda e, model, var, cache_dir: called.append(var))
        result = util.temporal._download_layer_chunk(entry, "copernicus_era5", ["cloud_cover", "precipitation"], "/tmp")
        assert called == ["cloud_cover", "precipitation"]
        assert result is entry


# ---------------------------------------------------------------------------
# build_chunk_index
# ---------------------------------------------------------------------------

class TestBuildChunkIndex:
    def test_basic_structure(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        assert idx.resolution == 3600.0
        assert idx.latest_end_time == _META["data_end_time"]
        assert len(idx.ranges) == 3

    def test_ranges_sorted_ascending(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        starts = [r.start for r in idx.ranges]
        assert starts == sorted(starts)

    def test_source_types(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        assert {r.source for r in idx.ranges} >= {"chunk", "year"}

    def test_min_year_filters_old_ranges(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation", min_year=2023)
        cutoff = datetime(2023, 1, 1, tzinfo=UTC).timestamp()
        for r in idx.ranges:
            assert r.end >= cutoff

    def test_cache_hit_returns_same_object(self, monkeypatch):
        _mock_s3(monkeypatch)
        idx1 = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        idx2 = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        assert idx1 is idx2

    def test_missing_end_time_raises(self, monkeypatch):
        util.temporal._CHUNK_INDEX_CACHE.clear()
        monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: {})
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        with pytest.raises(RuntimeError, match="Missing data_end_time"):
            util.temporal.build_chunk_index("copernicus_era5", "bad_var")

    def test_no_files_raises(self, monkeypatch):
        util.temporal._CHUNK_INDEX_CACHE.clear()
        monkeypatch.setattr("util.temporal._open_s3_json", lambda uri: _META.copy())
        mock_fs = MagicMock()
        mock_fs.ls.return_value = []
        monkeypatch.setattr(fsspec, "filesystem", lambda *a, **kw: mock_fs)
        with pytest.raises(RuntimeError, match="No .om files"):
            util.temporal.build_chunk_index("copernicus_era5", "empty_var")

    def test_malformed_chunk_filename_ignored(self, monkeypatch):
        listing = [
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/chunk_abc.om"},
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2022.om"},
        ]
        _mock_s3(monkeypatch, listing=listing)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
        assert len(idx.ranges) == 1

    def test_malformed_year_filename_ignored(self, monkeypatch):
        listing = [
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_xyz.om"},
            {"name": "s3://openmeteo/data/copernicus_era5/precipitation/year_2023.om"},
        ]
        _mock_s3(monkeypatch, listing=listing)
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
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
        idx = util.temporal.build_chunk_index("copernicus_era5", "precipitation")
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

    def _build(self, tmp_path, root_id, data_root, occ_filename, min_year=None, **kw):
        idx = tmp_path / "occ_index.parquet"
        util.temporal.build_occ_index(root_id, data_root, occ_filename, idx, min_year=min_year, **kw)
        return pq.read_table(idx)

    def test_unknown_root_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: None)
        with pytest.raises(RuntimeError, match="Unknown root taxon"):
            util.temporal.build_occ_index("bad", "/data", "occurrence.parquet", tmp_path / "idx.parquet")

    def test_empty_when_no_files(self, tmp_path, monkeypatch):
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = self._build(tmp_path, "1", str(tmp_path), "occurrence.parquet")
        assert result.num_rows == 0

    def test_scans_parquet(self, tmp_path, monkeypatch):
        self._write_occ(tmp_path / "occurrence.parquet", [52.52], [13.40], [1_000_000.0])
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = self._build(tmp_path, "1", str(tmp_path), "occurrence.parquet")
        assert result.num_rows == 1
        assert result["latitude"][0].as_py() == pytest.approx(52.52)

    def test_min_year_filters(self, tmp_path, monkeypatch):
        t_old = datetime(1990, 6, 1, tzinfo=UTC).timestamp()
        t_new = datetime(2020, 6, 1, tzinfo=UTC).timestamp()
        self._write_occ(tmp_path / "occurrence.parquet", [0.0, 1.0], [0.0, 1.0], [t_old, t_new])
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = self._build(tmp_path, "1", str(tmp_path), "occurrence.parquet", min_year=2000)
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
        result = self._build(tmp_path, "1", str(tmp_path), "occurrence.parquet")
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
        result = self._build(tmp_path, "1", str(tmp_path), "occurrence.parquet")
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
        entry = util.temporal.ChunkRange(chunk_num=1, start=0, end=3600, time_len=2, source="year")
        index = util.temporal.ChunkIndex(latest_end_time=3600, resolution=3600, ranges=[entry])
        result = util.temporal.map_to_worklist(empty, index, "lat_asc_lon_pm180", 0.25)
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

    def _build(self, tmp_path, data_root, **kw):
        idx = tmp_path / "occ_index.parquet"
        util.temporal.build_occ_index("1", data_root, "occurrence.parquet", idx, **kw)
        return pq.read_table(idx)

    def test_elevation_nan_when_column_absent(self, tmp_path, monkeypatch):
        pq.write_table(pa.table({
            "decimalLatitude": pa.array([52.52], type=pa.float64()),
            "decimalLongitude": pa.array([13.40], type=pa.float64()),
            "eventTimestamp": pa.array([1_000_000.0], type=pa.float64()),
        }), tmp_path / "occurrence.parquet")
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = self._build(tmp_path, str(tmp_path))
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
        result = self._build(tmp_path, str(tmp_path))
        assert result["elevation"][0].as_py() == pytest.approx(420.0)

    def test_empty_result_has_elevation_column(self, tmp_path, monkeypatch):
        node = self._node(tmp_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])
        result = self._build(tmp_path, str(tmp_path))
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
        entry = util.temporal.ChunkRange(chunk_num=0, start=0, end=7200, time_len=2, source="chunk")
        return util.temporal.ChunkIndex(latest_end_time=7200, resolution=3600, ranges=[entry])

    def test_elevation_passed_through(self):
        result = util.temporal.map_to_worklist(self._make_occ(elev=500.0), self._index(), "lat_asc_lon_pm180", 0.25)
        assert "elevation" in result.column_names
        assert result["elevation"][0].as_py() == pytest.approx(500.0)

    def test_elevation_nan_when_absent(self):
        result = util.temporal.map_to_worklist(self._make_occ(), self._index(), "lat_asc_lon_pm180", 0.25)
        assert "elevation" in result.column_names
        assert np.isnan(result["elevation"][0].as_py())


class TestReadModelElevation:
    def test_returns_nan_on_s3_failure(self, monkeypatch):
        monkeypatch.setitem(util.temporal._MODEL_ELEV_CACHE, "bad_model", None)
        # Clear so it tries to load
        util.temporal._MODEL_ELEV_CACHE.pop("bad_model", None)
        import fsspec as _fsspec
        monkeypatch.setattr(_fsspec, "open", lambda *a, **kw: (_ for _ in ()).throw(OSError("no s3")))
        result = util.temporal._read_model_elevation("bad_model", np.array([0]), np.array([0]))
        assert np.isnan(result[0])

    def test_uses_cached_grid(self, monkeypatch):
        grid = np.array([[100.0, 200.0], [300.0, 400.0]])
        util.temporal._MODEL_ELEV_CACHE["test_model"] = grid
        result = util.temporal._read_model_elevation("test_model", np.array([0, 1]), np.array([1, 0]))
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
        result = util.temporal._read_model_elevation("nodata_load_model", np.array([0, 0]), np.array([0, 1]))
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(50.0)
        util.temporal._MODEL_ELEV_CACHE.pop("nodata_load_model", None)


class TestElevationCorrectableVars:
    def test_temperature_2m_in_set(self):
        assert "temperature_2m" in util.temporal.ELEVATION_CORRECTABLE_VARS

    def test_dew_point_in_set(self):
        assert "dew_point_2m" in util.temporal.ELEVATION_CORRECTABLE_VARS

    def test_soil_temperatures_in_set(self):
        assert "soil_temperature_0_to_7cm" in util.temporal.ELEVATION_CORRECTABLE_VARS

    def test_precipitation_not_in_set(self):
        assert "precipitation" not in util.temporal.ELEVATION_CORRECTABLE_VARS

    def test_cloud_cover_not_in_set(self):
        assert "cloud_cover" not in util.temporal.ELEVATION_CORRECTABLE_VARS
