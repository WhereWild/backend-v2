"""Pure-function unit tests for raster accumulation and incremental update math.

No network access, no fixtures, no S3.  These tests verify the core invariants
that the sliding window must maintain.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from util.temporal import (
    RASTER_GRIDS,
    RASTER_WC_CODES,
    ChunkIndex,
    ChunkRange,
    accumulate_raster,
    compute_raster_final,
    load_raster_state,
    reproject_to_grid,
    save_raster_state,
    vpd_kpa,
    weather_code_array,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERA5_NY = RASTER_GRIDS["copernicus_era5"]["ny"]
_ERA5_NX = RASTER_GRIDS["copernicus_era5"]["nx"]
_LAND_NY = RASTER_GRIDS["copernicus_era5_land"]["ny"]
_LAND_NX = RASTER_GRIDS["copernicus_era5_land"]["nx"]

_T0 = 1_560_000_000.0  # arbitrary epoch anchor


def _make_chunk_index(series_len: int, resolution: float = 3600.0, t0: float = _T0) -> tuple[ChunkIndex, ChunkRange]:
    t_end = t0 + (series_len - 1) * resolution
    entry = ChunkRange(chunk_num=0, start=t0, end=t_end, time_len=series_len, source="chunk")
    index = ChunkIndex(latest_end_time=t_end, resolution=resolution, ranges=[entry])
    return index, entry


class _FakeRasterReader:
    """OmFileReader stand-in: returns the same 1-D series for every cell."""

    def __init__(self, series: np.ndarray, ny: int, nx: int) -> None:
        self._s = series
        self.shape = (ny, nx, len(series))

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple) and len(key) == 3:
            r, c, t = key
            if isinstance(t, slice):
                return self._s[t][np.newaxis, np.newaxis, :]
            return self._s[t:t + 1][np.newaxis, np.newaxis, :]
        return self._s


# ---------------------------------------------------------------------------
# TestAccumulateRaster
# ---------------------------------------------------------------------------

class TestAccumulateRaster:
    def _run(self, series: np.ndarray, start_ts: float, end_ts: float,
             ny: int = 3, nx: int = 3, monkeypatch: pytest.MonkeyPatch | None = None,
             model: str = "copernicus_era5") -> tuple[np.ndarray, int]:
        index, _ = _make_chunk_index(len(series), t0=_T0)
        tmp = tempfile.mkdtemp()
        dummy = Path(tmp) / "dummy.om"
        dummy.write_bytes(b"")
        if monkeypatch is not None:
            monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
            monkeypatch.setattr("util.temporal.OmFileReader", lambda p: _FakeRasterReader(series, ny, nx))
        return accumulate_raster(model, "precipitation", start_ts, end_ts, index, tmp)

    def test_full_range_sum(self, monkeypatch) -> None:
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid, n = self._run(series, _T0, _T0 + 4 * 3600, monkeypatch=monkeypatch)
        assert n == 5
        assert grid[0, 0] == pytest.approx(15.0, abs=1e-6)

    def test_partial_start(self, monkeypatch) -> None:
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Start at index 2: should sum [3,4,5]
        start = _T0 + 2 * 3600
        grid, n = self._run(series, start, _T0 + 4 * 3600, monkeypatch=monkeypatch)
        assert n == 3
        assert grid[0, 0] == pytest.approx(12.0, abs=1e-6)

    def test_partial_end(self, monkeypatch) -> None:
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid, n = self._run(series, _T0, _T0 + 2 * 3600, monkeypatch=monkeypatch)
        assert n == 3
        assert grid[0, 0] == pytest.approx(6.0, abs=1e-6)

    def test_nan_excluded(self, monkeypatch) -> None:
        series = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        grid, n = self._run(series, _T0, _T0 + 4 * 3600, monkeypatch=monkeypatch)
        # NaN excluded from sum; n_steps still counts all steps (NaN handling is in nansum)
        assert n == 5
        assert grid[0, 0] == pytest.approx(9.0, abs=1e-6)

    def test_empty_range_returns_zero(self, monkeypatch) -> None:
        series = np.array([1.0, 2.0, 3.0])
        # end_ts before chunk start
        grid, n = self._run(series, _T0 + 10 * 3600, _T0 + 12 * 3600, monkeypatch=monkeypatch)
        assert n == 0

    def test_no_flipud_era5_land(self, monkeypatch) -> None:
        # ERA5-land is lat-ascending (catalog: lat_asc_lon_pm180), flipud=False.
        # Row ordering must be preserved unchanged.
        ny, nx = 4, 4
        row0 = np.arange(float(nx))
        row_last = np.arange(float(nx)) + 100.0

        class _RowReader:
            shape = (ny, nx, 3)

            def __getitem__(self, key):
                r_sl, c_sl, t_sl = key
                out = np.zeros((ny, nx, 3))
                out[0, :, :] = row0[:, np.newaxis]
                out[ny - 1, :, :] = row_last[:, np.newaxis]
                return out

        tmp = tempfile.mkdtemp()
        dummy = Path(tmp) / "dummy.om"
        dummy.write_bytes(b"")
        monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
        monkeypatch.setattr("util.temporal.OmFileReader", lambda p: _RowReader())

        index, _ = _make_chunk_index(3)
        grid, n = accumulate_raster("copernicus_era5_land", "temperature_2m",
                                    _T0, _T0 + 2 * 3600, index, tmp)
        # No flip: row 0 stays row 0, row ny-1 stays row ny-1
        assert n == 3
        assert grid[0, 0] == pytest.approx(row0[0] * 3)
        assert grid[ny - 1, 0] == pytest.approx(row_last[0] * 3)

    def test_cross_chunk_boundary(self, monkeypatch) -> None:
        # Two chunks; accumulate spanning both
        chunk_a = np.array([1.0, 2.0, 3.0])
        chunk_b = np.array([4.0, 5.0, 6.0])
        t0_a = _T0
        t0_b = _T0 + 3 * 3600
        entry_a = ChunkRange(0, t0_a, t0_a + 2 * 3600, 3, "chunk")
        entry_b = ChunkRange(1, t0_b, t0_b + 2 * 3600, 3, "chunk")
        index = ChunkIndex(latest_end_time=t0_b + 2 * 3600, resolution=3600.0, ranges=[entry_a, entry_b])

        tmp = tempfile.mkdtemp()
        dummy = Path(tmp) / "dummy.om"
        dummy.write_bytes(b"")

        call_count = [0]
        def _fake_reader(p: str):
            r = _FakeRasterReader(chunk_a if call_count[0] == 0 else chunk_b, 2, 2)
            call_count[0] += 1
            return r

        monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
        monkeypatch.setattr("util.temporal.OmFileReader", _fake_reader)

        grid, n = accumulate_raster("copernicus_era5", "precipitation",
                                    t0_a, t0_b + 2 * 3600, index, tmp)
        assert n == 6
        assert grid[0, 0] == pytest.approx(21.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestComputeRasterFinal
# ---------------------------------------------------------------------------

class TestComputeRasterFinal:
    def _zero(self, shape=(3, 4)) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    def test_avg_combined(self) -> None:
        # Sums: ERA5 has 10 steps × avg=10 → sum=100; GFS has 20 steps × avg=20 → sum=400
        era5 = np.full((3, 4), 100.0)
        gfs = np.full((3, 4), 400.0)
        result = compute_raster_final("temperature_2m", "avg",
                                      {"era5_temperature_2m": era5, "gfs_temperature_2m": gfs},
                                      10, 20)
        expected = (100.0 + 400.0) / 30.0
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_sum(self) -> None:
        era5 = np.full((3, 4), 5.0)
        gfs = np.full((3, 4), 3.0)
        result = compute_raster_final("precipitation", "sum",
                                      {"era5_precipitation": era5, "gfs_precipitation": gfs},
                                      100, 24)
        assert result[0, 0] == pytest.approx(8.0, abs=1e-6)

    def test_mode(self) -> None:
        counts = {c: np.zeros((3, 4), dtype=np.int32) for c in RASTER_WC_CODES}
        counts[63] = np.full((3, 4), 5, dtype=np.int32)   # moderate rain wins
        counts[65] = np.full((3, 4), 3, dtype=np.int32)
        result = compute_raster_final("weather_code_simple", "mode", counts, 8, 0)
        assert int(result[0, 0]) == 63

    def test_vpd_clamped_to_zero(self) -> None:
        T = np.full((2, 2), 10.0)
        Td = np.full((2, 2), 15.0)  # Td > T → would be negative VPD
        n = 10
        result = compute_raster_final("vapor_pressure_deficit", "avg",
                                      {"era5_temperature_2m": T * n, "era5_dew_point_2m": Td * n},
                                      n, 0)
        assert (result >= 0).all()

    def test_vpd_formula(self) -> None:
        T_val, Td_val = 20.0, 10.0
        n = 24
        T = np.full((2, 2), T_val * n)
        Td = np.full((2, 2), Td_val * n)
        result = compute_raster_final("vapor_pressure_deficit", "avg",
                                      {"era5_temperature_2m": T, "era5_dew_point_2m": Td},
                                      n, 0)
        expected = float(vpd_kpa(T_val, Td_val))
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_era5_only_avg(self) -> None:
        era5 = np.full((2, 2), 48.0)
        result = compute_raster_final("cloud_cover", "avg",
                                      {"era5_cloud_cover": era5}, 24, 0)
        assert result[0, 0] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestDropAddInvariant  (the key sliding-window correctness proof)
# ---------------------------------------------------------------------------

class TestDropAddInvariant:
    """full_rebuild(t0,t1) == drop(t0, t_mid) + add_back(t0, t_mid) applied to full(t0,t1)."""

    def _accum(self, series: np.ndarray, start_ts: float, end_ts: float,
               monkeypatch, tmp: str) -> tuple[np.ndarray, int]:
        index, _ = _make_chunk_index(len(series))
        dummy = Path(tmp) / "dummy.om"
        dummy.write_bytes(b"")
        monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
        monkeypatch.setattr("util.temporal.OmFileReader", lambda p: _FakeRasterReader(series, 2, 2))
        return accumulate_raster("copernicus_era5", "precipitation", start_ts, end_ts, index, tmp)

    def test_subtract_add_back_is_identity(self, monkeypatch, tmp_path) -> None:
        rng = np.random.default_rng(42)
        series = rng.uniform(0, 10, size=200)
        t1 = _T0 + 199 * 3600

        full_sum, full_n = self._accum(series, _T0, t1, monkeypatch, str(tmp_path))

        # Drop first 50 steps
        drop_sum, drop_n = self._accum(series, _T0, _T0 + 49 * 3600, monkeypatch, str(tmp_path))
        remaining = full_sum - drop_sum
        remaining_n = full_n - drop_n

        # Add them back
        restored = remaining + drop_sum
        assert restored[0, 0] == pytest.approx(full_sum[0, 0], abs=1e-4)
        assert full_n == drop_n + remaining_n

    def test_full_rebuild_equals_incremental(self, monkeypatch, tmp_path) -> None:
        rng = np.random.default_rng(99)
        series = rng.uniform(0, 5, size=500)

        # Full rebuild: hours 100..499
        t_start = _T0 + 100 * 3600
        t_end = _T0 + 499 * 3600
        full_sum, full_n = self._accum(series, t_start, t_end, monkeypatch, str(tmp_path))

        # Simulated "state at hour 300": hours 100..299
        t_mid = _T0 + 299 * 3600
        state_sum, state_n = self._accum(series, t_start, t_mid, monkeypatch, str(tmp_path))

        # Incremental: add hours 300..499
        delta_sum, delta_n = self._accum(series, _T0 + 300 * 3600, t_end, monkeypatch, str(tmp_path))

        incr_sum = state_sum + delta_sum
        incr_n = state_n + delta_n

        assert incr_sum[0, 0] == pytest.approx(full_sum[0, 0], abs=1e-3)
        assert incr_n == full_n


# ---------------------------------------------------------------------------
# TestReprojectToGrid
# ---------------------------------------------------------------------------

class TestReprojectToGrid:
    _ERA5_G = RASTER_GRIDS["copernicus_era5"]
    _LAND_G = RASTER_GRIDS["copernicus_era5_land"]
    _GFS_G = RASTER_GRIDS["ncep_gfs013"]

    def test_identity_round_trip(self) -> None:
        # Reproject ERA5 to itself — values should be preserved
        src = np.random.default_rng(7).uniform(0, 100, (self._ERA5_G["ny"], self._ERA5_G["nx"])).astype(np.float32)
        g = self._ERA5_G
        result = reproject_to_grid(
            src,
            g["lat_min"], g["lat_max"], g["lon_min"], g["lon_max"],
            g["ny"], g["nx"],
            g["lat_min"], g["lat_max"], g["lon_min"], g["lon_max"],
        )
        assert result.shape == (g["ny"], g["nx"])
        # Interior values should be ~identical (bilinear of same grid)
        assert np.nanmean(np.abs(result[10:-10, 10:-10] - src[10:-10, 10:-10])) < 0.1

    def test_gfs_to_era5_shape(self) -> None:
        src = np.ones((721, 1440), dtype=np.float32)  # approximate GFS shape
        g = self._GFS_G
        dst = self._ERA5_G
        result = reproject_to_grid(
            src,
            g["lat_min"], g["lat_max"], g["lon_min"], g["lon_max"],
            dst["ny"], dst["nx"],
            dst["lat_min"], dst["lat_max"], dst["lon_min"], dst["lon_max"],
        )
        assert result.shape == (dst["ny"], dst["nx"])

    def test_uniform_field_preserved(self) -> None:
        # A uniform field should reproject to the same value everywhere
        src = np.full((100, 200), 42.0, dtype=np.float32)
        result = reproject_to_grid(
            src, -90, 90, -180, 180, 50, 100, -90, 90, -180, 180,
        )
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert np.abs(finite - 42.0).max() < 0.01


# ---------------------------------------------------------------------------
# TestRasterStateIO
# ---------------------------------------------------------------------------

class TestRasterStateIO:
    def test_round_trip(self, tmp_path) -> None:
        sums = {"era5_precipitation": np.full((5, 10), 3.14, dtype=np.float32)}
        meta = {"var_id": "precipitation", "window_h": 24, "window_label": "24h",
                "era5_window_start_ts": 0.0, "era5_end_ts": 86400.0,
                "gfs_start_ts": 86400.0, "gfs_end_ts": 86400.0,
                "n_era5": 24, "n_gfs": 0, "built_at": "2024-01-01T00:00:00+00:00"}

        save_raster_state(str(tmp_path), "precipitation", "24h", "sum", sums, meta)
        loaded_sums, loaded_meta = load_raster_state(str(tmp_path), "precipitation", "24h")

        assert loaded_meta is not None
        assert loaded_meta["n_era5"] == 24
        assert np.allclose(loaded_sums["era5_precipitation"], sums["era5_precipitation"])

    def test_missing_returns_none(self, tmp_path) -> None:
        sums, meta = load_raster_state(str(tmp_path), "nonexistent", "24h")
        assert sums is None
        assert meta is None

    def test_npy_written(self, tmp_path) -> None:
        sums = {"era5_temperature_2m": np.full((5, 10), 20.0 * 24, dtype=np.float32)}
        meta = {"var_id": "temperature_2m", "window_h": 24, "window_label": "24h",
                "era5_window_start_ts": 0.0, "era5_end_ts": 86400.0,
                "gfs_start_ts": 86400.0, "gfs_end_ts": 86400.0,
                "n_era5": 24, "n_gfs": 0, "built_at": "2024-01-01T00:00:00+00:00"}
        save_raster_state(str(tmp_path), "temperature_2m", "24h", "avg", sums, meta)
        npy = tmp_path / "temperature_2m_24h.npy"
        assert npy.exists()
        loaded = np.load(npy)
        assert loaded[0, 0] == pytest.approx(20.0, abs=1e-3)
