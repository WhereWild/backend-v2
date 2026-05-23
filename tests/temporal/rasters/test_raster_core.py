"""Pure-function unit tests for raster accumulation and incremental update math.

No network access, no fixtures, no S3.  These tests verify the core invariants
that the sliding window must maintain.
"""
from __future__ import annotations

import numpy as np
import pytest

from util.temporal import (
    RASTER_GRIDS,
    RASTER_WC_CODES,
    ChunkIndex,
    ChunkRange,
    accumulate_raster,
    accumulate_raster_mode,
    compute_raster_final,
    load_raster_state,
    reproject_to_grid,
    save_raster_state,
    vpd_kpa,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    def read_array(self, ranges: object) -> np.ndarray:
        ny, nx, _ = self.shape
        _, _, t = ranges  # type: ignore[misc]
        ts = self._s[t]
        return np.broadcast_to(ts[np.newaxis, np.newaxis, :], (ny, nx, len(ts))).copy()


class _FakeRasterReader3D:
    """OmFileReader stand-in that stores a full (ny, nx, time_len) array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self.shape = data.shape

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple) and len(key) == 3:
            r, c, t = key
            return self._data[r, c, t]
        return self._data

    def read_array(self, ranges: object) -> np.ndarray:
        r, c, t = ranges  # type: ignore[misc]
        return self._data[r, c, t]


# ---------------------------------------------------------------------------
# TestAccumulateRaster
# ---------------------------------------------------------------------------

class TestAccumulateRaster:
    def _run(self, series: np.ndarray, start_ts: float, end_ts: float,
             ny: int = 3, nx: int = 3, monkeypatch: pytest.MonkeyPatch | None = None,
             model: str = "copernicus_era5") -> tuple[np.ndarray, int]:
        index, _ = _make_chunk_index(len(series), t0=_T0)
        if monkeypatch is not None:
            monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: _FakeRasterReader(series, ny, nx))
        return accumulate_raster(model, "precipitation", start_ts, end_ts, index)

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
                _ = key
                out = np.zeros((ny, nx, 3))
                out[0, :, :] = row0[:, np.newaxis]
                out[ny - 1, :, :] = row_last[:, np.newaxis]
                return out

            def read_array(self, ranges):
                out = np.zeros((ny, nx, 3))
                out[0, :, :] = row0[:, np.newaxis]
                out[ny - 1, :, :] = row_last[:, np.newaxis]
                return out

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: _RowReader())

        index, _ = _make_chunk_index(3)
        grid, n = accumulate_raster("copernicus_era5_land", "temperature_2m",
                                    _T0, _T0 + 2 * 3600, index)
        # No flip: row 0 stays row 0, row ny-1 stays row ny-1
        assert n == 3
        assert grid[0, 0] == pytest.approx(row0[0] * 3)
        assert grid[ny - 1, 0] == pytest.approx(row_last[0] * 3)

    def test_break_early(self, monkeypatch) -> None:
        # Second chunk starts well after end_ts → break (line 1350), only first chunk opened
        chunk_a = np.array([1.0, 2.0, 3.0])
        t0_a = _T0
        t0_b = _T0 + 10 * 3600  # far past end_ts
        entry_a = ChunkRange(0, t0_a, t0_a + 2 * 3600, 3, "chunk")
        entry_b = ChunkRange(1, t0_b, t0_b + 2 * 3600, 3, "chunk")
        index = ChunkIndex(
            latest_end_time=t0_b + 2 * 3600, resolution=3600.0,
            ranges=[entry_a, entry_b],
        )
        call_count = [0]

        def _fake_open(entry, model, var) -> _FakeRasterReader:
            call_count[0] += 1
            return _FakeRasterReader(chunk_a, 2, 2)

        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)
        grid, n = accumulate_raster(
            "copernicus_era5", "precipitation", t0_a, t0_a + 2 * 3600, index
        )
        assert n == 3
        assert call_count[0] == 1  # break fired; second chunk never opened
        assert grid[0, 0] == pytest.approx(6.0, abs=1e-6)

    def test_continue_degenerate_range(self, monkeypatch) -> None:
        # start_ts > end_ts inside a matching chunk → t1 <= t0 → continue (line 1355)
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid, n = self._run(series, _T0 + 3 * 3600, _T0 + 1 * 3600, monkeypatch=monkeypatch)
        assert n == 0

    def test_flipud_applied(self, monkeypatch) -> None:
        # Monkeypatch copernicus_era5 to flipud=True → row ordering is reversed (line 1379)
        fake_grid = dict(RASTER_GRIDS["copernicus_era5"])
        fake_grid["flipud"] = True
        monkeypatch.setitem(RASTER_GRIDS, "copernicus_era5", fake_grid)

        ny, nx, n = 3, 3, 3
        data = np.zeros((ny, nx, n), dtype=np.float64)
        for row in range(ny):
            data[row, :, :] = float(row + 1)  # row 0 = 1.0, row 1 = 2.0, row 2 = 3.0

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: _FakeRasterReader3D(data))
        index, _ = _make_chunk_index(n)
        grid, steps = accumulate_raster(
            "copernicus_era5", "precipitation", _T0, _T0 + (n - 1) * 3600, index
        )
        assert steps == n
        # flipud: original row 0 (value 1.0) moves to grid[ny-1]
        assert grid[ny - 1, 0] == pytest.approx(1.0 * n, abs=1e-6)
        assert grid[0, 0] == pytest.approx(float(ny) * n, abs=1e-6)

    def test_cross_chunk_boundary(self, monkeypatch) -> None:
        # Two chunks; accumulate spanning both
        chunk_a = np.array([1.0, 2.0, 3.0])
        chunk_b = np.array([4.0, 5.0, 6.0])
        t0_a = _T0
        t0_b = _T0 + 3 * 3600
        entry_a = ChunkRange(0, t0_a, t0_a + 2 * 3600, 3, "chunk")
        entry_b = ChunkRange(1, t0_b, t0_b + 2 * 3600, 3, "chunk")
        index = ChunkIndex(latest_end_time=t0_b + 2 * 3600, resolution=3600.0, ranges=[entry_a, entry_b])

        call_count = [0]

        def _fake_open(entry, model, var):
            r = _FakeRasterReader(chunk_a if call_count[0] == 0 else chunk_b, 2, 2)
            call_count[0] += 1
            return r

        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        grid, n = accumulate_raster("copernicus_era5", "precipitation",
                                    t0_a, t0_b + 2 * 3600, index)
        assert n == 6
        assert grid[0, 0] == pytest.approx(21.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestAccumulateRasterMode
# ---------------------------------------------------------------------------

class TestAccumulateRasterMode:
    def _make_index(self, n: int, t0: float = _T0) -> ChunkIndex:
        t_end = t0 + (n - 1) * 3600.0
        entry = ChunkRange(chunk_num=0, start=t0, end=t_end, time_len=n, source="chunk")
        return ChunkIndex(latest_end_time=t_end, resolution=3600.0, ranges=[entry])

    def _make_fixtures(
        self, n: int, ny: int, nx: int, data_by_var: dict | None = None,
    ) -> tuple[dict, object]:
        if data_by_var is None:
            data_by_var = {
                "cloud_cover": np.full((ny, nx, n), 100.0),
                "precipitation": np.zeros((ny, nx, n)),
                "snowfall_water_equivalent": np.zeros((ny, nx, n)),
            }

        def _fake_open(entry, model, var):
            return _FakeRasterReader3D(data_by_var[var])

        return data_by_var, _fake_open

    def test_basic_code_counts(self, monkeypatch) -> None:
        n, ny, nx = 2, 2, 2
        _, _fake_open = self._make_fixtures(n, ny, nx)
        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        cloud_idx = self._make_index(n)
        precip_idx = self._make_index(n)
        swe_idx = self._make_index(n)

        result = accumulate_raster_mode(
            "copernicus_era5",
            _T0, _T0 + (n - 1) * 3600.0,
            cloud_idx, precip_idx, swe_idx,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        # Each cell must be assigned to exactly one code per step
        total = sum(result[c][0, 0] for c in RASTER_WC_CODES)
        assert total == n

    def test_with_temp_grid(self, monkeypatch) -> None:
        n, ny, nx = 1, 2, 2
        data_by_var = {
            "cloud_cover": np.zeros((ny, nx, n)),
            "precipitation": np.full((ny, nx, n), 2.0),
            "snowfall_water_equivalent": np.full((ny, nx, n), 0.5),
        }
        _, _fake_open = self._make_fixtures(n, ny, nx, data_by_var)
        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        cloud_idx = self._make_index(n)
        precip_idx = self._make_index(n)
        swe_idx = self._make_index(n)
        temp_grid = np.full((ny, nx), -5.0)  # below freezing

        result = accumulate_raster_mode(
            "copernicus_era5",
            _T0, _T0,
            cloud_idx, precip_idx, swe_idx,
            temp_grid_025=temp_grid,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        total = sum(result[c][0, 0] for c in RASTER_WC_CODES)
        assert total == n

    def test_break_at_second_chunk(self, monkeypatch) -> None:
        # Second cloud chunk starts after end_ts → break (line 1431 in accumulate_raster_mode)
        n, ny, nx = 2, 2, 2
        _, _fake_open = self._make_fixtures(n, ny, nx)
        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        t0_b = _T0 + 10 * 3600  # second chunk starts far after end_ts
        entry_a = ChunkRange(0, _T0, _T0 + (n - 1) * 3600.0, n, "chunk")
        entry_b = ChunkRange(1, t0_b, t0_b + (n - 1) * 3600.0, n, "chunk")
        cloud_idx = ChunkIndex(
            latest_end_time=t0_b + (n - 1) * 3600.0, resolution=3600.0,
            ranges=[entry_a, entry_b],
        )
        precip_idx = self._make_index(n)
        swe_idx = self._make_index(n)

        result = accumulate_raster_mode(
            "copernicus_era5",
            _T0, _T0 + (n - 1) * 3600.0,
            cloud_idx, precip_idx, swe_idx,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        total = sum(result[c][0, 0] for c in RASTER_WC_CODES)
        assert total == n  # only first chunk processed

    def test_no_matching_chunks_returns_zeros(self, monkeypatch) -> None:
        # Request range entirely past the data → continue fires (line 1429), result is zeros
        n = 2

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: None)

        cloud_idx = self._make_index(n, t0=_T0)
        precip_idx = self._make_index(n, t0=_T0)
        swe_idx = self._make_index(n, t0=_T0)

        far_future = _T0 + 1_000_000 * 3600.0
        result = accumulate_raster_mode(
            "copernicus_era5",
            far_future, far_future + 3600.0,
            cloud_idx, precip_idx, swe_idx,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        for arr in result.values():
            assert arr.sum() == 0

    def test_missing_precip_entry(self, monkeypatch) -> None:
        # precip/swe index has no range covering cloud timestamps →
        # _cidx_entry_for returns None,-1 (line 1425); pr/sw data stays zero
        n, ny, nx = 1, 2, 2
        data_by_var = {"cloud_cover": np.full((ny, nx, n), 50.0)}
        _, _fake_open = self._make_fixtures(n, ny, nx, data_by_var)
        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        cloud_idx = self._make_index(n, t0=_T0)
        far_future = _T0 + 1_000_000 * 3600.0
        precip_idx = self._make_index(n, t0=far_future)  # doesn't cover _T0
        swe_idx = self._make_index(n, t0=far_future)

        result = accumulate_raster_mode(
            "copernicus_era5", _T0, _T0,
            cloud_idx, precip_idx, swe_idx,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        total = sum(result[c][0, 0] for c in RASTER_WC_CODES)
        assert total == n

    def test_degenerate_range_skipped(self, monkeypatch) -> None:
        # Inverted start_ts > end_ts — chunk must span both values so the outer
        # guards pass, but t1 <= t0 fires (line 1437).  n=11 gives end=_T0+10h.
        n, ny, nx = 11, 2, 2
        _, _fake_open = self._make_fixtures(n, ny, nx)
        monkeypatch.setattr("util.temporal._open_chunk", _fake_open)

        cloud_idx = self._make_index(n, t0=_T0)   # covers _T0 .. _T0+10h
        precip_idx = self._make_index(n, t0=_T0)
        swe_idx = self._make_index(n, t0=_T0)

        # start_ts > end_ts but both within the chunk → t1 (2) < t0 (3) → continue
        result = accumulate_raster_mode(
            "copernicus_era5",
            _T0 + 3 * 3600, _T0 + 1 * 3600,
            cloud_idx, precip_idx, swe_idx,
        )

        assert set(result.keys()) == set(RASTER_WC_CODES)
        for arr in result.values():
            assert arr.sum() == 0


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
        t = np.full((2, 2), 10.0)
        td = np.full((2, 2), 15.0)  # td > t → would be negative VPD
        n = 10
        result = compute_raster_final("vapor_pressure_deficit", "avg",
                                      {"era5_temperature_2m": t * n, "era5_dew_point_2m": td * n},
                                      n, 0)
        assert (result >= 0).all()

    def test_vpd_formula(self) -> None:
        t_val, td_val = 20.0, 10.0
        n = 24
        t = np.full((2, 2), t_val * n)
        td = np.full((2, 2), td_val * n)
        result = compute_raster_final("vapor_pressure_deficit", "avg",
                                      {"era5_temperature_2m": t, "era5_dew_point_2m": td},
                                      n, 0)
        expected = float(vpd_kpa(t_val, td_val))
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_era5_only_avg(self) -> None:
        era5 = np.full((2, 2), 48.0)
        result = compute_raster_final("cloud_cover", "avg",
                                      {"era5_cloud_cover": era5}, 24, 0)
        assert result[0, 0] == pytest.approx(2.0, abs=1e-5)

    def test_vpd_with_gfs(self) -> None:
        n_era5, n_gfs = 24, 12
        t_e, td_e = 20.0, 15.0
        t_g, td_g = 25.0, 10.0
        sums = {
            "era5_temperature_2m": np.full((2, 2), t_e * n_era5),
            "era5_dew_point_2m":   np.full((2, 2), td_e * n_era5),
            "gfs_temperature_2m":  np.full((2, 2), t_g * n_gfs),
            "gfs_dew_point_2m":    np.full((2, 2), td_g * n_gfs),
        }
        result = compute_raster_final("vapor_pressure_deficit", "avg", sums, n_era5, n_gfs)
        expected = max(
            (float(vpd_kpa(t_e, td_e)) * n_era5 + float(vpd_kpa(t_g, td_g)) * n_gfs)
            / (n_era5 + n_gfs),
            0.0,
        )
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_dew_point_era5_only(self) -> None:
        n_era5 = 24
        td_val = 8.0
        sums = {"era5_dew_point_2m": np.full((2, 2), td_val * n_era5)}
        result = compute_raster_final("dew_point_2m", "avg", sums, n_era5, 0)
        assert result[0, 0] == pytest.approx(td_val, abs=1e-4)

    def test_dew_point_combined(self) -> None:
        n_era5, n_gfs = 24, 12
        td_e, td_g = 8.0, 12.0
        sums = {
            "era5_dew_point_2m": np.full((2, 2), td_e * n_era5),
            "gfs_dew_point_2m":  np.full((2, 2), td_g * n_gfs),
        }
        result = compute_raster_final("dew_point_2m", "avg", sums, n_era5, n_gfs)
        expected = (td_e * n_era5 + td_g * n_gfs) / (n_era5 + n_gfs)
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# TestDropAddInvariant  (the key sliding-window correctness proof)
# ---------------------------------------------------------------------------

class TestDropAddInvariant:
    """full_rebuild(t0,t1) == drop(t0, t_mid) + add_back(t0, t_mid) applied to full(t0,t1)."""

    def _accum(self, series: np.ndarray, start_ts: float, end_ts: float,
               monkeypatch) -> tuple[np.ndarray, int]:
        index, _ = _make_chunk_index(len(series))
        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: _FakeRasterReader(series, 2, 2))
        return accumulate_raster("copernicus_era5", "precipitation", start_ts, end_ts, index)

    def test_subtract_add_back_is_identity(self, monkeypatch) -> None:
        rng = np.random.default_rng(42)
        series = rng.uniform(0, 10, size=200)
        t1 = _T0 + 199 * 3600

        full_sum, full_n = self._accum(series, _T0, t1, monkeypatch)

        # Drop first 50 steps
        drop_sum, drop_n = self._accum(series, _T0, _T0 + 49 * 3600, monkeypatch)
        remaining = full_sum - drop_sum
        remaining_n = full_n - drop_n

        # Add them back
        restored = remaining + drop_sum
        assert restored[0, 0] == pytest.approx(full_sum[0, 0], abs=1e-4)
        assert full_n == drop_n + remaining_n

    def test_full_rebuild_equals_incremental(self, monkeypatch) -> None:
        rng = np.random.default_rng(99)
        series = rng.uniform(0, 5, size=500)

        # Full rebuild: hours 100..499
        t_start = _T0 + 100 * 3600
        t_end = _T0 + 499 * 3600
        full_sum, full_n = self._accum(series, t_start, t_end, monkeypatch)

        # Simulated "state at hour 300": hours 100..299
        t_mid = _T0 + 299 * 3600
        state_sum, state_n = self._accum(series, t_start, t_mid, monkeypatch)

        # Incremental: add hours 300..499
        delta_sum, delta_n = self._accum(series, _T0 + 300 * 3600, t_end, monkeypatch)

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
