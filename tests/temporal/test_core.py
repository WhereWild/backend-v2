"""Phase 2 unit tests for util/temporal.py — all RED until Phase 3.

Tests pure functions only: no S3, no file I/O, no fixtures.
Uses synthetic numpy arrays for deterministic, reproducible results.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from util.temporal import (
    _window_mode_batch,
    grid_indices,
    vpd_kpa,
    weather_code_array,
    weather_code_simple,
    window_stats_batch,
    window_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ERA5_STEP = 0.25
ERA5_NY = 721   # 90S to 90N inclusive at 0.25° → 721 rows
ERA5_NX = 1440  # 180W to 180E at 0.25° → 1440 columns


def _ramp(n: int, start: float = 1.0) -> np.ndarray:
    """Return an array [start, start+1, ..., start+n-1] as float32."""
    return np.arange(start, start + n, dtype=np.float32)


def _make_steps(window_hours: tuple[int, ...], resolution: float = 3600.0) -> dict[int, int]:
    return window_steps(resolution, window_hours)


# ---------------------------------------------------------------------------
# grid_indices
# ---------------------------------------------------------------------------

class TestGridIndices:
    """grid_indices(lat, lon, ny, nx, mode, step) → (lat_idx, lon_idx)."""

    # --- lat_asc_lon_pm180 (ERA5 default) ---

    def test_origin(self) -> None:
        li, lo = grid_indices(0.0, 0.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == 360  # (0 + 90) / 0.25
        assert lo == 720  # (0 + 180) / 0.25

    def test_south_pole(self) -> None:
        li, lo = grid_indices(-90.0, 0.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == 0

    def test_north_pole(self) -> None:
        li, lo = grid_indices(90.0, 0.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == ERA5_NY - 1  # 720

    def test_west_edge(self) -> None:
        # lon = -180 → lon_idx = 0
        _, lo = grid_indices(0.0, -180.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert lo == 0

    def test_east_edge(self) -> None:
        # lon = +180 would map to 1440, clamped to 1439
        _, lo = grid_indices(0.0, 180.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert lo == ERA5_NX - 1

    def test_dateline_near_180(self) -> None:
        # Tuvalu-like: lon ≈ +179.2 — should be near the east edge
        _, lo = grid_indices(-8.51, 179.2, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert lo >= ERA5_NX - 5

    def test_near_zero_meridian(self) -> None:
        # London: lon = -0.13 → round(179.87 / 0.25) = round(719.48) = 719
        _, lo = grid_indices(51.51, -0.13, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert lo == 719

    def test_known_slc(self) -> None:
        # Salt Lake City: lat=40.77, lon=-111.89
        li, lo = grid_indices(40.77, -111.89, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == round((40.77 + 90.0) / 0.25)
        assert lo == round((-111.89 + 180.0) / 0.25)

    # --- lat_asc_lon_360 ---

    def test_lon360_positive_lon(self) -> None:
        # lon = 151.21 (Sydney) → same as pm180 for positive lons
        li_pm, lo_pm = grid_indices(-33.87, 151.21, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        li_360, lo_360 = grid_indices(-33.87, 151.21, ERA5_NY, ERA5_NX, "lat_asc_lon_360", ERA5_STEP)
        assert li_pm == li_360
        # lon_360 = lon % 360 = 151.21; lon_pm180 = (lon + 180) / 0.25 — different indices
        assert lo_360 == round(151.21 / ERA5_STEP)

    def test_lon360_negative_lon_wraps(self) -> None:
        # lon = -111.89 → 248.11 in [0,360]
        _, lo = grid_indices(40.77, -111.89, ERA5_NY, ERA5_NX, "lat_asc_lon_360", ERA5_STEP)
        assert lo == round(((-111.89) % 360.0) / ERA5_STEP)

    # --- lat_desc_lon_pm180 ---

    def test_lat_desc_north_is_row0(self) -> None:
        li, _ = grid_indices(90.0, 0.0, ERA5_NY, ERA5_NX, "lat_desc_lon_pm180", ERA5_STEP)
        assert li == 0

    def test_lat_desc_south_is_last_row(self) -> None:
        li, _ = grid_indices(-90.0, 0.0, ERA5_NY, ERA5_NX, "lat_desc_lon_pm180", ERA5_STEP)
        assert li == ERA5_NY - 1

    # --- lat_desc_lon_360 ---

    def test_lat_desc_lon360(self) -> None:
        li, lo = grid_indices(52.52, 13.40, ERA5_NY, ERA5_NX, "lat_desc_lon_360", ERA5_STEP)
        assert li == round((90.0 - 52.52) / ERA5_STEP)
        assert lo == round(13.40 / ERA5_STEP)

    # --- Clamping ---

    def test_clamp_below_south_pole(self) -> None:
        li, _ = grid_indices(-91.0, 0.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == 0

    def test_clamp_above_north_pole(self) -> None:
        li, _ = grid_indices(91.0, 0.0, ERA5_NY, ERA5_NX, "lat_asc_lon_pm180", ERA5_STEP)
        assert li == ERA5_NY - 1


# ---------------------------------------------------------------------------
# window_steps
# ---------------------------------------------------------------------------

class TestWindowSteps:
    def test_hourly_resolution(self) -> None:
        result = window_steps(3600.0, (1, 24, 168, 2160))
        assert result[1] == 1
        assert result[24] == 24
        assert result[168] == 168
        assert result[2160] == 2160

    def test_sub_hourly_resolution(self) -> None:
        # 30-min model: 24h window = 48 steps
        result = window_steps(1800.0, (24,))
        assert result[24] == 48

    def test_returns_int_values(self) -> None:
        result = window_steps(3600.0, (24,))
        assert isinstance(result[24], int)


# ---------------------------------------------------------------------------
# window_stats_batch
# ---------------------------------------------------------------------------

class TestWindowStatsBatch:
    """window_stats_batch(series, time_indices, steps) → (sums, counts)."""

    def test_sum_single_window(self) -> None:
        # Series [1, 2, 3, 4, 5]; 3-step window ending at index 4 → sum(3,4,5) = 12
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        time_idx = np.array([4])
        steps = {3: 3}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert sums[3][0] == pytest.approx(12.0)
        assert counts[3][0] == 3

    def test_sum_24h_window(self) -> None:
        # 100 hours of precipitation = 1mm/hr; 24h window ending at hour 50
        series = np.ones(100, dtype=np.float32)
        time_idx = np.array([50])
        steps = _make_steps((24,))
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert sums[24][0] == pytest.approx(24.0)
        assert counts[24][0] == 24

    def test_avg_24h_window(self) -> None:
        # Ramp series; 24h avg ending at index 99
        series = _ramp(100, start=0.0)  # 0,1,...,99
        time_idx = np.array([99])
        steps = _make_steps((24,))
        sums, counts = window_stats_batch(series, time_idx, steps)
        # Window indices 76..99 inclusive → avg = (76+77+...+99)/24 = (76+99)*24/2/24 = 87.5
        assert sums[24][0] / counts[24][0] == pytest.approx(87.5)

    def test_sum_vs_avg_different_results(self) -> None:
        # Same series, same window; sum != avg (unless series is all-1s and we compare differently)
        series = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        time_idx = np.array([4])
        steps = {5: 5}
        sums, counts = window_stats_batch(series, time_idx, steps)
        total_sum = sums[5][0]
        computed_avg = sums[5][0] / counts[5][0]
        assert total_sum != computed_avg  # 30 != 6

    def test_partial_window_at_start(self) -> None:
        # 24-step window ending at index 5 → only 6 steps available (0..5)
        series = np.ones(100, dtype=np.float32)
        time_idx = np.array([5])
        steps = _make_steps((24,))
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert counts[24][0] == 6
        assert sums[24][0] == pytest.approx(6.0)

    def test_2160h_max_window(self) -> None:
        # 3000-step series, all ones; 2160-step window ending at step 2999
        series = np.ones(3000, dtype=np.float32)
        time_idx = np.array([2999])
        steps = _make_steps((2160,))
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert counts[2160][0] == 2160
        assert sums[2160][0] == pytest.approx(2160.0)

    def test_nan_values_excluded_from_count(self) -> None:
        # Every other value is NaN; 4-step window should only count non-NaN
        series = np.array([1.0, np.nan, 1.0, np.nan, 1.0], dtype=np.float32)
        time_idx = np.array([4])
        steps = {5: 5}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert counts[5][0] == 3
        assert sums[5][0] == pytest.approx(3.0)

    def test_multiple_time_indices(self) -> None:
        # Each observation gets its own window
        series = np.ones(50, dtype=np.float32)
        time_idx = np.array([10, 20, 30])
        steps = {5: 5}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert np.all(counts[5] == 5)
        assert np.all(sums[5] == pytest.approx(5.0))

    def test_empty_time_indices(self) -> None:
        series = np.ones(10, dtype=np.float32)
        time_idx = np.array([], dtype=np.intp)
        steps = {24: 24}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert sums[24].shape == (0,)
        assert counts[24].shape == (0,)

    def test_all_nan_series(self) -> None:
        series = np.full(50, np.nan, dtype=np.float32)
        time_idx = np.array([30])
        steps = {24: 24}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert counts[24][0] == 0
        assert sums[24][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Chunk boundary via tail buffer
# ---------------------------------------------------------------------------

class TestChunkBoundary:
    """
    The tail buffer pattern: after processing chunk N, retain the last
    max_window_steps timesteps. For chunk N+1, prepend the tail so that
    window_stats_batch sees a continuous series.

    Tests here verify that window_stats_batch produces correct values when
    operating on [tail | chunk] concatenated data.
    """

    def test_cross_chunk_sum(self) -> None:
        # Chunk N: 24 hours of 2.0mm precipitation
        # Chunk N+1: 24 hours of 1.0mm precipitation
        # Observation at chunk_n1 index 5, 24h window:
        #   combined index = 24+5 = 29; start_idx = 29-23 = 6
        #   combined[6..23] = 2.0 (18 values), combined[24..29] = 1.0 (6 values)
        chunk_n = np.full(24, 2.0, dtype=np.float32)
        chunk_n1 = np.full(24, 1.0, dtype=np.float32)
        tail_size = 24  # keep entire chunk N as tail (≥ window size)

        tail = chunk_n[-tail_size:]
        combined = np.concatenate([tail, chunk_n1])

        # Observation at chunk_n1 index 5 → combined index = tail_size + 5 = 29
        time_idx = np.array([tail_size + 5])
        steps = {24: 24}
        sums, counts = window_stats_batch(combined, time_idx, steps)

        expected = 18 * 2.0 + 6 * 1.0  # 18 from tail, 6 from chunk_n1
        assert counts[24][0] == 24
        assert sums[24][0] == pytest.approx(expected)

    def test_cross_chunk_2160h_window(self) -> None:
        # Minimal smoke test: 2160h window spanning two chunks.
        # Chunk N: 2160 steps of 1.0; tail = last 2160 steps
        # Chunk N+1: 100 steps of 3.0; obs at step 50 needs tail(2110) + 50+1 = 2161... clip to 2160
        max_win = 2160
        chunk_n = np.ones(max_win, dtype=np.float32)
        chunk_n1 = np.full(200, 3.0, dtype=np.float32)
        tail = chunk_n[-max_win:]
        combined = np.concatenate([tail, chunk_n1])

        # Observation at chunk_n1 index 99 → combined index = 2160 + 99 = 2259
        time_idx = np.array([max_win + 99])
        steps = {max_win: max_win}
        sums, counts = window_stats_batch(combined, time_idx, steps)

        # Window [100..2259]: 2160 = 2060 from tail + 100 from chunk_n1
        assert counts[max_win][0] == max_win
        expected = 2060 * 1.0 + 100 * 3.0
        assert sums[max_win][0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Trailing-NaN capping (ERA5 processing-lag zone)
# ---------------------------------------------------------------------------

class TestTrailingNanCap:
    """
    When the last N timesteps of a series are NaN (ERA5 hasn't processed them
    yet), observations that land in that zone should receive the last valid
    value's window rather than returning NaN.

    process_chunk / process_chunk_mode implement this by capping local_time
    to the index of the last finite value in series_slice before calling
    window_stats_batch.  These tests verify the capping logic directly via
    window_stats_batch to confirm that capped indices yield non-NaN output.
    """

    def test_capped_index_gives_non_nan(self) -> None:
        # Series: 10 valid values then 5 NaN (57% of 15-step series)
        series = np.array([1.0] * 10 + [np.nan] * 5, dtype=np.float32)
        # Observation at position 13 (in NaN zone); cap to last valid = 9
        capped_time = np.array([9])
        steps = {5: 5}
        sums, counts = window_stats_batch(series, capped_time, steps)
        assert counts[5][0] == 5
        assert sums[5][0] == pytest.approx(5.0)

    def test_uncapped_nan_zone_produces_nan(self) -> None:
        # Without capping, observation deep in NaN zone → count=0 → NaN
        # series: valid[0..9], NaN[10..14]; window=3 ending at 12 → [10,11,12] all NaN
        series = np.array([1.0] * 10 + [np.nan] * 5, dtype=np.float32)
        uncapped_time = np.array([12])
        steps = {3: 3}
        sums, counts = window_stats_batch(series, uncapped_time, steps)
        assert counts[3][0] == 0

    def test_all_valid_series_unaffected(self) -> None:
        # No trailing NaN — capping does nothing, normal output
        series = np.full(20, 2.0, dtype=np.float32)
        time_idx = np.array([15])
        steps = {5: 5}
        sums, counts = window_stats_batch(series, time_idx, steps)
        assert counts[5][0] == 5
        assert sums[5][0] == pytest.approx(10.0)

    def test_last_valid_index_detection(self) -> None:
        # Helper: np.flatnonzero(np.isfinite(series))[-1] should find the right boundary
        series = np.array([3.0, np.nan, 5.0, np.nan, np.nan], dtype=np.float32)
        finite_idx = np.flatnonzero(np.isfinite(series))
        assert int(finite_idx[-1]) == 2


# ---------------------------------------------------------------------------
# vpd_kpa
# ---------------------------------------------------------------------------

class TestVpdKpa:
    """vpd_kpa(temp_c, dew_c) → VPD in kPa.

    Formula (Magnus approximation, same as Open-Meteo):
        e_s = 0.6108 * exp(17.27 * T / (T + 237.3))
        VPD = e_s(temp) - e_s(dew)
    """

    def _es(self, t: float) -> float:
        return 0.6108 * math.exp(17.27 * t / (t + 237.3))

    def test_zero_vpd_when_equal(self) -> None:
        # temp == dew_point → RH = 100% → VPD = 0
        result = vpd_kpa(20.0, 20.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_positive_vpd(self) -> None:
        # temp=25, dew=15 → vpd > 0
        expected = self._es(25.0) - self._es(15.0)
        assert vpd_kpa(25.0, 15.0) == pytest.approx(expected, abs=1e-6)

    def test_hot_dry_conditions(self) -> None:
        # Desert scenario: temp=40, dew=5 → high VPD
        result = vpd_kpa(40.0, 5.0)
        assert result > 4.0  # physically plausible for arid conditions

    def test_nan_propagation_temp(self) -> None:
        result = vpd_kpa(np.nan, 15.0)
        assert math.isnan(result)

    def test_nan_propagation_dew(self) -> None:
        result = vpd_kpa(25.0, np.nan)
        assert math.isnan(result)

    def test_array_input(self) -> None:
        temp = np.array([20.0, 25.0, 30.0], dtype=np.float32)
        dew = np.array([15.0, 20.0, 20.0], dtype=np.float32)
        result = vpd_kpa(temp, dew)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(self._es(20.0) - self._es(15.0), abs=1e-4)
        assert result[1] == pytest.approx(self._es(25.0) - self._es(20.0), abs=1e-4)


# ---------------------------------------------------------------------------
# weather_code_simple
# ---------------------------------------------------------------------------

class TestWeatherCodeSimple:
    """weather_code_simple(cloudcover, precip_mm, snowfall_we_mm, model_dt_s) → WMO code.

    Rate thresholds (per hour):
      Snow (cm/h from snowfall_water_equivalent/10):
        0.01–0.2  → 71 (slight snow)
        0.2–0.8   → 73 (moderate snow)
        ≥0.8      → 75 (heavy snow)
      Rain (mm/h from precipitation):
        0.01–0.5  → 51 (slight drizzle)
        0.5–1.0   → 53 (moderate drizzle)
        1.0–1.3   → 55 (heavy drizzle)
        1.3–2.5   → 61 (slight rain)
        2.5–7.6   → 63 (moderate rain)
        ≥7.6      → 65 (heavy rain)
      Cloud cover (when no precip):
        <20%  → 0 (clear)
        20–50 → 1 (mainly clear)
        50–80 → 2 (partly cloudy)
        ≥80   → 3 (overcast)
    """

    def test_clear_sky(self) -> None:
        assert weather_code_simple(5.0, 0.0, 0.0, 3600) == 0

    def test_mainly_clear(self) -> None:
        assert weather_code_simple(35.0, 0.0, 0.0, 3600) == 1

    def test_partly_cloudy(self) -> None:
        assert weather_code_simple(65.0, 0.0, 0.0, 3600) == 2

    def test_overcast(self) -> None:
        assert weather_code_simple(90.0, 0.0, 0.0, 3600) == 3

    def test_slight_drizzle(self) -> None:
        # 0.3 mm in 1h → rate 0.3 mm/h → code 51
        assert weather_code_simple(80.0, 0.3, 0.0, 3600) == 51

    def test_moderate_drizzle(self) -> None:
        assert weather_code_simple(80.0, 0.7, 0.0, 3600) == 53

    def test_heavy_drizzle(self) -> None:
        assert weather_code_simple(80.0, 1.1, 0.0, 3600) == 55

    def test_slight_rain(self) -> None:
        assert weather_code_simple(80.0, 2.0, 0.0, 3600) == 61

    def test_moderate_rain(self) -> None:
        assert weather_code_simple(80.0, 5.0, 0.0, 3600) == 63

    def test_heavy_rain(self) -> None:
        assert weather_code_simple(80.0, 10.0, 0.0, 3600) == 65

    def test_slight_snow(self) -> None:
        # snowfall_water_equivalent = 1mm → 0.1cm → rate 0.1cm/h → code 71
        assert weather_code_simple(80.0, 0.0, 1.0, 3600) == 71

    def test_moderate_snow(self) -> None:
        # 5mm WE → 0.5cm → rate 0.5cm/h → code 73
        assert weather_code_simple(80.0, 0.0, 5.0, 3600) == 73

    def test_heavy_snow(self) -> None:
        # 10mm WE → 1.0cm → rate 1.0cm/h → code 75
        assert weather_code_simple(80.0, 0.0, 10.0, 3600) == 75

    def test_snow_takes_priority_over_rain(self) -> None:
        # Both precip and snow present: snow check runs first
        assert weather_code_simple(80.0, 5.0, 5.0, 3600) == 73

    def test_none_on_null_inputs(self) -> None:
        assert weather_code_simple(None, 0.0, 0.0, 3600) is None
        assert weather_code_simple(80.0, None, 0.0, 3600) is None
        assert weather_code_simple(80.0, 0.0, None, 3600) is None

    def test_none_on_nan_inputs(self) -> None:
        assert weather_code_simple(np.nan, 0.0, 0.0, 3600) is None

    def test_3h_model_dt(self) -> None:
        # model_dt_seconds = 10800 (3h); 15mm precip over 3h → 5mm/h → code 63
        assert weather_code_simple(80.0, 15.0, 0.0, 10800) == 63

    def test_no_precip_boundary(self) -> None:
        # Exactly 0.0 precip and 0.0 snow → use cloud cover
        assert weather_code_simple(10.0, 0.0, 0.0, 3600) == 0


# ---------------------------------------------------------------------------
# weather_code_array
# ---------------------------------------------------------------------------

class TestWeatherCodeArray:
    def test_matches_scalar_clear_sky(self) -> None:
        result = weather_code_array(
            np.array([5.0]), np.array([0.0]), np.array([0.0]), 3600.0
        )
        assert result[0] == pytest.approx(weather_code_simple(5.0, 0.0, 0.0, 3600))

    def test_matches_scalar_heavy_rain(self) -> None:
        result = weather_code_array(
            np.array([90.0]), np.array([10.0]), np.array([0.0]), 3600.0
        )
        assert result[0] == pytest.approx(65.0)

    def test_matches_scalar_heavy_snow(self) -> None:
        result = weather_code_array(
            np.array([90.0]), np.array([0.0]), np.array([10.0]), 3600.0
        )
        assert result[0] == pytest.approx(75.0)

    def test_nan_on_invalid_input(self) -> None:
        result = weather_code_array(
            np.array([np.nan]), np.array([0.0]), np.array([0.0]), 3600.0
        )
        assert np.isnan(result[0])

    def test_vectorized_mixed(self) -> None:
        cloud = np.array([5.0, 90.0, 90.0])
        precip = np.array([0.0, 0.0, 10.0])
        snow = np.zeros(3)
        result = weather_code_array(cloud, precip, snow, 3600.0)
        assert result[0] == pytest.approx(0.0)   # clear
        assert result[1] == pytest.approx(3.0)   # overcast
        assert result[2] == pytest.approx(65.0)  # heavy rain

    def test_3h_resolution(self) -> None:
        # 15 mm over 3h = 5 mm/h → code 63 (moderate rain)
        result = weather_code_array(
            np.array([90.0]), np.array([15.0]), np.array([0.0]), 10800.0
        )
        assert result[0] == pytest.approx(63.0)

    def test_snow_priority_over_rain(self) -> None:
        result = weather_code_array(
            np.array([80.0]), np.array([5.0]), np.array([5.0]), 3600.0
        )
        assert result[0] == pytest.approx(73.0)


# ---------------------------------------------------------------------------
# _window_mode_batch
# ---------------------------------------------------------------------------

class TestWindowModeBatch:
    def test_single_step_window_returns_value(self) -> None:
        series = np.array([0.0, 1.0, 2.0, 3.0])
        result = _window_mode_batch(series, np.array([2]), {1: 1})
        assert result[1][0] == pytest.approx(2.0)

    def test_mode_of_uniform_window(self) -> None:
        series = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = _window_mode_batch(series, np.array([4]), {5: 5})
        assert result[5][0] == pytest.approx(3.0)

    def test_mode_picks_majority(self) -> None:
        # window: [0, 0, 0, 65, 65] → mode = 0
        series = np.array([0.0, 0.0, 0.0, 65.0, 65.0])
        result = _window_mode_batch(series, np.array([4]), {5: 5})
        assert result[5][0] == pytest.approx(0.0)

    def test_nan_inputs_excluded(self) -> None:
        series = np.array([np.nan, np.nan, 3.0])
        result = _window_mode_batch(series, np.array([2]), {3: 3})
        assert result[3][0] == pytest.approx(3.0)

    def test_all_nan_returns_nan(self) -> None:
        series = np.array([np.nan, np.nan, np.nan])
        result = _window_mode_batch(series, np.array([2]), {3: 3})
        assert np.isnan(result[3][0])

    def test_empty_time_indices(self) -> None:
        series = np.array([1.0, 2.0, 3.0])
        result = _window_mode_batch(series, np.array([], dtype=int), {3: 3})
        assert result[3].size == 0

    def test_window_len_zero_returns_nan(self) -> None:
        series = np.array([1.0, 2.0])
        result = _window_mode_batch(series, np.array([1]), {0: 0})
        assert np.isnan(result[0][0])

    def test_multiple_windows(self) -> None:
        series = np.array([0.0, 65.0, 65.0, 65.0])
        result = _window_mode_batch(series, np.array([3]), {1: 1, 4: 4})
        assert result[1][0] == pytest.approx(65.0)
        assert result[4][0] == pytest.approx(65.0)
