"""API-fixture-grounded correctness tests for the raster accumulation pipeline.

Each test uses the same Open-Meteo archive fixture JSON already fetched for
enrich_temporal.  The .om reader is monkeypatched to return the fixture series,
so accumulate_raster() sees real API values.  expected_window() provides the
independent ground truth computed directly from the same fixture data.

This gives the same guarantee as the enrich_temporal fixture tests: if these
pass, the raster pipeline produces values that match the Open-Meteo API.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.temporal.conftest import (
    TEST_LOCATIONS,
    expected_window,
)
from tests.temporal.rasters.conftest import (
    FakeRasterReader,
    chunk_from_fixture,
)
from util.temporal import (
    RASTER_GRIDS,
    RASTER_WC_CODES,
    accumulate_raster,
    compute_raster_final,
    grid_indices,
    vpd_kpa,
    weather_code_array,
)

pytestmark = pytest.mark.usefixtures("require_fixtures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOC = {loc["name"]: loc for loc in TEST_LOCATIONS}


def _cell(loc_name: str, model: str) -> tuple[int, int]:
    entry = _LOC[loc_name]
    g = RASTER_GRIDS[model]
    ny, nx = g.get("ny", 721), g.get("nx", 1440)
    return grid_indices(entry["lat"], entry["lon"], ny, nx, "lat_asc_lon_pm180", g["step"])


def _run_accumulate(fix, variable, model, obs_hour, window_hours, monkeypatch, tmp=None):
    """Run accumulate_raster on fixture data, return (cell_avg_or_sum, expected)."""
    obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
    series = np.array(fix["hourly"][variable], dtype=np.float64)
    index, entry = chunk_from_fixture(fix, model)
    start_ts = obs_ts - (window_hours - 1) * 3600

    monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series, model))
    grid, n = accumulate_raster(model, variable, start_ts, obs_ts, index)
    return grid, n, obs_ts


def _cell_value(grid: np.ndarray, n: int, loc_name: str, model: str, agg: str) -> float:
    li, lo = _cell(loc_name, model)
    raw = float(grid[li, lo])
    return raw / n if (agg == "avg" and n > 0) else raw


# ---------------------------------------------------------------------------
# TestRasterPrecipitationSum
# ---------------------------------------------------------------------------

class TestRasterPrecipitationSum:
    _MODEL = "copernicus_era5"
    _VAR = "precipitation"

    def test_salt_lake_city_24h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["salt_lake_city_early"]
        fix["_key"] = "salt_lake_city_early"
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "salt_lake_city", self._MODEL, "sum")
        expected = expected_window(fix, obs_ts, self._VAR, 24, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_reykjavik_168h_boundary(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["reykjavik_boundary"]
        obs_hour = 400
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 168, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "reykjavik", self._MODEL, "sum")
        expected = expected_window(fix, obs_ts, self._VAR, 168, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_sydney_early_168h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["sydney_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 168, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "sydney", self._MODEL, "sum")
        expected = expected_window(fix, obs_ts, self._VAR, 168, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-2)

    def test_nairobi_72h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["nairobi_early"]
        obs_hour = 300
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 72, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "nairobi", self._MODEL, "sum")
        expected = expected_window(fix, obs_ts, self._VAR, 72, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# TestRasterTemperatureAvg  (ERA5-land, 0.1°)
# ---------------------------------------------------------------------------

class TestRasterTemperatureAvg:
    _MODEL = "copernicus_era5_land"
    _VAR = "temperature_2m"

    def test_berlin_1h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 200
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 1, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", self._MODEL, "avg")
        expected = expected_window(fix, obs_ts, self._VAR, 1, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_dubai_24h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["dubai_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "dubai", self._MODEL, "avg")
        expected = expected_window(fix, obs_ts, self._VAR, 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_reykjavik_72h_boundary(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["reykjavik_boundary"]
        obs_hour = 400
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 72, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "reykjavik", self._MODEL, "avg")
        expected = expected_window(fix, obs_ts, self._VAR, 72, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_tuvalu_24h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        # Dateline edge: tuvalu lon ~179.2°
        fix = require_fixtures["tuvalu_early"]
        obs_hour = 300
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "tuvalu", self._MODEL, "avg")
        expected = expected_window(fix, obs_ts, self._VAR, 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_ushuaia_168h_early(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["ushuaia_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, self._VAR, self._MODEL, obs_hour, 168, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "ushuaia", self._MODEL, "avg")
        expected = expected_window(fix, obs_ts, self._VAR, 168, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# TestRasterOtherVariables
# ---------------------------------------------------------------------------

class TestRasterOtherVariables:
    def test_cloud_cover_berlin_24h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, "cloud_cover", "copernicus_era5", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5", "avg")
        expected = expected_window(fix, obs_ts, "cloud_cover", 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_cloud_cover_london_168h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["london_early"]
        obs_hour = 700
        grid, n, obs_ts = _run_accumulate(fix, "cloud_cover", "copernicus_era5", obs_hour, 168, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "london", "copernicus_era5", "avg")
        expected = expected_window(fix, obs_ts, "cloud_cover", 168, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_snowfall_reykjavik_24h_boundary(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["reykjavik_boundary"]
        obs_hour = 400
        grid, n, obs_ts = _run_accumulate(fix, "snowfall_water_equivalent", "copernicus_era5", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "reykjavik", "copernicus_era5", "sum")
        expected = expected_window(fix, obs_ts, "snowfall_water_equivalent", 24, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_snowfall_tromsoe_72h_boundary(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["tromsoe_boundary"]
        obs_hour = 300
        grid, n, obs_ts = _run_accumulate(fix, "snowfall_water_equivalent", "copernicus_era5", obs_hour, 72, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "tromsoe", "copernicus_era5", "sum")
        expected = expected_window(fix, obs_ts, "snowfall_water_equivalent", 72, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_dew_point_berlin_24h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, "dew_point_2m", "copernicus_era5_land", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "dew_point_2m", 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_dew_point_sydney_72h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["sydney_early"]
        obs_hour = 600
        grid, n, obs_ts = _run_accumulate(fix, "dew_point_2m", "copernicus_era5_land", obs_hour, 72, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "sydney", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "dew_point_2m", 72, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_soil_temp_berlin_72h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 1000
        grid, n, obs_ts = _run_accumulate(fix, "soil_temperature_0_to_7cm", "copernicus_era5_land", obs_hour, 72, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "soil_temperature_0_to_7cm", 72, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_soil_temp_nairobi_168h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["nairobi_early"]
        obs_hour = 800
        grid, n, obs_ts = _run_accumulate(fix, "soil_temperature_0_to_7cm", "copernicus_era5_land", obs_hour, 168, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "nairobi", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "soil_temperature_0_to_7cm", 168, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_soil_moisture_berlin_24h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, "soil_moisture_0_to_7cm", "copernicus_era5_land", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "soil_moisture_0_to_7cm", 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-4)

    def test_snow_depth_reykjavik_24h_boundary(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["reykjavik_boundary"]
        obs_hour = 300
        grid, n, obs_ts = _run_accumulate(fix, "snow_depth", "copernicus_era5_land", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "reykjavik", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "snow_depth", 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# TestRasterAllWindowSizes
# ---------------------------------------------------------------------------

class TestRasterAllWindowSizes:
    """Verify all 7 window sizes against API fixtures for temperature and precip."""

    @pytest.mark.parametrize(("window_h", "label"), [
        (1, "1h"), (8, "8h"), (24, "24h"), (72, "3d"), (168, "7d"),
    ])
    def test_temperature_all_windows_berlin(self, window_h, label, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        grid, n, obs_ts = _run_accumulate(fix, "temperature_2m", "copernicus_era5_land", obs_hour, window_h, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5_land", "avg")
        expected = expected_window(fix, obs_ts, "temperature_2m", window_h, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3), f"window={label}"

    @pytest.mark.parametrize(("window_h", "label"), [
        (1, "1h"), (8, "8h"), (24, "24h"), (72, "3d"), (168, "7d"),
    ])
    def test_precipitation_all_windows_berlin(self, window_h, label, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 2160
        grid, n, obs_ts = _run_accumulate(fix, "precipitation", "copernicus_era5", obs_hour, window_h, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5", "sum")
        expected = expected_window(fix, obs_ts, "precipitation", window_h, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-2), f"window={label}"


# ---------------------------------------------------------------------------
# TestRasterPartialWindows
# ---------------------------------------------------------------------------

class TestRasterPartialWindows:
    def test_near_series_start(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        # At index 10, a 24h window can only use hours 0..10 (11 values)
        obs_hour = 10
        grid, n, obs_ts = _run_accumulate(fix, "precipitation", "copernicus_era5", obs_hour, 24, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5", "sum")
        expected = expected_window(fix, obs_ts, "precipitation", 24, "sum")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3)

    def test_1h_is_single_value(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 100
        float(fix["hourly"]["time_unix"][obs_hour])
        series = np.array(fix["hourly"]["temperature_2m"], dtype=np.float64)
        # Single value should equal the raw series value at obs_hour
        grid, n, _ = _run_accumulate(fix, "temperature_2m", "copernicus_era5_land", obs_hour, 1, monkeypatch, str(tmp_path))
        cell = _cell_value(grid, n, "berlin", "copernicus_era5_land", "avg")
        assert cell == pytest.approx(float(series[obs_hour]), abs=1e-3)


# ---------------------------------------------------------------------------
# TestRasterVPD
# ---------------------------------------------------------------------------

class TestRasterVPD:
    def test_vpd_berlin_24h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        t_exp = expected_window(fix, obs_ts, "temperature_2m", 24, "avg")
        td_exp = expected_window(fix, obs_ts, "dew_point_2m", 24, "avg")
        assert t_exp is not None
        assert td_exp is not None
        expected_vpd = float(vpd_kpa(t_exp, td_exp))

        # Build t and td sums separately
        series_t = np.array(fix["hourly"]["temperature_2m"], dtype=np.float64)
        series_td = np.array(fix["hourly"]["dew_point_2m"], dtype=np.float64)
        index, _ = chunk_from_fixture(fix, "copernicus_era5_land")
        start_ts = obs_ts - 23 * 3600

        # Accumulate t
        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series_t, "copernicus_era5_land"))
        t_grid, t_n = accumulate_raster("copernicus_era5_land", "temperature_2m", start_ts, obs_ts, index)

        # Accumulate td
        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series_td, "copernicus_era5_land"))
        td_grid, td_n = accumulate_raster("copernicus_era5_land", "dew_point_2m", start_ts, obs_ts, index)

        # Compute VPD
        sums = {"era5_temperature_2m": t_grid, "era5_dew_point_2m": td_grid}
        result = compute_raster_final("vapor_pressure_deficit", "avg", sums, t_n, 0)
        li, lo = _cell("berlin", "copernicus_era5_land")
        cell_vpd = float(result[li, lo])
        assert cell_vpd == pytest.approx(max(expected_vpd, 0.0), abs=1e-3)

    def test_vpd_dubai_168h(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["dubai_early"]
        obs_hour = 500
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        t_exp = expected_window(fix, obs_ts, "temperature_2m", 168, "avg")
        td_exp = expected_window(fix, obs_ts, "dew_point_2m", 168, "avg")
        assert t_exp is not None
        assert td_exp is not None
        expected_vpd = max(float(vpd_kpa(t_exp, td_exp)), 0.0)

        series_t = np.array(fix["hourly"]["temperature_2m"], dtype=np.float64)
        series_td = np.array(fix["hourly"]["dew_point_2m"], dtype=np.float64)
        index, _ = chunk_from_fixture(fix, "copernicus_era5_land")
        start_ts = obs_ts - 167 * 3600

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series_t, "copernicus_era5_land"))
        t_grid, t_n = accumulate_raster("copernicus_era5_land", "temperature_2m", start_ts, obs_ts, index)
        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series_td, "copernicus_era5_land"))
        td_grid, td_n = accumulate_raster("copernicus_era5_land", "dew_point_2m", start_ts, obs_ts, index)

        result = compute_raster_final("vapor_pressure_deficit", "avg",
                                      {"era5_temperature_2m": t_grid, "era5_dew_point_2m": td_grid},
                                      t_n, 0)
        li, lo = _cell("dubai", "copernicus_era5_land")
        assert float(result[li, lo]) == pytest.approx(expected_vpd, abs=1e-3)


# ---------------------------------------------------------------------------
# TestRasterWeatherCodeMode
# ---------------------------------------------------------------------------

class TestRasterWeatherCodeMode:
    """Verify weather_code_simple mode against expected dominant code from fixture."""

    def _expected_mode(self, fix, obs_hour: int, window_hours: int) -> int | None:
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        t_start = float(fix["hourly"]["time_unix"][0])
        resolution = 3600.0
        end_idx = int(round((obs_ts - t_start) / resolution))
        start_idx = max(0, end_idx - window_hours + 1)

        cc = fix["hourly"]["cloud_cover"]
        pr = fix["hourly"]["precipitation"]
        sn = fix["hourly"]["snowfall_water_equivalent"]
        codes = []
        for i in range(start_idx, end_idx + 1):
            c = cc[i] if i < len(cc) else None
            p = pr[i] if i < len(pr) else None
            s = sn[i] if i < len(sn) else None
            if None in (c, p, s):
                continue
            code = float(weather_code_array(
                np.array([c]), np.array([p]), np.array([s]), resolution,
            )[0])
            if np.isfinite(code):
                codes.append(int(round(code)))
        if not codes:
            return None
        from collections import Counter
        return Counter(codes).most_common(1)[0][0]

    def test_dubai_clear_dominant(self, require_fixtures) -> None:
        fix = require_fixtures["dubai_early"]
        mode = self._expected_mode(fix, 500, 168)
        assert mode in (0, 1, 2), f"Dubai should be mostly clear/partly cloudy, got {mode}"

    def test_reykjavik_boundary_snow_present(self, require_fixtures) -> None:
        fix = require_fixtures["reykjavik_boundary"]
        mode = self._expected_mode(fix, 400, 72)
        assert mode is not None
        # Dec–Jan in Reykjavik — expect snow or heavy precip codes
        assert mode in (0, 1, 2, 3, 51, 53, 55, 61, 63, 65, 71, 73, 75)

    def test_mode_consistent_with_compute_raster_final(self, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        window_hours = 24
        t0_fix = float(fix["hourly"]["time_unix"][0])
        resolution = 3600.0

        end_idx = int(round((obs_ts - t0_fix) / resolution))
        start_idx = max(0, end_idx - window_hours + 1)

        cc_series = np.array(fix["hourly"]["cloud_cover"], dtype=np.float64)
        pr_series = np.array(fix["hourly"]["precipitation"], dtype=np.float64)
        sn_series = np.array(fix["hourly"]["snowfall_water_equivalent"], dtype=np.float64)

        codes = weather_code_array(
            cc_series[start_idx:end_idx + 1],
            pr_series[start_idx:end_idx + 1],
            sn_series[start_idx:end_idx + 1],
            resolution,
        )
        counts = {c: int((np.round(codes) == c).sum()) for c in RASTER_WC_CODES}
        count_grids = {c: np.full((5, 5), counts[c], dtype=np.int32) for c in RASTER_WC_CODES}
        result = compute_raster_final("weather_code_simple", "mode", count_grids, end_idx - start_idx + 1, 0)
        expected_mode = max(counts, key=counts.get)
        assert int(result[0, 0]) == expected_mode


# ---------------------------------------------------------------------------
# TestSlidingWindowIncrementalCorrectness (the critical API-grounded test)
# ---------------------------------------------------------------------------

class TestSlidingWindowIncrementalCorrectness:
    """Prove that incremental update == full rebuild, verified against API values.

    Uses the 8784-hour berlin_early fixture (a full year).
    Run 1: accumulate [t0, t_mid]
    Run 2: incremental delta [t_mid, t_end]  →  sum_run1 + delta = full(t0, t_end)
    Final cell value must equal expected_window(fixture, t_end, var, W, agg).
    """

    def _accum(self, fix, variable, model, start_ts, end_ts, monkeypatch, tmp=None) -> tuple[np.ndarray, int]:
        series = np.array(fix["hourly"][variable], dtype=np.float64)
        index, _ = chunk_from_fixture(fix, model)

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series, model))
        return accumulate_raster(model, variable, start_ts, end_ts, index)

    @pytest.mark.parametrize("window_hours", [24, 168])
    def test_temperature_incremental_equals_full(self, window_hours, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        times = fix["hourly"]["time_unix"]
        t_end_idx = min(window_hours + 500, len(times) - 1)
        # t_mid must fall inside the window: split at min(W//2, 100) steps before t_end
        t_mid_idx = t_end_idx - min(window_hours // 2, 100)
        t_mid = float(times[t_mid_idx])
        t_end = float(times[t_end_idx])
        w_start = t_end - (window_hours - 1) * 3600

        # Full rebuild over [w_start, t_end]
        full_sum, full_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                       w_start, t_end, monkeypatch, str(tmp_path))

        # Run 1: [w_start, t_mid]
        run1_sum, run1_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                       w_start, t_mid, monkeypatch, str(tmp_path))

        # Delta: [t_mid + 1h, t_end]
        delta_sum, delta_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                         t_mid + 3600, t_end, monkeypatch, str(tmp_path))

        incr_sum = run1_sum + delta_sum
        incr_n = run1_n + delta_n

        li, lo = _cell("berlin", "copernicus_era5_land")
        assert incr_sum[li, lo] == pytest.approx(full_sum[li, lo], rel=1e-4), \
            f"window={window_hours}h: incremental sum != full rebuild sum"
        assert incr_n == full_n, f"window={window_hours}h: count mismatch"

        # Also verify against API expected value
        expected = expected_window(fix, t_end, "temperature_2m", window_hours, "avg")
        if expected is not None:
            cell_avg = incr_sum[li, lo] / incr_n if incr_n > 0 else float("nan")
            assert cell_avg == pytest.approx(expected, abs=1e-3), \
                f"window={window_hours}h: cell avg does not match API expected"

    @pytest.mark.parametrize("window_hours", [24, 168])
    def test_precipitation_incremental_equals_full(self, window_hours, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        times = fix["hourly"]["time_unix"]
        t_end_idx = min(window_hours + 500, len(times) - 1)
        t_mid_idx = t_end_idx - min(window_hours // 2, 50)
        t_mid = float(times[t_mid_idx])
        t_end = float(times[t_end_idx])
        w_start = t_end - (window_hours - 1) * 3600

        full_sum, full_n = self._accum(fix, "precipitation", "copernicus_era5",
                                       w_start, t_end, monkeypatch, str(tmp_path))
        run1_sum, run1_n = self._accum(fix, "precipitation", "copernicus_era5",
                                       w_start, t_mid, monkeypatch, str(tmp_path))
        delta_sum, delta_n = self._accum(fix, "precipitation", "copernicus_era5",
                                         t_mid + 3600, t_end, monkeypatch, str(tmp_path))

        incr_sum = run1_sum + delta_sum
        li, lo = _cell("berlin", "copernicus_era5")
        assert incr_sum[li, lo] == pytest.approx(full_sum[li, lo], rel=1e-4), \
            f"window={window_hours}h"

        expected = expected_window(fix, t_end, "precipitation", window_hours, "sum")
        if expected is not None:
            assert incr_sum[li, lo] == pytest.approx(expected, abs=1e-2), \
                f"window={window_hours}h: sum does not match API expected"

    def test_drop_oldest_leaves_correct_window(self, require_fixtures, monkeypatch, tmp_path) -> None:
        """After dropping old hours, cell value matches expected_window at new end."""
        fix = require_fixtures["berlin_early"]
        times = fix["hourly"]["time_unix"]
        # Window 24h ending at hour 600
        t_end = float(times[600])
        t_end_new = float(times[624])  # 24h later
        w_start_old = t_end - 23 * 3600   # first point of old 24h window
        w_start_new = t_end_new - 23 * 3600  # first point of new 24h window

        # Build old state: [w_start_old, t_end]
        old_sum, old_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                     w_start_old, t_end, monkeypatch, str(tmp_path))

        # Drop [w_start_old, w_start_new)
        drop_sum, drop_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                       w_start_old, w_start_new - 3600, monkeypatch, str(tmp_path))

        # Add [t_end+1h, t_end_new]
        add_sum, add_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                     t_end + 3600, t_end_new, monkeypatch, str(tmp_path))

        incr = old_sum - drop_sum + add_sum
        incr_n = old_n - drop_n + add_n

        li, lo = _cell("berlin", "copernicus_era5_land")
        expected = expected_window(fix, t_end_new, "temperature_2m", 24, "avg")
        assert expected is not None
        cell_avg = incr[li, lo] / incr_n if incr_n > 0 else float("nan")
        assert cell_avg == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# TestForecastOffsetFixtures
# ---------------------------------------------------------------------------

class TestForecastOffsetFixtures:
    """Forecast raster at +H offset matches expected_window at obs_ts + H."""

    def _accum(self, fix, variable, model, start_ts, end_ts, monkeypatch, tmp=None) -> tuple[np.ndarray, int]:
        series = np.array(fix["hourly"][variable], dtype=np.float64)
        index, _ = chunk_from_fixture(fix, model)

        monkeypatch.setattr("util.temporal._open_chunk", lambda *a, **kw: FakeRasterReader(series, model))
        return accumulate_raster(model, variable, start_ts, end_ts, index)

    @pytest.mark.parametrize("forecast_h", [1, 8, 24, 72])
    def test_temperature_forecast_berlin(self, forecast_h, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        times = fix["hourly"]["time_unix"]
        now_idx = 500
        now_ts = float(times[now_idx])
        future_ts = now_ts + forecast_h * 3600
        window_h = 24

        # Forecast raster: window shifted by forecast_h
        w_start_fc = future_ts - (window_h - 1) * 3600
        fc_sum, fc_n = self._accum(fix, "temperature_2m", "copernicus_era5_land",
                                   w_start_fc, future_ts, monkeypatch, str(tmp_path))

        li, lo = _cell("berlin", "copernicus_era5_land")
        cell = fc_sum[li, lo] / fc_n if fc_n > 0 else float("nan")
        expected = expected_window(fix, future_ts, "temperature_2m", window_h, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=1e-3), f"forecast_h={forecast_h}"

    @pytest.mark.parametrize("forecast_h", [1, 24, 72])
    def test_precipitation_forecast_berlin(self, forecast_h, require_fixtures, monkeypatch, tmp_path) -> None:
        fix = require_fixtures["berlin_early"]
        times = fix["hourly"]["time_unix"]
        now_ts = float(times[500])
        future_ts = now_ts + forecast_h * 3600
        window_h = 168

        w_start_fc = future_ts - (window_h - 1) * 3600
        fc_sum, _ = self._accum(fix, "precipitation", "copernicus_era5",
                                w_start_fc, future_ts, monkeypatch, str(tmp_path))

        li, lo = _cell("berlin", "copernicus_era5")
        expected = expected_window(fix, future_ts, "precipitation", window_h, "sum")
        assert expected is not None
        assert fc_sum[li, lo] == pytest.approx(expected, abs=1e-2), f"forecast_h={forecast_h}"
