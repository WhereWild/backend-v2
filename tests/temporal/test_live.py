"""
Gold-standard end-to-end live tests.

Validates that the complete pipeline (grid indexing → S3 .om range read →
windowed aggregation) produces the same values as the Open-Meteo archive API.
This is the only test that really matters: if these pass, the pipeline is the API.

Run:
    pt --temporal                           # run live tests
    pt --temporal --regenerate-live         # re-fetch API ground truth first

Excluded from the regular test suite (pt without --temporal skips them).
Requires anonymous S3 access; --regenerate-live also requires Open-Meteo archive
API access.

Test locations are chosen at exact 0.5° boundaries so they fall on grid nodes
for both 0.25° (ERA5) and 0.1° (ERA5-Land) grids simultaneously, eliminating
any spatial-interpolation ambiguity.

Comparison: round(pipeline_value, api_decimals) == round(api_value, api_decimals)
The .om files store int16 with scale_factor=20, giving 0.05 precision. The API
rounds to 1 decimal for temperature-like vars. Rounding both to the same precision
the API uses gives exact equality on every valid check.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import httpx
import numpy as np
import pytest
from omfiles import OmFileReader

from util.temporal import (
    grid_indices,
    window_stats_batch,
    window_steps,
)

pytestmark = pytest.mark.live

_FIXTURES_PATH = Path(__file__).parent / "live_fixtures.json"

_OBS_JUN = 1560556800  # 2019-06-15 00:00 UTC  (mid-year, no boundary issues)
_OBS_JAN = 1547510400  # 2019-01-15 00:00 UTC  (winter, for snow cases)
_OBS_CHK = 1710460800  # 2024-03-15 00:00 UTC  (chunk-file era, time_idx≈260 within chunk)

# Each entry: what to test.  Expected values live in live_fixtures.json.
# model        — S3 model path segment and API models= param
# grid_step    — degrees per cell (0.25 for ERA5, 0.1 for ERA5-Land)
# api_decimals — precision the API rounds to; we round omfile values to match
LIVE_CASES: dict[str, dict[str, Any]] = {
    # --- ERA5 (0.25°), year-file era ---
    "berlin_temp_24h": dict(
        variable="temperature_2m", model="copernicus_era5", grid_step=0.25,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="avg", api_decimals=1,
    ),
    "berlin_temp_72h": dict(
        variable="temperature_2m", model="copernicus_era5", grid_step=0.25,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=72, agg="avg", api_decimals=1,
    ),
    "berlin_precip_24h": dict(
        variable="precipitation", model="copernicus_era5", grid_step=0.25,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="sum", api_decimals=1,
    ),
    "sydney_precip_24h": dict(
        variable="precipitation", model="copernicus_era5", grid_step=0.25,
        lat=-33.5, lon=151.5, obs_ts=_OBS_JUN, window_hours=24, agg="sum", api_decimals=1,
    ),
    "nairobi_temp_72h": dict(
        variable="temperature_2m", model="copernicus_era5", grid_step=0.25,
        lat=-1.5, lon=37.0, obs_ts=_OBS_JUN, window_hours=72, agg="avg", api_decimals=1,
    ),
    "berlin_cloud_24h": dict(
        variable="cloud_cover", model="copernicus_era5", grid_step=0.25,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="avg", api_decimals=1,
    ),
    "reykjavik_snowfall_24h": dict(
        variable="snowfall_water_equivalent", model="copernicus_era5", grid_step=0.25,
        lat=64.5, lon=-22.0, obs_ts=_OBS_JAN, window_hours=24, agg="sum", api_decimals=1,
    ),
    # --- ERA5-Land (0.1°), year-file era ---
    "berlin_dewpoint_24h": dict(
        variable="dew_point_2m", model="copernicus_era5_land", grid_step=0.1,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="avg", api_decimals=1,
    ),
    "berlin_soiltemp_24h": dict(
        variable="soil_temperature_0_to_7cm", model="copernicus_era5_land", grid_step=0.1,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="avg", api_decimals=1,
    ),
    "berlin_soiltemp_72h": dict(
        variable="soil_temperature_0_to_7cm", model="copernicus_era5_land", grid_step=0.1,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=72, agg="avg", api_decimals=1,
    ),
    "berlin_soilmoist_24h": dict(
        variable="soil_moisture_0_to_7cm", model="copernicus_era5_land", grid_step=0.1,
        lat=52.5, lon=13.5, obs_ts=_OBS_JUN, window_hours=24, agg="avg", api_decimals=3,
    ),
    "nairobi_soilmoist_72h": dict(
        variable="soil_moisture_0_to_7cm", model="copernicus_era5_land", grid_step=0.1,
        lat=-1.5, lon=37.0, obs_ts=_OBS_JUN, window_hours=72, agg="avg", api_decimals=3,
    ),
    "reykjavik_snow_24h": dict(
        variable="snow_depth", model="copernicus_era5_land", grid_step=0.1,
        lat=64.5, lon=-22.0, obs_ts=_OBS_JAN, window_hours=24, agg="avg", api_decimals=3,
    ),
    # --- chunk-file era (2024-03-15, time_idx≈260 within chunk, well clear of boundary) ---
    "berlin_temp_chunk_24h": dict(
        variable="temperature_2m", model="copernicus_era5", grid_step=0.25,
        lat=52.5, lon=13.5, obs_ts=_OBS_CHK, window_hours=24, agg="avg", api_decimals=1,
    ),
    "berlin_soiltemp_chunk_24h": dict(
        variable="soil_temperature_0_to_7cm", model="copernicus_era5_land", grid_step=0.1,
        lat=52.5, lon=13.5, obs_ts=_OBS_CHK, window_hours=24, agg="avg", api_decimals=1,
    ),
}


# ---------------------------------------------------------------------------
# Module-level gate: skip unless --live was passed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _live_gate(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live"):
        pytest.skip("live S3 tests skipped — use: pt --temporal")


# ---------------------------------------------------------------------------
# Expected-value fixture (loads or regenerates live_fixtures.json)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_expected(request: pytest.FixtureRequest) -> dict[str, float]:
    if request.config.getoption("--regenerate-live") or not _FIXTURES_PATH.exists():
        data = _regenerate()
        _FIXTURES_PATH.write_text(json.dumps(data, indent=2))
        print(f"\n[live] wrote {_FIXTURES_PATH}")
        return data
    return json.loads(_FIXTURES_PATH.read_text())


# ---------------------------------------------------------------------------
# Ground-truth: fetch from Open-Meteo archive API and compute windowed value
# ---------------------------------------------------------------------------

def _api_window_value(case: dict[str, Any]) -> float:
    obs_ts = case["obs_ts"]
    obs_hour = (obs_ts // 3600) * 3600
    start_ts = obs_hour - case["window_hours"] * 3600
    start_dt = datetime.fromtimestamp(start_ts, UTC)
    end_dt   = datetime.fromtimestamp(obs_hour, UTC)

    resp = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   case["lat"],
            "longitude":  case["lon"],
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date":   end_dt.strftime("%Y-%m-%d"),
            "hourly":     case["variable"],
            "models":     case["model"],
            "timezone":   "UTC",
            "elevation":  "NaN",
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    times_unix = [
        int(datetime.fromisoformat(t).replace(tzinfo=UTC).timestamp())
        for t in data["hourly"]["time"]
    ]
    vals = data["hourly"][case["variable"]]
    t0 = times_unix[0]
    end_idx = round((obs_hour - t0) / 3600)
    if end_idx < 0 or end_idx >= len(times_unix):
        raise ValueError(f"obs_ts not in API response for {case['variable']}")

    start_idx = max(0, end_idx - case["window_hours"] + 1)
    window_vals = [v for v in vals[start_idx : end_idx + 1] if v is not None]
    if not window_vals:
        raise ValueError(f"all-null window for {case['variable']} at ({case['lat']},{case['lon']})")

    raw = sum(window_vals) if case["agg"] == "sum" else sum(window_vals) / len(window_vals)
    return round(raw, case["api_decimals"])


def _regenerate() -> dict[str, float]:
    print("\n[live] fetching ground truth from Open-Meteo archive API...")
    result: dict[str, float] = {}
    for key, case in LIVE_CASES.items():
        print(f"  {key} ... ", end="", flush=True)
        try:
            val = _api_window_value(case)
            result[key] = val
            print(f"{val}")
        except Exception as exc:
            print(f"FAILED ({exc})")
    return result


# ---------------------------------------------------------------------------
# Pipeline: compute windowed value via the real S3 .om file
# ---------------------------------------------------------------------------

# Epoch of first chunk file for copernicus_era5 (chunk_904 = 2021-12-23 00:00 UTC).
# Observations at or after this timestamp live in chunk_NNN.om files, not year files.
_CHUNK_ERA_START = 1640217600  # 2021-12-23 00:00 UTC

_meta_cache: dict[str, dict] = {}


def _get_meta(model: str) -> dict:
    if model not in _meta_cache:
        uri = f"s3://openmeteo/data/{model}/static/meta.json"
        with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
            _meta_cache[model] = json.loads(fh.read())
    return _meta_cache[model]


def _pipeline_value(case: dict[str, Any]) -> float:
    obs_ts = case["obs_ts"]
    step   = case["grid_step"]
    model  = case["model"]
    variable = case["variable"]
    mode   = "lat_asc_lon_pm180"

    ny = int(round(180.0 / step)) + 1
    nx = int(round(360.0 / step)) + 1
    lat_idx, lon_idx = grid_indices(case["lat"], case["lon"], ny, nx, mode, step)

    if obs_ts < _CHUNK_ERA_START:
        # Year file — Jan 1 aligned
        year = datetime.fromtimestamp(obs_ts, UTC).year
        uri = f"s3://openmeteo/data/{model}/{variable}/year_{year}.om"
        year_start = int(datetime(year, 1, 1, tzinfo=UTC).timestamp())
        time_idx = int((obs_ts - year_start) // 3600)
    else:
        # Chunk file — epoch-aligned: chunk_N starts at N * chunk_time_len * resolution
        meta = _get_meta(model)
        chunk_time_len = int(meta["chunk_time_length"])
        resolution = float(meta.get("temporal_resolution_seconds", 3600))
        chunk_num = int(obs_ts // (chunk_time_len * resolution))
        uri = f"s3://openmeteo/data/{model}/{variable}/chunk_{chunk_num}.om"
        chunk_start = chunk_num * chunk_time_len * resolution
        time_idx = int((obs_ts - chunk_start) // resolution)

    with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
        reader = OmFileReader(fh)
        file_ny, file_nx, _ = reader.shape
        li = min(lat_idx, file_ny - 1)
        lo = min(lon_idx, file_nx - 1)
        series = np.asarray(reader[li, lo, :], dtype=np.float64)

    steps = window_steps(3600.0, (case["window_hours"],))
    sums, counts = window_stats_batch(series, np.array([time_idx]), steps)
    cnt = int(counts[case["window_hours"]][0])
    if cnt == 0:
        return float("nan")
    s = float(sums[case["window_hours"]][0])
    raw = s if case["agg"] == "sum" else s / cnt
    return round(raw, case["api_decimals"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(LIVE_CASES))
def test_pipeline_matches_api(
    key: str,
    live_expected: dict[str, float],
) -> None:
    case     = LIVE_CASES[key]
    expected = live_expected.get(key)
    if expected is None:
        pytest.skip(f"no expected value stored for {key!r}")

    result = _pipeline_value(case)

    assert not np.isnan(result), (
        f"{key}: pipeline returned NaN — likely wrong grid cell "
        f"(grid_step={case['grid_step']}, model={case['model']!r})"
    )
    assert result == expected, (
        f"{key}: pipeline={result}  api={expected}  "
        f"[model={case['model']!r}, grid_step={case['grid_step']}]"
    )
