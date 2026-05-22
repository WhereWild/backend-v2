"""Live end-to-end tests for the raster accumulation pipeline.

Validates that accumulate_raster() + grid_indices() produces the same cell
values as the Open-Meteo archive API, using real S3 .om files.

Run:
    pt --temporal                           # run live tests
    pt --temporal --regenerate-live         # re-fetch API ground truth first

Skipped from regular pt runs (no --temporal flag).

Test locations sit on exact 0.5° boundaries, falling on grid nodes for both
0.25° (ERA5) and 0.1° (ERA5-Land) grids simultaneously — no spatial
interpolation ambiguity.

Comparison: round(pipeline_cell_value, api_decimals) == round(api_value, api_decimals)
Both sides rounded to the same precision the API uses gives exact equality.

Dynamic timestamps: obs_ts is stored in live_raster_fixtures.json at
regeneration time as era5_end_ts - 30 * 24 * 3600 (30 days before current ERA5
end). Since ERA5 data for that past timestamp is frozen, there's no drift on
subsequent runs.
"""
from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import httpx
import numpy as np
import pytest

from util.temporal import (
    RASTER_GRIDS,
    accumulate_raster,
    build_chunk_index,
    grid_indices,
)

pytestmark = pytest.mark.live

_FIXTURES_PATH = Path(__file__).parent / "live_raster_fixtures.json"

# ---------------------------------------------------------------------------
# Case definitions — expected values stored in live_raster_fixtures.json.
# obs_ts is "dynamic": regeneration detects it from the current ERA5 end_time.
# ---------------------------------------------------------------------------

# ERA5 model S3 path used to detect current era5_end_ts at regeneration time.
_ERA5_MODEL = "copernicus_era5"
_ERA5_LAND_MODEL = "copernicus_era5_land"


def _model_end_ts(model: str) -> float:
    """Fetch data_end_time from S3 meta.json for the given model."""
    import json as _json
    uri = f"s3://openmeteo/data/{model}/static/meta.json"
    with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
        meta = _json.loads(fh.read())
    raw = meta.get("data_end_time")
    if raw is None:
        raise RuntimeError(f"Missing data_end_time in {model} meta.json")
    return float(raw)


# Each case: model, variable, lat, lon, window_hours, agg, api_decimals.
# obs_ts is filled in at regeneration time from era5_end_ts - 30 days.
_CASE_TEMPLATES: list[dict[str, Any]] = [
    # --- ERA5 0.25° ---
    dict(key="berlin_temp_24h",        variable="temperature_2m",           model=_ERA5_MODEL,      lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_temp_168h",       variable="temperature_2m",           model=_ERA5_MODEL,      lat=52.5,  lon=13.5,  window_hours=168, agg="avg", api_decimals=1),
    dict(key="berlin_precip_24h",      variable="precipitation",            model=_ERA5_MODEL,      lat=52.5,  lon=13.5,  window_hours=24,  agg="sum", api_decimals=1),
    dict(key="berlin_precip_168h",     variable="precipitation",            model=_ERA5_MODEL,      lat=52.5,  lon=13.5,  window_hours=168, agg="sum", api_decimals=1),
    dict(key="sydney_precip_24h",      variable="precipitation",            model=_ERA5_MODEL,      lat=-33.5, lon=151.5, window_hours=24,  agg="sum", api_decimals=1),
    dict(key="nairobi_temp_72h",       variable="temperature_2m",           model=_ERA5_MODEL,      lat=-1.5,  lon=37.0,  window_hours=72,  agg="avg", api_decimals=1),
    dict(key="berlin_cloud_24h",       variable="cloud_cover",              model=_ERA5_MODEL,      lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_snowfall_24h",    variable="snowfall_water_equivalent", model=_ERA5_MODEL,     lat=52.5,  lon=13.5,  window_hours=24,  agg="sum", api_decimals=1),
    # --- ERA5-Land 0.1° ---
    dict(key="berlin_dewpoint_24h",    variable="dew_point_2m",             model=_ERA5_LAND_MODEL, lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_soiltemp_24h",    variable="soil_temperature_0_to_7cm",model=_ERA5_LAND_MODEL, lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_soiltemp_168h",   variable="soil_temperature_0_to_7cm",model=_ERA5_LAND_MODEL, lat=52.5,  lon=13.5,  window_hours=168, agg="avg", api_decimals=1),
    dict(key="berlin_soilmoist_24h",   variable="soil_moisture_0_to_7cm",   model=_ERA5_LAND_MODEL, lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=3),
    dict(key="nairobi_soilmoist_72h",  variable="soil_moisture_0_to_7cm",   model=_ERA5_LAND_MODEL, lat=-1.5,  lon=37.0,  window_hours=72,  agg="avg", api_decimals=3),
    dict(key="nairobi_soiltemp_24h",   variable="soil_temperature_0_to_7cm",model=_ERA5_LAND_MODEL, lat=-1.5,  lon=37.0,  window_hours=24,  agg="avg", api_decimals=1),
]


# ---------------------------------------------------------------------------
# Gate: skip unless --live
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _live_gate(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live"):
        pytest.skip("live S3 raster tests skipped — use: pt --temporal")


# ---------------------------------------------------------------------------
# Expected-value fixture (loads or regenerates live_raster_fixtures.json)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_raster_expected(request: pytest.FixtureRequest) -> dict[str, Any]:
    if request.config.getoption("--regenerate-live") or not _FIXTURES_PATH.exists():
        data = _regenerate()
        _FIXTURES_PATH.write_text(json.dumps(data, indent=2))
        print(f"\n[live-raster] wrote {_FIXTURES_PATH}")
        return data
    return json.loads(_FIXTURES_PATH.read_text())


# ---------------------------------------------------------------------------
# Regeneration: fetch API values and store obs_ts + expected
# ---------------------------------------------------------------------------

def _regenerate() -> dict[str, Any]:
    """Build cases with dynamic obs_ts and fetch API ground truth."""
    era5_end = _model_end_ts(_ERA5_MODEL)
    era5_land_end = _model_end_ts(_ERA5_LAND_MODEL)
    # Use the earlier of ERA5 and ERA5-Land end times, then go back 60 days.
    # ERA5-Land S3 chunk files lag ~2 months behind the reported data_end_time;
    # 60 days gives a safe margin beyond partially-filled recent chunks.
    anchor = min(era5_end, era5_land_end)
    obs_ts = int(anchor) - 60 * 24 * 3600
    # Align to hour boundary
    obs_ts = (obs_ts // 3600) * 3600

    print(f"\n[live-raster] obs_ts = {obs_ts} ({datetime.fromtimestamp(obs_ts, UTC).isoformat()})")
    print(f"[live-raster] era5_end = {era5_end} ({datetime.fromtimestamp(era5_end, UTC).isoformat()})")
    print(f"[live-raster] era5_land_end = {era5_land_end} ({datetime.fromtimestamp(era5_land_end, UTC).isoformat()})")

    result: dict[str, Any] = {"obs_ts": obs_ts}
    for tmpl in _CASE_TEMPLATES:
        key = tmpl["key"]
        print(f"  {key} ... ", end="", flush=True)
        try:
            val = _api_window_value(tmpl, obs_ts)
            result[key] = val
            print(f"{val}")
        except Exception as exc:
            print(f"FAILED ({exc})")
    return result


def _api_window_value(case: dict[str, Any], obs_ts: int) -> float:
    """Compute the API's expected windowed value for one case."""
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


# ---------------------------------------------------------------------------
# Pipeline: call accumulate_raster, extract cell value
# ---------------------------------------------------------------------------

def _pipeline_value(case: dict[str, Any], obs_ts: int, tmp_dir: str) -> float:
    """Run accumulate_raster on real S3 data and return the cell value."""
    model = case["model"]
    variable = case["variable"]
    window_hours = case["window_hours"]
    lat = case["lat"]
    lon = case["lon"]
    agg = case["agg"]
    api_decimals = case["api_decimals"]

    g = RASTER_GRIDS[model]
    ny, nx = g["ny"], g["nx"]
    lat_idx, lon_idx = grid_indices(lat, lon, ny, nx, "lat_asc_lon_pm180", g["step"])

    obs_hour = (obs_ts // 3600) * 3600
    start_ts = float(obs_hour - (window_hours - 1) * 3600)
    end_ts   = float(obs_hour)

    chunk_index = build_chunk_index(model, variable)
    sum_grid, n_steps = accumulate_raster(model, variable, start_ts, end_ts, chunk_index, tmp_dir)

    if n_steps == 0:
        return float("nan")

    cell_sum = float(sum_grid[lat_idx, lon_idx])
    raw = cell_sum if agg == "sum" else cell_sum / n_steps
    return round(raw, api_decimals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [t["key"] for t in _CASE_TEMPLATES])
def test_raster_pipeline_matches_api(
    key: str,
    live_raster_expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    obs_ts = live_raster_expected.get("obs_ts")
    if obs_ts is None:
        pytest.skip("obs_ts missing in live_raster_fixtures.json — regenerate with --regenerate-live")

    expected = live_raster_expected.get(key)
    if expected is None:
        pytest.skip(f"no expected value for {key!r} — regenerate with --regenerate-live")

    case = next(t for t in _CASE_TEMPLATES if t["key"] == key)
    result = _pipeline_value(case, int(obs_ts), str(tmp_path))

    assert not np.isnan(result), (
        f"{key}: raster pipeline returned NaN "
        f"(model={case['model']!r}, variable={case['variable']!r})"
    )
    assert result == expected, (
        f"{key}: raster={result}  api={expected}  "
        f"[model={case['model']!r}, variable={case['variable']!r}, "
        f"window={case['window_hours']}h, agg={case['agg']!r}]"
    )
