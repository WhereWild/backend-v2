# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import httpx
import numpy as np
import pytest

import scripts.build_temporal as bt
from util.temporal import (
    RASTER_GRIDS,
    accumulate_raster,
    build_chunk_index,
    grid_indices,
    load_raster_state,
    save_raster_state,
    set_raster_chunk_cache,
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
    dict(key="berlin_temp_24h",     variable="temperature_2m",            model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_temp_168h",    variable="temperature_2m",            model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=168, agg="avg", api_decimals=1),
    dict(key="berlin_precip_24h",   variable="precipitation",             model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="sum", api_decimals=1),
    dict(key="berlin_precip_168h",  variable="precipitation",             model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=168, agg="sum", api_decimals=1),
    dict(key="sydney_precip_24h",   variable="precipitation",             model=_ERA5_MODEL,
         lat=-33.5, lon=151.5, window_hours=24,  agg="sum", api_decimals=1),
    dict(key="nairobi_temp_72h",    variable="temperature_2m",            model=_ERA5_MODEL,
         lat=-1.5,  lon=37.0,  window_hours=72,  agg="avg", api_decimals=1),
    dict(key="berlin_cloud_24h",    variable="cloud_cover",               model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_snowfall_24h", variable="snowfall_water_equivalent",  model=_ERA5_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="sum", api_decimals=1),
    # --- ERA5-Land 0.1° ---
    dict(key="berlin_dewpoint_24h",   variable="dew_point_2m",              model=_ERA5_LAND_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_soiltemp_24h",   variable="soil_temperature_0_to_7cm", model=_ERA5_LAND_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=1),
    dict(key="berlin_soiltemp_168h",  variable="soil_temperature_0_to_7cm", model=_ERA5_LAND_MODEL,
         lat=52.5,  lon=13.5,  window_hours=168, agg="avg", api_decimals=1),
    dict(key="berlin_soilmoist_24h",  variable="soil_moisture_0_to_7cm",    model=_ERA5_LAND_MODEL,
         lat=52.5,  lon=13.5,  window_hours=24,  agg="avg", api_decimals=3),
    dict(key="nairobi_soilmoist_72h", variable="soil_moisture_0_to_7cm",    model=_ERA5_LAND_MODEL,
         lat=-1.5,  lon=37.0,  window_hours=72,  agg="avg", api_decimals=3),
    dict(key="nairobi_soiltemp_24h",  variable="soil_temperature_0_to_7cm", model=_ERA5_LAND_MODEL,
         lat=-1.5,  lon=37.0,  window_hours=24,  agg="avg", api_decimals=1),
]


# ---------------------------------------------------------------------------
# Gate: skip unless --live
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _live_gate(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live"):
        pytest.skip("live S3 raster tests skipped — use: pt --temporal")


@pytest.fixture(scope="session")
def live_chunk_cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Session-scoped local chunk cache so S3 chunks are downloaded once per session."""
    cache = str(tmp_path_factory.mktemp("chunk_cache"))
    set_raster_chunk_cache(cache)
    return cache


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

def _pipeline_value(case: dict[str, Any], obs_ts: int) -> float:
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
    sum_grid, n_steps = accumulate_raster(model, variable, start_ts, end_ts, chunk_index)

    if n_steps == 0:
        return float("nan")

    cell_sum = float(sum_grid[lat_idx, lon_idx])
    raw = cell_sum if agg == "sum" else cell_sum / n_steps
    return round(raw, api_decimals)




# ---------------------------------------------------------------------------
# Incremental correctness: 1h cloud_cover with real S3 GFS data
# ---------------------------------------------------------------------------

def test_cloud_cover_24h_incremental_bounds(
    live_raster_expected: dict[str, Any],
    live_chunk_cache: str,
    tmp_path: Path,
) -> None:
    """Seed 24h cloud_cover state from real ERA5 data, slide the window twice, assert bounds.

    Uses the same frozen obs_ts as the API-matching tests (60 days before ERA5 end)
    so the year file chunk is already warm and the S3 reads are fast.
    Validates that drop-oldest / add-newest keeps n_total constant and output
    stays within 0–100 % across two 1h advances.
    """
    obs_ts_raw = live_raster_expected.get("obs_ts")
    if obs_ts_raw is None:
        pytest.skip("obs_ts missing — regenerate with --regenerate-live")

    # obs_ts is ~60 days before era5_end; use obs_ts-26h as "now" so there's
    # room to advance 2 more hours fully inside ERA5 territory.
    obs_ts = float(int(obs_ts_raw) // 3600 * 3600) - 26 * 3600
    era5_end = int(_model_end_ts(_ERA5_MODEL)) // 3600 * 3600
    # Pretend ERA5 ends 2h after obs_ts so each slide adds exactly 1 ERA5 hour.
    fake_era5_end = obs_ts + 2 * 3600

    era5_cidx_obj = build_chunk_index("copernicus_era5", "cloud_cover")
    era5_cidx = {"cloud_cover": era5_cidx_obj}
    gfs_cidx_obj = build_chunk_index("ncep_gfs013", "cloud_cover")
    gfs_cidx = {"cloud_cover": gfs_cidx_obj}
    cfg = bt.VAR_CONFIGS["cloud_cover"]
    out_dir = str(tmp_path)

    # Seed: accumulate 24h ERA5 window ending at obs_ts
    w_start = obs_ts - 23 * 3600
    era5_sum, n_era5 = accumulate_raster("copernicus_era5", "cloud_cover",
                                          w_start, obs_ts, era5_cidx_obj)
    sums: dict[str, Any] = {
        "era5_cloud_cover": era5_sum,
        "gfs_cloud_cover": np.zeros_like(era5_sum),
        "gfs_forecast_cloud_cover": np.zeros_like(era5_sum),
    }
    meta: dict[str, Any] = {
        "var_id": "cloud_cover",
        "window_h": 24,
        "window_label": "24h",
        "era5_window_start_ts": w_start,
        "era5_end_ts": fake_era5_end,
        "gfs_start_ts": fake_era5_end + 3600,
        "gfs_end_ts": obs_ts,   # gfs_end < gfs_start → 0 GFS steps in seed
        "gfs_cycle_init_ts": obs_ts,
        "n_era5": n_era5,
        "n_gfs": 0,
        "n_gfs_stable": 0,
        "n_gfs_forecast": 0,
        "built_at": datetime.fromtimestamp(obs_ts, UTC).isoformat(),
    }
    save_raster_state(out_dir, "cloud_cover", "24h", "avg", sums, meta)

    for step in range(2):
        now_ts = obs_ts + (step + 1) * 3600
        loaded_sums, loaded_meta = load_raster_state(out_dir, "cloud_cover", "24h")
        bt._incremental_update(
            "cloud_cover", cfg, 24, "24h",
            sums={k: v.copy() for k, v in loaded_sums.items()},
            old_meta=loaded_meta,
            now_ts=now_ts,
            era5_end_ts=fake_era5_end,
            gfs_end_ts=obs_ts,   # frozen — drops oldest ERA5, adds no GFS
            era5_cidx=era5_cidx,
            gfs_cidx=gfs_cidx,
            out_dir=out_dir,
        )

        result = np.load(tmp_path / "cloud_cover_24h.npy")
        lo_bad = int((result < -1.0).sum())
        hi_bad = int((result > 101.0).sum())
        assert lo_bad == 0, f"step {step+1}: {lo_bad} pixels below -1 % (min={result.min():.2f})"
        assert hi_bad == 0, f"step {step+1}: {hi_bad} pixels above 101 % (max={result.max():.2f})"


# ---------------------------------------------------------------------------
# Incremental correctness: 1h cloud_cover, stale build (11h gap)
# ---------------------------------------------------------------------------

def test_cloud_cover_1h_stale_incremental(
    live_raster_expected: dict[str, Any],
    live_chunk_cache: str,
    tmp_path: Path,
) -> None:
    """Seed a 1h cloud_cover window then advance 11h in one stale build.

    This is the exact scenario that caused persistent bad n_gfs counts in
    production: the build falls behind by many hours (restart, sleep, etc.)
    and the catch-up incremental adds all hours since old_gfs_end instead of
    only the hours inside the new window.

    Key assertion: after the stale advance, meta["n_gfs"] == 1 (not 11).
    A 1h window should always hold exactly 1 GFS step.
    """
    obs_ts_raw = live_raster_expected.get("obs_ts")
    if obs_ts_raw is None:
        pytest.skip("obs_ts missing — regenerate with --regenerate-live")

    obs_ts = float(int(obs_ts_raw) // 3600 * 3600)
    era5_cidx_obj = build_chunk_index("copernicus_era5", "cloud_cover")
    era5_cidx = {"cloud_cover": era5_cidx_obj}
    gfs_cidx_obj = build_chunk_index("ncep_gfs013", "cloud_cover")
    gfs_cidx = {"cloud_cover": gfs_cidx_obj}
    cfg = bt.VAR_CONFIGS["cloud_cover"]
    out_dir = str(tmp_path)

    # Seed: 1h window fully inside ERA5 territory, 0 GFS steps.
    # old_gfs_end = obs_ts; gfs_start > gfs_end means no GFS in window.
    era5_sum, n_era5 = accumulate_raster(
        "copernicus_era5", "cloud_cover", obs_ts, obs_ts, era5_cidx_obj
    )
    sums: dict[str, Any] = {
        "era5_cloud_cover": era5_sum,
        "gfs_cloud_cover": np.zeros_like(era5_sum),
        "gfs_forecast_cloud_cover": np.zeros_like(era5_sum),
    }
    meta: dict[str, Any] = {
        "var_id": "cloud_cover",
        "window_h": 1,
        "window_label": "1h",
        "era5_window_start_ts": obs_ts,
        "era5_end_ts": obs_ts,
        "gfs_start_ts": obs_ts + 3600,   # no GFS in seed
        "gfs_end_ts": obs_ts,
        "gfs_cycle_init_ts": obs_ts,
        "n_era5": n_era5,
        "n_gfs": 0,
        "n_gfs_stable": 0,
        "n_gfs_forecast": 0,
        "built_at": datetime.fromtimestamp(obs_ts, UTC).isoformat(),
    }
    save_raster_state(out_dir, "cloud_cover", "1h", "avg", sums, meta)

    # Stale build: advance 11h in one shot.
    # new_w_start = obs_ts + 11h (1h window), so only obs_ts+11h should be added.
    now_ts = obs_ts + 11 * 3600
    gfs_end_ts = obs_ts + 11 * 3600
    loaded_sums, loaded_meta = load_raster_state(out_dir, "cloud_cover", "1h")
    bt._incremental_update(
        "cloud_cover", cfg, 1, "1h",
        sums={k: v.copy() for k, v in loaded_sums.items()},
        old_meta=loaded_meta,
        now_ts=now_ts,
        era5_end_ts=obs_ts,       # ERA5 hasn't advanced
        gfs_end_ts=gfs_end_ts,
        era5_cidx=era5_cidx,
        gfs_cidx=gfs_cidx,
        out_dir=out_dir,
    )

    _, out_meta = load_raster_state(out_dir, "cloud_cover", "1h")
    n_gfs_actual = int(out_meta["n_gfs"])
    assert n_gfs_actual == 1, (
        f"Expected n_gfs=1 after 11h-stale 1h-window build, got {n_gfs_actual}. "
        f"The stale add step is accumulating hours outside the window."
    )

    result = np.load(tmp_path / "cloud_cover_1h.npy")
    lo_bad = int((result < -1.0).sum())
    hi_bad = int((result > 101.0).sum())
    assert lo_bad == 0, f"stale 1h build: {lo_bad} pixels below -1 % (min={result.min():.2f})"
    assert hi_bad == 0, f"stale 1h build: {hi_bad} pixels above 101 % (max={result.max():.2f})"


# ---------------------------------------------------------------------------
# GFS cycle rederive: stale forecast sums discarded, correct values rebuilt
# ---------------------------------------------------------------------------

def test_gfs_cycle_rederive_discards_stale_forecast(
    live_raster_expected: dict[str, Any],
    live_chunk_cache: str,
    tmp_path: Path,
) -> None:
    """Seed state with corrupt gfs_forecast_cloud_cover, run _gfs_rederive, assert clean output.

    Simulates the core cycle-rederive scenario:
    - Old cycle init at T-6h (old_cycle_init_ts), new cycle at T (new_cycle_init_ts)
    - gfs_forecast_cloud_cover is seeded with garbage (very large values) to represent
      stale data from the previous GFS cycle
    - After rederive, the output must match direct accumulation from the current chunk
    - gfs_* (stable) sums are preserved; gfs_forecast_* are rebuilt
    """
    obs_ts_raw = live_raster_expected.get("obs_ts")
    if obs_ts_raw is None:
        pytest.skip("obs_ts missing — regenerate with --regenerate-live")

    # Use a timestamp well inside ERA5 territory for stable chunk reads.
    # obs_ts is ~60 days before era5_end, so chunk is in the year file.
    obs_ts = float(int(obs_ts_raw) // 3600 * 3600)

    # Floor to nearest 6h boundary → simulates a real GFS cycle init
    old_cycle_init_ts = float((int(obs_ts) // (6 * 3600)) * (6 * 3600))
    new_cycle_init_ts = old_cycle_init_ts + 6 * 3600  # next cycle

    era5_cidx_obj = build_chunk_index("copernicus_era5", "cloud_cover")
    era5_cidx = {"cloud_cover": era5_cidx_obj}
    gfs_cidx_obj = build_chunk_index("ncep_gfs013", "cloud_cover")
    gfs_cidx = {"cloud_cover": gfs_cidx_obj}
    cfg = bt.VAR_CONFIGS["cloud_cover"]
    out_dir = str(tmp_path)

    window_h = 24
    w_start = obs_ts - (window_h - 1) * 3600

    # Accumulate real ERA5 stable sum for the ERA5 portion
    era5_sum, n_era5 = accumulate_raster("copernicus_era5", "cloud_cover",
                                          w_start, obs_ts, era5_cidx_obj)

    # Accumulate real GFS stable sum: [gfs_start, old_cycle_init_ts)
    gfs_start = obs_ts + 3600
    gfs_end_ts = old_cycle_init_ts + 6 * 3600  # 6h past old cycle = new cycle

    gfs_stable_sum = np.zeros_like(era5_sum, dtype=np.float64)
    n_gfs_stable = 0
    if gfs_start < old_cycle_init_ts:
        from util.temporal import RASTER_GRIDS, reproject_to_grid
        gfs_g = RASTER_GRIDS["ncep_gfs013"]
        era5_g = RASTER_GRIDS["copernicus_era5"]
        acc, n_gfs_stable = accumulate_raster("ncep_gfs013", "cloud_cover",
                                               gfs_start, old_cycle_init_ts - 3600, gfs_cidx_obj)
        gfs_stable_sum = reproject_to_grid(
            acc.astype(np.float32),
            gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
            era5_g["ny"], era5_g["nx"],
            era5_g["lat_min"], era5_g["lat_max"], era5_g["lon_min"], era5_g["lon_max"],
        ).astype(np.float64)

    # Corrupt gfs_forecast_cloud_cover with garbage — simulates stale old-cycle data
    corrupt_forecast = np.full_like(era5_sum, fill_value=9999.0, dtype=np.float64)

    sums: dict[str, Any] = {
        "era5_cloud_cover": era5_sum.copy(),
        "gfs_cloud_cover": gfs_stable_sum.copy(),
        "gfs_forecast_cloud_cover": corrupt_forecast,
    }
    meta: dict[str, Any] = {
        "var_id": "cloud_cover",
        "window_h": window_h,
        "window_label": "24h",
        "era5_window_start_ts": w_start,
        "era5_end_ts": obs_ts,
        "gfs_start_ts": gfs_start,
        "gfs_end_ts": gfs_end_ts,
        "gfs_cycle_init_ts": old_cycle_init_ts,
        "n_era5": n_era5,
        "n_gfs": n_gfs_stable,
        "n_gfs_stable": n_gfs_stable,
        "n_gfs_forecast": 0,
        "built_at": datetime.fromtimestamp(obs_ts, UTC).isoformat(),
    }
    save_raster_state(out_dir, "cloud_cover", "24h", "avg", sums, meta)

    # Run rederive with new_cycle_init_ts
    bt._gfs_rederive(
        "cloud_cover", cfg, window_h, "24h",
        existing_sums=sums,
        existing_meta=meta,
        now_ts=obs_ts,
        era5_end_ts=obs_ts,
        gfs_end_ts=gfs_end_ts,
        era5_cidx=era5_cidx,
        gfs_cidx=gfs_cidx,
        out_dir=out_dir,
        new_cycle_init_ts=new_cycle_init_ts,
    )

    result_sums, result_meta = load_raster_state(out_dir, "cloud_cover", "24h")

    # Corrupt values must be gone — no pixel should have 9999 residual
    if "gfs_forecast_cloud_cover" in result_sums:
        max_fc = float(np.max(np.abs(result_sums["gfs_forecast_cloud_cover"])))
        assert max_fc < 10000.0, f"gfs_forecast_cloud_cover still has corrupt values (max_abs={max_fc})"

    # ERA5 stable sum must be preserved exactly
    np.testing.assert_array_equal(
        result_sums["era5_cloud_cover"], era5_sum,
        err_msg="ERA5 sum was modified during rederive — ERA5 is immutable",
    )

    # GFS stable sum must include graduation: old_cycle_init_ts..new_cycle_init_ts
    # The stable sum should be >= original stable (graduation only adds)
    if n_gfs_stable > 0:
        stable_increased = np.any(result_sums["gfs_cloud_cover"] >= gfs_stable_sum)
        assert stable_increased, "GFS stable sum should have grown after graduation"

    # n_gfs_stable must be >= original (graduation transferred some forecast→stable)
    assert int(result_meta["n_gfs_stable"]) >= n_gfs_stable

    # Output raster must be within bounds
    result = np.load(tmp_path / "cloud_cover_24h.npy")
    lo_bad = int((result < -1.0).sum())
    hi_bad = int((result > 101.0).sum())
    assert lo_bad == 0, f"rederive output: {lo_bad} pixels below -1 % (min={result.min():.2f})"
    assert hi_bad == 0, f"rederive output: {hi_bad} pixels above 101 % (max={result.max():.2f})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [t["key"] for t in _CASE_TEMPLATES])
def test_raster_pipeline_matches_api(
    key: str,
    live_raster_expected: dict[str, Any],
) -> None:
    obs_ts = live_raster_expected.get("obs_ts")
    if obs_ts is None:
        pytest.skip("obs_ts missing in live_raster_fixtures.json — regenerate with --regenerate-live")

    expected = live_raster_expected.get(key)
    if expected is None:
        pytest.skip(f"no expected value for {key!r} — regenerate with --regenerate-live")

    case = next(t for t in _CASE_TEMPLATES if t["key"] == key)
    result = _pipeline_value(case, int(obs_ts))

    assert not np.isnan(result), (
        f"{key}: raster pipeline returned NaN "
        f"(model={case['model']!r}, variable={case['variable']!r})"
    )
    assert result == expected, (
        f"{key}: raster={result}  api={expected}  "
        f"[model={case['model']!r}, variable={case['variable']!r}, "
        f"window={case['window_hours']}h, agg={case['agg']!r}]"
    )
