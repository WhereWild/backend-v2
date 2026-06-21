# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fixture-grounded correctness tests for _full_build → _incremental_update continuity.

Each test:
  1. Patches util.temporal._open_chunk to return the fixture series for every cell.
  2. Calls bt._full_build at t0 with a realistic ERA5/GFS split.
  3. Calls bt._incremental_update at t1 (a few hours later, fixed ERA5 end → no
     quality swap; or advancing ERA5 end → quality swap fires).
  4. Asserts the raster cell value matches expected_window(fixture, t1_ts, ...).

This is the gap identified in the test suite: accumulate_raster is tested against
fixtures, but the sliding-window incremental math (drop-oldest, add-newest, ERA5
quality swap) was only covered by synthetic-data unit tests.  These tests use real
Open-Meteo values so any arithmetic error in the incremental path produces a
measurable residual.

Test layout:
  - TestIncrementalTemperatureAvg  — drop + add, no quality swap (ERA5-land 0.1°)
  - TestIncrementalPrecipitationSum — drop + add, no quality swap (ERA5 0.25°)
  - TestIncrementalCloudCoverAvg   — drop + add; this is the var showing > 100% in prod
  - TestIncrementalQualitySwap     — ERA5 end advances between runs
  - TestIncrementalCumulativeDrift — 10 sequential updates, checks accumulated error
"""
from __future__ import annotations

import numpy as np
import pytest

import scripts.build_temporal as bt
from tests.temporal.conftest import TEST_LOCATIONS, expected_window
from tests.temporal.rasters.conftest import FakeRasterReader, chunk_from_fixture
from util.temporal import (
    RASTER_GRIDS,
    ChunkIndex,
    grid_indices,
    load_raster_state,
)

pytestmark = pytest.mark.usefixtures("require_fixtures")

_LOC = {loc["name"]: loc for loc in TEST_LOCATIONS}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cidx(fixture: dict, model: str) -> ChunkIndex:
    """Wrap the full fixture time range as a single ChunkIndex for model."""
    idx, _ = chunk_from_fixture(fixture, model)
    return idx


def _cell_at(npy_path, lat: float, lon: float, model: str) -> float:
    g = RASTER_GRIDS[model]
    ny, nx = g.get("ny", 721), g.get("nx", 1440)
    li, lo = grid_indices(lat, lon, ny, nx, "lat_asc_lon_pm180", g["step"])
    return float(np.load(npy_path)[li, lo])


def _patch_open_chunk(monkeypatch, series: np.ndarray) -> None:
    """Patch _open_chunk so every model/variable/cell returns the fixture series."""
    monkeypatch.setattr(
        "util.temporal._open_chunk",
        lambda entry, model, variable, n_steps=-1: FakeRasterReader(series, model),
    )


def _run_full_then_incremental(
    *,
    fixture: dict,
    var_id: str,
    cfg: dict,
    window_h: int,
    window_label: str,
    t0_hour: int,
    t1_hour: int,
    era5_end_hour: int,      # simulated ERA5 lag; fixed between t0 and t1
    era5_end_hour_t1: int | None = None,  # if set, ERA5 end advances → quality swap fires
    loc_name: str,
    monkeypatch,
    tmp_path,
    tol: float = 0.01,
) -> None:
    """
    Full build at t0, incremental at t1, assert cell ≈ expected_window(t1).

    Both ERA5 and GFS are backed by the same fixture series (monkeypatched).
    Since FakeRasterReader broadcasts the same 1-D series to every cell and
    reproject_to_grid is a no-op for uniform fields, the raster cell value at
    the fixture location must equal the expected_window ground truth.
    """
    times = fixture["hourly"]["time_unix"]
    t0_ts = float(times[t0_hour])
    t1_ts = float(times[t1_hour])
    era5_end_ts_t0 = float(times[era5_end_hour])
    era5_end_ts_t1 = float(times[era5_end_hour_t1]) if era5_end_hour_t1 is not None else era5_end_ts_t0

    era5_model = cfg["era5_model"]
    era5_var = cfg.get("era5_var", var_id)
    gfs_var = cfg.get("gfs_var", var_id)

    era5_cidx_map = {era5_var: _cidx(fixture, era5_model)}
    gfs_cidx_map = {gfs_var: _cidx(fixture, "ncep_gfs013")}

    series = np.array(fixture["hourly"][era5_var], dtype=np.float64)
    _patch_open_chunk(monkeypatch, series)

    out_dir = str(tmp_path)

    # Phase 1 — full build at t0
    bt._full_build(
        var_id, cfg, window_h, window_label,
        now_ts=t0_ts,
        era5_end_ts=era5_end_ts_t0,
        gfs_end_ts=t0_ts,
        era5_cidx=era5_cidx_map,
        gfs_cidx=gfs_cidx_map,
        out_dir=out_dir,
    )

    sums, meta = load_raster_state(out_dir, var_id, window_label)
    assert sums is not None, "_full_build produced no state files"

    # Phase 2 — incremental at t1
    bt._incremental_update(
        var_id, cfg, window_h, window_label,
        sums={k: v.copy() for k, v in sums.items()},
        old_meta=meta,
        now_ts=t1_ts,
        era5_end_ts=era5_end_ts_t1,
        gfs_end_ts=t1_ts,
        era5_cidx=era5_cidx_map,
        gfs_cidx=gfs_cidx_map,
        out_dir=out_dir,
    )

    loc = _LOC[loc_name]
    npy_path = tmp_path / f"{var_id}_{window_label}.npy"
    cell = _cell_at(npy_path, loc["lat"], loc["lon"], era5_model)
    expected = expected_window(fixture, t1_ts, era5_var, window_h, cfg["agg"])

    assert expected is not None, (
        f"expected_window returned None for {loc_name} t1_hour={t1_hour} window={window_label}"
    )
    assert cell == pytest.approx(expected, abs=tol), (
        f"{var_id} {window_label} [{loc_name}]: "
        f"incremental={cell:.4f}  expected={expected:.4f}  diff={abs(cell - expected):.4f}\n"
        f"  t0={t0_hour}  t1={t1_hour}  era5_end={era5_end_hour}  delta={t1_hour - t0_hour}h"
    )


# ---------------------------------------------------------------------------
# temperature_2m — avg, ERA5-land 0.1°
#
# Window layout (24h, t0=600, era5_end=580):
#   w_start = hour 577; ERA5[577..580]=4h; GFS[580..600]=20h
#   Incremental at t1=610: drop ERA5[577..586], add GFS[601..610]
# ---------------------------------------------------------------------------

_TEMP_CFG = {
    "era5_model": "copernicus_era5_land",
    "era5_var": "temperature_2m",
    "gfs_var": "temperature_2m",
    "agg": "avg",
}


class TestIncrementalTemperatureAvg:
    def test_berlin_24h_advance_10h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=610, era5_end_hour=580,
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_berlin_168h_advance_24h(self, require_fixtures, monkeypatch, tmp_path):
        # 168h window, t0=700, era5_end=600: ERA5[533..600]=68h, GFS[600..700]=100h
        # incremental at t1=724: drop ERA5[533..556], add GFS[701..724]
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=168, window_label="7d",
            t0_hour=700, t1_hour=724, era5_end_hour=600,
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_sydney_24h_advance_10h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["sydney_early"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=610, era5_end_hour=580,
            loc_name="sydney", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_reykjavik_24h_advance_1h(self, require_fixtures, monkeypatch, tmp_path):
        # Minimal advance: single-hour step — the most common production cadence
        _run_full_then_incremental(
            fixture=require_fixtures["reykjavik_boundary"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=24, window_label="24h",
            t0_hour=400, t1_hour=401, era5_end_hour=380,
            loc_name="reykjavik", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )


# ---------------------------------------------------------------------------
# precipitation — sum, ERA5 0.25°
# ---------------------------------------------------------------------------

_PRECIP_CFG = {
    "era5_model": "copernicus_era5",
    "era5_var": "precipitation",
    "gfs_var": "precipitation",
    "agg": "sum",
}


class TestIncrementalPrecipitationSum:
    def test_berlin_24h_advance_10h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="precipitation", cfg=_PRECIP_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=610, era5_end_hour=580,
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_reykjavik_168h_advance_24h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["reykjavik_boundary"],
            var_id="precipitation", cfg=_PRECIP_CFG,
            window_h=168, window_label="7d",
            t0_hour=500, t1_hour=524, era5_end_hour=400,
            loc_name="reykjavik", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_nairobi_72h_advance_12h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["nairobi_early"],
            var_id="precipitation", cfg=_PRECIP_CFG,
            window_h=72, window_label="3d",
            t0_hour=500, t1_hour=512, era5_end_hour=450,
            loc_name="nairobi", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )


# ---------------------------------------------------------------------------
# cloud_cover — avg, ERA5 0.25°
#
# This is the variable showing physically impossible values (> 100%) in
# production.  If the incremental math double-counts or fails to drop hours
# correctly, the avg will exceed 100.
# ---------------------------------------------------------------------------

_CLOUD_CFG = {
    "era5_model": "copernicus_era5",
    "era5_var": "cloud_cover",
    "gfs_var": "cloud_cover",
    "agg": "avg",
}


class TestIncrementalCloudCoverAvg:
    def test_london_24h_advance_10h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["london_early"],
            var_id="cloud_cover", cfg=_CLOUD_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=610, era5_end_hour=580,
            loc_name="london", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_london_168h_advance_48h(self, require_fixtures, monkeypatch, tmp_path):
        _run_full_then_incremental(
            fixture=require_fixtures["london_early"],
            var_id="cloud_cover", cfg=_CLOUD_CFG,
            window_h=168, window_label="7d",
            t0_hour=700, t1_hour=748, era5_end_hour=600,
            loc_name="london", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_london_24h_result_within_valid_range(self, require_fixtures, monkeypatch, tmp_path):
        """Cloud cover avg must stay in [0, 100] after incremental update."""
        fixture = require_fixtures["london_early"]
        times = fixture["hourly"]["time_unix"]

        series = np.array(fixture["hourly"]["cloud_cover"], dtype=np.float64)
        _patch_open_chunk(monkeypatch, series)

        cidx_map = {"cloud_cover": _cidx(fixture, "copernicus_era5")}
        gfs_cidx_map = {"cloud_cover": _cidx(fixture, "ncep_gfs013")}
        out_dir = str(tmp_path)

        bt._full_build(
            "cloud_cover", _CLOUD_CFG, 24, "24h",
            now_ts=float(times[600]),
            era5_end_ts=float(times[580]),
            gfs_end_ts=float(times[600]),
            era5_cidx=cidx_map,
            gfs_cidx=gfs_cidx_map,
            out_dir=out_dir,
        )
        for step in range(1, 25):
            sums, meta = load_raster_state(out_dir, "cloud_cover", "24h")
            bt._incremental_update(
                "cloud_cover", _CLOUD_CFG, 24, "24h",
                sums={k: v.copy() for k, v in sums.items()},
                old_meta=meta,
                now_ts=float(times[600 + step]),
                era5_end_ts=float(times[580]),
                gfs_end_ts=float(times[600 + step]),
                era5_cidx=cidx_map,
                gfs_cidx=gfs_cidx_map,
                out_dir=out_dir,
            )

        loc = _LOC["london"]
        npy_path = tmp_path / "cloud_cover_24h.npy"
        cell = _cell_at(npy_path, loc["lat"], loc["lon"], "copernicus_era5")
        assert 0.0 <= cell <= 100.0, f"cloud cover out of range after 24 updates: {cell:.2f}%"


# ---------------------------------------------------------------------------
# ERA5 quality swap: ERA5 end advances between full build and incremental
#
# The swap path adds ERA5 sums and subtracts GFS sums for the overlap period.
# Since both are backed by the same fixture series here, the net change is
# zero (ERA5 value == GFS value for every cell) and the final result must
# still match expected_window.
# ---------------------------------------------------------------------------

class TestIncrementalQualitySwap:
    def test_precipitation_swap_5h(self, require_fixtures, monkeypatch, tmp_path):
        """ERA5 advances 5h between full build (t0) and incremental (t1)."""
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="precipitation", cfg=_PRECIP_CFG,
            window_h=168, window_label="7d",
            t0_hour=700, t1_hour=710,
            era5_end_hour=600,
            era5_end_hour_t1=605,  # ERA5 advances 5h → quality swap fires
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_temperature_swap_1h(self, require_fixtures, monkeypatch, tmp_path):
        """ERA5 advances exactly 1h — the single-step swap case.

        At t0=600 era5_end=580: w_start=577, ERA5[577..580]=4h, GFS[581..600]=20h.
        At t1=601 era5_end=581: swap fires for [581..581] (1 step) replacing GFS[581]
        with ERA5[581]; then drop ERA5[577], add GFS[602].
        Since fixture data is the same for both sources, net swap change is zero and
        the final value must equal expected_window(t1=601, 24h).
        """
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=601,
            era5_end_hour=580,
            era5_end_hour_t1=581,  # ERA5 advances exactly 1h → tests swap_end >= swap_start
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_temperature_swap_advances_past_window_start(self, require_fixtures, monkeypatch, tmp_path):
        """ERA5 end crosses the window start boundary during the swap."""
        # 24h window at t0=600: w_start=577, era5_end_t0=570 (7h before w_start)
        # era5_end_t1=585 (8h into the window — swap covers [577..585])
        _run_full_then_incremental(
            fixture=require_fixtures["berlin_early"],
            var_id="temperature_2m", cfg=_TEMP_CFG,
            window_h=24, window_label="24h",
            t0_hour=600, t1_hour=610,
            era5_end_hour=570,
            era5_end_hour_t1=585,
            loc_name="berlin", monkeypatch=monkeypatch, tmp_path=tmp_path,
        )

    def test_cloud_cover_swap_no_overflow(self, require_fixtures, monkeypatch, tmp_path):
        """Quality swap must not push cloud cover avg above 100."""
        fixture = require_fixtures["london_early"]
        times = fixture["hourly"]["time_unix"]

        series = np.array(fixture["hourly"]["cloud_cover"], dtype=np.float64)
        _patch_open_chunk(monkeypatch, series)

        cidx_era5 = {"cloud_cover": _cidx(fixture, "copernicus_era5")}
        cidx_gfs = {"cloud_cover": _cidx(fixture, "ncep_gfs013")}
        out_dir = str(tmp_path)

        bt._full_build(
            "cloud_cover", _CLOUD_CFG, 168, "7d",
            now_ts=float(times[700]),
            era5_end_ts=float(times[600]),
            gfs_end_ts=float(times[700]),
            era5_cidx=cidx_era5,
            gfs_cidx=cidx_gfs,
            out_dir=out_dir,
        )

        sums, meta = load_raster_state(out_dir, "cloud_cover", "7d")
        bt._incremental_update(
            "cloud_cover", _CLOUD_CFG, 168, "7d",
            sums={k: v.copy() for k, v in sums.items()},
            old_meta=meta,
            now_ts=float(times[710]),
            era5_end_ts=float(times[620]),  # ERA5 advanced 20h → large swap
            gfs_end_ts=float(times[710]),
            era5_cidx=cidx_era5,
            gfs_cidx=cidx_gfs,
            out_dir=out_dir,
        )

        loc = _LOC["london"]
        npy_path = tmp_path / "cloud_cover_7d.npy"
        cell = _cell_at(npy_path, loc["lat"], loc["lon"], "copernicus_era5")
        expected = expected_window(fixture, float(times[710]), "cloud_cover", 168, "avg")
        assert expected is not None
        assert 0.0 <= cell <= 100.0, f"cloud cover overflow after swap: {cell:.2f}%"
        assert cell == pytest.approx(expected, abs=0.01), (
            f"quality swap: cell={cell:.4f} expected={expected:.4f}"
        )


# ---------------------------------------------------------------------------
# Cumulative drift: N sequential hourly updates
#
# Runs 10 consecutive hourly incremental updates from a cold full build.
# Checks that floating-point accumulation error stays below the tolerance
# used throughout the fixture test suite (abs=0.01).
# ---------------------------------------------------------------------------

class TestIncrementalCumulativeDrift:
    def test_temperature_10_hourly_updates(self, require_fixtures, monkeypatch, tmp_path):
        fixture = require_fixtures["berlin_early"]
        times = fixture["hourly"]["time_unix"]
        start_hour = 600
        n_steps = 10
        era5_end_ts = float(times[580])

        series = np.array(fixture["hourly"]["temperature_2m"], dtype=np.float64)
        _patch_open_chunk(monkeypatch, series)

        cidx_era5 = {"temperature_2m": _cidx(fixture, "copernicus_era5_land")}
        cidx_gfs = {"temperature_2m": _cidx(fixture, "ncep_gfs013")}
        out_dir = str(tmp_path)

        bt._full_build(
            "temperature_2m", _TEMP_CFG, 24, "24h",
            now_ts=float(times[start_hour]),
            era5_end_ts=era5_end_ts,
            gfs_end_ts=float(times[start_hour]),
            era5_cidx=cidx_era5,
            gfs_cidx=cidx_gfs,
            out_dir=out_dir,
        )

        for step in range(1, n_steps + 1):
            sums, meta = load_raster_state(out_dir, "temperature_2m", "24h")
            assert sums is not None, f"state missing at step {step}"
            bt._incremental_update(
                "temperature_2m", _TEMP_CFG, 24, "24h",
                sums={k: v.copy() for k, v in sums.items()},
                old_meta=meta,
                now_ts=float(times[start_hour + step]),
                era5_end_ts=era5_end_ts,
                gfs_end_ts=float(times[start_hour + step]),
                era5_cidx=cidx_era5,
                gfs_cidx=cidx_gfs,
                out_dir=out_dir,
            )

        final_hour = start_hour + n_steps
        loc = _LOC["berlin"]
        npy_path = tmp_path / "temperature_2m_24h.npy"
        cell = _cell_at(npy_path, loc["lat"], loc["lon"], "copernicus_era5_land")
        expected = expected_window(fixture, float(times[final_hour]), "temperature_2m", 24, "avg")
        assert expected is not None
        assert cell == pytest.approx(expected, abs=0.01), (
            f"after {n_steps} hourly updates: cell={cell:.4f} expected={expected:.4f} "
            f"drift={abs(cell - expected):.4f}"
        )

    def test_cloud_cover_10_hourly_updates(self, require_fixtures, monkeypatch, tmp_path):
        fixture = require_fixtures["london_early"]
        times = fixture["hourly"]["time_unix"]
        start_hour = 600
        n_steps = 10
        era5_end_ts = float(times[580])

        series = np.array(fixture["hourly"]["cloud_cover"], dtype=np.float64)
        _patch_open_chunk(monkeypatch, series)

        cidx_era5 = {"cloud_cover": _cidx(fixture, "copernicus_era5")}
        cidx_gfs = {"cloud_cover": _cidx(fixture, "ncep_gfs013")}
        out_dir = str(tmp_path)

        bt._full_build(
            "cloud_cover", _CLOUD_CFG, 24, "24h",
            now_ts=float(times[start_hour]),
            era5_end_ts=era5_end_ts,
            gfs_end_ts=float(times[start_hour]),
            era5_cidx=cidx_era5,
            gfs_cidx=cidx_gfs,
            out_dir=out_dir,
        )

        for step in range(1, n_steps + 1):
            sums, meta = load_raster_state(out_dir, "cloud_cover", "24h")
            assert sums is not None
            bt._incremental_update(
                "cloud_cover", _CLOUD_CFG, 24, "24h",
                sums={k: v.copy() for k, v in sums.items()},
                old_meta=meta,
                now_ts=float(times[start_hour + step]),
                era5_end_ts=era5_end_ts,
                gfs_end_ts=float(times[start_hour + step]),
                era5_cidx=cidx_era5,
                gfs_cidx=cidx_gfs,
                out_dir=out_dir,
            )

        final_hour = start_hour + n_steps
        loc = _LOC["london"]
        npy_path = tmp_path / "cloud_cover_24h.npy"
        cell = _cell_at(npy_path, loc["lat"], loc["lon"], "copernicus_era5")
        expected = expected_window(fixture, float(times[final_hour]), "cloud_cover", 24, "avg")
        assert expected is not None
        assert 0.0 <= cell <= 100.0, f"cloud cover out of range: {cell:.2f}%"
        assert cell == pytest.approx(expected, abs=0.01), (
            f"after {n_steps} hourly updates: cell={cell:.4f} expected={expected:.4f}"
        )
