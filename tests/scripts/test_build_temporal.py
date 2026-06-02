"""Tests for scripts/build_temporal.py — 100% line coverage target."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import scripts.build_temporal as bt
from util.temporal import RASTER_WC_CODES, ChunkIndex, ChunkRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_index(resolution: float = 3600.0) -> ChunkIndex:
    r = ChunkRange(
        chunk_num=0,
        start=0.0,
        end=100 * resolution,
        time_len=101,
        source="year",
    )
    return ChunkIndex(latest_end_time=100 * resolution, resolution=resolution, ranges=[r])


def _zeros4() -> np.ndarray:
    return np.zeros((4, 4), dtype=np.float64)


def _zeros4_f32() -> np.ndarray:
    return np.zeros((4, 4), dtype=np.float32)


def _zeros4_i32() -> np.ndarray:
    return np.zeros((4, 4), dtype=np.int32)


def _wc_counts() -> dict:
    return {c: _zeros4_i32() for c in RASTER_WC_CODES}


# Common mock for accumulate_raster: returns (zeros_array, 10)
def _fake_accumulate_raster(*args, **kwargs):
    return _zeros4(), 10


# Common mock for accumulate_raster_mode
def _fake_accumulate_raster_mode(*args, **kwargs):
    return _wc_counts()


# Common mock for reproject_to_grid
def _fake_reproject(*args, **kwargs):
    return _zeros4_f32()


# ---------------------------------------------------------------------------
# _era5_raw_vars
# ---------------------------------------------------------------------------

def test_era5_raw_vars_has_era5_var():
    cfg = {"era5_var": "temperature_2m", "agg": "avg"}
    assert bt._era5_raw_vars(cfg) == ["temperature_2m"]


def test_era5_raw_vars_uses_derived_needs():
    cfg = {"era5_derived_needs": ["temperature_2m", "dew_point_2m"], "agg": "avg"}
    assert bt._era5_raw_vars(cfg) == ["temperature_2m", "dew_point_2m"]


def test_era5_raw_vars_empty_when_neither():
    cfg = {"agg": "mode"}
    assert bt._era5_raw_vars(cfg) == []


# ---------------------------------------------------------------------------
# _gfs_raw_vars
# ---------------------------------------------------------------------------

def test_gfs_raw_vars_has_gfs_var():
    cfg = {"gfs_var": "temperature_2m", "agg": "avg"}
    assert bt._gfs_raw_vars(cfg) == ["temperature_2m"]


def test_gfs_raw_vars_uses_derived_needs():
    cfg = {"gfs_derived_needs": ["temperature_2m", "relative_humidity_2m"], "agg": "avg"}
    assert bt._gfs_raw_vars(cfg) == ["temperature_2m", "relative_humidity_2m"]


def test_gfs_raw_vars_empty_when_neither():
    cfg = {"agg": "mode"}
    assert bt._gfs_raw_vars(cfg) == []


# ---------------------------------------------------------------------------
# _derive_dew_point
# ---------------------------------------------------------------------------

def test_derive_dew_point_basic():
    t = np.array([20.0, 0.0], dtype=np.float32)
    rh = np.array([50.0, 100.0], dtype=np.float32)
    result = bt._derive_dew_point(t, rh)
    assert result.dtype == np.float32
    assert result.shape == (2,)
    # At 100% RH, dew point should equal temperature (approx)
    assert abs(result[1] - 0.0) < 0.1


def test_derive_dew_point_clips_rh():
    t = np.array([20.0], dtype=np.float32)
    # rh < 1 should be clipped to 1
    rh = np.array([0.0], dtype=np.float32)
    result = bt._derive_dew_point(t, rh)
    assert result.shape == (1,)
    assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# _gfs_grid
# ---------------------------------------------------------------------------

def test_gfs_grid_default():
    g = bt._gfs_grid()
    assert "lat_min" in g
    assert "lat_max" in g


def test_gfs_grid_explicit_model():
    g = bt._gfs_grid("ncep_gfs013")
    assert g is bt.RASTER_GRIDS["ncep_gfs013"]


# ---------------------------------------------------------------------------
# _reproject_gfs_to
# ---------------------------------------------------------------------------

def test_reproject_gfs_to(monkeypatch):
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    src = np.zeros((4, 4), dtype=np.float32)
    result = bt._reproject_gfs_to(src, "copernicus_era5")
    assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# _full_build — mode path
# ---------------------------------------------------------------------------

NOW_TS = 1_700_000_000.0
ERA5_END = NOW_TS - 5 * 3600
GFS_END = NOW_TS + 2 * 3600


def _mode_cfg():
    return bt.VAR_CONFIGS["weather_code_simple"].copy()


def _make_era5_cidx_mode(include_t: bool = True, include_cc: bool = True):
    cidx: dict[str, ChunkIndex] = {}
    if include_cc:
        cidx["cloud_cover"] = _make_chunk_index()
        cidx["precipitation"] = _make_chunk_index()
        cidx["snowfall_water_equivalent"] = _make_chunk_index()
    if include_t:
        cidx["_temperature_for_wc"] = _make_chunk_index()
    return cidx


def _make_gfs_cidx_mode():
    return {
        "cloud_cover": _make_chunk_index(),
        "precipitation": _make_chunk_index(),
        "snowfall_water_equivalent": _make_chunk_index(),
    }


def test_full_build_mode_no_cc_cidx_returns_early(monkeypatch):
    """Mode path with no cloud_cover in era5_cidx → early return, no save."""
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state", lambda *a, **kw: saved.append(1))
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {},  # empty era5_cidx → no cc_cidx
        {},
        "/tmp/test_out",
    )
    assert saved == []


def test_full_build_mode_with_t_cidx(monkeypatch):
    """Mode path: t_cidx present, t_n > 0 → temp_grid_025 computed."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(kw or a))
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        _make_era5_cidx_mode(include_t=True),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_mode_without_t_cidx(monkeypatch):
    """Mode path: t_cidx absent → temp_grid_025 = None."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_mode_t_cidx_n_zero(monkeypatch):
    """Mode path: t_cidx present but accumulate returns n=0 → temp_grid_025 stays None."""
    def _acc_zero(*a, **kw):
        return _zeros4(), 0

    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _acc_zero)
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        _make_era5_cidx_mode(include_t=True),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_mode_no_gfs_gap_fill(monkeypatch):
    """Mode path: gfs_mode_start >= gfs_end_ts → skip GFS gap-fill."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    # gfs_end_ts < era5_end_ts → gfs_mode_start == era5_end_ts > gfs_end_ts
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, ERA5_END - 3600,  # gfs_end < era5_end
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_mode_gfs_missing_vars(monkeypatch):
    """Mode path: GFS cidx missing some vars → no GFS gap-fill."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        _make_era5_cidx_mode(include_t=False),
        {"cloud_cover": _make_chunk_index()},  # missing precip + swe
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _full_build — scalar paths
# ---------------------------------------------------------------------------

def _temperature_cfg():
    return bt.VAR_CONFIGS["temperature_2m"].copy()


def _dew_point_cfg():
    return bt.VAR_CONFIGS["dew_point_2m"].copy()


def _vpd_cfg():
    return bt.VAR_CONFIGS["vapor_pressure_deficit"].copy()


def _precip_cfg():
    return bt.VAR_CONFIGS["precipitation"].copy()


def test_full_build_scalar_no_era5_data_returns_early(monkeypatch):
    """Scalar path: no ERA5 cidx → early return."""
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {},  # no ERA5 data
        {},
        "/tmp/test_out",
    )
    assert saved == []


def test_full_build_scalar_temperature_basic(monkeypatch):
    """Scalar path: generic var with GFS raw vars, gfs_start < gfs_end_ts."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_scalar_gfs_start_gte_gfs_end(monkeypatch):
    """Scalar path: gfs_start >= gfs_end_ts → skip GFS."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    # Make era5_end_ts > gfs_end_ts so gfs_start >= gfs_end_ts
    bt._full_build(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        NOW_TS, NOW_TS + 3600, NOW_TS,  # era5_end > gfs_end → gfs_start == era5_end > gfs_end
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_scalar_gfs_var_no_cidx(monkeypatch):
    """Scalar path: GFS var not in cidx → skip that gv."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {},  # no GFS cidx
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_dew_point_with_gfs(monkeypatch):
    """Scalar dew_point_2m path: t_cidx + rh_cidx present, n > 0."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "dew_point_2m", _dew_point_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {"dew_point_2m": _make_chunk_index()},
        {
            "temperature_2m": _make_chunk_index(),
            "relative_humidity_2m": _make_chunk_index(),
        },
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_dew_point_gfs_n_zero(monkeypatch):
    """Scalar dew_point_2m: n == 0 → no GFS dew point added."""
    def _acc_zero(*a, **kw):
        return _zeros4(), 0

    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _acc_zero)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    # Patch back to return 10 only for ERA5 call but 0 for GFS
    call_count = {"n": 0}

    def _acc_mixed(*a, **kw):
        call_count["n"] += 1
        if a[0] == "copernicus_era5_land":
            return _zeros4(), 10
        return _zeros4(), 0

    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _acc_mixed)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "dew_point_2m", _dew_point_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {"dew_point_2m": _make_chunk_index()},
        {
            "temperature_2m": _make_chunk_index(),
            "relative_humidity_2m": _make_chunk_index(),
        },
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_dew_point_no_gfs_cidx(monkeypatch):
    """Scalar dew_point_2m: missing GFS t_cidx → skip GFS branch."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "dew_point_2m", _dew_point_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {"dew_point_2m": _make_chunk_index()},
        {},  # no GFS
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_vpd_with_gfs(monkeypatch):
    """Scalar vapor_pressure_deficit: both t_cidx + rh_cidx present, n > 0."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "vapor_pressure_deficit", _vpd_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {
            "temperature_2m": _make_chunk_index(),
            "dew_point_2m": _make_chunk_index(),
        },
        {
            "temperature_2m": _make_chunk_index(),
            "relative_humidity_2m": _make_chunk_index(),
        },
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_vpd_gfs_n_zero(monkeypatch):
    """Scalar VPD: n == 0 → GFS branch skipped."""
    def _acc(model, *a, **kw):
        if model == "copernicus_era5_land":
            return _zeros4(), 10
        return _zeros4(), 0

    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _acc)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "vapor_pressure_deficit", _vpd_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {
            "temperature_2m": _make_chunk_index(),
            "dew_point_2m": _make_chunk_index(),
        },
        {
            "temperature_2m": _make_chunk_index(),
            "relative_humidity_2m": _make_chunk_index(),
        },
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_vpd_no_gfs_cidx(monkeypatch):
    """Scalar VPD: missing GFS cidx → skip."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._full_build(
        "vapor_pressure_deficit", _vpd_cfg(), 24, "24h",
        NOW_TS, ERA5_END, GFS_END,
        {
            "temperature_2m": _make_chunk_index(),
            "dew_point_2m": _make_chunk_index(),
        },
        {},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_full_build_with_suffix(monkeypatch):
    """suffix is forwarded to save_raster_state."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    suffixes = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda out, vid, wl, agg, sums, meta, suffix="": suffixes.append(suffix))
    bt._full_build(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
        suffix="__f001h",
    )
    assert suffixes == ["__f001h"]


# ---------------------------------------------------------------------------
# _incremental_update — mode path
# ---------------------------------------------------------------------------

def _old_meta_mode(now_ts: float = NOW_TS) -> dict:
    return {
        "era5_window_start_ts": now_ts - 48 * 3600,
        "era5_end_ts": now_ts - 10 * 3600,
        "gfs_start_ts": now_ts - 10 * 3600,
        "gfs_end_ts": now_ts - 2 * 3600,
        "n_era5": 38,
        "n_gfs": 8,
    }


def _sums_mode() -> dict:
    return {c: np.zeros((4, 4), dtype=np.int32) for c in RASTER_WC_CODES}


def test_incremental_mode_no_cc_cidx_returns_early(monkeypatch):
    """Mode incremental: no cloud_cover in era5_cidx → early return."""
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))
    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), _old_meta_mode(),
        NOW_TS, ERA5_END, GFS_END,
        {},  # no era5_cidx
        {},
        "/tmp/test_out",
    )
    assert saved == []


def test_incremental_mode_era5_quality_swap(monkeypatch):
    """Mode incremental: ERA5 quality swap path (era5_end_ts > old_era5_end + 3600)."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_mode()
    old_era5_end = float(old_meta["era5_end_ts"])
    # new era5_end is much later
    new_era5_end = old_era5_end + 5 * 3600

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), old_meta,
        NOW_TS, new_era5_end, GFS_END,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_mode_drop_oldest(monkeypatch):
    """Mode incremental: drop oldest hours path (new_w_start > old_w_start)."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_mode(NOW_TS - 48 * 3600)
    # Use a large window_h so new_w_start is later than old_w_start
    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 1, "1h",  # 1h window → new_w_start = now - 0h
        _sums_mode(), old_meta,
        NOW_TS, ERA5_END, GFS_END,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_mode_add_newest_gfs(monkeypatch):
    """Mode incremental: add newest GFS hours (gfs_end_ts > old_gfs_end)."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_mode()
    old_gfs_end = float(old_meta["gfs_end_ts"])
    new_gfs_end = old_gfs_end + 4 * 3600

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), old_meta,
        NOW_TS, ERA5_END, new_gfs_end,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_mode_add_newest_gfs_missing_vars(monkeypatch):
    """Mode incremental: GFS add branch returns None when vars missing."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_mode()
    old_gfs_end = float(old_meta["gfs_end_ts"])

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), old_meta,
        NOW_TS, ERA5_END, old_gfs_end + 4 * 3600,
        _make_era5_cidx_mode(include_t=False),
        {"cloud_cover": _make_chunk_index()},  # missing precip + swe → _mode_accumulate returns None
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_mode_swap_add_and_sub_none(monkeypatch):
    """Mode incremental swap: _mode_accumulate returns None for add/sub (missing GFS vars)."""
    # When GFS vars are missing, add/sub are None → no update to sums
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_mode()
    old_era5_end = float(old_meta["era5_end_ts"])
    old_gfs_end = float(old_meta["gfs_end_ts"])

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), old_meta,
        NOW_TS, old_era5_end + 5 * 3600, old_gfs_end + 2 * 3600,
        _make_era5_cidx_mode(include_t=False),
        {},  # no GFS → sub = None in swap
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _incremental_update — scalar path
# ---------------------------------------------------------------------------

def _old_meta_scalar(now_ts: float = NOW_TS) -> dict:
    return {
        "era5_window_start_ts": now_ts - 48 * 3600,
        "era5_end_ts": now_ts - 10 * 3600,
        "gfs_start_ts": now_ts - 10 * 3600,
        "gfs_end_ts": now_ts - 2 * 3600,
        "n_era5": 38,
        "n_gfs": 8,
    }


def _sums_temperature() -> dict:
    return {"era5_temperature_2m": _zeros4(), "gfs_temperature_2m": _zeros4()}


def test_incremental_scalar_era5_quality_swap(monkeypatch):
    """Scalar incremental: ERA5 quality swap (era5_end_ts > old_era5_end + 3600)."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_scalar()
    old_era5_end = float(old_meta["era5_end_ts"])
    new_era5_end = old_era5_end + 5 * 3600

    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        _sums_temperature(), old_meta,
        NOW_TS, new_era5_end, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_drop_oldest(monkeypatch):
    """Scalar incremental: drop oldest hours."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # Use 1h window so new_w_start = NOW_TS, old_w_start = much earlier
    old_meta = _old_meta_scalar(NOW_TS - 72 * 3600)
    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        _sums_temperature(), old_meta,
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_add_newest_gfs(monkeypatch):
    """Scalar incremental: add newest GFS hours."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_scalar()
    old_gfs_end = float(old_meta["gfs_end_ts"])
    new_gfs_end = old_gfs_end + 4 * 3600

    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        _sums_temperature(), old_meta,
        NOW_TS, ERA5_END, new_gfs_end,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_no_era5_cidx(monkeypatch):
    """Scalar incremental: resolution fallback when era5_cidx is empty."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    old_meta = _old_meta_scalar()
    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        _sums_temperature(), old_meta,
        NOW_TS, ERA5_END, GFS_END,
        {},  # no era5_cidx → resolution defaults to 3600
        {},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_gfs_var_missing_from_sums(monkeypatch):
    """Scalar drop: gfs key not in sums → skip subtraction gracefully."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # Use 1h window to trigger drop, but sums has no gfs key
    old_meta = _old_meta_scalar(NOW_TS - 72 * 3600)
    sums = {"era5_temperature_2m": _zeros4()}  # no gfs key

    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        sums, old_meta,
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_accum_returns_none_when_end_lte_start(monkeypatch):
    """_accum_era5 / _accum_gfs_reproj return None when end <= start."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # Force swap_end <= swap_start by making old_gfs_end < old_era5_end
    old_meta = {
        "era5_window_start_ts": NOW_TS - 48 * 3600,
        "era5_end_ts": NOW_TS - 10 * 3600,
        "gfs_start_ts": NOW_TS - 10 * 3600,
        "gfs_end_ts": NOW_TS - 12 * 3600,  # gfs_end < era5_end → swap_end = min(era5, old_gfs) < swap_start
        "n_era5": 38,
        "n_gfs": 0,
    }
    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 24, "24h",
        _sums_temperature(), old_meta,
        NOW_TS, NOW_TS - 5 * 3600, GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _build_forecast_aggregates
# ---------------------------------------------------------------------------

def _minimal_var_configs():
    return {
        "temperature_2m": bt.VAR_CONFIGS["temperature_2m"].copy(),
        "weather_code_simple": bt.VAR_CONFIGS["weather_code_simple"].copy(),
    }


def _minimal_windows():
    return [(1, "1h")]


def test_forecast_mode_uses_full_build_when_no_state(monkeypatch):
    """mode agg with no existing state → _full_build is called."""
    full_build_called = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_called.append(a[0]))
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (None, None))
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: None)

    bt._build_forecast_aggregates(
        {"weather_code_simple": bt.VAR_CONFIGS["weather_code_simple"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, GFS_END,
        {},
        {},
        "/tmp/test_out",
    )
    assert "weather_code_simple" in full_build_called


def test_forecast_mode_slides_counts(monkeypatch):
    """mode agg with existing state → per-code counts are updated via add/subtract."""
    grid_shape = (2, 4)
    existing_sums = {c: np.ones(grid_shape, dtype=np.int32) * 5 for c in RASTER_WC_CODES}
    existing_meta = {
        "var_id": "weather_code_simple", "window_h": 1, "window_label": "1h",
        "era5_window_start_ts": ERA5_END - 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": ERA5_END + 3600,
        "n_era5": 1, "n_gfs": 1,
        "built_at": "2023-01-01T00:00:00+00:00",
    }

    fake_delta = {c: np.ones(grid_shape, dtype=np.int32) for c in RASTER_WC_CODES}
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: ({**existing_sums}, {**existing_meta}))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode",
                        lambda *a, **kw: fake_delta)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid",
                        lambda arr, *a, **kw: arr)
    saved = {}
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda od, vid, wl, agg, sums, meta, suffix="": saved.update({"sums": sums, "meta": meta}))

    fake_cc_cidx = ChunkIndex(
        latest_end_time=1.0,
        ranges=[ChunkRange(chunk_num=2023, start=0, end=1, time_len=1, source="year")],
        resolution=3600.0,
    )
    era5_cidx_by_var = {"weather_code_simple": {
        "cloud_cover": fake_cc_cidx,
        "precipitation": fake_cc_cidx,
        "snowfall_water_equivalent": fake_cc_cidx,
    }}
    gfs_cidx = {
        "cloud_cover": fake_cc_cidx,
        "precipitation": fake_cc_cidx,
        "snowfall_water_equivalent": fake_cc_cidx,
    }

    bt._build_forecast_aggregates(
        {"weather_code_simple": bt.VAR_CONFIGS["weather_code_simple"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, NOW_TS + 8 * 3600,
        era5_cidx_by_var,
        gfs_cidx,
        "/tmp/test_out",
    )
    assert saved, "save_raster_state was not called"
    # counts should have changed (drop old + add new)
    for c in RASTER_WC_CODES:
        assert saved["sums"][c] is not None


def test_forecast_state_already_up_to_date_skips(monkeypatch):
    """Forecast state gfs_end_ts >= gfs_end_for_fc → early return, no save."""
    # gfs_end_for_fc = min(GFS_END, NOW_TS + forecast_h). For forecast_h=1,
    # gfs_end_for_fc = min(NOW_TS+2h, NOW_TS+1h) = NOW_TS+1h.
    # Set gfs_end_ts = GFS_END = NOW_TS+2h so it's already past gfs_end_for_fc.
    now_sums = {"era5_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": GFS_END,  # already ahead of any gfs_end_for_fc
        "n_era5": 0, "n_gfs": 1,
    }
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {"temperature_2m": _make_chunk_index()}},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert saved == []


def test_forecast_no_existing_state_calls_full_build(monkeypatch):
    """No existing state → _full_build is called."""
    full_build_called = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_called.append(a[0]))
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (None, None))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, GFS_END,
        {},
        {},
        "/tmp/test_out",
    )
    assert "temperature_2m" in full_build_called


def test_forecast_with_existing_state_incremental(monkeypatch):
    """Existing state → incremental drops/adds applied, save_raster_state called."""
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    now_sums = {"era5_temperature_2m": _zeros4(), "gfs_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS - 48 * 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": NOW_TS - 2 * 3600,
        "n_era5": 38,
        "n_gfs": 8,
    }

    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {"temperature_2m": _make_chunk_index()}},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) >= 1


def test_forecast_exception_in_process_printed(monkeypatch, capsys):
    """Exception in _process is caught and printed per-future."""
    def _raise(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.build_temporal._full_build", _raise)
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (None, None))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        [(1, "1h")],
        NOW_TS, ERA5_END, GFS_END,
        {},
        {},
        "/tmp/test_out",
    )
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_forecast_mode_slides_counts_missing_cidx(monkeypatch):
    """mode agg with missing era5/gfs cidx → None guards on lines 582, 587 are hit."""
    grid_shape = (2, 4)
    existing_sums = {c: np.ones(grid_shape, dtype=np.int32) * 5 for c in RASTER_WC_CODES}
    existing_meta = {
        "var_id": "weather_code_simple", "window_h": 1, "window_label": "1h",
        "era5_window_start_ts": ERA5_END - 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": ERA5_END + 3600,
        "n_era5": 1, "n_gfs": 1,
        "built_at": "2023-01-01T00:00:00+00:00",
    }
    saved = {}
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: ({**existing_sums}, {**existing_meta}))
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda od, vid, wl, agg, sums, meta, suffix="": saved.update({"sums": sums}))

    # Empty cidx → cc_cidx=None, gfs_cc=None → both delta functions return None early
    bt._build_forecast_aggregates(
        {"weather_code_simple": bt.VAR_CONFIGS["weather_code_simple"]},
        _minimal_windows(),
        NOW_TS, ERA5_END, NOW_TS + 8 * 3600,
        {"weather_code_simple": {}},
        {},
        "/tmp/test_out",
    )
    assert saved  # save was still called, counts unchanged


def test_forecast_era5_drop_end_lte_old_w_start(monkeypatch):
    """era5_drop_end <= old_w_start → ERA5 drop loop not entered."""
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    # Make old_w_start = future_ts so era5_drop_end = min(new_w_start=future+fc, era5_end) may be <= old_w_start
    now_sums = {"era5_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS + 100 * 3600,  # future window start
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": NOW_TS - 2 * 3600,
        "n_era5": 0,
        "n_gfs": 0,
    }

    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        [(1, "1h")],
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {"temperature_2m": _make_chunk_index()}},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) >= 1


def test_forecast_gfs_add_missing_cidx(monkeypatch):
    """GFS add: gv cidx missing → skip that var."""
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    now_sums = {"era5_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS - 10 * 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": NOW_TS - 20 * 3600,  # old_gfs_end < gfs_end_for_fc → add branch
        "n_era5": 5,
        "n_gfs": 0,
    }

    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        [(1, "1h")],
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {"temperature_2m": _make_chunk_index()}},
        {},  # no GFS cidx
        "/tmp/test_out",
    )
    assert len(saved) >= 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class _FakeCfg:
    temporal_raster_out_dir = "/tmp/test_rasters"
    temporal_raster_force_rebuild = True
    temporal_raster_vars = "temperature_2m"
    temporal_raster_windows = "1h"
    temporal_raster_upload_enabled = False
    temporal_raster_b2_dest = ""


def _patch_main_base(monkeypatch, tmp_path: Path, force: bool = True,
                     vars_str: str = "temperature_2m",
                     windows_str: str = "1h"):
    """Patch all external I/O needed for main()."""
    cfg = _FakeCfg()
    cfg.temporal_raster_out_dir = str(tmp_path)
    cfg.temporal_raster_force_rebuild = force
    cfg.temporal_raster_vars = vars_str
    cfg.temporal_raster_windows = windows_str

    monkeypatch.setenv("TEMPORAL_RASTER_NO_PUSH", "1")
    monkeypatch.setattr("scripts.build_temporal.load_config", lambda _: cfg)

    # fsspec.filesystem mock
    fs_mock = MagicMock()
    fs_mock.ls.return_value = []

    meta_data = json.dumps({"data_end_time": str(NOW_TS - 5 * 3600)}).encode()

    class _FakeFile:
        def __init__(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return meta_data

    fs_mock.open.return_value = _FakeFile()
    monkeypatch.setattr("scripts.build_temporal.fsspec.filesystem", lambda *a, **kw: fs_mock)

    cidx = _make_chunk_index()
    monkeypatch.setattr("scripts.build_temporal.build_chunk_index", lambda *a, **kw: cidx)
    monkeypatch.setattr("scripts.build_temporal.load_raster_state", lambda *a, **kw: (None, None))
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._incremental_update", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._build_forecast_aggregates", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._push_temporal_state", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._push_rasters", lambda *a, **kw: None)

    return cfg


def test_main_force_true_skips_stale_check(monkeypatch, tmp_path: Path):
    """force=True → runs full build, no stale check."""
    _patch_main_base(monkeypatch, tmp_path, force=True)
    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    assert "temperature_2m" in full_build_calls


def test_main_force_false_data_unchanged_skips(monkeypatch, tmp_path: Path):
    """force=False, prior state completed with same timestamps → early return."""
    era5_end = NOW_TS - 5 * 3600
    gfs_end = NOW_TS - 5 * 3600

    _patch_main_base(monkeypatch, tmp_path, force=False)

    # Write a completed state file with matching timestamps
    state_file = tmp_path / "temporal_state.json"
    state_file.write_text(json.dumps({
        "status": "completed",
        "era5_end_ts": era5_end,
        "gfs_end_ts": gfs_end,
    }))

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))

    bt.main()
    assert full_build_calls == []


def test_main_prior_state_corrupt_runs_normally(monkeypatch, tmp_path: Path):
    """Corrupt temporal_state.json → prior_state={} → runs normally."""
    _patch_main_base(monkeypatch, tmp_path, force=True)
    (tmp_path / "temporal_state.json").write_text("{not valid json{{")

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    assert "temperature_2m" in full_build_calls


def test_main_resumes_skips_completed_vars(monkeypatch, tmp_path: Path):
    """Prior run was interrupted (status=running) with same timestamps → resume skips done vars."""
    era5_end = NOW_TS - 5 * 3600
    gfs_end = NOW_TS - 5 * 3600

    _patch_main_base(monkeypatch, tmp_path, force=False)

    (tmp_path / "temporal_state.json").write_text(json.dumps({
        "status": "running",
        "era5_end_ts": era5_end,
        "gfs_end_ts": gfs_end,
        "completed_vars": ["temperature_2m_1h"],
        "forecast_completed": False,
    }))

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    # temperature_2m_1h was already done → should NOT be in full_build_calls
    assert "temperature_2m" not in full_build_calls


def test_main_resumes_skips_forecast_if_done(monkeypatch, tmp_path: Path):
    """Prior run completed vars but not forecast → forecast skipped on resume."""
    era5_end = NOW_TS - 5 * 3600
    gfs_end = NOW_TS - 5 * 3600

    _patch_main_base(monkeypatch, tmp_path, force=False)

    (tmp_path / "temporal_state.json").write_text(json.dumps({
        "status": "running",
        "era5_end_ts": era5_end,
        "gfs_end_ts": gfs_end,
        "completed_vars": [],
        "forecast_completed": True,
    }))

    forecast_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._build_forecast_aggregates",
                        lambda *a, **kw: forecast_calls.append(1))
    bt.main()
    assert forecast_calls == []


def test_main_force_false_no_existing_meta_runs(monkeypatch, tmp_path: Path, capsys):
    """force=False but no existing meta files → runs normally."""
    _patch_main_base(monkeypatch, tmp_path, force=False)
    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    # No meta files → skips the stale check → proceeds to build
    assert "temperature_2m" in full_build_calls


def test_main_force_false_data_changed_runs(monkeypatch, tmp_path: Path):
    """force=False, data changed → runs."""
    _patch_main_base(monkeypatch, tmp_path, force=False)

    # Write a meta.json with OLD timestamps
    sample_meta = {"era5_end_ts": 0.0, "gfs_end_ts": 0.0}
    meta_file = tmp_path / "temperature_2m_1h.meta.json"
    meta_file.write_text(json.dumps(sample_meta))

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    assert "temperature_2m" in full_build_calls


def test_main_incremental_path(monkeypatch, tmp_path: Path):
    """force=False, load_raster_state returns existing data → incremental update."""
    _patch_main_base(monkeypatch, tmp_path, force=False)

    # Write a stale meta file so stale check passes
    sample_meta = {"era5_end_ts": 0.0, "gfs_end_ts": 0.0}
    meta_file = tmp_path / "temperature_2m_1h.meta.json"
    meta_file.write_text(json.dumps(sample_meta))

    # load_raster_state returns existing state → incremental path
    existing_sums = {"era5_temperature_2m": _zeros4()}
    existing_meta = {
        "era5_window_start_ts": NOW_TS - 48 * 3600,
        "era5_end_ts": NOW_TS - 10 * 3600,
        "gfs_start_ts": NOW_TS - 10 * 3600,
        "gfs_end_ts": NOW_TS - 2 * 3600,
        "n_era5": 38,
        "n_gfs": 8,
    }
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (existing_sums, existing_meta))

    incremental_calls = []
    monkeypatch.setattr("scripts.build_temporal._incremental_update",
                        lambda *a, **kw: incremental_calls.append(a[0]))
    bt.main()
    assert "temperature_2m" in incremental_calls


def test_main_gfs_snow_depth_available(monkeypatch, tmp_path: Path):
    """GFS snow_depth ls() succeeds → gfs_var stays in config."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="snow_depth")

    fs_mock = MagicMock()
    fs_mock.ls.return_value = ["s3://openmeteo/data/ncep_gfs013/snow_depth/some_file"]
    meta_data = json.dumps({"data_end_time": str(NOW_TS - 5 * 3600)}).encode()

    class _FakeFile:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return meta_data

    fs_mock.open.return_value = _FakeFile()
    monkeypatch.setattr("scripts.build_temporal.fsspec.filesystem", lambda *a, **kw: fs_mock)

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    # snow_depth should be in gfs_raw_needed
    assert "snow_depth" in full_build_calls


def test_main_gfs_snow_depth_unavailable(monkeypatch, tmp_path: Path, capsys):
    """GFS snow_depth ls() raises → gfs_var removed from snow_depth config."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="snow_depth")

    fs_mock = MagicMock()
    fs_mock.ls.side_effect = FileNotFoundError("not found")
    meta_data = json.dumps({"data_end_time": str(NOW_TS - 5 * 3600)}).encode()

    class _FakeFile:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return meta_data

    fs_mock.open.return_value = _FakeFile()
    monkeypatch.setattr("scripts.build_temporal.fsspec.filesystem", lambda *a, **kw: fs_mock)

    bt.main()
    out = capsys.readouterr().out
    assert "not found" in out or "ERA5-land only" in out




def test_main_empty_vars_uses_all(monkeypatch, tmp_path: Path):
    """Empty vars string → all VAR_CONFIGS used."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="", windows_str="1h")

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    # Should include multiple variables
    assert len(set(full_build_calls)) > 1


def test_main_empty_windows_uses_all(monkeypatch, tmp_path: Path):
    """Empty windows string → all WINDOW_HOURS used."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="temperature_2m", windows_str="")

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()
    # Should include multiple windows for temperature_2m
    assert len(full_build_calls) > 1


def test_main_build_chunk_index_exception(monkeypatch, tmp_path: Path, capsys):
    """build_chunk_index raises → exception printed, continues."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="temperature_2m", windows_str="1h")
    monkeypatch.setattr("scripts.build_temporal.build_chunk_index",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("S3 error")))
    # Should not raise
    bt.main()
    out = capsys.readouterr().out
    assert "S3 error" in out


def test_main_weather_code_temperature_prefetch_exception(monkeypatch, tmp_path: Path, capsys):
    """weather_code_simple: ERA5-land temperature prefetch fails → logged, continues."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="weather_code_simple", windows_str="1h")

    call_count = {"n": 0}

    def _selective_fail(model, var):
        call_count["n"] += 1
        if model == "copernicus_era5_land" and var == "temperature_2m":
            raise RuntimeError("temperature unavailable")
        return _make_chunk_index()

    monkeypatch.setattr("scripts.build_temporal.build_chunk_index", _selective_fail)
    bt.main()
    out = capsys.readouterr().out
    assert "temperature unavailable" in out


def test_main_process_var_exception_printed(monkeypatch, tmp_path: Path, capsys):
    """Exception in _process_var is caught and printed."""
    _patch_main_base(monkeypatch, tmp_path, force=True, vars_str="temperature_2m", windows_str="1h")

    def _raise(*a, **kw):
        raise RuntimeError("full build exploded")

    monkeypatch.setattr("scripts.build_temporal._full_build", _raise)
    bt.main()
    out = capsys.readouterr().out
    assert "ERROR" in out


# ---------------------------------------------------------------------------
# _incremental_update — mode: GFS drop branch (lines 418-422)
# ---------------------------------------------------------------------------

def test_incremental_mode_gfs_drop_branch(monkeypatch):
    """Mode incremental: drop oldest covers GFS range (gfs_drop_end >= gfs_drop_start).

    Setup (no swap branch — era5_end_ts == old_era5_end):
      base = NOW_TS - 48h
      old_w_start = base = NOW_TS - 48h
      old_era5_end = base + 8h = NOW_TS - 40h
      old_gfs_start = base + 8h = NOW_TS - 40h
      old_gfs_end = base + 20h = NOW_TS - 28h
      era5_end_ts == old_era5_end → no swap branch
      window_h = 1 → new_w_start = NOW_TS
      gfs_drop_start = max(base, old_gfs_start) = NOW_TS - 40h
      gfs_drop_end = min(NOW_TS - 3600, NOW_TS - 28h) = NOW_TS - 28h
      gfs_drop_end >= gfs_drop_start → NOW_TS-28h >= NOW_TS-40h ✓
    """
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    base = NOW_TS - 48 * 3600
    old_era5_end = base + 8 * 3600
    old_meta = {
        "era5_window_start_ts": base,
        "era5_end_ts": old_era5_end,
        "gfs_start_ts": old_era5_end,
        "gfs_end_ts": base + 20 * 3600,
        "n_era5": 8,
        "n_gfs": 12,
    }

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 1, "1h",
        _sums_mode(), old_meta,
        NOW_TS, old_era5_end,  # era5_end_ts == old_era5_end → no swap
        GFS_END,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_mode_gfs_drop_branch_sub_none(monkeypatch):
    """Mode incremental: GFS drop branch entered but sub=None (missing precip/swe)."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    base = NOW_TS - 48 * 3600
    old_era5_end = base + 8 * 3600
    old_meta = {
        "era5_window_start_ts": base,
        "era5_end_ts": old_era5_end,
        "gfs_start_ts": old_era5_end,
        "gfs_end_ts": base + 20 * 3600,
        "n_era5": 8,
        "n_gfs": 12,
    }

    # No GFS precipitation/swe → _mode_accumulate returns None for GFS sub
    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 1, "1h",
        _sums_mode(), old_meta,
        NOW_TS, old_era5_end,  # no swap
        GFS_END,
        _make_era5_cidx_mode(include_t=False),
        {"cloud_cover": _make_chunk_index()},  # missing precip + swe → None
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _incremental_update — mode: _mode_accumulate end <= start (line 371)
# ---------------------------------------------------------------------------

def test_incremental_mode_accumulate_end_lte_start(monkeypatch):
    """_mode_accumulate(start, end, ...) where end <= start → returns None."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster_mode", _fake_accumulate_raster_mode)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # Trigger via add newest GFS: gfs_end_ts > old_gfs_end but gfs_end_ts - old_gfs_end < resolution
    # so _mode_accumulate(old_gfs_end + resolution, gfs_end_ts) has end < start → returns None
    old_meta = _old_meta_mode()
    old_gfs_end = float(old_meta["gfs_end_ts"])
    # gfs_end_ts = old_gfs_end + 0.5h → 0.5 * resolution
    new_gfs_end = old_gfs_end + 1800  # < 3600 (resolution) → end < start in _mode_accumulate

    bt._incremental_update(
        "weather_code_simple", _mode_cfg(), 24, "24h",
        _sums_mode(), old_meta,
        NOW_TS, ERA5_END, new_gfs_end,
        _make_era5_cidx_mode(include_t=False),
        _make_gfs_cidx_mode(),
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _incremental_update — scalar: GFS drop branch (lines 492-499)
# ---------------------------------------------------------------------------

def test_incremental_scalar_gfs_drop_branch(monkeypatch):
    """Scalar incremental: drop oldest covers GFS range (gfs_drop_end >= gfs_drop_start).

    Setup (no swap — era5_end_ts == old_era5_end):
      base = NOW_TS - 48h, old_era5_end = base+8h, old_gfs_end = base+20h
      window_h=1 → new_w_start = NOW_TS
      gfs_drop_start = max(base, old_era5_end) = NOW_TS - 40h
      gfs_drop_end = min(NOW_TS-1h, NOW_TS-28h) = NOW_TS-28h  ≥ gfs_drop_start ✓
    """
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    base = NOW_TS - 48 * 3600
    old_era5_end = base + 8 * 3600
    old_meta = {
        "era5_window_start_ts": base,
        "era5_end_ts": old_era5_end,
        "gfs_start_ts": old_era5_end,
        "gfs_end_ts": base + 20 * 3600,
        "n_era5": 8,
        "n_gfs": 12,
    }
    sums = {"era5_temperature_2m": _zeros4(), "gfs_temperature_2m": _zeros4()}

    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        sums, old_meta,
        NOW_TS, old_era5_end,  # no swap
        GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


def test_incremental_scalar_gfs_drop_key_not_in_sums(monkeypatch):
    """Scalar GFS drop: GFS drop branch entered but key not in sums → skip subtraction."""
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    base = NOW_TS - 48 * 3600
    old_era5_end = base + 8 * 3600
    old_meta = {
        "era5_window_start_ts": base,
        "era5_end_ts": old_era5_end,
        "gfs_start_ts": old_era5_end,
        "gfs_end_ts": base + 20 * 3600,
        "n_era5": 8,
        "n_gfs": 12,
    }
    sums = {"era5_temperature_2m": _zeros4()}  # no gfs key

    bt._incremental_update(
        "temperature_2m", _temperature_cfg(), 1, "1h",
        sums, old_meta,
        NOW_TS, old_era5_end,  # no swap
        GFS_END,
        {"temperature_2m": _make_chunk_index()},
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# _build_forecast_aggregates: ERA5 cidx missing (line 580) and GFS cidx missing (line 593)
# ---------------------------------------------------------------------------

def test_forecast_era5_cidx_missing_for_rv(monkeypatch):
    """Forecast drop ERA5: rv not in era5_cidx → continue (line 580)."""
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    now_sums = {"era5_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS - 10 * 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": ERA5_END,
        "gfs_end_ts": ERA5_END,  # stale so guard doesn't short-circuit
        "n_era5": 5,
        "n_gfs": 3,
    }
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # era5_cidx_by_var has temperature_2m var but its cidx dict is empty → rv not found
    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        [(1, "1h")],
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {}},  # empty era5_cidx → cidx = None → continue
        {"temperature_2m": _make_chunk_index()},
        "/tmp/test_out",
    )
    assert len(saved) >= 1


def test_forecast_gfs_cidx_missing_for_gv(monkeypatch):
    """Forecast GFS drop: gfs_drop_end > old_gfs_start but gv not in gfs_cidx → continue (line 593).

    Set old_gfs_start very early and new_w_start past it so gfs_drop_end > old_gfs_start.
    old_w_start = NOW_TS - 48h, forecast_h=1 → new_w_start = NOW_TS - 47h
    old_gfs_start = NOW_TS - 50h (earlier than new_w_start)
    old_gfs_end = NOW_TS - 45h
    gfs_drop_end = min(new_w_start=NOW_TS-47h, old_gfs_end=NOW_TS-45h) = NOW_TS-47h
    gfs_drop_end > old_gfs_start → NOW_TS-47h > NOW_TS-50h ✓
    """
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    now_sums = {"era5_temperature_2m": _zeros4(), "gfs_temperature_2m": _zeros4()}
    now_meta = {
        "era5_window_start_ts": NOW_TS - 48 * 3600,
        "era5_end_ts": ERA5_END,
        "gfs_start_ts": NOW_TS - 50 * 3600,  # very early
        "gfs_end_ts": NOW_TS - 45 * 3600,
        "n_era5": 5,
        "n_gfs": 3,
    }
    monkeypatch.setattr("scripts.build_temporal.load_raster_state",
                        lambda *a, **kw: (now_sums, now_meta))
    monkeypatch.setattr("scripts.build_temporal.accumulate_raster", _fake_accumulate_raster)
    monkeypatch.setattr("scripts.build_temporal.reproject_to_grid", _fake_reproject)
    saved = []
    monkeypatch.setattr("scripts.build_temporal.save_raster_state",
                        lambda *a, **kw: saved.append(1))

    # gfs_cidx is empty → gv not found → continue on line 593
    bt._build_forecast_aggregates(
        {"temperature_2m": bt.VAR_CONFIGS["temperature_2m"]},
        [(1, "1h")],
        NOW_TS, ERA5_END, GFS_END,
        {"temperature_2m": {"temperature_2m": _make_chunk_index()}},
        {},  # empty gfs_cidx → cidx = None → continue
        "/tmp/test_out",
    )
    assert len(saved) >= 1


def test_main_writes_failed_state_on_exception(monkeypatch, tmp_path: Path):
    """Unhandled exception in main work block → temporal_state.json written with status=failed."""
    _patch_main_base(monkeypatch, tmp_path, force=True)
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)
    monkeypatch.setattr("scripts.build_temporal._build_forecast_aggregates",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        bt.main()

    state = json.loads((tmp_path / "temporal_state.json").read_text())
    assert state["status"] == "failed"
    assert state["error"] == "boom"


def test_main_writes_completed_state(monkeypatch, tmp_path: Path):
    """Successful run → temporal_state.json written with status=completed."""
    _patch_main_base(monkeypatch, tmp_path, force=True)
    monkeypatch.setattr("scripts.build_temporal._full_build", lambda *a, **kw: None)

    bt.main()

    state = json.loads((tmp_path / "temporal_state.json").read_text())
    assert state["status"] == "completed"
    assert "completed_at" in state
    assert state["skipped"] is False


def test_main_exits_if_pid_alive(monkeypatch, tmp_path: Path, capsys):
    """status=running with a live PID → exit immediately, no build."""
    import os
    _patch_main_base(monkeypatch, tmp_path, force=False)
    (tmp_path / "temporal_state.json").write_text(json.dumps({
        "status": "running",
        "pid": os.getpid(),
        "era5_end_ts": NOW_TS - 5 * 3600,
        "gfs_end_ts": NOW_TS - 5 * 3600,
        "completed_vars": [],
        "forecast_completed": False,
    }))

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()

    assert full_build_calls == []
    assert "already running" in capsys.readouterr().out


def test_main_resumes_if_pid_dead(monkeypatch, tmp_path: Path):
    """status=running with a dead PID → treat as crashed, proceed with run."""
    _patch_main_base(monkeypatch, tmp_path, force=False)
    era5_end = NOW_TS - 5 * 3600
    gfs_end = NOW_TS - 5 * 3600
    (tmp_path / "temporal_state.json").write_text(json.dumps({
        "status": "running",
        "pid": 99999999,
        "era5_end_ts": era5_end,
        "gfs_end_ts": gfs_end,
        "completed_vars": [],
        "forecast_completed": False,
    }))

    full_build_calls = []
    monkeypatch.setattr("scripts.build_temporal._full_build",
                        lambda *a, **kw: full_build_calls.append(a[0]))
    bt.main()

    assert "temperature_2m" in full_build_calls
