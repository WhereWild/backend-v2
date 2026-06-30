# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build and incrementally update aggregate rasters for all temporal variables.

On first run: full rebuild from ERA5 + GFS013 chunks.
On subsequent runs: incremental sliding-window update —
  1. Drop oldest hours that fell off the window
  2. Add newest hours since last run
  3. Recompute final output and save

Sidecar files per raster:
  {var}_{window}.npy       — final output (avg, sum, or mode); read by the API
  {var}_{window}.meta.json — bookkeeping timestamps and counts
  {var}_{window}.sums.npz  — raw component sums / mode count grids (for incremental updates)

Sources:
  - ERA5 (copernicus_era5, 0.25°): temp, dew_point, soil, vpd, cloud, precip, swe, weather_code
  - GFS013 (ncep_gfs013, 0.125°): ~6-day ERA5 gap fill
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import httpx
import numpy as np

from config.config import load_config
from util.temporal import (
    RASTER_GRIDS,
    RASTER_WC_CODES,
    ChunkIndex,
    _chunk_entry_for_time,
    _download_chunk,
    accumulate_raster,
    accumulate_raster_mode,
    accumulate_vpd_raster,
    accumulate_vpd_raster_gfs,
    build_chunk_index,
    load_raster_state,
    reproject_to_grid,
    save_raster_state,
    set_raster_chunk_cache,
)

_TEMPORAL_STATUS_PUSH_URL = os.environ.get("WHEREWILD_STATUS_PUSH_URL", "")


def _floor_to_6h(last_modified: datetime) -> float:
    h = (last_modified.hour // 6) * 6
    return last_modified.replace(hour=h, minute=0, second=0, microsecond=0).timestamp()


def _gfs_detect_cycle(
    gfs_cidx: dict[str, "ChunkIndex"],
    gfs_end_ts: float,
    max_window_h: int,
    fs: "fsspec.AbstractFileSystem",
    prior_last_modified: dict[str, str],
) -> tuple[bool, float, dict[str, str]]:
    """Check S3 LastModified for all GFS chunks that could affect this run.

    Returns (changed, new_cycle_init_ts, new_last_modified_dict).
    new_cycle_init_ts = floor(max(new LastModified), 6h) across all changed chunks.
    """
    unique_chunks: dict[str, tuple[object, str, str]] = {}
    for gv, cidx in gfs_cidx.items():
        window_start = gfs_end_ts - max_window_h * 3600
        for ts in [window_start, gfs_end_ts]:
            entry, _ = _chunk_entry_for_time(cidx, ts)
            if entry is not None and entry.source != "year":
                key = f"{gv}_{entry.chunk_num}"
                if key not in unique_chunks:
                    unique_chunks[key] = (entry, "ncep_gfs013", gv)

    new_lm: dict[str, str] = {}
    max_changed_lm: datetime | None = None

    for key, (entry, model, gv) in unique_chunks.items():
        fname = f"chunk_{entry.chunk_num}.om"
        uri = f"s3://openmeteo/data/{model}/{gv}/{fname}"
        try:
            info = fs.info(uri)
            lm = info.get("LastModified")
            if lm is None:
                continue
            lm_str = str(lm)
            new_lm[key] = lm_str
            if lm_str != prior_last_modified.get(key):
                if max_changed_lm is None or lm > max_changed_lm:
                    max_changed_lm = lm
        except Exception as e:
            print(f"  [cycle-detect] HEAD {uri}: {e}")

    changed = max_changed_lm is not None
    cycle_init_ts = _floor_to_6h(max_changed_lm) if changed else 0.0
    return changed, cycle_init_ts, new_lm


def _push_temporal_state(state: dict) -> None:
    if not _TEMPORAL_STATUS_PUSH_URL:
        return
    try:
        url = _TEMPORAL_STATUS_PUSH_URL.rstrip("/") + "/internal/temporal-state"
        httpx.post(url, json=state, timeout=5)
    except Exception as exc:
        print(f"temporal status push: failed: {exc}")


# ---------------------------------------------------------------------------
# Variable config
# ---------------------------------------------------------------------------

WINDOW_HOURS = [1, 8, 24, 72, 168, 720, 2160]
WINDOW_LABELS: dict[int, str] = {1: "1h", 8: "8h", 24: "24h", 72: "3d", 168: "7d", 720: "30d", 2160: "90d"}
FORECAST_HOURS = [1, 8, 24, 72, 168]

VAR_CONFIGS: dict[str, dict] = {
    "temperature_2m": {
        "era5_model": "copernicus_era5",
        "era5_var": "temperature_2m",
        "gfs_var": "temperature_2m",
        "agg": "avg",
    },
    "dew_point_2m": {
        "era5_model": "copernicus_era5",
        "era5_var": "dew_point_2m",
        "gfs_derived_needs": ["temperature_2m", "relative_humidity_2m"],
        "agg": "avg",
    },
    "soil_temperature_0_to_7cm": {
        "era5_model": "copernicus_era5",
        "era5_var": "soil_temperature_0_to_7cm",
        "gfs_var": "soil_temperature_0_to_10cm",
        "agg": "avg",
    },
    "soil_moisture_0_to_7cm": {
        "era5_model": "copernicus_era5",
        "era5_var": "soil_moisture_0_to_7cm",
        "gfs_var": "soil_moisture_0_to_10cm",
        "agg": "avg",
    },
    "cloud_cover": {
        "era5_model": "copernicus_era5",
        "era5_var": "cloud_cover",
        "gfs_var": "cloud_cover",
        "agg": "avg",
    },
    "precipitation": {
        "era5_model": "copernicus_era5",
        "era5_var": "precipitation",
        "gfs_var": "precipitation",
        "agg": "sum",
    },
    "snowfall_water_equivalent": {
        "era5_model": "copernicus_era5",
        "era5_var": "snowfall_water_equivalent",
        "gfs_var": "snowfall_water_equivalent",
        "agg": "sum",
    },
    "vapor_pressure_deficit": {
        "era5_model": "copernicus_era5",
        "era5_derived_needs": ["temperature_2m", "dew_point_2m"],
        "gfs_derived_needs": ["temperature_2m", "relative_humidity_2m"],
        "agg": "avg",
    },
    "weather_code_simple": {
        "era5_model": "copernicus_era5",
        "agg": "mode",
        # sources: cloud_cover, precipitation, snowfall_water_equivalent, temperature_2m
    },
}

# ---------------------------------------------------------------------------
# Helpers: raw variable lists
# ---------------------------------------------------------------------------

def _era5_raw_vars(cfg: dict) -> list[str]:
    if "era5_var" in cfg:
        return [cfg["era5_var"]]
    return list(cfg.get("era5_derived_needs", []))


def _gfs_raw_vars(cfg: dict) -> list[str]:
    if "gfs_var" in cfg:
        return [cfg["gfs_var"]]
    return list(cfg.get("gfs_derived_needs", []))


def _derive_dew_point(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rh_c = np.clip(rh, 1.0, 100.0)
    gamma = np.log(rh_c / 100.0) + 17.625 * t / (243.04 + t)
    return (243.04 * gamma / (17.625 - gamma)).astype(np.float32)


def _gfs_grid(model: str = "ncep_gfs013") -> dict:
    return RASTER_GRIDS[model]


def _reproject_gfs_to(src: np.ndarray, dst_model: str) -> np.ndarray:
    """Reproject a lat-ascending GFS 0.125° grid to the target model's native grid."""
    gfs = RASTER_GRIDS["ncep_gfs013"]
    dst = RASTER_GRIDS[dst_model]
    return reproject_to_grid(
        src,
        gfs["lat_min"], gfs["lat_max"], gfs["lon_min"], gfs["lon_max"],
        dst["ny"], dst["nx"],
        dst["lat_min"], dst["lat_max"], dst["lon_min"], dst["lon_max"],
    )


# ---------------------------------------------------------------------------
# Full build for one var + window
# ---------------------------------------------------------------------------

def _full_build(
    var_id: str,
    cfg: dict,
    window_h: int,
    window_label: str,
    now_ts: float,
    era5_end_ts: float,
    gfs_end_ts: float,
    era5_cidx: dict[str, ChunkIndex],
    gfs_cidx: dict[str, ChunkIndex],
    out_dir: str,
    suffix: str = "",
    cycle_init_ts: float = 0.0,
) -> None:
    era5_model = cfg["era5_model"]
    agg = cfg["agg"]
    w_start = now_ts - (window_h - 1) * 3600

    sums: dict[str, np.ndarray] = {}
    n_era5, n_gfs_stable, n_gfs_forecast = 0, 0, 0

    # GFS starts one step AFTER era5_end_ts so the boundary hour is not counted
    # in both ERA5 and GFS sums simultaneously.
    resolution = next((c.resolution for c in list(era5_cidx.values()) + list(gfs_cidx.values()) if c is not None), 3600.0)
    gfs_start = max(era5_end_ts + resolution, w_start)

    # stable = [gfs_start, cycle_init_ts); forecast = [cycle_init_ts, gfs_end_ts]
    # If cycle_init_ts == 0 (unknown), treat entire GFS window as forecast.
    gfs_stable_end = (cycle_init_ts - resolution) if cycle_init_ts > gfs_start else (gfs_start - resolution)
    gfs_fc_start = cycle_init_ts if cycle_init_ts > gfs_start else gfs_start

    if agg == "mode":
        cc_cidx = era5_cidx.get("cloud_cover")
        pr_cidx = era5_cidx.get("precipitation")
        sw_cidx = era5_cidx.get("snowfall_water_equivalent")
        t_cidx = era5_cidx.get("_temperature_for_wc")  # ERA5-land temp, pre-fetched

        if not cc_cidx:
            print(f"  [{window_label}] {var_id}: no ERA5 cloud_cover, skipping")
            return

        # Accumulate mean temperature at ERA5 0.25° for snow/rain cutoff
        temp_grid_025: np.ndarray | None = None
        if t_cidx:
            t_sum, t_n = accumulate_raster("copernicus_era5", "temperature_2m",
                                           w_start, era5_end_ts, t_cidx)
            if t_n > 0:
                temp_grid_025 = (t_sum / t_n).astype(np.float32)

        era5_counts = accumulate_raster_mode(
            era5_model, w_start, era5_end_ts,
            cc_cidx, pr_cidx, sw_cidx,  # type: ignore[arg-type]
            temp_grid_025=temp_grid_025,
        )
        for c in RASTER_WC_CODES:
            sums[c] = era5_counts[c]
        n_era5 = max(0, int(round((era5_end_ts - w_start) / cc_cidx.resolution)) + 1)

        # GFS gap fill for mode (no stable/forecast split for mode counts)
        gfs_cc = gfs_cidx.get("cloud_cover")
        gfs_pr = gfs_cidx.get("precipitation")
        gfs_sw = gfs_cidx.get("snowfall_water_equivalent")
        gfs_mode_start = gfs_start
        if gfs_cc and gfs_pr and gfs_sw and gfs_mode_start <= gfs_end_ts:
            gfs_counts = accumulate_raster_mode(
                "ncep_gfs013", gfs_mode_start, gfs_end_ts,
                gfs_cc, gfs_pr, gfs_sw,
            )
            era5_grid = RASTER_GRIDS["copernicus_era5"]
            for c in RASTER_WC_CODES:
                gfs_reproj = reproject_to_grid(
                    gfs_counts[c].astype(np.float32),
                    RASTER_GRIDS["ncep_gfs013"]["lat_min"], RASTER_GRIDS["ncep_gfs013"]["lat_max"],
                    RASTER_GRIDS["ncep_gfs013"]["lon_min"], RASTER_GRIDS["ncep_gfs013"]["lon_max"],
                    era5_grid["ny"], era5_grid["nx"],
                    era5_grid["lat_min"], era5_grid["lat_max"],
                    era5_grid["lon_min"], era5_grid["lon_max"],
                    resampling="nearest",
                )
                sums[c] = sums[c] + np.round(gfs_reproj).astype(np.int32)
            n_gfs_forecast = max(0, int(round((gfs_end_ts - gfs_mode_start) / gfs_cc.resolution)) + 1)

    elif var_id == "vapor_pressure_deficit":
        t_cidx = era5_cidx.get("temperature_2m")
        td_cidx = era5_cidx.get("dew_point_2m")
        if not t_cidx or not td_cidx:
            print(f"  [{window_label}] {var_id}: missing ERA5-land temperature/dew_point index, skipping")
            return
        vpd_sum, n = accumulate_vpd_raster(era5_model, w_start, era5_end_ts, t_cidx, td_cidx)
        sums["era5_vpd"] = vpd_sum
        n_era5 = n

        t_cidx_gfs = gfs_cidx.get("temperature_2m")
        rh_cidx_gfs = gfs_cidx.get("relative_humidity_2m")
        if t_cidx_gfs and rh_cidx_gfs:
            if gfs_start <= gfs_stable_end:
                vpd_stable, n_s = accumulate_vpd_raster_gfs(
                    gfs_start, gfs_stable_end, t_cidx_gfs, rh_cidx_gfs, era5_model
                )
                sums["gfs_vpd"] = vpd_stable
                n_gfs_stable = n_s
            if gfs_fc_start <= gfs_end_ts:
                vpd_fc, n_f = accumulate_vpd_raster_gfs(
                    gfs_fc_start, gfs_end_ts, t_cidx_gfs, rh_cidx_gfs, era5_model
                )
                sums["gfs_forecast_vpd"] = vpd_fc
                n_gfs_forecast = n_f

        if not sums:
            print(f"  [{window_label}] {var_id}: no ERA5 data, skipping")
            return

    else:
        # Scalar accumulation: ERA5 portion
        for rv in _era5_raw_vars(cfg):
            cidx = era5_cidx.get(rv)
            if not cidx:
                continue
            acc, n = accumulate_raster(era5_model, rv, w_start, era5_end_ts, cidx)
            sums[f"era5_{rv}"] = acc
            n_era5 = max(n_era5, n)

        if not sums:
            print(f"  [{window_label}] {var_id}: no ERA5 data, skipping")
            return

        dst_g = RASTER_GRIDS[era5_model]
        gfs_g = RASTER_GRIDS["ncep_gfs013"]

        def _accum_gfs_reproj_full(gv: str, t0: float, t1: float) -> tuple[np.ndarray | None, int]:
            cidx = gfs_cidx.get(gv)
            if not cidx or t0 > t1:
                return None, 0
            acc, n = accumulate_raster("ncep_gfs013", gv, t0, t1, cidx)
            reproj = reproject_to_grid(
                acc.astype(np.float32),
                gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                dst_g["ny"], dst_g["nx"],
                dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
            )
            return reproj.astype(np.float64), n

        if var_id == "dew_point_2m":
            t_cidx = gfs_cidx.get("temperature_2m")
            rh_cidx = gfs_cidx.get("relative_humidity_2m")

            def _accum_td(t0: float, t1: float) -> tuple[np.ndarray | None, int]:
                if not t_cidx or not rh_cidx or t0 > t1:
                    return None, 0
                t_sum, t_n = accumulate_raster("ncep_gfs013", "temperature_2m", t0, t1, t_cidx)
                rh_sum, rh_n = accumulate_raster("ncep_gfs013", "relative_humidity_2m", t0, t1, rh_cidx)
                n = min(t_n, rh_n)
                if n == 0:
                    return None, 0
                td = _derive_dew_point((t_sum / n).astype(np.float32), (rh_sum / n).astype(np.float32))
                r = reproject_to_grid(td, gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                                      dst_g["ny"], dst_g["nx"], dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"])
                return r.astype(np.float64) * n, n

            if gfs_start <= gfs_stable_end:
                arr, n = _accum_td(gfs_start, gfs_stable_end)
                if arr is not None:
                    sums["gfs_dew_point_2m"] = arr
                    n_gfs_stable = n
            if gfs_fc_start <= gfs_end_ts:
                arr, n = _accum_td(gfs_fc_start, gfs_end_ts)
                if arr is not None:
                    sums["gfs_forecast_dew_point_2m"] = arr
                    n_gfs_forecast = n

        else:
            for gv in _gfs_raw_vars(cfg):
                if gfs_start <= gfs_stable_end:
                    arr, n = _accum_gfs_reproj_full(gv, gfs_start, gfs_stable_end)
                    if arr is not None:
                        sums[f"gfs_{gv}"] = arr
                        n_gfs_stable = max(n_gfs_stable, n)
                if gfs_fc_start <= gfs_end_ts:
                    arr, n = _accum_gfs_reproj_full(gv, gfs_fc_start, gfs_end_ts)
                    if arr is not None:
                        sums[f"gfs_forecast_{gv}"] = arr
                        n_gfs_forecast = max(n_gfs_forecast, n)

    n_gfs = n_gfs_stable + n_gfs_forecast
    meta = {
        "var_id": var_id, "window_h": window_h, "window_label": window_label,
        "era5_window_start_ts": w_start, "era5_end_ts": era5_end_ts,
        "gfs_start_ts": gfs_start,
        "gfs_end_ts": gfs_end_ts,
        "gfs_cycle_init_ts": cycle_init_ts,
        "n_era5": n_era5, "n_gfs": n_gfs,
        "n_gfs_stable": n_gfs_stable, "n_gfs_forecast": n_gfs_forecast,
        "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
    }
    save_raster_state(out_dir, var_id, window_label, agg, sums, meta, suffix=suffix)
    print(f"  [{window_label}] {var_id}: {n_era5}h ERA5 + {n_gfs_stable}h GFS-stable + {n_gfs_forecast}h GFS-fc → {out_dir}/{var_id}_{window_label}{suffix}.npy")


# ---------------------------------------------------------------------------
# Incremental update for one var + window
# ---------------------------------------------------------------------------

def _incremental_update(
    var_id: str,
    cfg: dict,
    window_h: int,
    window_label: str,
    sums: dict[str, np.ndarray],
    old_meta: dict,
    now_ts: float,
    era5_end_ts: float,
    gfs_end_ts: float,
    era5_cidx: dict[str, ChunkIndex],
    gfs_cidx: dict[str, ChunkIndex],
    out_dir: str,
) -> None:
    agg = cfg["agg"]
    era5_model = cfg["era5_model"]
    new_w_start = now_ts - (window_h - 1) * 3600
    old_w_start = float(old_meta["era5_window_start_ts"])
    old_era5_end = float(old_meta["era5_end_ts"])
    old_gfs_start = float(old_meta["gfs_start_ts"])
    old_gfs_end = float(old_meta["gfs_end_ts"])
    n_era5 = int(old_meta["n_era5"])
    n_gfs_stable = int(old_meta.get("n_gfs_stable", old_meta["n_gfs"]))
    n_gfs_forecast = int(old_meta.get("n_gfs_forecast", 0))
    n_gfs = n_gfs_stable + n_gfs_forecast
    cycle_init_ts = float(old_meta.get("gfs_cycle_init_ts", 0.0))

    if agg == "mode":
        # Mode incremental: sums contains {wc_code: count_grid}
        cc_cidx = era5_cidx.get("cloud_cover")
        pr_cidx = era5_cidx.get("precipitation")
        sw_cidx = era5_cidx.get("snowfall_water_equivalent")
        if not cc_cidx:
            return

        resolution = cc_cidx.resolution
        gfs_g = RASTER_GRIDS["ncep_gfs013"]

        def _mode_accumulate(start: float, end: float, use_gfs: bool) -> dict[int, np.ndarray] | None:
            if end < start:
                return None
            model = "ncep_gfs013" if use_gfs else era5_model
            _cc = gfs_cidx.get("cloud_cover") if use_gfs else cc_cidx
            _pr = gfs_cidx.get("precipitation") if use_gfs else pr_cidx
            _sw = gfs_cidx.get("snowfall_water_equivalent") if use_gfs else sw_cidx
            if not _cc or not _pr or not _sw:
                return None
            delta = accumulate_raster_mode(model, start, end, _cc, _pr, _sw)
            if use_gfs:
                era5_g = RASTER_GRIDS["copernicus_era5"]
                for c in RASTER_WC_CODES:
                    delta[c] = np.round(reproject_to_grid(
                        delta[c].astype(np.float32),
                        gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                        era5_g["ny"], era5_g["nx"],
                        era5_g["lat_min"], era5_g["lat_max"], era5_g["lon_min"], era5_g["lon_max"],
                        resampling="nearest",
                    )).astype(np.int32)
            return delta

        # ERA5 quality swap: replace GFS with ERA5 for [old_era5_end+1h, new_era5_end]
        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start:
                add = _mode_accumulate(swap_start, swap_end, use_gfs=False)
                sub = _mode_accumulate(swap_start, swap_end, use_gfs=True)
                swap_n = int(round((swap_end - swap_start) / resolution)) + 1
                if add and sub:
                    for c in RASTER_WC_CODES:
                        sums[c] = np.maximum(0, sums[c] + add[c] - sub[c])
                n_era5 += swap_n
                n_gfs -= swap_n
                old_gfs_start = era5_end_ts + resolution

        # Drop oldest hours — new_w_start is the first point of the new window, so drop up to new_w_start - 1h
        if new_w_start > old_w_start:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                sub = _mode_accumulate(old_w_start, era5_drop_end, use_gfs=False)
                if sub:
                    for c in RASTER_WC_CODES:
                        sums[c] = np.maximum(0, sums[c] - sub[c])
                n_era5 -= int(round((era5_drop_end - old_w_start) / resolution)) + 1

            gfs_drop_start = max(old_w_start, old_gfs_start)
            gfs_drop_end = min(new_w_start - resolution, old_gfs_end)
            if gfs_drop_end >= gfs_drop_start:
                sub = _mode_accumulate(gfs_drop_start, gfs_drop_end, use_gfs=True)
                if sub:
                    for c in RASTER_WC_CODES:
                        sums[c] = np.maximum(0, sums[c] - sub[c])
                n_gfs -= int(round((gfs_drop_end - gfs_drop_start) / resolution)) + 1

        # Add newest GFS hours — start at old_gfs_end + 1h to avoid double-counting.
        # Cap at new_w_start so a stale build doesn't add hours before the window.
        if gfs_end_ts > old_gfs_end:
            add_start = max(old_gfs_end + resolution, new_w_start)
            if gfs_end_ts >= add_start:
                add = _mode_accumulate(add_start, gfs_end_ts, use_gfs=True)
                if add:
                    for c in RASTER_WC_CODES:
                        sums[c] = sums[c] + add[c]
                n_gfs += int(round((gfs_end_ts - add_start) / cc_cidx.resolution)) + 1

    elif var_id == "vapor_pressure_deficit":
        # VPD incremental: accumulate per-timestep VPD directly (not T/Td separately)
        t_cidx = era5_cidx.get("temperature_2m")
        td_cidx = era5_cidx.get("dew_point_2m")
        t_cidx_gfs = gfs_cidx.get("temperature_2m")
        rh_cidx_gfs = gfs_cidx.get("relative_humidity_2m")
        resolution = t_cidx.resolution if t_cidx else 3600.0

        def _vpd_era5(start: float, end: float) -> np.ndarray | None:
            if not t_cidx or not td_cidx or end < start:
                return None
            acc, _ = accumulate_vpd_raster(era5_model, start, end, t_cidx, td_cidx)
            return acc.astype(np.float32)

        def _vpd_gfs(start: float, end: float) -> np.ndarray | None:
            if not t_cidx_gfs or not rh_cidx_gfs or end < start:
                return None
            acc, _ = accumulate_vpd_raster_gfs(start, end, t_cidx_gfs, rh_cidx_gfs, era5_model)
            return acc.astype(np.float32)

        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start:
                swap_n = int(round((swap_end - swap_start) / resolution)) + 1
                added = _vpd_era5(swap_start, swap_end)
                if added is not None:
                    sums["era5_vpd"] = sums.get("era5_vpd", np.zeros_like(added)) + added
                if cycle_init_ts > 0:
                    stable_swap_end = min(swap_end, cycle_init_ts - resolution)
                    fc_swap_start = max(swap_start, cycle_init_ts)
                else:
                    stable_swap_end = swap_start - 1
                    fc_swap_start = swap_start
                stable_swap_n = max(0, int(round((stable_swap_end - swap_start) / resolution)) + 1) if stable_swap_end >= swap_start else 0
                fc_swap_n = max(0, int(round((swap_end - fc_swap_start) / resolution)) + 1) if fc_swap_start <= swap_end else 0
                if stable_swap_end >= swap_start:
                    dropped = _vpd_gfs(swap_start, stable_swap_end)
                    if dropped is not None and "gfs_vpd" in sums:
                        sums["gfs_vpd"] = sums["gfs_vpd"] - dropped
                if fc_swap_start <= swap_end:
                    dropped = _vpd_gfs(fc_swap_start, swap_end)
                    if dropped is not None and "gfs_forecast_vpd" in sums:
                        sums["gfs_forecast_vpd"] = sums["gfs_forecast_vpd"] - dropped
                n_era5 += swap_n
                n_gfs_stable -= stable_swap_n
                n_gfs_forecast -= fc_swap_n
                old_gfs_start = era5_end_ts + resolution

        # Drops: split at cycle_init_ts — stable bucket vs forecast bucket
        if new_w_start > old_w_start:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                drop_n = int(round((era5_drop_end - old_w_start) / resolution)) + 1
                dropped = _vpd_era5(old_w_start, era5_drop_end)
                if dropped is not None and "era5_vpd" in sums:
                    sums["era5_vpd"] = sums["era5_vpd"] - dropped
                n_era5 -= drop_n

            gfs_drop_start = max(old_w_start, old_gfs_start)
            gfs_drop_end = min(new_w_start - resolution, old_gfs_end)
            if gfs_drop_end >= gfs_drop_start:
                if cycle_init_ts > 0:
                    stable_drop_end = min(gfs_drop_end, cycle_init_ts - resolution)
                    fc_drop_start = max(gfs_drop_start, cycle_init_ts)
                else:
                    stable_drop_end = gfs_drop_start - 1
                    fc_drop_start = gfs_drop_start
                if stable_drop_end >= gfs_drop_start:
                    dropped = _vpd_gfs(gfs_drop_start, stable_drop_end)
                    sn = int(round((stable_drop_end - gfs_drop_start) / resolution)) + 1
                    if dropped is not None and "gfs_vpd" in sums:
                        sums["gfs_vpd"] = sums["gfs_vpd"] - dropped
                    n_gfs_stable -= sn
                if fc_drop_start <= gfs_drop_end:
                    dropped = _vpd_gfs(fc_drop_start, gfs_drop_end)
                    fn = int(round((gfs_drop_end - fc_drop_start) / resolution)) + 1
                    if dropped is not None and "gfs_forecast_vpd" in sums:
                        sums["gfs_forecast_vpd"] = sums["gfs_forecast_vpd"] - dropped
                    n_gfs_forecast -= fn

        # Adds always go to forecast bucket
        if gfs_end_ts > old_gfs_end:
            add_start = max(old_gfs_end + resolution, new_w_start)
            if gfs_end_ts >= add_start:
                add_n = int(round((gfs_end_ts - add_start) / resolution)) + 1
                added = _vpd_gfs(add_start, gfs_end_ts)
                if added is not None:
                    sums["gfs_forecast_vpd"] = sums.get("gfs_forecast_vpd", np.zeros_like(added)) + added
                n_gfs_forecast += add_n

    else:
        # Scalar incremental
        resolution = next(iter(era5_cidx.values())).resolution if era5_cidx else 3600.0
        dst_g = RASTER_GRIDS[era5_model]
        gfs_g = RASTER_GRIDS["ncep_gfs013"]

        def _accum_era5(rv: str, start: float, end: float) -> np.ndarray | None:
            cidx = era5_cidx.get(rv)
            if not cidx or end < start:
                return None
            acc, _ = accumulate_raster(era5_model, rv, start, end, cidx)
            return acc.astype(np.float32)

        def _accum_gfs_reproj(gv: str, start: float, end: float) -> np.ndarray | None:
            cidx = gfs_cidx.get(gv)
            if not cidx or end < start:
                return None
            acc, _ = accumulate_raster("ncep_gfs013", gv, start, end, cidx)
            return reproject_to_grid(
                acc.astype(np.float32),
                gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                dst_g["ny"], dst_g["nx"],
                dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
            )

        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start:
                swap_n = int(round((swap_end - swap_start) / resolution)) + 1
                for rv in _era5_raw_vars(cfg):
                    added = _accum_era5(rv, swap_start, swap_end)
                    if added is not None:
                        key = f"era5_{rv}"
                        sums[key] = sums.get(key, np.zeros_like(added)) + added
                # ERA5 quality swap: split GFS drop between stable and forecast buckets.
                # When cycle_init_ts=0 (unknown), everything is in the forecast bucket.
                if cycle_init_ts > 0:
                    stable_swap_end = min(swap_end, cycle_init_ts - resolution)
                    fc_swap_start = max(swap_start, cycle_init_ts)
                else:
                    stable_swap_end = swap_start - 1  # nothing in stable
                    fc_swap_start = swap_start          # all in forecast
                stable_swap_n = max(0, int(round((stable_swap_end - swap_start) / resolution)) + 1) if stable_swap_end >= swap_start else 0
                fc_swap_n = max(0, int(round((swap_end - fc_swap_start) / resolution)) + 1) if fc_swap_start <= swap_end else 0
                for gv in _gfs_raw_vars(cfg):
                    if stable_swap_end >= swap_start:
                        dropped = _accum_gfs_reproj(gv, swap_start, stable_swap_end)
                        if dropped is not None and f"gfs_{gv}" in sums:
                            sums[f"gfs_{gv}"] = sums[f"gfs_{gv}"] - dropped
                    if fc_swap_start <= swap_end:
                        dropped = _accum_gfs_reproj(gv, fc_swap_start, swap_end)
                        if dropped is not None and f"gfs_forecast_{gv}" in sums:
                            sums[f"gfs_forecast_{gv}"] = sums[f"gfs_forecast_{gv}"] - dropped
                n_era5 += swap_n
                n_gfs_stable -= stable_swap_n
                n_gfs_forecast -= fc_swap_n
                old_gfs_start = era5_end_ts + resolution

        if new_w_start > old_w_start:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                drop_n = int(round((era5_drop_end - old_w_start) / resolution)) + 1
                for rv in _era5_raw_vars(cfg):
                    dropped = _accum_era5(rv, old_w_start, era5_drop_end)
                    if dropped is not None:
                        key = f"era5_{rv}"
                        if key in sums:
                            sums[key] = sums[key] - dropped
                n_era5 -= drop_n

            gfs_drop_start = max(old_w_start, old_gfs_start)
            gfs_drop_end = min(new_w_start - resolution, old_gfs_end)
            if gfs_drop_end >= gfs_drop_start:
                if cycle_init_ts > 0:
                    stable_drop_end = min(gfs_drop_end, cycle_init_ts - resolution)
                    fc_drop_start = max(gfs_drop_start, cycle_init_ts)
                else:
                    stable_drop_end = gfs_drop_start - 1  # nothing in stable
                    fc_drop_start = gfs_drop_start          # all in forecast
                for gv in _gfs_raw_vars(cfg):
                    if stable_drop_end >= gfs_drop_start:
                        dropped = _accum_gfs_reproj(gv, gfs_drop_start, stable_drop_end)
                        if dropped is not None and f"gfs_{gv}" in sums:
                            sums[f"gfs_{gv}"] = sums[f"gfs_{gv}"] - dropped
                    if fc_drop_start <= gfs_drop_end:
                        dropped = _accum_gfs_reproj(gv, fc_drop_start, gfs_drop_end)
                        if dropped is not None and f"gfs_forecast_{gv}" in sums:
                            sums[f"gfs_forecast_{gv}"] = sums[f"gfs_forecast_{gv}"] - dropped
                stable_dn = max(0, int(round((stable_drop_end - gfs_drop_start) / resolution)) + 1) if stable_drop_end >= gfs_drop_start else 0
                fc_dn = max(0, int(round((gfs_drop_end - fc_drop_start) / resolution)) + 1) if fc_drop_start <= gfs_drop_end else 0
                n_gfs_stable -= stable_dn
                n_gfs_forecast -= fc_dn

        # Adds always go to forecast bucket (new data >= cycle_init_ts)
        if gfs_end_ts > old_gfs_end:
            add_start = max(old_gfs_end + resolution, new_w_start)
            if gfs_end_ts >= add_start:
                add_n = int(round((gfs_end_ts - add_start) / resolution)) + 1
                for gv in _gfs_raw_vars(cfg):
                    added = _accum_gfs_reproj(gv, add_start, gfs_end_ts)
                    if added is not None:
                        key = f"gfs_forecast_{gv}"
                        sums[key] = sums.get(key, np.zeros_like(added)) + added
                n_gfs_forecast += add_n

    n_era5 = max(n_era5, 0)
    if agg == "mode":
        # Mode doesn't split stable/forecast; n_gfs was updated directly above.
        n_gfs = max(n_gfs, 0)
        n_gfs_stable = n_gfs
        n_gfs_forecast = 0
    else:
        n_gfs_stable = max(n_gfs_stable, 0)
        n_gfs_forecast = max(n_gfs_forecast, 0)
        n_gfs = n_gfs_stable + n_gfs_forecast
    meta = {
        **old_meta,
        "era5_end_ts": era5_end_ts,
        "era5_window_start_ts": new_w_start,
        "gfs_start_ts": max(era5_end_ts + resolution, new_w_start),
        "gfs_end_ts": gfs_end_ts,
        "gfs_cycle_init_ts": cycle_init_ts,
        "n_era5": n_era5, "n_gfs": n_gfs,
        "n_gfs_stable": n_gfs_stable, "n_gfs_forecast": n_gfs_forecast,
        "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
    }
    save_raster_state(out_dir, var_id, window_label, agg, sums, meta)


# ---------------------------------------------------------------------------
# GFS cycle re-derive: rebuild only the GFS portion when frontier chunk changes
# ---------------------------------------------------------------------------

def _gfs_rederive(
    var_id: str,
    cfg: dict,
    window_h: int,
    window_label: str,
    existing_sums: dict,
    existing_meta: dict,
    now_ts: float,
    era5_end_ts: float,
    gfs_end_ts: float,
    era5_cidx: dict,
    gfs_cidx: dict,
    out_dir: str,
    new_cycle_init_ts: float = 0.0,
) -> None:
    """Re-accumulate GFS forecast sums after a GFS cycle update.

    Strategy (stable/forecast split):
    - Graduate [old_cycle_init_ts, new_cycle_init_ts) from forecast → stable (re-read ~6h, now frozen)
    - Discard all gfs_forecast_* sums
    - Rebuild gfs_forecast_* from [new_cycle_init_ts, new_gfs_end_ts]
    - gfs_* (stable) sums are preserved as-is (chunk values frozen at ≤ old cycle init)
    - ERA5 quality swap + ERA5 window drop applied normally
    - For mode vars: can't separate ERA5/GFS in count arrays → fall back to _full_build
    """
    agg = cfg["agg"]

    if agg == "mode":
        _full_build(var_id, cfg, window_h, window_label, now_ts, era5_end_ts, gfs_end_ts,
                    era5_cidx, gfs_cidx, out_dir, cycle_init_ts=new_cycle_init_ts)
        return

    era5_model = cfg["era5_model"]
    resolution = 3600.0
    dst_g = RASTER_GRIDS[era5_model]
    gfs_g = RASTER_GRIDS["ncep_gfs013"]

    new_w_start = now_ts - (window_h - 1) * 3600
    new_gfs_start = max(era5_end_ts + resolution, new_w_start)

    old_w_start = float(existing_meta["era5_window_start_ts"])
    old_era5_end = float(existing_meta["era5_end_ts"])
    old_gfs_end = float(existing_meta["gfs_end_ts"])
    old_cycle_init_ts = float(existing_meta.get("gfs_cycle_init_ts", 0.0))

    # Preserve ERA5 and stable GFS sums; zero only forecast sums.
    sums: dict = {}
    for k, v in existing_sums.items():
        sums[k] = np.zeros_like(v) if str(k).startswith("gfs_forecast_") else v.copy()

    n_gfs_stable = int(existing_meta.get("n_gfs_stable", existing_meta.get("n_gfs", 0)))
    n_gfs_forecast = 0

    def _gfs_acc(gv: str, start: float, end: float) -> np.ndarray | None:
        cidx = gfs_cidx.get(gv)
        if not cidx or end < start:
            return None
        acc, _ = accumulate_raster("ncep_gfs013", gv, start, end, cidx)
        return reproject_to_grid(
            acc.astype(np.float32),
            gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
            dst_g["ny"], dst_g["nx"],
            dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
        )

    if var_id == "vapor_pressure_deficit":
        t_cidx = era5_cidx.get("temperature_2m")
        td_cidx = era5_cidx.get("dew_point_2m")
        t_cidx_gfs = gfs_cidx.get("temperature_2m")
        rh_cidx_gfs = gfs_cidx.get("relative_humidity_2m")

        def _vpd_era5_acc(start: float, end: float) -> np.ndarray | None:
            if not t_cidx or not td_cidx or end < start:
                return None
            acc, _ = accumulate_vpd_raster(era5_model, start, end, t_cidx, td_cidx)
            return acc.astype(np.float32)

        def _vpd_gfs_acc(start: float, end: float) -> np.ndarray | None:
            if not t_cidx_gfs or not rh_cidx_gfs or end < start:
                return None
            acc, _ = accumulate_vpd_raster_gfs(start, end, t_cidx_gfs, rh_cidx_gfs, era5_model)
            return acc.astype(np.float32)

        # ERA5 quality swap
        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start:
                added = _vpd_era5_acc(swap_start, swap_end)
                if added is not None:
                    sums["era5_vpd"] = sums.get("era5_vpd", np.zeros_like(added)) + added

        # ERA5 window drop
        if new_w_start > old_w_start:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                dropped = _vpd_era5_acc(old_w_start, era5_drop_end)
                if dropped is not None and "era5_vpd" in sums:
                    sums["era5_vpd"] = sums["era5_vpd"] - dropped

        # Graduate old forecast → stable, then rebuild new forecast
        if old_cycle_init_ts > 0 and new_cycle_init_ts > old_cycle_init_ts:
            grad_end = new_cycle_init_ts - resolution
            if grad_end >= old_cycle_init_ts:
                graduated = _vpd_gfs_acc(old_cycle_init_ts, grad_end)
                grad_n = int(round((grad_end - old_cycle_init_ts) / resolution)) + 1
                if graduated is not None:
                    sums["gfs_vpd"] = sums.get("gfs_vpd", np.zeros_like(graduated)) + graduated
                    n_gfs_stable += grad_n

        if new_cycle_init_ts > 0 and new_cycle_init_ts <= gfs_end_ts:
            fc_start = max(new_cycle_init_ts, new_gfs_start)
            if fc_start <= gfs_end_ts:
                vpd_fc, n_f = accumulate_vpd_raster_gfs(fc_start, gfs_end_ts, t_cidx_gfs, rh_cidx_gfs, era5_model)
                sums["gfs_forecast_vpd"] = vpd_fc
                n_gfs_forecast = n_f
        elif new_gfs_start <= gfs_end_ts and t_cidx_gfs and rh_cidx_gfs:
            vpd_fc, n_gfs_forecast = accumulate_vpd_raster_gfs(
                new_gfs_start, gfs_end_ts, t_cidx_gfs, rh_cidx_gfs, era5_model
            )
            sums["gfs_forecast_vpd"] = vpd_fc

    elif var_id == "dew_point_2m":
        td_era5_cidx = era5_cidx.get("dew_point_2m")
        t_cidx_gfs = gfs_cidx.get("temperature_2m")
        rh_cidx_gfs = gfs_cidx.get("relative_humidity_2m")

        def _accum_td(start: float, end: float) -> tuple[np.ndarray | None, int]:
            if not t_cidx_gfs or not rh_cidx_gfs or start > end:
                return None, 0
            t_sum, t_n = accumulate_raster("ncep_gfs013", "temperature_2m", start, end, t_cidx_gfs)
            rh_sum, rh_n = accumulate_raster("ncep_gfs013", "relative_humidity_2m", start, end, rh_cidx_gfs)
            n = min(t_n, rh_n)
            if n == 0:
                return None, 0
            td = _derive_dew_point((t_sum / n).astype(np.float32), (rh_sum / n).astype(np.float32))
            r = reproject_to_grid(td, gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                                  dst_g["ny"], dst_g["nx"], dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"])
            return r.astype(np.float64) * n, n

        # ERA5 quality swap
        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start and td_era5_cidx:
                added, _ = accumulate_raster(era5_model, "dew_point_2m", swap_start, swap_end, td_era5_cidx)
                key = "era5_dew_point_2m"
                sums[key] = sums.get(key, np.zeros_like(added)) + added

        # ERA5 window drop
        if new_w_start > old_w_start and td_era5_cidx:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                dropped, _ = accumulate_raster(era5_model, "dew_point_2m", old_w_start, era5_drop_end, td_era5_cidx)
                if "era5_dew_point_2m" in sums:
                    sums["era5_dew_point_2m"] = sums["era5_dew_point_2m"] - dropped

        # Graduate old forecast → stable
        if old_cycle_init_ts > 0 and new_cycle_init_ts > old_cycle_init_ts:
            grad_end = new_cycle_init_ts - resolution
            if grad_end >= old_cycle_init_ts:
                arr, grad_n = _accum_td(old_cycle_init_ts, grad_end)
                if arr is not None:
                    sums["gfs_dew_point_2m"] = sums.get("gfs_dew_point_2m", np.zeros_like(arr)) + arr
                    n_gfs_stable += grad_n

        # Rebuild new forecast
        fc_start = max(new_cycle_init_ts, new_gfs_start) if new_cycle_init_ts > 0 else new_gfs_start
        if fc_start <= gfs_end_ts:
            arr, n_gfs_forecast = _accum_td(fc_start, gfs_end_ts)
            if arr is not None:
                sums["gfs_forecast_dew_point_2m"] = arr

    else:
        # Standard scalar vars
        def _era5_acc(rv: str, start: float, end: float) -> np.ndarray | None:
            cidx = era5_cidx.get(rv)
            if not cidx or end < start:
                return None
            acc, _ = accumulate_raster(era5_model, rv, start, end, cidx)
            return acc.astype(np.float32)

        # ERA5 quality swap
        if era5_end_ts > old_era5_end:
            swap_start = max(old_era5_end + resolution, old_w_start)
            swap_end = min(era5_end_ts, old_gfs_end)
            if swap_end >= swap_start:
                for rv in _era5_raw_vars(cfg):
                    added = _era5_acc(rv, swap_start, swap_end)
                    if added is not None:
                        key = f"era5_{rv}"
                        sums[key] = sums.get(key, np.zeros_like(added)) + added

        # ERA5 window drop
        if new_w_start > old_w_start:
            era5_drop_end = min(new_w_start - resolution, era5_end_ts)
            if era5_drop_end >= old_w_start:
                for rv in _era5_raw_vars(cfg):
                    dropped = _era5_acc(rv, old_w_start, era5_drop_end)
                    if dropped is not None:
                        key = f"era5_{rv}"
                        if key in sums:
                            sums[key] = sums[key] - dropped

        # Graduate [old_cycle_init_ts, new_cycle_init_ts) from forecast → stable
        if old_cycle_init_ts > 0 and new_cycle_init_ts > old_cycle_init_ts:
            grad_end = new_cycle_init_ts - resolution
            if grad_end >= old_cycle_init_ts:
                grad_n = int(round((grad_end - old_cycle_init_ts) / resolution)) + 1
                for gv in _gfs_raw_vars(cfg):
                    arr = _gfs_acc(gv, old_cycle_init_ts, grad_end)
                    if arr is not None:
                        sums[f"gfs_{gv}"] = sums.get(f"gfs_{gv}", np.zeros_like(arr)) + arr.astype(np.float64)
                n_gfs_stable += grad_n

        # Rebuild forecast sums from [new_cycle_init_ts, gfs_end_ts]
        fc_start = max(new_cycle_init_ts, new_gfs_start) if new_cycle_init_ts > 0 else new_gfs_start
        if fc_start <= gfs_end_ts:
            fc_n = int(round((gfs_end_ts - fc_start) / resolution)) + 1
            for gv in _gfs_raw_vars(cfg):
                arr = _gfs_acc(gv, fc_start, gfs_end_ts)
                if arr is not None:
                    sums[f"gfs_forecast_{gv}"] = arr.astype(np.float64)
                    n_gfs_forecast = max(n_gfs_forecast, fc_n)

    n_era5 = (
        int(round((era5_end_ts - new_w_start) / resolution)) + 1
        if new_w_start <= era5_end_ts else 0
    )
    n_era5 = max(n_era5, 0)
    n_gfs_stable = max(n_gfs_stable, 0)
    n_gfs_forecast = max(n_gfs_forecast, 0)
    n_gfs = n_gfs_stable + n_gfs_forecast

    meta = {
        **existing_meta,
        "era5_end_ts": era5_end_ts,
        "era5_window_start_ts": new_w_start,
        "gfs_start_ts": new_gfs_start,
        "gfs_end_ts": gfs_end_ts,
        "gfs_cycle_init_ts": new_cycle_init_ts,
        "n_era5": n_era5,
        "n_gfs": n_gfs,
        "n_gfs_stable": n_gfs_stable,
        "n_gfs_forecast": n_gfs_forecast,
        "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
    }
    save_raster_state(out_dir, var_id, window_label, agg, sums, meta)
    print(f"  [{window_label}] {var_id}: GFS cycle rederive → {n_gfs_stable}h stable + {n_gfs_forecast}h forecast, ERA5 {n_era5}h preserved", flush=True)


# ---------------------------------------------------------------------------
# Forecast aggregates
# ---------------------------------------------------------------------------

def _build_forecast_aggregates(
    var_configs: dict,
    windows: list[tuple[int, str]],
    now_ts: float,
    era5_end_ts: float,
    gfs_data_end_ts: float,
    era5_cidx_by_var: dict[str, dict[str, ChunkIndex]],
    gfs_cidx: dict[str, ChunkIndex],
    out_dir: str,
) -> None:
    print("\n=== forecast aggregate builds ===")
    for forecast_h in FORECAST_HOURS:
        future_ts = now_ts + forecast_h * 3600
        gfs_end_for_fc = min(gfs_data_end_ts, future_ts)
        suffix = f"__f{forecast_h:03d}h"
        print(f"\n--- +{forecast_h}h forecast ---", flush=True)
        t0 = time.perf_counter()

        combos = [(vid, cfg, wh, wl) for vid, cfg in var_configs.items() for wh, wl in windows]

        def _process(vid: str, cfg: dict, wh: int, wl: str) -> None:
            agg = cfg["agg"]
            now_sums, now_meta = load_raster_state(out_dir, vid, wl, suffix=suffix)
            if now_sums is None:
                if wh >= forecast_h:
                    # Window is large enough that the new window start doesn't overshoot
                    # old_gfs_end — safe to slide from the base window state.
                    now_sums, now_meta = load_raster_state(out_dir, vid, wl)
                if now_sums is None:
                    # window_h < forecast_h: sliding would add the entire forecast period
                    # instead of just the window slice at the horizon, so full-build instead.
                    _full_build(vid, cfg, wh, wl, future_ts, era5_end_ts, gfs_end_for_fc,
                                era5_cidx_by_var.get(vid, {}), gfs_cidx, out_dir, suffix=suffix)
                    return
            if float(now_meta.get("gfs_end_ts", 0)) >= gfs_end_for_fc:
                return  # forecast state already up to date

            sums = {k: v.copy() for k, v in now_sums.items()}
            n_era5 = int(now_meta["n_era5"])
            n_gfs = int(now_meta["n_gfs"])
            old_w_start = float(now_meta["era5_window_start_ts"])
            old_gfs_end = float(now_meta["gfs_end_ts"])
            old_gfs_start = float(now_meta["gfs_start_ts"])
            # Target window start for this forecast horizon — works whether we
            # loaded from the base state or from an existing forecast state.
            new_w_start = future_ts - wh * 3600

            era5_cidx = era5_cidx_by_var.get(vid, {})
            era5_model = cfg["era5_model"]
            dst_g = RASTER_GRIDS[era5_model]
            gfs_g = RASTER_GRIDS["ncep_gfs013"]
            resolution = next((v.resolution for k, v in era5_cidx.items() if not k.startswith("_")), 3600.0)

            if agg == "mode":
                cc_cidx = era5_cidx.get("cloud_cover")
                pr_cidx = era5_cidx.get("precipitation")
                sw_cidx = era5_cidx.get("snowfall_water_equivalent")
                gfs_cc = gfs_cidx.get("cloud_cover")
                gfs_pr = gfs_cidx.get("precipitation")
                gfs_sw = gfs_cidx.get("snowfall_water_equivalent")

                def _mode_delta_era5(start: float, end: float) -> dict | None:
                    if end < start or not cc_cidx:
                        return None
                    return accumulate_raster_mode(era5_model, start, end, cc_cidx, pr_cidx, sw_cidx)

                def _mode_delta_gfs(start: float, end: float) -> dict | None:
                    if end < start or not (gfs_cc and gfs_pr and gfs_sw):
                        return None
                    d = accumulate_raster_mode("ncep_gfs013", start, end, gfs_cc, gfs_pr, gfs_sw)
                    era5_g = RASTER_GRIDS["copernicus_era5"]
                    return {c: np.round(reproject_to_grid(
                        d[c].astype(np.float32),
                        gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                        era5_g["ny"], era5_g["nx"],
                        era5_g["lat_min"], era5_g["lat_max"], era5_g["lon_min"], era5_g["lon_max"],
                        resampling="nearest",
                    )).astype(np.int32) for c in RASTER_WC_CODES}

                era5_drop_end = min(new_w_start - resolution, era5_end_ts)
                if era5_drop_end >= old_w_start:
                    delta = _mode_delta_era5(old_w_start, era5_drop_end)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = np.maximum(0, sums[c] - delta[c])
                    n_era5 -= int(round((era5_drop_end - old_w_start) / resolution)) + 1

                gfs_drop_end = min(new_w_start - resolution, old_gfs_end)
                if gfs_drop_end >= old_gfs_start:
                    delta = _mode_delta_gfs(old_gfs_start, gfs_drop_end)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = np.maximum(0, sums[c] - delta[c])
                    n_gfs -= int(round((gfs_drop_end - old_gfs_start) / resolution)) + 1

                if gfs_end_for_fc > old_gfs_end:
                    delta = _mode_delta_gfs(old_gfs_end + resolution, gfs_end_for_fc)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = sums.get(c, np.zeros_like(delta[c])) + delta[c]
                    n_gfs += int(round((gfs_end_for_fc - old_gfs_end) / resolution))

            else:
                cycle_init_ts = float(now_meta.get("gfs_cycle_init_ts", 0.0))
                n_gfs_stable = int(now_meta.get("n_gfs_stable", n_gfs))
                n_gfs_forecast = int(now_meta.get("n_gfs_forecast", 0))

                def _gfs_reproj_fc(gv: str, start: float, end: float) -> np.ndarray | None:
                    cidx = gfs_cidx.get(gv)
                    if not cidx or end < start:
                        return None
                    acc, _ = accumulate_raster("ncep_gfs013", gv, start, end, cidx)
                    return reproject_to_grid(
                        acc.astype(np.float32),
                        gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                        dst_g["ny"], dst_g["nx"],
                        dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
                    )

                # Drop oldest hours from ERA5 tail (exclude new_w_start — it stays in window)
                era5_drop_end = min(new_w_start - resolution, era5_end_ts)
                if era5_drop_end >= old_w_start:
                    drop_n = int(round((era5_drop_end - old_w_start) / resolution)) + 1
                    for rv in _era5_raw_vars(cfg):
                        cidx = era5_cidx.get(rv)
                        if not cidx:
                            continue
                        dropped, _ = accumulate_raster(era5_model, rv, old_w_start, era5_drop_end, cidx)
                        key = f"era5_{rv}"
                        if key in sums:
                            sums[key] = np.maximum(0.0, sums[key] - dropped)
                    n_era5 -= drop_n

                # GFS drops — split at cycle_init_ts
                gfs_drop_end = min(new_w_start - resolution, old_gfs_end)
                if gfs_drop_end >= old_gfs_start:
                    if cycle_init_ts > 0:
                        stable_drop_end = min(gfs_drop_end, cycle_init_ts - resolution)
                        fc_drop_start = max(old_gfs_start, cycle_init_ts)
                    else:
                        stable_drop_end = old_gfs_start - 1  # nothing in stable
                        fc_drop_start = old_gfs_start         # all in forecast
                    for gv in _gfs_raw_vars(cfg):
                        if stable_drop_end >= old_gfs_start:
                            r = _gfs_reproj_fc(gv, old_gfs_start, stable_drop_end)
                            if r is not None and f"gfs_{gv}" in sums:
                                sums[f"gfs_{gv}"] = np.maximum(0.0, sums[f"gfs_{gv}"] - r)
                        if fc_drop_start <= gfs_drop_end:
                            r = _gfs_reproj_fc(gv, fc_drop_start, gfs_drop_end)
                            if r is not None and f"gfs_forecast_{gv}" in sums:
                                sums[f"gfs_forecast_{gv}"] = np.maximum(0.0, sums[f"gfs_forecast_{gv}"] - r)
                    sdn = max(0, int(round((stable_drop_end - old_gfs_start) / resolution)) + 1) if stable_drop_end >= old_gfs_start else 0
                    fdn = max(0, int(round((gfs_drop_end - fc_drop_start) / resolution)) + 1) if fc_drop_start <= gfs_drop_end else 0
                    n_gfs_stable -= sdn
                    n_gfs_forecast -= fdn

                # Add GFS forecast hours → always to forecast bucket (start from old_gfs_end+1 step)
                if gfs_end_for_fc > old_gfs_end:
                    add_n = int(round((gfs_end_for_fc - old_gfs_end) / resolution))
                    for gv in _gfs_raw_vars(cfg):
                        r = _gfs_reproj_fc(gv, old_gfs_end + resolution, gfs_end_for_fc)
                        if r is not None:
                            key = f"gfs_forecast_{gv}"
                            sums[key] = sums.get(key, np.zeros_like(r)) + r
                    n_gfs_forecast += add_n

                n_era5 = max(n_era5, 0)
                n_gfs_stable = max(n_gfs_stable, 0)
                n_gfs_forecast = max(n_gfs_forecast, 0)
                n_gfs = n_gfs_stable + n_gfs_forecast
                fc_meta = {
                    **now_meta,
                    "era5_window_start_ts": new_w_start,
                    "gfs_start_ts": max(era5_end_ts, new_w_start),
                    "gfs_end_ts": gfs_end_for_fc,
                    "n_era5": n_era5, "n_gfs": n_gfs,
                    "n_gfs_stable": n_gfs_stable, "n_gfs_forecast": n_gfs_forecast,
                    "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
                }
                save_raster_state(out_dir, vid, wl, agg, sums, fc_meta, suffix=suffix)
                return

            n_era5 = max(n_era5, 0)
            n_gfs = max(n_gfs, 0)
            fc_meta = {
                **now_meta,
                "era5_window_start_ts": new_w_start,
                "gfs_start_ts": max(era5_end_ts, new_w_start),
                "gfs_end_ts": gfs_end_for_fc,
                "n_era5": n_era5, "n_gfs": n_gfs,
                "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
            }
            save_raster_state(out_dir, vid, wl, agg, sums, fc_meta, suffix=suffix)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_process, vid, cfg, wh, wl): (vid, wl) for vid, cfg, wh, wl in combos}
            for fut in as_completed(futures):
                if exc := fut.exception():
                    vid, wl = futures[fut]
                    print(f"  ERROR [{wl}] {vid} +{forecast_h}h: {exc}", flush=True)

        print(f"  +{forecast_h}h done in {time.perf_counter() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _push_rasters(out_dir: str) -> None:
    """Sync temporal rasters to the configured destination.

    Set WW_TEMPORAL_RASTER_DEST to an rclone remote path to push directly to
    the production server (e.g. "gambaby:/home/gambel/wherewild/backend-v2/data/gis/temporal/rasters").
    Falls back to B2 if unset.
    """
    transfers = os.environ.get("WW_RCLONE_TRANSFERS", "16")
    dest = os.environ.get("WW_TEMPORAL_RASTER_DEST")
    if not dest:
        remote = os.environ.get("WW_B2_WRITER_REMOTE", "wherewild-localdev-writer")
        bucket = os.environ.get("WW_B2_BUCKET", "wherewild-data")
        prefix = os.environ.get("WW_B2_PREFIX", "data")
        dest = f"{remote}:{bucket}/{prefix}/gis/temporal/rasters" if prefix else f"{remote}:{bucket}/gis/temporal/rasters"
    print(f"\n=== pushing rasters to: {dest} ===")
    result = subprocess.run([
        "rclone", "sync", out_dir, dest,
        "--fast-list",
        "--transfers", transfers,
        "--stats-one-line",
        "--stats", "1m",
    ])
    if result.returncode != 0:
        print(f"  rclone sync exited {result.returncode}")
    else:
        print("  upload complete")


def _prefetch_chunks(
    gfs_cidx: dict[str, ChunkIndex],
    era5_cidx_flat: dict[tuple[str, str], ChunkIndex],  # (era5_model, raw_var) → ChunkIndex
    gfs_end_ts: float,
    era5_end_ts: float,
    old_era5_end_ts: float,
    window_hours: list[int],
    cache_dir: str,
) -> set[str]:
    """Download all chunk files needed for this run in parallel before processing.

    Covers:
    - GFS: add position (gfs_end_ts) + all window drop positions
    - ERA5: quality-swap zone ([old_era5_end, era5_end_ts]) when ERA5 has advanced

    Multiple timestamps that resolve to the same file are deduplicated.
    Returns the set of local filenames that are needed (used by cleanup).
    """
    # needed maps local_filename → (entry, model, raw_var, is_frontier)
    # is_frontier=True triggers an S3 freshness check on the cached file so that
    # new GFS model cycles (which rewrite the rolling frontier chunk in-place on
    # S3) are picked up without waiting for the chunk to age out of the cache.
    needed: dict[str, tuple[object, str, str, bool]] = {}

    # GFS: add step + all window drop steps
    for gv, cidx in gfs_cidx.items():
        # The chunk containing gfs_end_ts is the live frontier — Open-Meteo
        # rewrites it on every GFS cycle run, so we must check S3 freshness.
        frontier_entry, _ = _chunk_entry_for_time(cidx, gfs_end_ts)
        frontier_chunk_num = frontier_entry.chunk_num if frontier_entry is not None else None

        timestamps = [gfs_end_ts] + [gfs_end_ts - wh * 3600 for wh in window_hours]
        for ts in timestamps:
            entry, _ = _chunk_entry_for_time(cidx, ts)
            if entry is not None and entry.source != "year":
                key = f"ncep_gfs013_{gv}_chunk_{entry.chunk_num}.om"
                is_frontier = entry.chunk_num == frontier_chunk_num
                # Preserve is_frontier=True if already set by an earlier timestamp
                if key not in needed or is_frontier:
                    needed[key] = (entry, "ncep_gfs013", gv, is_frontier)

    # ERA5: quality-swap zone — when ERA5 has advanced since last run we need
    # both the ERA5 data to add and the ERA5 chunk files covering that period.
    # ERA5 chunks are sealed/immutable once written; no freshness check needed.
    if era5_end_ts > old_era5_end_ts:
        for (era5_model, rv), cidx in era5_cidx_flat.items():
            for ts in (old_era5_end_ts + cidx.resolution, era5_end_ts):
                entry, _ = _chunk_entry_for_time(cidx, ts)
                if entry is not None and entry.source != "year":
                    key = f"{era5_model}_{rv}_chunk_{entry.chunk_num}.om"
                    if key not in needed:
                        needed[key] = (entry, era5_model, rv, False)

    if not needed:
        return set()

    frontier_count = sum(1 for _, _, _, f in needed.values() if f)
    print(f"  pre-fetching {len(needed)} chunk(s) ({frontier_count} frontier) in parallel …", flush=True)
    t0 = time.perf_counter()

    def _fetch(entry: object, model: str, rv: str, is_frontier: bool) -> None:
        _download_chunk(entry, model, rv, cache_dir, check_freshness=is_frontier)

    with ThreadPoolExecutor(max_workers=len(needed)) as pool:
        futures = [pool.submit(_fetch, e, m, v, f) for e, m, v, f in needed.values()]
        for f in futures:
            f.result()

    print(f"  pre-fetch done in {time.perf_counter() - t0:.1f}s", flush=True)
    return set(needed.keys())


def _cleanup_chunk_cache(cache_dir: Path, needed_filenames: set[str]) -> None:
    """Remove chunk_*.om files that aren't needed for this run or the next.

    Any chunk not in needed_filenames is stale — either too old to fall within
    any aggregation window or superseded by a newer chunk.  year_*.om files are
    never touched (they're immutable and kept forever).
    """
    if not cache_dir.exists():
        return

    removed = 0
    for p in cache_dir.rglob("chunk_*.om"):
        if p.name not in needed_filenames:
            try:
                p.unlink()
                removed += 1
            except Exception as exc:
                print(f"  [cleanup] could not remove {p}: {exc}")

    if removed:
        print(f"  [cleanup] removed {removed} stale chunk_*.om file(s) from {cache_dir}")


def main() -> None:
    cfg = load_config("global")
    out_dir = cfg.temporal_raster_out_dir
    force = cfg.temporal_raster_force_rebuild
    only_vars = [v.strip() for v in cfg.temporal_raster_vars.split(",") if v.strip()] or None
    only_windows = [w.strip() for w in cfg.temporal_raster_windows.split(",") if w.strip()] or None

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    set_raster_chunk_cache(str(Path(out_dir).parent / "chunks"))

    t_main = time.perf_counter()

    var_configs = {k: v for k, v in VAR_CONFIGS.items() if only_vars is None or k in only_vars}
    windows = [(h, WINDOW_LABELS[h]) for h in WINDOW_HOURS if only_windows is None or WINDOW_LABELS[h] in only_windows]

    # ── Probe GFS for snow_depth ──────────────────────────────────────────
    fs = fsspec.filesystem("s3", anon=True)
    try:
        fs.ls("s3://openmeteo/data/ncep_gfs013/snow_depth")
        print("GFS snow_depth: available")
    except Exception:
        if "snow_depth" in var_configs:
            var_configs["snow_depth"] = {k: v for k, v in var_configs["snow_depth"].items() if k != "gfs_var"}
        print("GFS snow_depth: not found, ERA5-land only")

    # ── Read S3 model metadata ────────────────────────────────────────────
    def _read_meta(model: str) -> dict:
        with fs.open(f"s3://openmeteo/data/{model}/static/meta.json") as fh:
            return json.loads(fh.read())

    era5_meta = _read_meta("copernicus_era5")
    era5_land_meta = _read_meta("copernicus_era5_land")
    gfs_meta = _read_meta("ncep_gfs013")

    era5_end_ts = float(era5_meta["data_end_time"])
    era5_land_end_ts = float(era5_land_meta["data_end_time"])
    gfs_data_end_ts = float(gfs_meta["data_end_time"])

    now_ts = round(datetime.now(UTC).timestamp() / 3600) * 3600
    gfs_end_ts = min(gfs_data_end_ts, now_ts)
    max_window_h = max(h for h, _ in windows)

    print(f"ERA5      ends: {datetime.fromtimestamp(era5_end_ts, tz=UTC).strftime('%Y-%m-%dT%HZ')}")
    print(f"ERA5-land ends: {datetime.fromtimestamp(era5_land_end_ts, tz=UTC).strftime('%Y-%m-%dT%HZ')}")
    print(f"GFS       ends: {datetime.fromtimestamp(gfs_end_ts, tz=UTC).strftime('%Y-%m-%dT%HZ')}")
    print(f"Now:            {datetime.fromtimestamp(now_ts, tz=UTC).strftime('%Y-%m-%dT%HZ')}")
    print(f"Max window: {max_window_h}h  Force: {force}\n")

    state_path = Path(out_dir) / "temporal_state.json"
    started_at = datetime.now(UTC)

    # Read prior state before overwriting it
    prior_state: dict = {}
    if state_path.exists():
        try:
            prior_state = json.loads(state_path.read_text())
        except Exception:
            pass

    # If a prior run is still in-progress and its PID is alive, exit cleanly.
    if prior_state.get("status") == "running":
        running_pid = prior_state.get("pid")
        if running_pid is not None:
            try:
                os.kill(int(running_pid), 0)
                alive = True
            except (ProcessLookupError, PermissionError):
                alive = False
            if alive:
                print(f"build_temporal already running (pid {running_pid}), exiting.")
                return

    same_data = (
        prior_state.get("era5_end_ts") == era5_end_ts
        and prior_state.get("gfs_end_ts") == gfs_end_ts
    )

    # Skip entirely only if last run completed successfully with the same data
    if not force and prior_state.get("status") == "completed" and same_data:
        print("=== no new S3 data since last completed run, skipping ===")
        return

    # Resume vars already finished in a prior interrupted run with the same data
    resume_completed: set[str] = set()
    if same_data and prior_state.get("status") in ("running", "failed"):
        resume_completed = set(prior_state.get("completed_vars", []))
        if resume_completed:
            print(f"=== resuming: skipping {len(resume_completed)} already-completed var(s) ===")

    completed_vars: list[str] = list(resume_completed)
    forecast_done: list[bool] = [
        same_data and bool(prior_state.get("forecast_completed"))
    ]
    gfs_chunk_last_modified: dict = {}  # populated after cycle detection
    new_cycle_init_ts: float = 0.0     # set after cycle detection

    def _write_state(status: str, *, skipped: bool = False, error: str | None = None) -> None:
        state: dict = {
            "status": status,
            "pid": os.getpid() if status == "running" else None,
            "started_at": started_at.isoformat(),
            "era5_end_ts": era5_end_ts,
            "gfs_end_ts": gfs_end_ts,
            "skipped": skipped,
            "completed_vars": completed_vars,
            "forecast_completed": forecast_done[0],
            "gfs_chunk_last_modified": gfs_chunk_last_modified,
        }
        if status == "completed":
            state["completed_at"] = datetime.now(UTC).isoformat()
            state["duration_s"] = round(time.perf_counter() - t_main)
        if error is not None:
            state["error"] = error
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_path)
        _push_temporal_state(state)

    _write_state("running")

    try:
        # ── Build chunk indices ────────────────────────────────────────────────
        print("=== building chunk indices ===")

        # Collect all GFS vars needed
        gfs_raw_needed: set[str] = set()
        for vcfg in var_configs.values():
            gfs_raw_needed.update(_gfs_raw_vars(vcfg))
        # weather_code needs mode GFS sources
        if "weather_code_simple" in var_configs:
            gfs_raw_needed.update(["cloud_cover", "precipitation", "snowfall_water_equivalent"])

        gfs_cidx: dict[str, ChunkIndex] = {}
        for gv in sorted(gfs_raw_needed):
            try:
                gfs_cidx[gv] = build_chunk_index("ncep_gfs013", gv)
                print(f"  GFS {gv}: {len(gfs_cidx[gv].ranges)} range(s)")
            except Exception as e:
                print(f"  GFS {gv}: unavailable ({e})")

        # Detect GFS cycle update via S3 LastModified comparison.
        # chunk_num only changes ~every 20 days; LastModified changes every 6h cycle.
        import fsspec as _fsspec
        _fs = _fsspec.filesystem("s3", anon=True)
        prev_lm: dict = prior_state.get("gfs_chunk_last_modified", {})
        max_window_h = max((h for h, _ in windows), default=168)
        gfs_cycle_changed, new_cycle_init_ts, gfs_chunk_last_modified = _gfs_detect_cycle(
            gfs_cidx, gfs_end_ts, max_window_h, _fs, prev_lm
        )
        if gfs_cycle_changed:
            from datetime import datetime as _dt
            cycle_label = _dt.fromtimestamp(new_cycle_init_ts, tz=UTC).strftime("%Y-%m-%dT%HZ")
            print(f"  GFS cycle changed → new cycle_init_ts={cycle_label}, will re-derive GFS sums", flush=True)

        era5_cidx_by_var: dict[str, dict[str, ChunkIndex]] = {}
        for var_id, vcfg in var_configs.items():
            era5_model = vcfg["era5_model"]
            era5_cidx: dict[str, ChunkIndex] = {}
            raw_vars = _era5_raw_vars(vcfg)
            # For weather_code, also pre-fetch temperature (ERA5 0.25° for snow/rain cutoff)
            if var_id == "weather_code_simple":
                raw_vars = ["cloud_cover", "precipitation", "snowfall_water_equivalent"]
                try:
                    era5_cidx["_temperature_for_wc"] = build_chunk_index("copernicus_era5", "temperature_2m")
                except Exception as e:
                    print(f"  ERA5 temperature_2m (for weather_code): {e}")
            for rv in raw_vars:
                try:
                    era5_cidx[rv] = build_chunk_index(era5_model, rv)
                    print(f"  ERA5 {var_id}/{rv}: {len(era5_cidx[rv].ranges)} range(s)")
                except Exception as e:
                    print(f"  ERA5 {var_id}/{rv}: {e}")
            era5_cidx_by_var[var_id] = era5_cidx

        # Flatten ERA5 indices to (model, raw_var) → ChunkIndex for the prefetch.
        era5_cidx_flat: dict[tuple[str, str], ChunkIndex] = {}
        for var_id, vcfg in var_configs.items():
            era5_model = vcfg["era5_model"]
            for rv, cidx in era5_cidx_by_var.get(var_id, {}).items():
                if not rv.startswith("_"):
                    era5_cidx_flat[(era5_model, rv)] = cidx

        old_era5_end_ts = float(prior_state.get("era5_end_ts") or 0)

        # ── Pre-fetch all needed chunks in parallel ────────────────────────────
        chunk_cache_dir = Path(out_dir).parent / "chunks"
        _needed_chunks = _prefetch_chunks(
            gfs_cidx, era5_cidx_flat,
            gfs_end_ts, era5_end_ts, old_era5_end_ts,
            [h for h, _ in windows],
            str(chunk_cache_dir),
        )

        # ── Main window loop ───────────────────────────────────────────────────
        _state_lock = threading.Lock()

        def _process_var(var_id: str, vcfg: dict, window_h: int, wl: str) -> None:
            var_key = f"{var_id}_{wl}"
            if var_key in resume_completed:
                print(f"  [{wl}] {var_id} already completed, skipping", flush=True)
                return
            era5_model = vcfg["era5_model"]
            era5_end = era5_land_end_ts if era5_model == "copernicus_era5_land" else era5_end_ts
            era5_cidx = era5_cidx_by_var.get(var_id, {})
            existing_sums, existing_meta = load_raster_state(out_dir, var_id, wl)
            # Migration: state missing gfs_cycle_init_ts → treat as stale, force full build
            needs_migration = (existing_meta is not None
                               and "gfs_cycle_init_ts" not in existing_meta
                               and vcfg.get("agg") != "mode")
            if force or existing_sums is None or needs_migration:
                reason = "migration" if needs_migration else ("force" if force else "no state")
                print(f"  [{wl}] {var_id} full build ({reason}) …", flush=True)
                _full_build(var_id, vcfg, window_h, wl, now_ts, era5_end, gfs_end_ts,
                            era5_cidx, gfs_cidx, out_dir, cycle_init_ts=new_cycle_init_ts)
            elif gfs_cycle_changed:
                print(f"  [{wl}] {var_id} GFS cycle re-derive …", flush=True)
                _gfs_rederive(var_id, vcfg, window_h, wl, existing_sums, existing_meta,
                              now_ts, era5_end, gfs_end_ts, era5_cidx, gfs_cidx, out_dir,
                              new_cycle_init_ts=new_cycle_init_ts)
            else:
                stale_h = (now_ts - float(existing_meta["gfs_end_ts"])) / 3600
                print(f"  [{wl}] {var_id} incremental ({stale_h:.1f}h stale) …", flush=True)
                _incremental_update(var_id, vcfg, window_h, wl, existing_sums, existing_meta,
                                    now_ts, era5_end, gfs_end_ts, era5_cidx, gfs_cidx, out_dir)
            with _state_lock:
                completed_vars.append(var_key)
                _write_state("running")

        t_windows = time.perf_counter()
        all_combos = [(vid, vcfg, wh, wl) for wh, wl in windows for vid, vcfg in var_configs.items()]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_process_var, vid, vcfg, wh, wl): (vid, wl)
                       for vid, vcfg, wh, wl in all_combos}
            for fut in as_completed(futures):
                if exc := fut.exception():
                    vid, wl = futures[fut]
                    print(f"  ERROR [{wl}] {vid}: {exc}", flush=True)
        print(f"\n=== windows done in {time.perf_counter() - t_windows:.1f}s ===")

        # ── Forecast aggregates ────────────────────────────────────────────────
        if forecast_done[0]:
            print("=== forecast aggregates already completed, skipping ===")
        else:
            _build_forecast_aggregates(
                var_configs, windows, now_ts, era5_end_ts, gfs_data_end_ts,
                era5_cidx_by_var, gfs_cidx, out_dir,
            )
            forecast_done[0] = True
            _write_state("running")

        print(f"\n=== total {time.perf_counter() - t_main:.1f}s ===")

        # ── Clean up raster chunk cache ────────────────────────────────────────
        # Keep the latest chunk per variable (reused next run); delete older ones.
        _cleanup_chunk_cache(chunk_cache_dir, _needed_chunks)

        # ── Push rasters to B2 ────────────────────────────────────────────────
        if os.environ.get("TEMPORAL_RASTER_NO_PUSH", "0") != "1":
            _push_rasters(out_dir)

        _write_state("completed")

    except Exception as exc:
        _write_state("failed", error=str(exc))
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
