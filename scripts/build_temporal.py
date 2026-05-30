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
  - ERA5-land (copernicus_era5_land, 0.1°): temp, soil, dew_point, snow_depth
  - ERA5      (copernicus_era5,     0.25°): cloud, precip, swe, weather_code
  - GFS013    (ncep_gfs013,         0.125°): ~6-day ERA5 gap fill

Output resolutions match the primary source (no blanket 0.25° downsampling).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np

from config.config import load_config
from util.temporal import (
    RASTER_GRIDS,
    RASTER_WC_CODES,
    ChunkIndex,
    accumulate_raster,
    accumulate_raster_mode,
    build_chunk_index,
    load_raster_state,
    reproject_to_grid,
    save_raster_state,
    set_raster_chunk_cache,
)

# ---------------------------------------------------------------------------
# Variable config
# ---------------------------------------------------------------------------

WINDOW_HOURS = [1, 8, 24, 72, 168, 720, 2160]
WINDOW_LABELS: dict[int, str] = {1: "1h", 8: "8h", 24: "24h", 72: "3d", 168: "7d", 720: "30d", 2160: "90d"}
FORECAST_HOURS = [1, 8, 24, 72, 168]

VAR_CONFIGS: dict[str, dict] = {
    "temperature_2m": {
        "era5_model": "copernicus_era5_land",
        "era5_var": "temperature_2m",
        "gfs_var": "temperature_2m",
        "agg": "avg",
    },
    "dew_point_2m": {
        "era5_model": "copernicus_era5_land",
        "era5_var": "dew_point_2m",
        "gfs_derived_needs": ["temperature_2m", "relative_humidity_2m"],
        "agg": "avg",
    },
    "soil_temperature_0_to_7cm": {
        "era5_model": "copernicus_era5_land",
        "era5_var": "soil_temperature_0_to_7cm",
        "gfs_var": "soil_temperature_0_to_10cm",
        "agg": "avg",
    },
    "soil_moisture_0_to_7cm": {
        "era5_model": "copernicus_era5_land",
        "era5_var": "soil_moisture_0_to_7cm",
        "gfs_var": "soil_moisture_0_to_10cm",
        "agg": "avg",
    },
    "snow_depth": {
        "era5_model": "copernicus_era5_land",
        "era5_var": "snow_depth",
        "gfs_var": "snow_depth",  # probed at startup; removed if absent from GFS S3
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
        "era5_model": "copernicus_era5_land",
        "era5_derived_needs": ["temperature_2m", "dew_point_2m"],
        "gfs_derived_needs": ["temperature_2m", "relative_humidity_2m"],
        "agg": "avg",
    },
    "weather_code_simple": {
        "era5_model": "copernicus_era5",
        "agg": "mode",
        # sources: cloud_cover, precipitation, snowfall_water_equivalent
        # temperature_2m reprojected from ERA5-land 0.1° → 0.25° for snow/rain cutoff
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
) -> None:
    era5_model = cfg["era5_model"]
    agg = cfg["agg"]
    w_start = now_ts - max(window_h - 1, 1) * 3600

    sums: dict[str, np.ndarray] = {}
    n_era5, n_gfs = 0, 0

    # GFS starts one step AFTER era5_end_ts so the boundary hour is not counted
    # in both ERA5 and GFS sums simultaneously.
    resolution = next((c.resolution for c in list(era5_cidx.values()) + list(gfs_cidx.values()) if c is not None), 3600.0)
    gfs_start = max(era5_end_ts + resolution, w_start)

    if agg == "mode":
        cc_cidx = era5_cidx.get("cloud_cover")
        pr_cidx = era5_cidx.get("precipitation")
        sw_cidx = era5_cidx.get("snowfall_water_equivalent")
        t_cidx = era5_cidx.get("_temperature_for_wc")  # ERA5-land temp, pre-fetched

        if not cc_cidx:
            print(f"  [{window_label}] {var_id}: no ERA5 cloud_cover, skipping")
            return

        # Reproject temperature to 0.25° ERA5 grid once per window if available
        temp_grid_025: np.ndarray | None = None
        if t_cidx:
            t_sum, t_n = accumulate_raster("copernicus_era5_land", "temperature_2m",
                                           w_start, era5_end_ts, t_cidx)
            if t_n > 0:
                t_avg = (t_sum / t_n).astype(np.float32)
                temp_grid_025 = _reproject_gfs_to(t_avg, "copernicus_era5")

        era5_counts = accumulate_raster_mode(
            era5_model, w_start, era5_end_ts,
            cc_cidx, pr_cidx, sw_cidx,  # type: ignore[arg-type]
            temp_grid_025=temp_grid_025,
        )
        for c in RASTER_WC_CODES:
            sums[c] = era5_counts[c]
        n_era5 = max(0, int(round((era5_end_ts - w_start) / cc_cidx.resolution)))

        # GFS gap fill for mode
        gfs_cc = gfs_cidx.get("cloud_cover")
        gfs_pr = gfs_cidx.get("precipitation")
        gfs_sw = gfs_cidx.get("snowfall_water_equivalent")
        gfs_mode_start = gfs_start
        if gfs_cc and gfs_pr and gfs_sw and gfs_mode_start < gfs_end_ts:
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
                )
                sums[c] = sums[c] + np.round(gfs_reproj).astype(np.int32)
            n_gfs = max(0, int(round((gfs_end_ts - gfs_mode_start) / gfs_cc.resolution)))

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

        # GFS portion: starts one step after era5_end_ts (gfs_start computed above)
        dst_g = RASTER_GRIDS[era5_model]
        gfs_g = RASTER_GRIDS["ncep_gfs013"]

        if var_id == "dew_point_2m":
            # Derive GFS dew point from T + RH
            t_cidx = gfs_cidx.get("temperature_2m")
            rh_cidx = gfs_cidx.get("relative_humidity_2m")
            if t_cidx and rh_cidx and gfs_start < gfs_end_ts:
                t_sum, t_n = accumulate_raster("ncep_gfs013", "temperature_2m", gfs_start, gfs_end_ts, t_cidx)
                rh_sum, rh_n = accumulate_raster("ncep_gfs013", "relative_humidity_2m", gfs_start, gfs_end_ts, rh_cidx)
                n = min(t_n, rh_n)
                if n > 0:
                    t_avg = (t_sum / n).astype(np.float32)
                    rh_avg = (rh_sum / n).astype(np.float32)
                    td_gfs = _derive_dew_point(t_avg, rh_avg)
                    td_repr = reproject_to_grid(
                        td_gfs,
                        gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                        dst_g["ny"], dst_g["nx"],
                        dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
                    )
                    sums["gfs_dew_point_2m"] = td_repr.astype(np.float64) * n
                    n_gfs = n

        elif var_id == "vapor_pressure_deficit":
            t_cidx = gfs_cidx.get("temperature_2m")
            rh_cidx = gfs_cidx.get("relative_humidity_2m")
            if t_cidx and rh_cidx and gfs_start < gfs_end_ts:
                t_sum, t_n = accumulate_raster("ncep_gfs013", "temperature_2m", gfs_start, gfs_end_ts, t_cidx)
                rh_sum, rh_n = accumulate_raster("ncep_gfs013", "relative_humidity_2m", gfs_start, gfs_end_ts, rh_cidx)
                n = min(t_n, rh_n)
                if n > 0:
                    t_avg = (t_sum / n).astype(np.float32)
                    rh_avg = (rh_sum / n).astype(np.float32)
                    td_gfs = _derive_dew_point(t_avg, rh_avg)
                    t_repr = reproject_to_grid(
                        t_avg,
                        gfs_g["lat_min"], gfs_g["lat_max"],
                        gfs_g["lon_min"], gfs_g["lon_max"],
                        dst_g["ny"], dst_g["nx"],
                        dst_g["lat_min"], dst_g["lat_max"],
                        dst_g["lon_min"], dst_g["lon_max"],
                    )
                    td_repr = reproject_to_grid(
                        td_gfs,
                        gfs_g["lat_min"], gfs_g["lat_max"],
                        gfs_g["lon_min"], gfs_g["lon_max"],
                        dst_g["ny"], dst_g["nx"],
                        dst_g["lat_min"], dst_g["lat_max"],
                        dst_g["lon_min"], dst_g["lon_max"],
                    )
                    sums["gfs_temperature_2m"] = t_repr.astype(np.float64) * n
                    sums["gfs_dew_point_2m"] = td_repr.astype(np.float64) * n
                    n_gfs = n

        else:
            for gv in _gfs_raw_vars(cfg):
                cidx = gfs_cidx.get(gv)
                if not cidx or gfs_start >= gfs_end_ts:
                    continue
                acc, n = accumulate_raster("ncep_gfs013", gv, gfs_start, gfs_end_ts, cidx)
                reproj = reproject_to_grid(
                    acc.astype(np.float32),
                    gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                    dst_g["ny"], dst_g["nx"],
                    dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
                )
                sums[f"gfs_{gv}"] = reproj.astype(np.float64)
                n_gfs = max(n_gfs, n)

    meta = {
        "var_id": var_id, "window_h": window_h, "window_label": window_label,
        "era5_window_start_ts": w_start, "era5_end_ts": era5_end_ts,
        "gfs_start_ts": gfs_start,
        "gfs_end_ts": gfs_end_ts,
        "n_era5": n_era5, "n_gfs": n_gfs,
        "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
    }
    save_raster_state(out_dir, var_id, window_label, agg, sums, meta, suffix=suffix)
    print(f"  [{window_label}] {var_id}: {n_era5}h ERA5 + {n_gfs}h GFS → {out_dir}/{var_id}_{window_label}{suffix}.npy")


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
    n_gfs = int(old_meta["n_gfs"])

    dst_g = RASTER_GRIDS[era5_model]
    gfs_g = RASTER_GRIDS["ncep_gfs013"]

    if agg == "mode":
        # Mode incremental: sums contains {wc_code: count_grid}
        cc_cidx = era5_cidx.get("cloud_cover")
        pr_cidx = era5_cidx.get("precipitation")
        sw_cidx = era5_cidx.get("snowfall_water_equivalent")
        if not cc_cidx:
            return

        resolution = cc_cidx.resolution

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

        # Add newest GFS hours — start at old_gfs_end + 1h to avoid double-counting
        if gfs_end_ts > old_gfs_end:
            add = _mode_accumulate(old_gfs_end + resolution, gfs_end_ts, use_gfs=True)
            if add:
                for c in RASTER_WC_CODES:
                    sums[c] = sums[c] + add[c]
            n_gfs += int(round((gfs_end_ts - old_gfs_end) / cc_cidx.resolution))

    else:
        # Scalar incremental
        resolution = next(iter(era5_cidx.values())).resolution if era5_cidx else 3600.0

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

        # ERA5 quality swap: replace GFS with ERA5 for [old_era5_end+1h, new_era5_end]
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
                for gv in _gfs_raw_vars(cfg):
                    dropped = _accum_gfs_reproj(gv, swap_start, swap_end)
                    if dropped is not None:
                        key = f"gfs_{gv}"
                        if key in sums:
                            sums[key] = sums[key] - dropped
                n_era5 += swap_n
                n_gfs -= swap_n
                old_gfs_start = era5_end_ts + resolution

        # Drop oldest hours — new_w_start is the first point of the new window, so drop up to new_w_start - 1h
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
                drop_n = int(round((gfs_drop_end - gfs_drop_start) / resolution)) + 1
                for gv in _gfs_raw_vars(cfg):
                    dropped = _accum_gfs_reproj(gv, gfs_drop_start, gfs_drop_end)
                    if dropped is not None:
                        key = f"gfs_{gv}"
                        if key in sums:
                            sums[key] = sums[key] - dropped
                n_gfs -= drop_n

        # Add newest GFS hours — start at old_gfs_end + 1h to avoid double-counting
        if gfs_end_ts > old_gfs_end:
            add_n = int(round((gfs_end_ts - old_gfs_end) / resolution))
            for gv in _gfs_raw_vars(cfg):
                added = _accum_gfs_reproj(gv, old_gfs_end + resolution, gfs_end_ts)
                if added is not None:
                    key = f"gfs_{gv}"
                    sums[key] = sums.get(key, np.zeros_like(added)) + added
            n_gfs += add_n

    n_era5 = max(n_era5, 0)
    n_gfs = max(n_gfs, 0)
    meta = {
        **old_meta,
        "era5_end_ts": era5_end_ts,
        "era5_window_start_ts": new_w_start,
        "gfs_start_ts": max(era5_end_ts + resolution, new_w_start),
        "gfs_end_ts": gfs_end_ts,
        "n_era5": n_era5, "n_gfs": n_gfs,
        "built_at": datetime.fromtimestamp(now_ts, tz=UTC).isoformat(),
    }
    save_raster_state(out_dir, var_id, window_label, agg, sums, meta)


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
            # Try the existing forecast state first (incremental update path).
            # Fall back to base state only if no forecast state exists yet.
            now_sums, now_meta = load_raster_state(out_dir, vid, wl, suffix=suffix)
            if now_sums is None:
                now_sums, now_meta = load_raster_state(out_dir, vid, wl)
            if now_sums is None:
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
                    )).astype(np.int32) for c in RASTER_WC_CODES}

                era5_drop_end = min(new_w_start, era5_end_ts)
                if era5_drop_end > old_w_start:
                    delta = _mode_delta_era5(old_w_start, era5_drop_end)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = np.maximum(0, sums[c] - delta[c])
                    n_era5 -= int(round((era5_drop_end - old_w_start) / resolution))

                gfs_drop_end = min(new_w_start, old_gfs_end)
                if gfs_drop_end > old_gfs_start:
                    delta = _mode_delta_gfs(old_gfs_start, gfs_drop_end)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = np.maximum(0, sums[c] - delta[c])
                    n_gfs -= int(round((gfs_drop_end - old_gfs_start) / resolution))

                if gfs_end_for_fc > old_gfs_end:
                    delta = _mode_delta_gfs(old_gfs_end, gfs_end_for_fc)
                    if delta:
                        for c in RASTER_WC_CODES:
                            sums[c] = sums.get(c, np.zeros_like(delta[c])) + delta[c]
                    n_gfs += int(round((gfs_end_for_fc - old_gfs_end) / resolution))

            else:
                # Drop oldest forecast_h hours from ERA5 tail
                era5_drop_end = min(new_w_start, era5_end_ts)
                if era5_drop_end > old_w_start:
                    drop_n = int(round((era5_drop_end - old_w_start) / resolution))
                    for rv in _era5_raw_vars(cfg):
                        cidx = era5_cidx.get(rv)
                        if not cidx:
                            continue
                        dropped, _ = accumulate_raster(era5_model, rv, old_w_start, era5_drop_end, cidx)
                        key = f"era5_{rv}"
                        if key in sums:
                            sums[key] = np.maximum(0.0, sums[key] - dropped)
                    n_era5 -= drop_n

                gfs_drop_end = min(new_w_start, old_gfs_end)
                if gfs_drop_end > old_gfs_start:
                    drop_n = int(round((gfs_drop_end - old_gfs_start) / resolution))
                    for gv in _gfs_raw_vars(cfg):
                        cidx = gfs_cidx.get(gv)
                        if not cidx:
                            continue
                        dropped, _ = accumulate_raster("ncep_gfs013", gv, old_gfs_start, gfs_drop_end, cidx)
                        reproj = reproject_to_grid(
                            dropped.astype(np.float32),
                            gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                            dst_g["ny"], dst_g["nx"],
                            dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
                        )
                        key = f"gfs_{gv}"
                        if key in sums:
                            sums[key] = np.maximum(0.0, sums[key] - reproj)
                    n_gfs -= drop_n

                # Add GFS forecast hours [old_gfs_end → future_ts]
                if gfs_end_for_fc > old_gfs_end:
                    add_n = int(round((gfs_end_for_fc - old_gfs_end) / resolution))
                    for gv in _gfs_raw_vars(cfg):
                        cidx = gfs_cidx.get(gv)
                        if not cidx:
                            continue
                        added, _ = accumulate_raster("ncep_gfs013", gv, old_gfs_end, gfs_end_for_fc, cidx)
                        reproj = reproject_to_grid(
                            added.astype(np.float32),
                            gfs_g["lat_min"], gfs_g["lat_max"], gfs_g["lon_min"], gfs_g["lon_max"],
                            dst_g["ny"], dst_g["nx"],
                            dst_g["lat_min"], dst_g["lat_max"], dst_g["lon_min"], dst_g["lon_max"],
                        )
                        key = f"gfs_{gv}"
                        sums[key] = sums.get(key, np.zeros_like(reproj)) + reproj
                    n_gfs += add_n

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

        with ThreadPoolExecutor(max_workers=4) as pool:
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
        }
        if status == "completed":
            state["completed_at"] = datetime.now(UTC).isoformat()
            state["duration_s"] = round(time.perf_counter() - t_main)
        if error is not None:
            state["error"] = error
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_path)

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

        era5_cidx_by_var: dict[str, dict[str, ChunkIndex]] = {}
        for var_id, vcfg in var_configs.items():
            era5_model = vcfg["era5_model"]
            era5_cidx: dict[str, ChunkIndex] = {}
            raw_vars = _era5_raw_vars(vcfg)
            # For weather_code, also pre-fetch temperature from ERA5-land
            if var_id == "weather_code_simple":
                raw_vars = ["cloud_cover", "precipitation", "snowfall_water_equivalent"]
                try:
                    era5_cidx["_temperature_for_wc"] = build_chunk_index("copernicus_era5_land", "temperature_2m")
                except Exception as e:
                    print(f"  ERA5-land temperature_2m (for weather_code): {e}")
            for rv in raw_vars:
                try:
                    era5_cidx[rv] = build_chunk_index(era5_model, rv)
                    print(f"  ERA5 {var_id}/{rv}: {len(era5_cidx[rv].ranges)} range(s)")
                except Exception as e:
                    print(f"  ERA5 {var_id}/{rv}: {e}")
            era5_cidx_by_var[var_id] = era5_cidx

        # ── Main window loop ───────────────────────────────────────────────────
        def _process_var(var_id: str, vcfg: dict, window_h: int, wl: str) -> None:
            var_key = f"{var_id}_{wl}"
            if var_key in resume_completed:
                print(f"  [{wl}] {var_id} already completed, skipping", flush=True)
                return
            era5_model = vcfg["era5_model"]
            era5_end = era5_land_end_ts if era5_model == "copernicus_era5_land" else era5_end_ts
            era5_cidx = era5_cidx_by_var.get(var_id, {})
            existing_sums, existing_meta = load_raster_state(out_dir, var_id, wl)
            if force or existing_sums is None:
                print(f"  [{wl}] {var_id} full build …", flush=True)
                _full_build(var_id, vcfg, window_h, wl, now_ts, era5_end, gfs_end_ts,
                            era5_cidx, gfs_cidx, out_dir)
            else:
                stale_h = (now_ts - float(existing_meta["gfs_end_ts"])) / 3600
                print(f"  [{wl}] {var_id} incremental ({stale_h:.1f}h stale) …", flush=True)
                _incremental_update(var_id, vcfg, window_h, wl, existing_sums, existing_meta,
                                    now_ts, era5_end, gfs_end_ts, era5_cidx, gfs_cidx, out_dir)
            completed_vars.append(var_key)
            _write_state("running")

        t_windows = time.perf_counter()
        for window_h, wl in windows:
            print(f"\n=== window {wl} ===")
            for vid, vcfg in var_configs.items():
                try:
                    _process_var(vid, vcfg, window_h, wl)
                except Exception as exc:
                    print(f"  ERROR {vid}: {exc}", flush=True)
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

        # ── Push rasters to B2 ────────────────────────────────────────────────
        if os.environ.get("TEMPORAL_RASTER_NO_PUSH", "0") != "1":
            _push_rasters(out_dir)

        _write_state("completed")

    except Exception as exc:
        _write_state("failed", error=str(exc))
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
