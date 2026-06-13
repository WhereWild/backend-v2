# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Temporal enrichment utilities.

Enriches occurrence parquets with time-windowed weather statistics from
Open-Meteo ERA5 data (s3://openmeteo/data/, public/anonymous).

Processing model: chunks are processed sequentially in ascending time order.
Each chunk is downloaded on-demand, processed, then deleted. A tail buffer
(last max_window_steps timesteps per active grid cell) is kept in memory
across chunk boundaries so 2160h windows spanning two chunks are handled
correctly without re-downloaded.

Elevation correction: lapse-rate correction (model_elev - obs_elev) * 0.0065 °C/m
is applied to temperature-like variables.  Model elevation comes from
s3://openmeteo/data/{model}/static/HSURF.om (cached per model).  Observation
elevation comes from the `elevation` column in occurrence parquets, written by
the DEM pipeline (not yet built).  Until that column exists the correction is a
no-op: obs_elev is NaN → offset is 0.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from omfiles import OmFileReader
from rasterio.crs import CRS as _CRS
from rasterio.transform import from_bounds as _from_bounds
from rasterio.warp import Resampling as _Resampling
from rasterio.warp import reproject as _reproject

from config.config import load_config
from util.taxa import get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

_LAT_COL = "decimalLatitude"
_LON_COL = "decimalLongitude"
_TIME_COL = "eventTimestamp"


# Variables that receive lapse-rate elevation correction (°C/m × 0.0065).
# Precipitation, cloud cover, and other flux/ratio variables are unaffected.
ELEVATION_CORRECTABLE_VARS: frozenset[str] = frozenset({
    "temperature_2m",
    "dew_point_2m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
})

_LAPSE_RATE = 0.0065  # °C per metre

# Per-model HSURF elevation grid cache {model: np.ndarray shape (ny, nx)}.
_MODEL_ELEV_CACHE: dict[str, np.ndarray] = {}


def _read_model_elevation(model: str, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """Return model surface elevation (m) at the given grid indices.

    Loads HSURF.om from S3 once per model and caches the full grid in RAM
    (~8 MB for ERA5).  Returns NaN for nodata cells (value <= -900 m).
    """
    grid = _MODEL_ELEV_CACHE.get(model)
    if grid is None:
        uri = f"s3://openmeteo/data/{model}/static/HSURF.om"
        try:
            with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
                reader = OmFileReader(fh)
                grid = np.asarray(reader[:, :], dtype=np.float64)
        except Exception:
            grid = np.full((1, 1), np.nan)
        grid = np.where(grid <= -900, np.nan, grid)
        _MODEL_ELEV_CACHE[model] = grid
    ny, nx = grid.shape
    li = np.clip(lat_idx, 0, ny - 1)
    lo = np.clip(lon_idx, 0, nx - 1)
    return grid[li, lo]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkRange:
    chunk_num: int
    start: float    # Unix timestamp of first step
    end: float      # Unix timestamp of last step
    time_len: int   # Number of timesteps
    source: str     # "chunk" or "year"


@dataclass
class ChunkIndex:
    latest_end_time: float
    resolution: float           # seconds per step
    ranges: list[ChunkRange]    # sorted ascending by start


@dataclass
class TemporalLayer:
    id: str
    model: str
    grid_mode: str
    agg: str
    windows: list[int]
    derived: bool = False
    sources: list[str] = field(default_factory=list)
    grid_step: float = 0.25


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def load_temporal_layers(catalog_path: str | Path) -> list[TemporalLayer]:
    """Return all temporal layers from catalog.json."""
    with open(catalog_path) as f:
        cat = json.load(f)
    category_windows: list[int] = []
    layers: list[TemporalLayer] = []
    for category in cat.get("categories", []):
        if category.get("id") != "temporal":
            continue
        category_windows = category.get("windows", [])
        for layer in category.get("layers", []):
            windows = layer.get("windows", category_windows)
            layers.append(TemporalLayer(
                id=layer["id"],
                model=layer.get("model", ""),
                grid_mode=layer.get("grid_mode", "lat_asc_lon_pm180"),
                agg=layer.get("agg", "avg"),
                windows=list(windows),
                derived=bool(layer.get("derived", False)),
                sources=list(layer.get("sources", [])),
                grid_step=float(layer.get("grid_step", 0.25)),
            ))
    return layers


# ---------------------------------------------------------------------------
# Pure math: grid indexing
# ---------------------------------------------------------------------------

def grid_indices(
    lat: float,
    lon: float,
    ny: int,
    nx: int,
    mode: str,
    step: float = 0.25,
) -> tuple[int, int]:
    """Map (lat, lon) to (lat_idx, lon_idx) for a given ERA5 grid mode.

    Modes:
        lat_asc_lon_pm180  — latitude ascending, longitude -180..+180
        lat_asc_lon_360    — latitude ascending, longitude 0..360
        lat_desc_lon_pm180 — latitude descending, longitude -180..+180
        lat_desc_lon_360   — latitude descending, longitude 0..360
    """
    if mode == "lat_asc_lon_360":
        li = int(round((lat + 90.0) / step))
        lo = int(round((lon % 360.0) / step))
    elif mode == "lat_asc_lon_pm180":
        li = int(round((lat + 90.0) / step))
        lo = int(round((lon + 180.0) / step))
    elif mode == "lat_desc_lon_360":
        li = int(round((90.0 - lat) / step))
        lo = int(round((lon % 360.0) / step))
    else:  # lat_desc_lon_pm180
        li = int(round((90.0 - lat) / step))
        lo = int(round((lon + 180.0) / step))
    return max(0, min(li, ny - 1)), max(0, min(lo, nx - 1))


def _grid_indices_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    ny: int,
    nx: int,
    mode: str,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version of grid_indices for arrays."""
    if mode == "lat_asc_lon_360":
        lat_idx = np.rint((lats + 90.0) / step).astype(np.int32)
        lon_idx = np.rint((np.mod(lons, 360.0)) / step).astype(np.int32)
    elif mode == "lat_asc_lon_pm180":
        lat_idx = np.rint((lats + 90.0) / step).astype(np.int32)
        lon_idx = np.rint((lons + 180.0) / step).astype(np.int32)
    elif mode == "lat_desc_lon_360":
        lat_idx = np.rint((90.0 - lats) / step).astype(np.int32)
        lon_idx = np.rint((np.mod(lons, 360.0)) / step).astype(np.int32)
    else:  # lat_desc_lon_pm180
        lat_idx = np.rint((90.0 - lats) / step).astype(np.int32)
        lon_idx = np.rint((lons + 180.0) / step).astype(np.int32)
    return np.clip(lat_idx, 0, ny - 1), np.clip(lon_idx, 0, nx - 1)


# ---------------------------------------------------------------------------
# Pure math: windowed aggregation
# ---------------------------------------------------------------------------

def window_steps(resolution: float, window_hours: tuple[int, ...]) -> dict[int, int]:
    """Convert window sizes from hours to timestep counts."""
    return {hours: int(round((hours * 3600) / resolution)) for hours in window_hours}


def window_stats_batch(
    series: np.ndarray,
    time_indices: np.ndarray,
    steps: dict[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute window sums and finite-value counts for many timestamps.

    Uses prefix sums so each (observation, window) lookup is O(1).

    Args:
        series:       1-D float array (one grid cell's full time series).
        time_indices: 1-D int array of indices into series (one per observation).
        steps:        {window_hours: window_steps} from window_steps().

    Returns:
        (sums, counts) — each a dict keyed by window_hours with shape (n_obs,).
        NaN values in series are excluded from both sum and count.
    """
    finite = np.isfinite(series)
    clean = np.where(finite, series, 0.0)
    cumsum = np.cumsum(clean.astype(np.float64))
    ccount = np.cumsum(finite.astype(np.int64))

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}

    for hours, window_len in steps.items():
        if window_len <= 0 or time_indices.size == 0:
            sums[hours] = np.full(time_indices.shape, np.nan, dtype=np.float64)
            counts[hours] = np.zeros(time_indices.shape, dtype=np.int64)
            continue

        end_idx = time_indices
        start_idx = np.clip(end_idx - (window_len - 1), 0, len(clean) - 1)

        prefix_sum = np.where(start_idx > 0, cumsum[start_idx - 1], 0.0)
        prefix_cnt = np.where(start_idx > 0, ccount[start_idx - 1], np.int64(0))
        sums[hours] = (cumsum[end_idx] - prefix_sum).astype(np.float64)
        counts[hours] = (ccount[end_idx] - prefix_cnt).astype(np.int64)

    return sums, counts


def _window_mode_batch(
    series: np.ndarray,
    time_indices: np.ndarray,
    steps: dict[int, int],
) -> dict[int, np.ndarray]:
    """Sliding-window mode for an integer-valued (nominal) series.

    Builds a (n+1, n_codes) prefix-count matrix so that each (observation,
    window) lookup is O(n_codes) with pure array indexing — no Python loop
    over observations.  For weather codes (0–75, n_codes≤76) this is ~20×
    faster than per-observation slicing when observations are dense.
    """
    result: dict[int, np.ndarray] = {}
    n = len(series)

    if time_indices.size == 0 or n == 0:
        for hours in steps:
            result[hours] = np.full(0, np.nan)
        return result

    finite = np.isfinite(series)
    if not finite.any():
        for hours in steps:
            result[hours] = np.full(len(time_indices), np.nan)
        return result

    int_vals = np.where(finite, series, 0).astype(np.int64)
    n_codes = int(int_vals[finite].max()) + 1

    # Build one-hot matrix then cumsum to get prefix counts.
    valid = np.flatnonzero(finite)
    one_hot = np.zeros((n, n_codes), dtype=np.int32)
    one_hot[valid, int_vals[valid]] = 1
    prefix = np.empty((n + 1, n_codes), dtype=np.int32)
    prefix[0] = 0
    np.cumsum(one_hot, axis=0, out=prefix[1:])

    ti = np.asarray(time_indices, dtype=np.int64)
    ends = np.clip(ti + 1, 0, n)

    for hours, window_len in steps.items():
        modes = np.full(len(ti), np.nan)
        if window_len <= 0:
            result[hours] = modes
            continue
        starts = np.maximum(0, ti - window_len + 1)
        counts = prefix[ends] - prefix[starts]   # (m, n_codes)
        has_any = counts.any(axis=1)
        modes[has_any] = counts[has_any].argmax(axis=1).astype(float)
        result[hours] = modes

    return result


# ---------------------------------------------------------------------------
# Pure math: derived variables
# ---------------------------------------------------------------------------

def vpd_kpa(temp_c: Any, dew_c: Any) -> Any:
    """Vapour-pressure deficit (kPa) from temperature and dew-point (°C).

    VPD = e_s(temp) − e_s(dew), where e_s is the Magnus saturation formula.
    Works on scalars and numpy arrays; NaN propagates naturally.
    """
    def _es(t: Any) -> Any:
        return 0.6108 * np.exp(17.27 * t / (t + 237.3))
    result = _es(np.asarray(temp_c, dtype=float)) - _es(np.asarray(dew_c, dtype=float))
    # Return scalar float for scalar inputs so math.isnan() works
    if result.ndim == 0:
        return float(result)
    return result


def weather_code_simple(
    cloudcover: float | None,
    precipitation: float | None,
    snowfall_water_equivalent: float | None,
    model_dt_seconds: float,
    temperature_2m: float | None = None,
) -> int | None:
    """Derive simplified WMO weather code from 1-timestep aggregates.

    Args:
        cloudcover:               Cloud cover percent (0–100).
        precipitation:            Precipitation (mm) over model_dt_seconds.
        snowfall_water_equivalent: Snowfall water equivalent (mm) over model_dt_seconds.
        model_dt_seconds:         Timestep length in seconds (e.g. 3600 for 1h).
        temperature_2m:           Air temperature (°C). When provided, applies a hard
                                  snow/rain cutoff: snow codes → rain when >0°C; rain
                                  codes 61/63/65 → snow when <0°C.

    Returns:
        WMO code (int) or None if any core input is null/NaN.

    Code table:
        Snow:  71 slight / 73 moderate / 75 heavy
        Rain:  51 / 53 / 55 slight–heavy drizzle; 61 / 63 / 65 slight–heavy rain
        Cloud: 0 clear / 1 mainly clear / 2 partly cloudy / 3 overcast
    """
    if not all(
        v is not None and np.isfinite(float(v))
        for v in (cloudcover, precipitation, snowfall_water_equivalent)
    ):
        return None

    dt_hours = model_dt_seconds / 3600.0
    snow_cm_h = (float(snowfall_water_equivalent) / 10.0) / dt_hours

    if 0.01 <= snow_cm_h < 0.2:
        code = 71
    elif 0.2 <= snow_cm_h < 0.8:
        code = 73
    elif snow_cm_h >= 0.8:
        code = 75
    else:
        rain_mm_h = float(precipitation) / dt_hours
        if 0.01 <= rain_mm_h < 0.5:
            code = 51
        elif 0.5 <= rain_mm_h < 1.0:
            code = 53
        elif 1.0 <= rain_mm_h < 1.3:
            code = 55
        elif 1.3 <= rain_mm_h < 2.5:
            code = 61
        elif 2.5 <= rain_mm_h < 7.6:
            code = 63
        elif rain_mm_h >= 7.6:
            code = 65
        else:
            cc = float(cloudcover)
            if cc < 20.0:
                code = 0
            elif cc < 50.0:
                code = 1
            elif cc < 80.0:
                code = 2
            else:
                code = 3

    if temperature_2m is not None and np.isfinite(float(temperature_2m)):
        t = float(temperature_2m)
        if t > 0:
            if code == 75:
                code = 65
            elif code == 73:
                code = 63
            elif code == 71:
                code = 61
        elif t < 0:
            if code == 65:
                code = 75
            elif code == 63:
                code = 73
            elif code == 61:
                code = 71

    return code


def weather_code_array(
    cloud: np.ndarray,
    precip: np.ndarray,
    snow: np.ndarray,
    resolution: float,
    temp: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized per-timestep weather codes (NaN where any input is non-finite).

    Same code table as weather_code_simple; uses np.select for speed.
    When temp is provided, applies a hard snow/rain cutoff: snow codes → rain
    when >0°C; rain codes 61/63/65 → snow when <0°C. Drizzle codes are unaffected.
    """
    c = np.asarray(cloud, dtype=float)
    p = np.asarray(precip, dtype=float)
    s = np.asarray(snow, dtype=float)
    dt_hours = resolution / 3600.0
    snow_cm_h = (s / 10.0) / dt_hours
    rain_mm_h = p / dt_hours
    valid = np.isfinite(c) & np.isfinite(p) & np.isfinite(s)
    result = np.select(
        [
            ~valid,
            snow_cm_h >= 0.8,
            snow_cm_h >= 0.2,
            snow_cm_h >= 0.01,
            rain_mm_h >= 7.6,
            rain_mm_h >= 2.5,
            rain_mm_h >= 1.3,
            rain_mm_h >= 1.0,
            rain_mm_h >= 0.5,
            rain_mm_h >= 0.01,
            c >= 80.0,
            c >= 50.0,
            c >= 20.0,
        ],
        [np.nan, 75.0, 73.0, 71.0, 65.0, 63.0, 61.0, 55.0, 53.0, 51.0, 3.0, 2.0, 1.0],
        default=0.0,
    )
    if temp is not None:
        t = np.asarray(temp, dtype=float)
        warm = np.isfinite(t) & (t > 0)
        cold = np.isfinite(t) & (t < 0)
        result = np.where(warm & (result == 75), 65, result)
        result = np.where(warm & (result == 73), 63, result)
        result = np.where(warm & (result == 71), 61, result)
        result = np.where(cold & (result == 65), 75, result)
        result = np.where(cold & (result == 63), 73, result)
        result = np.where(cold & (result == 61), 71, result)
    return result


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _open_s3_json(uri: str) -> dict[str, Any] | None:
    try:
        with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
            return json.loads(fh.read())
    except Exception:
        return None


def _parse_s3_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def _chunk_filename(chunk_entry: ChunkRange) -> str:
    if chunk_entry.source == "year":
        return f"year_{chunk_entry.chunk_num}.om"
    return f"chunk_{chunk_entry.chunk_num}.om"


# When set, accumulate_raster downloads chunks here before reading (faster for large grids).
_RASTER_CHUNK_CACHE_DIR: str | None = None


def set_raster_chunk_cache(path: str) -> None:
    global _RASTER_CHUNK_CACHE_DIR
    Path(path).mkdir(parents=True, exist_ok=True)
    _RASTER_CHUNK_CACHE_DIR = path


def _open_chunk(entry: ChunkRange, model: str, variable: str) -> OmFileReader:
    """Return an OmFileReader for the chunk.

    If _RASTER_CHUNK_CACHE_DIR is set, downloads to disk first (fast local reads).
    Otherwise streams directly from S3 via from_fsspec.
    """
    if _RASTER_CHUNK_CACHE_DIR is not None:
        local = _download_chunk(entry, model, variable, _RASTER_CHUNK_CACHE_DIR)
        return OmFileReader(str(local))
    filename = _chunk_filename(entry)
    path = f"openmeteo/data/{model}/{variable}/{filename}"
    fs = fsspec.filesystem("s3", anon=True)
    return OmFileReader.from_fsspec(fs, path)


def _open_chunk_s3(entry: ChunkRange, model: str, variable: str) -> OmFileReader:
    """Open a chunk directly from S3 via HTTP range requests — no local download."""
    filename = _chunk_filename(entry)
    path = f"openmeteo/data/{model}/{variable}/{filename}"
    fs = fsspec.filesystem("s3", anon=True)
    return OmFileReader.from_fsspec(fs, path)


def _download_chunk(
    chunk_entry: ChunkRange,
    model: str,
    variable: str,
    cache_dir: str,
) -> Path:
    """Download a single .om chunk to local disk via fsspec and return the path."""
    filename = _chunk_filename(chunk_entry)
    uri = f"s3://openmeteo/data/{model}/{variable}/{filename}"

    dest_dir = Path(cache_dir) / "chunks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{model}_{variable}_{filename}"

    if target.exists():
        return target

    fs = fsspec.filesystem("s3", anon=True)
    fs.get(uri, str(target))
    return target


_PREFETCH_WORKERS = 8
_RANGE_FETCH_WORKERS = 8


def _download_layer_chunk(
    chunk_entry: ChunkRange,
    model: str,
    variables: list[str],
    cache_dir: str,
) -> ChunkRange:
    """Download all variable files for one chunk; return the entry (for futures)."""
    for var in variables:
        _download_chunk(chunk_entry, model, var, cache_dir)
    return chunk_entry


# ---------------------------------------------------------------------------
# Chunk index (S3 metadata only — no .om downloads)
# ---------------------------------------------------------------------------

_CHUNK_INDEX_CACHE: dict[tuple[str, str], ChunkIndex] = {}


def build_chunk_index(
    model: str,
    variable: str,
    *,
    min_date: str | None = None,
) -> ChunkIndex:
    """Build a time-ordered index of all .om files for a model/variable.

    Fetches only meta.json and the S3 directory listing; does not download
    any .om data.
    """
    cache_key = (model, variable)
    cached = _CHUNK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    meta_uri = f"s3://openmeteo/data/{model}/static/meta.json"
    meta = _open_s3_json(meta_uri) or {}
    end_time = _parse_s3_time(meta.get("data_end_time"))
    resolution = float(_parse_s3_time(meta.get("temporal_resolution_seconds")) or 3600.0)
    chunk_time_len = meta.get("chunk_time_length")
    if not isinstance(chunk_time_len, (int, float)):
        chunk_time_len = None

    if end_time is None:
        raise RuntimeError(f"Missing data_end_time in static/meta.json for {model}")

    fs = fsspec.filesystem("s3", anon=True)
    base = f"s3://openmeteo/data/{model}/{variable}"
    listing = fs.ls(base)

    chunk_nums: list[int] = []
    year_files: list[int] = []
    for item in listing:
        leaf = (item.get("name") if isinstance(item, dict) else item).split("/")[-1]
        if leaf.startswith("chunk_") and leaf.endswith(".om"):
            try:
                chunk_nums.append(int(leaf[6:-3]))
            except ValueError:
                pass  # ignore malformed chunk filenames from directory listings
        elif leaf.startswith("year_") and leaf.endswith(".om"):
            try:
                year_files.append(int(leaf[5:-3]))
            except ValueError:
                pass  # ignore malformed year filenames; only numeric suffixes are valid

    ranges: list[ChunkRange] = []

    # chunk_* files: epoch-aligned formula — chunk_N starts at N * chunk_time_len * resolution
    if chunk_nums:
        for chunk_num in sorted(chunk_nums, reverse=True):
            if chunk_time_len is not None:
                tlen = int(chunk_time_len)
            else:
                # Fall back to reading shape from the file (rare)
                uri = f"{base}/chunk_{chunk_num}.om"
                with fsspec.open(uri, mode="rb", s3={"anon": True}) as fh:
                    reader = OmFileReader(fh)
                    tlen = reader.shape[2]
            start = float(chunk_num) * float(tlen) * resolution
            end   = start + (tlen - 1) * resolution
            ranges.append(ChunkRange(
                chunk_num=chunk_num,
                start=start,
                end=end,
                time_len=tlen,
                source="chunk",
            ))

    # year_* files: calendar-aligned Jan 1 boundaries
    for year in sorted(year_files):
        start_dt = datetime(year, 1, 1, tzinfo=UTC)
        end_dt = datetime(year + 1, 1, 1, tzinfo=UTC)
        tlen = int(round((end_dt - start_dt).total_seconds() / resolution))
        start = start_dt.timestamp()
        end = start + (tlen - 1) * resolution
        ranges.append(ChunkRange(
            chunk_num=year,
            start=start,
            end=end,
            time_len=tlen,
            source="year",
        ))

    if not ranges:
        raise RuntimeError(f"No .om files found for {model}/{variable}")

    # Year files are authoritative archival data. Where a rolling chunk overlaps
    # a year file, clip the chunk's effective start to just after the year file
    # ends so searchsorted always prefers the complete year file for that period.
    year_ranges = [r for r in ranges if r.source == "year"]
    if year_ranges:
        clipped: list[ChunkRange] = []
        for r in ranges:
            if r.source != "chunk":
                clipped.append(r)
                continue
            # Find the latest year-file end that overlaps this chunk
            overlap_end = max(
                (y.end for y in year_ranges if y.start <= r.end and y.end >= r.start),
                default=None,
            )
            if overlap_end is None:
                clipped.append(r)  # no overlap — keep as-is
                continue
            new_start = overlap_end + resolution
            if new_start > r.end:
                continue  # chunk entirely within year file(s) — drop it
            new_time_idx = int(round((new_start - r.start) / resolution))
            clipped.append(ChunkRange(
                chunk_num=r.chunk_num,
                start=new_start,
                end=r.end,
                time_len=r.time_len - new_time_idx,
                source=r.source,
            ))
        ranges = clipped

    ranges.sort(key=lambda r: r.start)

    if min_date is not None:
        cutoff = datetime.fromisoformat(min_date).replace(tzinfo=UTC).timestamp()
        ranges = [r for r in ranges if r.end >= cutoff]

    result = ChunkIndex(
        latest_end_time=float(end_time),
        resolution=resolution,
        ranges=ranges,
    )
    _CHUNK_INDEX_CACHE[cache_key] = result
    return result


def _chunk_entry_for_time(idx: ChunkIndex, ts: float) -> tuple[ChunkRange | None, int]:
    """Return the ChunkRange containing ts and the 0-based time index within it."""
    for entry in idx.ranges:
        if entry.start <= ts <= entry.end:
            return entry, int(round((ts - entry.start) / idx.resolution))
    return None, -1


# ---------------------------------------------------------------------------
# Occurrence index
# ---------------------------------------------------------------------------

_OCC_INDEX_SCHEMA = pa.schema([
    pa.field("taxon_path", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("timestamp", pa.float64()),
    pa.field("elevation", pa.float64()),
])


def iter_occ_index_batches(index_path: Path, batch_rows: int) -> Iterable[pa.Table]:
    """Yield occurrence index batches from a parquet file written by build_occ_index."""
    pf = pq.ParquetFile(index_path)
    for batch in pf.iter_batches(batch_size=batch_rows):
        yield pa.Table.from_batches([batch]).combine_chunks()


def build_occ_index(
    root_taxon_id: str,
    data_root: str,
    occ_filename: str,
    index_path: Path,
    min_date: str | None = None,
    skip_if_cols: list[list[str]] | None = None,
) -> int:
    """Scan all descendant occurrence parquets and write a flat index to disk.

    Streams one taxon at a time so memory usage is bounded regardless of total
    observation count. Returns the total number of rows written.

    skip_if_cols: list of per-layer column groups. A row is excluded only when
    every group is fully non-null (i.e. every active layer is already enriched
    for that row). Rows needing enrichment for any one layer are included.
    """
    root = get_taxon_by_id(root_taxon_id)
    if root is None:
        raise RuntimeError(f"Unknown root taxon {root_taxon_id}")

    cutoff = (
        datetime.fromisoformat(min_date).replace(tzinfo=UTC).timestamp()
        if min_date is not None
        else None
    )

    tree_root = Path(data_root) / "taxonomy" / "tree"
    total_rows = 0
    total_skipped = 0
    n_files = 0
    t0 = time.monotonic()
    last_report = t0

    with pq.ParquetWriter(index_path, _OCC_INDEX_SCHEMA) as writer:
        for node in iter_descendants(root, include_self=True):
            occ_path = tree_root / node["path"] / occ_filename
            if not occ_path.exists():
                continue
            n_files += 1
            schema = pq.read_schema(occ_path)
            read_cols = [_LAT_COL, _LON_COL, _TIME_COL]
            has_elev = "elevation" in schema.names
            if has_elev:
                read_cols.append("elevation")
            table = pq.read_table(occ_path, columns=read_cols)
            df = table.to_pandas()

            valid = df[_TIME_COL].notna() & df[_LAT_COL].notna() & df[_LON_COL].notna()
            if cutoff is not None:
                valid &= df[_TIME_COL] >= cutoff

            if skip_if_cols:
                all_skip_cols = list({c for group in skip_if_cols for c in group})
                present_set = {c for c in all_skip_cols if c in schema.names}
                if present_set:
                    skip_table = pq.read_table(occ_path, columns=list(present_set))
                    # A row is skippable only when ALL layer groups are complete.
                    # Within each group: done = all columns non-null (AND).
                    # Across groups: skippable = all groups done (AND).
                    # NaN is a real float value in Arrow so is_null returns False —
                    # NaN sentinels ("tried, no coverage") correctly count as done.
                    already_done = np.ones(skip_table.num_rows, dtype=bool)
                    for group in skip_if_cols:
                        present_g = [c for c in group if c in present_set]
                        if len(present_g) < len(group):
                            # Layer columns absent — not yet enriched, include all rows
                            already_done[:] = False
                            break
                        layer_done = np.ones(skip_table.num_rows, dtype=bool)
                        for c in present_g:
                            layer_done &= np.asarray(pc.invert(pc.is_null(skip_table.column(c))))
                        already_done &= layer_done
                    total_skipped += int(already_done.sum())
                    valid &= ~already_done

            if not valid.any():
                continue

            row_idx = df.index[valid].to_numpy(dtype=np.int64)
            times = df.loc[valid, _TIME_COL].to_numpy(dtype=np.float64)
            lats = df.loc[valid, _LAT_COL].to_numpy(dtype=np.float64)
            lons = df.loc[valid, _LON_COL].to_numpy(dtype=np.float64)
            elevs = (
                df.loc[valid, "elevation"].to_numpy(dtype=np.float64)
                if has_elev
                else np.full(len(row_idx), np.nan)
            )
            path_str = str(occ_path)

            chunk_table = pa.table({
                "taxon_path": pa.array([path_str] * len(row_idx), type=pa.string()),
                "row_idx":    pa.array(row_idx,  type=pa.int64()),
                "latitude":   pa.array(lats,     type=pa.float64()),
                "longitude":  pa.array(lons,     type=pa.float64()),
                "timestamp":  pa.array(times,    type=pa.float64()),
                "elevation":  pa.array(elevs,    type=pa.float64()),
            })
            writer.write_table(chunk_table)
            total_rows += len(row_idx)

            now = time.monotonic()
            if now - last_report >= 30:
                print(
                    f"[occ_index] {n_files} files scanned  "
                    f"found={total_rows}  skipped={total_skipped}  "
                    f"elapsed={now - t0:.0f}s"
                )
                last_report = now

    elapsed = time.monotonic() - t0
    print(
        f"[occ_index] done: {n_files} files  found={total_rows}  "
        f"skipped={total_skipped}  elapsed={elapsed:.1f}s"
    )
    return total_rows


# ---------------------------------------------------------------------------
# Worklist construction
# ---------------------------------------------------------------------------

def map_to_worklist(
    occ_table: pa.Table,
    chunk_index: ChunkIndex,
    grid_mode: str,
    step: float,
) -> pa.Table:
    """Project occurrence index onto a model chunk grid.

    Returns a table with columns:
        taxon_path, row_idx, chunk_num, lat_idx, lon_idx, time_idx
    """
    times = occ_table.column("timestamp").to_numpy()
    lats = occ_table.column("latitude").to_numpy()
    lons = occ_table.column("longitude").to_numpy()
    row_idx = occ_table.column("row_idx").to_numpy()
    taxon_path_col = occ_table.column("taxon_path")
    elevation = (
        occ_table.column("elevation").to_numpy()
        if "elevation" in occ_table.schema.names
        else np.full(len(times), np.nan, dtype=np.float64)
    )

    if times.size == 0:
        return pa.table({
            "taxon_path": pa.array([], type=pa.string()),
            "row_idx": pa.array([], type=pa.int64()),
            "chunk_num": pa.array([], type=pa.int32()),
            "lat_idx": pa.array([], type=pa.int32()),
            "lon_idx": pa.array([], type=pa.int32()),
            "time_idx": pa.array([], type=pa.int32()),
            "elevation": pa.array([], type=pa.float64()),
        })

    # Pick ny, nx, step from a sample range (latest) to check bounds
    # grid dimensions are validated per-chunk when reading the .om file;
    # here we just store raw indices (clamping happens per-chunk)
    # Derive loose upper bounds from the grid step so non-0.25° grids
    # (e.g. ERA5-Land at 0.1°) are not prematurely clipped here.
    max_ny = int(round(180.0 / step)) + 1
    max_nx = int(round(360.0 / step)) + 1

    lat_idx, lon_idx = _grid_indices_batch(lats, lons, max_ny, max_nx, grid_mode, step)

    # Map timestamps → chunk_num and per-chunk time index
    asc_starts = np.array([r.start for r in chunk_index.ranges], dtype=np.float64)
    asc_chunk_nums = np.array([r.chunk_num for r in chunk_index.ranges], dtype=np.int32)
    asc_time_lens = np.array([r.time_len for r in chunk_index.ranges], dtype=np.int32)

    chunk_lookup = np.searchsorted(asc_starts, times, side="right") - 1
    chunk_lookup = np.clip(chunk_lookup, 0, len(asc_starts) - 1)

    chunk_nums = asc_chunk_nums[chunk_lookup]
    chunk_starts = asc_starts[chunk_lookup]
    chunk_time_lens = asc_time_lens[chunk_lookup]

    # Floor to hour boundary (ERA5 is hourly)
    time_indices = np.floor((times - chunk_starts) / chunk_index.resolution).astype(np.int32)
    time_indices = np.clip(time_indices, 0, chunk_time_lens - 1)

    return pa.table({
        "taxon_path": taxon_path_col,
        "row_idx": row_idx,
        "chunk_num": chunk_nums,
        "lat_idx": lat_idx,
        "lon_idx": lon_idx,
        "time_idx": time_indices,
        "elevation": elevation,
    })


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

# TailBuffer: {(lat_idx, lon_idx): last max_window_steps of series}
TailBuffer = dict[tuple[int, int], np.ndarray]


def process_chunk(
    chunk_entry: ChunkRange,
    worklist_slice: pa.Table,
    tail_buffer: TailBuffer,
    model: str,
    variable: str,
    steps: dict[int, int],
    agg_mode: str,
    cache_dir: str,
    range_request: bool = False,
) -> tuple[dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]], TailBuffer]:
    """Download, process, and delete one .om chunk.

    For each active grid cell:
      1. Read the cell's full time series from the .om file.
      2. If the cell has a tail from the previous chunk, prepend it so
         windows spanning the chunk boundary are computed correctly.
      3. Run window_stats_batch to get sums and counts.
      4. Save the last max_window_steps values to new_tail for the next chunk.
      5. Delete the downloaded file on exit.

    Args:
        chunk_entry:    Metadata for the chunk to process.
        worklist_slice: Observations assigned to this chunk (from map_to_worklist).
        tail_buffer:    {(lat_idx, lon_idx): tail_array} from the previous chunk.
        model, variable: Identifies the ERA5 dataset.
        steps:          {window_hours: window_steps} from window_steps().
        agg_mode:       "sum" or "avg".
        cache_dir:      Directory for temporary .om downloads.

    Returns:
        (updates, new_tail_buffer)
        updates: {taxon_path: {column: [(row_ids, values)]}}
        new_tail_buffer: tails from this chunk (pass to the next call).
    """
    max_window_steps = max(steps.values()) if steps else 0

    reader = (_open_chunk_s3(chunk_entry, model, variable) if range_request
              else OmFileReader(str(_download_chunk(chunk_entry, model, variable, cache_dir))))
    ny, nx, _ = reader.shape

    data = worklist_slice.to_pydict()
    lat = np.asarray(data["lat_idx"], dtype=np.int32)
    lon = np.asarray(data["lon_idx"], dtype=np.int32)
    time_idx = np.asarray(data["time_idx"], dtype=np.int32)
    taxon_path = np.asarray(data["taxon_path"])
    row_idx = np.asarray(data["row_idx"], dtype=np.int64)
    obs_elev = np.asarray(
        data.get("elevation", np.full(len(lat), np.nan)), dtype=np.float64
    )

    # Re-clamp grid indices to actual file dimensions
    lat = np.clip(lat, 0, ny - 1)
    lon = np.clip(lon, 0, nx - 1)

    # Elevation correction: compute per-observation offset upfront.
    # No-op while obs_elev is all-NaN (DEM pipeline not yet built).
    do_elev = variable in ELEVATION_CORRECTABLE_VARS
    elev_correction: np.ndarray | None = None
    if do_elev and np.isfinite(obs_elev).any():
        model_elev = _read_model_elevation(model, lat, lon)
        raw_corr = (model_elev - obs_elev) * _LAPSE_RATE
        elev_correction = np.where(np.isfinite(raw_corr), raw_corr, 0.0)

    # Sort by (lat, lon) to process one grid cell at a time
    order = np.lexsort((lon, lat))
    lat = lat[order]
    lon = lon[order]
    time_idx = time_idx[order]
    taxon_path = taxon_path[order]
    row_idx = row_idx[order]
    if elev_correction is not None:
        elev_correction = elev_correction[order]

    change = np.empty(len(lat), dtype=bool)
    change[0] = True
    change[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    group_starts = np.flatnonzero(change)
    group_ends = np.append(group_starts[1:], len(lat))

    # Parallel-prefetch all unique grid cells upfront when using range requests,
    # avoiding O(cells) serial round trips.
    _cell_cache: dict[tuple[int, int], np.ndarray] = {}
    if range_request and group_starts.size > 0:
        _unique = [(int(lat[s]), int(lon[s])) for s in group_starts]
        def _fetch(li_lo: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
            try:
                return li_lo, np.asarray(reader[li_lo[0], li_lo[1], :], dtype=np.float64)
            except Exception:
                return li_lo, np.empty(0, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=_RANGE_FETCH_WORKERS) as _ex:
            _cell_cache = dict(_ex.map(_fetch, _unique))

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    new_tail: TailBuffer = {}

    for s, e in zip(group_starts, group_ends):
        li = int(lat[s])
        lo = int(lon[s])

        if range_request:
            series = _cell_cache.get((li, lo), np.empty(0, dtype=np.float64))
        else:
            try:
                series = np.asarray(reader[li, lo, :], dtype=np.float64)
            except Exception:
                continue
        if series.size == 0:
            continue

        # Save tail from current chunk for next chunk's boundary handling
        if max_window_steps > 0:
            new_tail[(li, lo)] = series[-max_window_steps:].copy()

        # Prepend previous chunk's tail if observations near chunk start
        time_slice = time_idx[s:e]
        prev_tail = tail_buffer.get((li, lo))
        prev_len = 0

        if prev_tail is not None and max_window_steps > 1:
            min_t = int(time_slice.min())
            need = (max_window_steps - 1) - min_t
            if need > 0:
                prev_len = min(int(need), len(prev_tail))
                series = np.concatenate([prev_tail[-prev_len:], series])

        # Narrow the series slice to only what the windows need
        min_t = int(time_slice.min())
        max_t = int(time_slice.max())
        slice_start = max(0, (min_t + prev_len) - (max_window_steps - 1))
        slice_end = min(series.size - 1, max_t + prev_len)
        series_slice = series[slice_start : slice_end + 1]
        local_time = np.clip(
            (time_slice + prev_len) - slice_start, 0, series_slice.size - 1
        )

        # Cap observations that fall in a trailing NaN zone (e.g. ERA5 processing lag)
        # to the last valid timestep so they receive data instead of NaN.
        if series_slice.size > 0 and not np.isfinite(series_slice[-1]):
            finite_in_slice = np.flatnonzero(np.isfinite(series_slice))
            if finite_in_slice.size == 0:
                continue
            last_valid = int(finite_in_slice[-1])
            local_time = np.minimum(local_time, last_valid)

        window_sums, window_counts = window_stats_batch(series_slice, local_time, steps)

        paths_slice = taxon_path[s:e]
        rows_slice = row_idx[s:e]
        corr_slice = elev_correction[s:e] if elev_correction is not None else None
        for tpath in np.unique(paths_slice):
            mask = paths_slice == tpath
            row_ids = rows_slice[mask]
            for hours, sums in window_sums.items():
                cnts = window_counts[hours]
                if agg_mode == "sum":
                    values = np.where(cnts > 0, sums, np.nan)
                else:
                    values = np.full_like(sums, np.nan, dtype=np.float64)
                    np.divide(sums, cnts, out=values, where=cnts > 0)
                    if corr_slice is not None:
                        values = values + corr_slice
                col = f"{variable}_{agg_mode}_{hours}h"
                updates.setdefault(str(tpath), {}).setdefault(col, []).append(
                    (row_ids, values[mask])
                )

    return updates, new_tail


def process_chunk_mode(
    chunk_entry: ChunkRange,
    worklist_slice: pa.Table,
    tail_buffer: TailBuffer,
    model: str,
    source_variables: list[str],
    col_prefix: str,
    steps: dict[int, int],
    resolution: float,
    cache_dir: str,
    secondary_indices: dict[str, ChunkIndex] | None = None,
    range_request: bool = False,
) -> tuple[dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]], TailBuffer]:
    """Download multiple .om files, derive a per-timestep series, apply sliding-window mode.

    Downloads one chunk file per source variable, computes weather_code_array
    per timestep, then applies _window_mode_batch for each window size.
    Columns are written as {col_prefix}_mode_{W}h.
    """
    max_window_steps = max(steps.values()) if steps else 0

    _sv = source_variables
    _cloud_idx  = _sv.index("cloud_cover")
    _precip_idx = _sv.index("precipitation")
    _snow_idx   = _sv.index("snowfall_water_equivalent")

    def _open(entry: ChunkRange, var: str) -> OmFileReader:
        return (_open_chunk_s3(entry, model, var) if range_request
                else OmFileReader(str(_download_chunk(entry, model, var, cache_dir))))

    primary_var = source_variables[0]
    readers: list[OmFileReader | None] = [_open(chunk_entry, primary_var)]
    sec_offsets: list[int] = [0]
    # (ext_reader, ext_offset, ext_steps) — stitch when secondary ends before primary
    _sec_exts: list[tuple[OmFileReader | None, int, int]] = [(None, 0, 0)]
    for var in source_variables[1:]:
        if secondary_indices is not None and var in secondary_indices:
            sec_entry, sec_t0 = _chunk_entry_for_time(secondary_indices[var], chunk_entry.start)
            if sec_entry is not None:
                readers.append(_open(sec_entry, var))
                sec_offsets.append(sec_t0)
                sec_available = sec_entry.time_len - sec_t0
                if sec_available < chunk_entry.time_len:
                    _ext_e, _ext_t0 = _chunk_entry_for_time(
                        secondary_indices[var], sec_entry.end + resolution
                    )
                    if _ext_e is not None:
                        _sec_exts.append((_open(_ext_e, var), _ext_t0, chunk_entry.time_len - sec_available))
                    else:
                        _sec_exts.append((None, 0, 0))
                else:
                    _sec_exts.append((None, 0, 0))
            else:
                readers.append(None)
                sec_offsets.append(0)
                _sec_exts.append((None, 0, 0))
        else:
            readers.append(_open(chunk_entry, var))
            sec_offsets.append(0)
            _sec_exts.append((None, 0, 0))
    ny, nx, _ = readers[0].shape

    data = worklist_slice.to_pydict()
    lat = np.asarray(data["lat_idx"], dtype=np.int32)
    lon = np.asarray(data["lon_idx"], dtype=np.int32)
    time_idx = np.asarray(data["time_idx"], dtype=np.int32)
    taxon_path = np.asarray(data["taxon_path"])
    row_idx = np.asarray(data["row_idx"], dtype=np.int64)
    obs_elev = np.asarray(
        data.get("elevation", np.full(len(lat), np.nan)), dtype=np.float64
    )

    lat = np.clip(lat, 0, ny - 1)
    lon = np.clip(lon, 0, nx - 1)

    # Per-observation temperature correction (no-op until DEM pipeline populates elevation).
    temp_var_idx = next(
        (i for i, v in enumerate(source_variables) if v in ELEVATION_CORRECTABLE_VARS), None
    )
    elev_correction: np.ndarray | None = None
    if temp_var_idx is not None and np.isfinite(obs_elev).any():
        model_elev = _read_model_elevation(model, lat, lon)
        raw_corr = (model_elev - obs_elev) * _LAPSE_RATE
        elev_correction = np.where(np.isfinite(raw_corr), raw_corr, 0.0)

    order = np.lexsort((lon, lat))
    lat, lon = lat[order], lon[order]
    time_idx = time_idx[order]
    taxon_path, row_idx = taxon_path[order], row_idx[order]
    if elev_correction is not None:
        elev_correction = elev_correction[order]

    change = np.empty(len(lat), dtype=bool)
    change[0] = True
    change[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    group_starts = np.flatnonzero(change)
    group_ends = np.append(group_starts[1:], len(lat))

    # Parallel-prefetch all (reader, cell) combinations at once when range-requesting.
    _reader_caches: list[dict[tuple[int, int], np.ndarray]] = [{} for _ in readers]
    if range_request and group_starts.size > 0:
        _unique = [(int(lat[s]), int(lon[s])) for s in group_starts]
        _tasks = [(r_idx, r, li, lo) for r_idx, r in enumerate(readers) if r is not None
                  for li, lo in _unique]
        def _fetch_multi(task: tuple) -> tuple:
            r_idx, r, li, lo = task
            try:
                return r_idx, li, lo, np.asarray(r[li, lo, :], dtype=np.float64)
            except Exception:
                return r_idx, li, lo, np.empty(0, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=_RANGE_FETCH_WORKERS) as _ex:
            for _r_idx, _li, _lo, _arr in _ex.map(_fetch_multi, _tasks):
                _reader_caches[_r_idx][(_li, _lo)] = _arr

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    new_tail: TailBuffer = {}

    for s, e in zip(group_starts, group_ends):
        li, lo = int(lat[s]), int(lon[s])

        try:
            if range_request:
                primary_arr = _reader_caches[0].get((li, lo), np.empty(0, dtype=np.float64))
            else:
                primary_arr = np.asarray(readers[0][li, lo, :], dtype=np.float64)
            primary_len = len(primary_arr)
            raw: list[np.ndarray] = [primary_arr]
            for r_idx, (r, off) in enumerate(zip(readers[1:], sec_offsets[1:]), start=1):
                if r is None:
                    raw.append(np.full(primary_len, np.nan, dtype=np.float64))
                else:
                    if range_request:
                        arr = _reader_caches[r_idx].get((li, lo), np.empty(0, dtype=np.float64))
                    else:
                        arr = np.asarray(r[li, lo, :], dtype=np.float64)
                    sliced = arr[off:off + primary_len]
                    if len(sliced) < primary_len:
                        ext_r, ext_off, ext_n = _sec_exts[r_idx]
                        if ext_r is not None and ext_n > 0:
                            try:
                                ext_arr = np.asarray(ext_r[li, lo, :], dtype=np.float64)
                                ext_slice = ext_arr[ext_off:ext_off + ext_n]
                                sliced = np.concatenate([sliced, ext_slice])
                            except Exception:
                                pass
                        if len(sliced) < primary_len:
                            padded = np.full(primary_len, np.nan, dtype=np.float64)
                            padded[:len(sliced)] = sliced
                            raw.append(padded)
                        else:
                            raw.append(sliced[:primary_len])
                    else:
                        raw.append(sliced)
        except Exception:
            continue
        if any(a.size == 0 for a in raw):
            continue

        # Pass temperature to weather_code_array, applying the median
        # elevation correction for the cell.  Median is used because
        # observations within a 0.25° cell may span different elevations.
        temp_arr: np.ndarray | None = raw[temp_var_idx] if temp_var_idx is not None else None
        if temp_arr is not None and elev_correction is not None:
            cell_offset = float(np.median(elev_correction[s:e]))
            temp_arr = temp_arr + cell_offset

        derived = weather_code_array(raw[_cloud_idx], raw[_precip_idx], raw[_snow_idx], resolution, temp=temp_arr)

        if max_window_steps > 0:
            new_tail[(li, lo)] = derived[-max_window_steps:].copy()

        time_slice = time_idx[s:e]
        prev_tail = tail_buffer.get((li, lo))
        prev_len = 0
        if prev_tail is not None and max_window_steps > 1:
            min_t = int(time_slice.min())
            need = (max_window_steps - 1) - min_t
            if need > 0:
                prev_len = min(int(need), len(prev_tail))
                derived = np.concatenate([prev_tail[-prev_len:], derived])

        min_t = int(time_slice.min())
        max_t = int(time_slice.max())
        slice_start = max(0, (min_t + prev_len) - (max_window_steps - 1))
        slice_end = min(derived.size - 1, max_t + prev_len)
        series_slice = derived[slice_start : slice_end + 1]
        local_time = np.clip(
            (time_slice + prev_len) - slice_start, 0, series_slice.size - 1
        )

        if series_slice.size > 0 and not np.isfinite(series_slice[-1]):
            finite_in_slice = np.flatnonzero(np.isfinite(series_slice))
            if finite_in_slice.size == 0:
                continue
            last_valid = int(finite_in_slice[-1])
            local_time = np.minimum(local_time, last_valid)

        window_modes = _window_mode_batch(series_slice, local_time, steps)

        paths_slice = taxon_path[s:e]
        rows_slice = row_idx[s:e]
        for tpath in np.unique(paths_slice):
            mask = paths_slice == tpath
            rids = rows_slice[mask]
            for hours, modes in window_modes.items():
                col = f"{col_prefix}_mode_{hours}h"
                updates.setdefault(str(tpath), {}).setdefault(col, []).append(
                    (rids, modes[mask])
                )

    return updates, new_tail


def process_chunk_vpd(
    chunk_entry: ChunkRange,
    worklist_slice: pa.Table,
    tail_buffer: TailBuffer,
    model: str,
    source_variables: list[str],
    col_prefix: str,
    steps: dict[int, int],
    resolution: float,
    cache_dir: str,
    secondary_indices: dict[str, ChunkIndex] | None = None,
    range_request: bool = False,
) -> tuple[dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]], TailBuffer]:
    """Derive VPD per-timestep from temperature_2m and dew_point_2m, then avg over windows.

    source_variables must be ['temperature_2m', 'dew_point_2m'] (in that order).
    Secondary sources use time-range lookup via secondary_indices, same as process_chunk_mode.
    """
    max_window_steps = max(steps.values()) if steps else 0

    def _open(entry: ChunkRange, var: str) -> OmFileReader:
        return (_open_chunk_s3(entry, model, var) if range_request
                else OmFileReader(str(_download_chunk(entry, model, var, cache_dir))))

    primary_var = source_variables[0]
    readers: list[OmFileReader | None] = [_open(chunk_entry, primary_var)]
    sec_offsets: list[int] = [0]
    # (ext_reader, ext_offset, ext_steps) — stitch when secondary ends before primary
    _sec_exts_vpd: list[tuple[OmFileReader | None, int, int]] = [(None, 0, 0)]
    for var in source_variables[1:]:
        if secondary_indices is not None and var in secondary_indices:
            sec_entry, sec_t0 = _chunk_entry_for_time(secondary_indices[var], chunk_entry.start)
            if sec_entry is not None:
                readers.append(_open(sec_entry, var))
                sec_offsets.append(sec_t0)
                sec_available = sec_entry.time_len - sec_t0
                if sec_available < chunk_entry.time_len:
                    _ext_e, _ext_t0 = _chunk_entry_for_time(
                        secondary_indices[var], sec_entry.end + resolution
                    )
                    if _ext_e is not None:
                        _sec_exts_vpd.append((_open(_ext_e, var), _ext_t0, chunk_entry.time_len - sec_available))
                    else:
                        _sec_exts_vpd.append((None, 0, 0))
                else:
                    _sec_exts_vpd.append((None, 0, 0))
            else:
                readers.append(None)
                sec_offsets.append(0)
                _sec_exts_vpd.append((None, 0, 0))
        else:
            readers.append(_open(chunk_entry, var))
            sec_offsets.append(0)
            _sec_exts_vpd.append((None, 0, 0))

    ny, nx, _ = readers[0].shape

    data = worklist_slice.to_pydict()
    lat = np.asarray(data["lat_idx"], dtype=np.int32)
    lon = np.asarray(data["lon_idx"], dtype=np.int32)
    time_idx = np.asarray(data["time_idx"], dtype=np.int32)
    taxon_path = np.asarray(data["taxon_path"])
    row_idx = np.asarray(data["row_idx"], dtype=np.int64)
    obs_elev = np.asarray(
        data.get("elevation", np.full(len(lat), np.nan)), dtype=np.float64
    )

    lat = np.clip(lat, 0, ny - 1)
    lon = np.clip(lon, 0, nx - 1)

    elev_correction: np.ndarray | None = None
    if np.isfinite(obs_elev).any():
        model_elev = _read_model_elevation(model, lat, lon)
        raw_corr = (model_elev - obs_elev) * _LAPSE_RATE
        elev_correction = np.where(np.isfinite(raw_corr), raw_corr, 0.0)

    order = np.lexsort((lon, lat))
    lat, lon = lat[order], lon[order]
    time_idx = time_idx[order]
    taxon_path, row_idx = taxon_path[order], row_idx[order]
    if elev_correction is not None:
        elev_correction = elev_correction[order]

    change = np.empty(len(lat), dtype=bool)
    change[0] = True
    change[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    group_starts = np.flatnonzero(change)
    group_ends = np.append(group_starts[1:], len(lat))

    # Parallel-prefetch all (reader, cell) combinations at once when range-requesting.
    _reader_caches_vpd: list[dict[tuple[int, int], np.ndarray]] = [{} for _ in readers]
    if range_request and group_starts.size > 0:
        _unique = [(int(lat[s]), int(lon[s])) for s in group_starts]
        _tasks = [(r_idx, r, li, lo) for r_idx, r in enumerate(readers) if r is not None
                  for li, lo in _unique]
        def _fetch_vpd(task: tuple) -> tuple:
            r_idx, r, li, lo = task
            try:
                return r_idx, li, lo, np.asarray(r[li, lo, :], dtype=np.float64)
            except Exception:
                return r_idx, li, lo, np.empty(0, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=_RANGE_FETCH_WORKERS) as _ex:
            for _r_idx, _li, _lo, _arr in _ex.map(_fetch_vpd, _tasks):
                _reader_caches_vpd[_r_idx][(_li, _lo)] = _arr

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    new_tail: TailBuffer = {}

    for s, e in zip(group_starts, group_ends):
        li, lo = int(lat[s]), int(lon[s])

        try:
            if range_request:
                primary_arr = _reader_caches_vpd[0].get((li, lo), np.empty(0, dtype=np.float64))
            else:
                primary_arr = np.asarray(readers[0][li, lo, :], dtype=np.float64)
            primary_len = len(primary_arr)
            raw: list[np.ndarray] = [primary_arr]
            for r_idx, (r, off) in enumerate(zip(readers[1:], sec_offsets[1:]), start=1):
                if r is None:
                    raw.append(np.full(primary_len, np.nan, dtype=np.float64))
                else:
                    if range_request:
                        arr = _reader_caches_vpd[r_idx].get((li, lo), np.empty(0, dtype=np.float64))
                    else:
                        arr = np.asarray(r[li, lo, :], dtype=np.float64)
                    sliced = arr[off:off + primary_len]
                    if len(sliced) < primary_len:
                        ext_r, ext_off, ext_n = _sec_exts_vpd[r_idx]
                        if ext_r is not None and ext_n > 0:
                            try:
                                ext_arr = np.asarray(ext_r[li, lo, :], dtype=np.float64)
                                ext_slice = ext_arr[ext_off:ext_off + ext_n]
                                sliced = np.concatenate([sliced, ext_slice])
                            except Exception:
                                pass
                        if len(sliced) < primary_len:
                            padded = np.full(primary_len, np.nan, dtype=np.float64)
                            padded[:len(sliced)] = sliced
                            raw.append(padded)
                        else:
                            raw.append(sliced[:primary_len])
                    else:
                        raw.append(sliced)
        except Exception:
            continue
        if any(a.size == 0 for a in raw):
            continue

        temp_arr = raw[0].copy()
        dew_arr = raw[1].copy() if len(raw) > 1 else np.full(len(temp_arr), np.nan)
        if elev_correction is not None:
            cell_offset = float(np.median(elev_correction[s:e]))
            temp_arr += cell_offset
            dew_arr += cell_offset

        derived = vpd_kpa(temp_arr, dew_arr)

        if max_window_steps > 0:
            new_tail[(li, lo)] = derived[-max_window_steps:].copy()

        time_slice = time_idx[s:e]
        prev_tail = tail_buffer.get((li, lo))
        prev_len = 0
        if prev_tail is not None and max_window_steps > 1:
            min_t = int(time_slice.min())
            need = (max_window_steps - 1) - min_t
            if need > 0:
                prev_len = min(int(need), len(prev_tail))
                derived = np.concatenate([prev_tail[-prev_len:], derived])

        min_t = int(time_slice.min())
        max_t = int(time_slice.max())
        slice_start = max(0, (min_t + prev_len) - (max_window_steps - 1))
        slice_end = min(derived.size - 1, max_t + prev_len)
        series_slice = derived[slice_start : slice_end + 1]
        local_time = np.clip(
            (time_slice + prev_len) - slice_start, 0, series_slice.size - 1
        )

        if series_slice.size > 0 and not np.isfinite(series_slice[-1]):
            finite_in_slice = np.flatnonzero(np.isfinite(series_slice))
            if finite_in_slice.size == 0:
                continue
            last_valid = int(finite_in_slice[-1])
            local_time = np.minimum(local_time, last_valid)

        window_sums, window_counts = window_stats_batch(series_slice, local_time, steps)

        paths_slice = taxon_path[s:e]
        rows_slice = row_idx[s:e]
        for tpath in np.unique(paths_slice):
            mask = paths_slice == tpath
            rids = rows_slice[mask]
            for hours, sums in window_sums.items():
                cnts = window_counts[hours]
                values = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1), np.nan)
                col = f"{col_prefix}_avg_{hours}h"
                updates.setdefault(str(tpath), {}).setdefault(col, []).append(
                    (rids, values[mask])
                )

    return updates, new_tail


# ---------------------------------------------------------------------------
# Parquet write-back
# ---------------------------------------------------------------------------

def _atomic_write(parquet_path: Path, table: pa.Table) -> None:
    from util.storage import atomic_write_parquet
    atomic_write_parquet(parquet_path, table, row_group_size=50_000)


def _apply_updates_arrow(
    table: pa.Table,
    updates: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> pa.Table:
    """Apply row-wise value updates to a pyarrow table, adding new columns as needed."""
    length = table.num_rows
    cols: list[Any] = []
    names: list[str] = []

    for name in table.column_names:
        if name in updates:
            arr = table[name].combine_chunks()
            np_arr = np.array(arr.to_numpy(zero_copy_only=False), copy=True, dtype=float)
            for row_ids, vals in updates[name]:
                np_arr[row_ids] = vals
            cols.append(pa.array(np_arr, type=pa.float64()))
        else:
            cols.append(table[name])
        names.append(name)

    for name, chunks in updates.items():
        if name in table.column_names:
            continue
        np_arr = np.full(length, np.nan, dtype=np.float64)
        for row_ids, vals in chunks:
            np_arr[row_ids] = vals
        cols.append(pa.array(np_arr, type=pa.float64()))
        names.append(name)

    return pa.table(cols, names=names)


def write_back(
    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]],
    max_workers: int = 8,
) -> None:
    """Write accumulated column updates back to occurrence parquets atomically.

    Pops entries from updates as they are submitted so colmap memory is freed
    progressively during the flush rather than held until all writes complete.
    """
    def _write_one(tpath: str, colmap: dict) -> None:
        parquet_path = Path(tpath)
        table = pq.read_table(parquet_path).combine_chunks()
        updated = _apply_updates_arrow(table, colmap)
        _atomic_write(parquet_path, updated)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: list = []
        while updates:
            tpath, colmap = updates.popitem()
            pending.append(pool.submit(_write_one, tpath, colmap))
            if len(pending) >= max_workers * 4:
                for fut in pending:
                    fut.result()
                pending.clear()
        for fut in pending:
            fut.result()


# ---------------------------------------------------------------------------
# Derived variables (post-hoc, from already-written windowed columns)
# ---------------------------------------------------------------------------

def derive_vpd(
    root_taxon_id: str,
    data_root: str,
    occ_filename: str,
    window_hours: list[int],
) -> None:
    """Compute vapor_pressure_deficit_avg_{h}h from temperature_2m and dew_point_2m averages."""
    root = get_taxon_by_id(root_taxon_id)
    if root is None:
        raise RuntimeError(f"Unknown root taxon {root_taxon_id}")

    tree_root = Path(data_root) / "taxonomy" / "tree"
    for node in iter_descendants(root, include_self=True):
        path = tree_root / node["path"] / occ_filename
        if not path.exists():
            continue
        table = pq.read_table(path).combine_chunks()
        df = table.to_pandas()
        if df.empty:
            continue

        updated_any = False
        for hours in window_hours:
            t_col = f"temperature_2m_avg_{hours}h"
            td_col = f"dew_point_2m_avg_{hours}h"
            vpd_col = f"vapor_pressure_deficit_avg_{hours}h"
            if t_col not in df.columns or td_col not in df.columns:
                continue
            t = df[t_col].to_numpy(dtype=float)
            td = df[td_col].to_numpy(dtype=float)
            vpd = vpd_kpa(t, td)
            vpd[~np.isfinite(vpd)] = np.nan
            df[vpd_col] = vpd
            updated_any = True

        if updated_any:
            _atomic_write(path, pa.Table.from_pandas(df, preserve_index=False))


# ---------------------------------------------------------------------------
# Raster builder helpers
# ---------------------------------------------------------------------------


_WGS84 = _CRS.from_epsg(4326)

# Per-model grid metadata. GFS ny/nx are determined at runtime from the .om shape
# because its bounds don't align to exact integer steps.
RASTER_GRIDS: dict[str, dict] = {
    "copernicus_era5": {
        "ny": 721, "nx": 1440,
        "lat_min": -90.0, "lat_max": 90.0,
        "lon_min": -180.0, "lon_max": 180.0,
        "step": 0.25, "flipud": False,
    },
    "copernicus_era5_land": {
        "ny": 1801, "nx": 3600,
        "lat_min": -90.0, "lat_max": 90.0,
        "lon_min": -180.0, "lon_max": 180.0,
        "step": 0.1, "flipud": False,
    },
    "ncep_gfs013": {
        # GFS .om files are stored flat: (ny*nx, time). ny=1536, nx=3072 confirmed from shape.
        "ny": 1536, "nx": 3072,
        "lat_min": -89.912125, "lat_max": 89.912125,
        "lon_min": -180.0, "lon_max": 179.88281,
        "step": 0.117188, "flipud": False,
    },
}

# WC code values in the order used for mode count stacks.
RASTER_WC_CODES: tuple[int, ...] = (0, 1, 2, 3, 51, 53, 55, 61, 63, 65, 71, 73, 75)
_WC_CODE_TO_IDX: dict[int, int] = {c: i for i, c in enumerate(RASTER_WC_CODES)}


def accumulate_raster(
    model: str,
    variable: str,
    start_ts: float,
    end_ts: float,
    chunk_index: ChunkIndex,
) -> tuple[np.ndarray, int]:
    """Sum all hourly slices in [start_ts, end_ts] across the full native grid.

    Streams each chunk directly from S3 via fsspec (no local download).

    Returns:
        (sum_grid: float64 shape (ny, nx), n_steps: int)
        NaN values in the source are excluded from the sum.
    """
    grid = RASTER_GRIDS[model]
    flipud = grid["flipud"]

    total_sum: np.ndarray | None = None
    n_steps = 0
    resolution = chunk_index.resolution

    for entry in chunk_index.ranges:
        if entry.end < start_ts:
            continue
        if entry.start > end_ts:
            break

        t0 = max(0, int(round((max(entry.start, start_ts) - entry.start) / resolution)))
        t1 = min(entry.time_len, int(round((min(entry.end, end_ts) - entry.start) / resolution)) + 1)
        if t1 <= t0:
            continue

        reader = _open_chunk(entry, model, variable)
        ny, nx, _ = reader.shape

        if total_sum is None:
            total_sum = np.zeros((ny, nx), dtype=np.float64)

        sub = 24
        for ts in range(t0, t1, sub):
            te = min(ts + sub, t1)
            chunk_data = np.asarray(
                reader.read_array((slice(0, ny), slice(0, nx), slice(ts, te))),
                dtype=np.float64,
            )
            total_sum += np.nansum(chunk_data, axis=2)

        n_steps += t1 - t0

    if total_sum is None:
        g = RASTER_GRIDS[model]
        ny, nx = g.get("ny", 1), g.get("nx", 1)
        return np.zeros((ny, nx), dtype=np.float64), 0

    if flipud:
        total_sum = np.flipud(total_sum)
    return total_sum, n_steps


def accumulate_vpd_raster(
    model: str,
    start_ts: float,
    end_ts: float,
    t_cidx: ChunkIndex,
    td_cidx: ChunkIndex,
) -> tuple[np.ndarray, int]:
    """Accumulate sum of vpd_kpa(T[t], Td[t]) per timestep across the full native grid.

    Both temperature_2m and dew_point_2m are on the same model (e.g. copernicus_era5_land).
    Uses time-range lookup for Td chunks so chunk file naming differences are handled.

    Returns (vpd_sum: float64 (ny, nx), n_steps: int).
    """
    grid = RASTER_GRIDS[model]
    flipud = grid["flipud"]
    resolution = t_cidx.resolution

    total_sum: np.ndarray | None = None
    n_steps = 0

    for t_entry in t_cidx.ranges:
        if t_entry.end < start_ts:
            continue
        if t_entry.start > end_ts:
            break

        t0 = max(0, int(round((max(t_entry.start, start_ts) - t_entry.start) / resolution)))
        t1 = min(t_entry.time_len, int(round((min(t_entry.end, end_ts) - t_entry.start) / resolution)) + 1)
        if t1 <= t0:
            continue

        t_reader = _open_chunk(t_entry, model, "temperature_2m")
        ny, nx, _ = t_reader.shape

        if total_sum is None:
            total_sum = np.zeros((ny, nx), dtype=np.float64)

        sub = 24
        for ts in range(t0, t1, sub):
            te = min(ts + sub, t1)
            step_ts = t_entry.start + ts * resolution

            t_data = np.asarray(
                t_reader.read_array((slice(0, ny), slice(0, nx), slice(ts, te))),
                dtype=np.float64,
            )

            td_entry, td_t0 = _chunk_entry_for_time(td_cidx, step_ts)
            if td_entry is None:
                continue
            td_reader = _open_chunk(td_entry, model, "dew_point_2m")
            batch_len = te - ts
            td_data = np.asarray(
                td_reader.read_array((slice(0, ny), slice(0, nx), slice(td_t0, td_t0 + batch_len))),
                dtype=np.float64,
            )
            if td_data.shape[2] < batch_len:
                t_data = t_data[:, :, :td_data.shape[2]]

            vpd_batch = vpd_kpa(t_data, td_data)
            total_sum += np.nansum(vpd_batch, axis=2)

        n_steps += t1 - t0

    if total_sum is None:
        g = RASTER_GRIDS[model]
        ny, nx = g.get("ny", 1), g.get("nx", 1)
        return np.zeros((ny, nx), dtype=np.float64), 0

    if flipud:
        total_sum = np.flipud(total_sum)
    return total_sum, n_steps


def _rh_to_dew_point(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Magnus formula: dew_point (°C) from temperature (°C) and relative humidity (%)."""
    rh_c = np.clip(rh, 1.0, 100.0)
    gamma = np.log(rh_c / 100.0) + 17.625 * t / (243.04 + t)
    return (243.04 * gamma / (17.625 - gamma)).astype(np.float64)


def accumulate_vpd_raster_gfs(
    start_ts: float,
    end_ts: float,
    t_cidx: ChunkIndex,
    rh_cidx: ChunkIndex,
    dst_model: str,
) -> tuple[np.ndarray, int]:
    """Accumulate sum of vpd_kpa(T[t], derive_td(T[t], RH[t])) per GFS timestep.

    Derives dew_point from GFS temperature + relative_humidity per timestep, then
    computes VPD. Result is reprojected to dst_model native grid.

    Returns (vpd_sum reprojected to dst_model: float64 (ny, nx), n_steps: int).
    """
    gfs_model = "ncep_gfs013"
    gfs_grid = RASTER_GRIDS[gfs_model]
    dst_grid = RASTER_GRIDS[dst_model]
    flipud = gfs_grid["flipud"]
    resolution = t_cidx.resolution

    total_sum_gfs: np.ndarray | None = None
    n_steps = 0

    for t_entry in t_cidx.ranges:
        if t_entry.end < start_ts:
            continue
        if t_entry.start > end_ts:
            break

        t0 = max(0, int(round((max(t_entry.start, start_ts) - t_entry.start) / resolution)))
        t1 = min(t_entry.time_len, int(round((min(t_entry.end, end_ts) - t_entry.start) / resolution)) + 1)
        if t1 <= t0:
            continue

        t_reader = _open_chunk(t_entry, gfs_model, "temperature_2m")
        ny, nx, _ = t_reader.shape

        if total_sum_gfs is None:
            total_sum_gfs = np.zeros((ny, nx), dtype=np.float64)

        sub = 24
        for ts in range(t0, t1, sub):
            te = min(ts + sub, t1)
            step_ts = t_entry.start + ts * resolution

            t_data = np.asarray(
                t_reader.read_array((slice(0, ny), slice(0, nx), slice(ts, te))),
                dtype=np.float64,
            )

            rh_entry, rh_t0 = _chunk_entry_for_time(rh_cidx, step_ts)
            if rh_entry is None:
                continue
            rh_reader = _open_chunk(rh_entry, gfs_model, "relative_humidity_2m")
            batch_len = te - ts
            rh_data = np.asarray(
                rh_reader.read_array((slice(0, ny), slice(0, nx), slice(rh_t0, rh_t0 + batch_len))),
                dtype=np.float64,
            )
            if rh_data.shape[2] < batch_len:
                t_data = t_data[:, :, :rh_data.shape[2]]

            for i in range(t_data.shape[2]):
                td_slice = _rh_to_dew_point(t_data[:, :, i], rh_data[:, :, i])
                vpd_slice = vpd_kpa(t_data[:, :, i], td_slice)
                total_sum_gfs += np.where(np.isfinite(vpd_slice), vpd_slice, 0.0)

        n_steps += t1 - t0

    if total_sum_gfs is None:
        ny, nx = gfs_grid.get("ny", 1), gfs_grid.get("nx", 1)
        total_sum_gfs = np.zeros((ny, nx), dtype=np.float64)

    if flipud:
        total_sum_gfs = np.flipud(total_sum_gfs)

    reprojected = reproject_to_grid(
        total_sum_gfs.astype(np.float32),
        gfs_grid["lat_min"], gfs_grid["lat_max"], gfs_grid["lon_min"], gfs_grid["lon_max"],
        dst_grid["ny"], dst_grid["nx"],
        dst_grid["lat_min"], dst_grid["lat_max"], dst_grid["lon_min"], dst_grid["lon_max"],
    )
    return reprojected.astype(np.float64), n_steps


def accumulate_raster_mode(
    model: str,
    start_ts: float,
    end_ts: float,
    cloud_index: ChunkIndex,
    precip_index: ChunkIndex,
    swe_index: ChunkIndex,
    temp_grid_025: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Accumulate per-code timestep counts across the full native ERA5 0.25° grid.

    For each timestep in [start_ts, end_ts]: derives weather_code_array from
    cloud/precip/snow, then increments the count grid for the resulting code.

    Args:
        temp_grid_025: Optional temperature grid already on the ERA5 0.25° grid
                       (reprojected from ERA5-land 0.1°). Passed to weather_code_array
                       for the snow/rain temperature cutoff.

    Returns:
        {wc_code: count_array (ny, nx)} for each code in RASTER_WC_CODES.
        Arrays are lat-ascending (flipud applied).
    """
    grid = RASTER_GRIDS[model]
    flipud = grid["flipud"]
    resolution = cloud_index.resolution

    counts: dict[int, np.ndarray | None] = {c: None for c in RASTER_WC_CODES}

    def _cidx_entry_for(cidx: ChunkIndex, ts: float) -> tuple[ChunkRange | None, int]:
        for entry in cidx.ranges:
            if entry.start <= ts <= entry.end:
                idx = int(round((ts - entry.start) / cidx.resolution))
                return entry, idx
        return None, -1

    for cc_entry in cloud_index.ranges:
        if cc_entry.end < start_ts:
            continue
        if cc_entry.start > end_ts:
            break

        t0 = max(0, int(round((max(cc_entry.start, start_ts) - cc_entry.start) / resolution)))
        t1 = min(cc_entry.time_len, int(round((min(cc_entry.end, end_ts) - cc_entry.start) / resolution)) + 1)
        if t1 <= t0:
            continue

        # Open cc reader once for the whole chunk but read one step at a time to
        # avoid materialising a (ny, nx, n_steps) float64 array for large windows.
        cc_reader = _open_chunk(cc_entry, model, "cloud_cover")
        rny, rnx, _ = cc_reader.shape
        ny, nx = rny, rnx

        if counts[0] is None:
            for c in RASTER_WC_CODES:
                counts[c] = np.zeros((ny, nx), dtype=np.int32)

        # Cache open readers for pr/sw so we don't reopen the same file every step.
        _pr_readers: dict[int, object] = {}
        _sw_readers: dict[int, object] = {}

        for i in range(t1 - t0):
            step_ts = cc_entry.start + (t0 + i) * resolution

            cc_slice = np.asarray(
                cc_reader.read_array((slice(0, ny), slice(0, nx), slice(t0 + i, t0 + i + 1))),
                dtype=np.float64,
            )[:, :, 0]

            pr_slice = np.zeros((ny, nx), dtype=np.float64)
            pr_entry, pr_idx = _cidx_entry_for(precip_index, step_ts)
            if pr_entry is not None:
                if pr_entry.chunk_num not in _pr_readers:
                    _pr_readers[pr_entry.chunk_num] = _open_chunk(pr_entry, model, "precipitation")
                pr_slice = np.asarray(
                    _pr_readers[pr_entry.chunk_num].read_array(
                        (slice(0, ny), slice(0, nx), slice(pr_idx, pr_idx + 1))
                    ),
                    dtype=np.float64,
                )[:, :, 0]

            sw_slice = np.zeros((ny, nx), dtype=np.float64)
            sw_entry, sw_idx = _cidx_entry_for(swe_index, step_ts)
            if sw_entry is not None:
                if sw_entry.chunk_num not in _sw_readers:
                    _sw_readers[sw_entry.chunk_num] = _open_chunk(sw_entry, model, "snowfall_water_equivalent")
                sw_slice = np.asarray(
                    _sw_readers[sw_entry.chunk_num].read_array(
                        (slice(0, ny), slice(0, nx), slice(sw_idx, sw_idx + 1))
                    ),
                    dtype=np.float64,
                )[:, :, 0]

            codes = weather_code_array(
                cc_slice, pr_slice, sw_slice,
                resolution, temp=temp_grid_025,
            )
            for c in RASTER_WC_CODES:
                counts[c] += (np.round(codes) == c).astype(np.int32)

    # Initialise to zero arrays if no data was found
    g = RASTER_GRIDS[model]
    ny0, nx0 = g.get("ny", 1), g.get("nx", 1)
    result: dict[int, np.ndarray] = {}
    for c in RASTER_WC_CODES:
        arr = counts[c] if counts[c] is not None else np.zeros((ny0, nx0), dtype=np.int32)
        result[c] = np.flipud(arr) if (flipud and counts[c] is not None) else arr
    return result


def reproject_to_grid(
    src: np.ndarray,
    src_lat_min: float,
    src_lat_max: float,
    src_lon_min: float,
    src_lon_max: float,
    dst_ny: int,
    dst_nx: int,
    dst_lat_min: float,
    dst_lat_max: float,
    dst_lon_min: float,
    dst_lon_max: float,
) -> np.ndarray:
    """Bilinear reproject src (lat-ascending float array) onto a new WGS84 grid.

    Both grids must be in geographic coordinates (degrees).  src must already
    be lat-ascending (flipud applied before calling).
    """
    src_f = np.asarray(src, dtype=np.float32)
    src_ny, src_nx = src_f.shape

    src_transform = _from_bounds(src_lon_min, src_lat_min, src_lon_max, src_lat_max, src_nx, src_ny)
    dst_transform = _from_bounds(dst_lon_min, dst_lat_min, dst_lon_max, dst_lat_max, dst_nx, dst_ny)

    dst = np.empty((dst_ny, dst_nx), dtype=np.float32)
    _reproject(
        source=src_f, destination=dst,
        src_transform=src_transform, src_crs=_WGS84,
        dst_transform=dst_transform, dst_crs=_WGS84,
        resampling=_Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst


def compute_raster_final(
    var_id: str,
    agg: str,
    sums: dict[str, np.ndarray],
    n_era5: int,
    n_gfs: int,
) -> np.ndarray:
    """Compute the final output raster from accumulated sums.

    For mode vars sums contains {wc_code: count_grid}.
    For scalar vars sums contains {era5_{raw_var}: sum_grid, gfs_{raw_var}: sum_grid}.
    For VPD, sums contains era5_temperature_2m, era5_dew_point_2m (and optionally gfs_* variants).
    """
    n_total = max(n_era5 + n_gfs, 1)

    if agg == "mode":
        stack = np.stack([sums.get(c, np.zeros_like(next(iter(sums.values()))))
                          for c in RASTER_WC_CODES], axis=0)
        best = np.argmax(stack, axis=0)
        return np.array(RASTER_WC_CODES, dtype=np.float32)[best]

    zero = np.zeros_like(next(iter(sums.values())), dtype=np.float64)

    if var_id == "vapor_pressure_deficit":
        vpd_sum = sums.get("era5_vpd", zero) + sums.get("gfs_vpd", zero)
        result = vpd_sum / n_total
        return np.maximum(result, 0.0).astype(np.float32)

    if var_id == "dew_point_2m":
        era5_val = sums.get("era5_dew_point_2m", zero) / max(n_era5, 1)
        if n_gfs > 0 and "gfs_dew_point_2m" in sums:
            gfs_val = sums["gfs_dew_point_2m"] / n_gfs
            return ((era5_val * n_era5 + gfs_val * n_gfs) / n_total).astype(np.float32)
        return era5_val.astype(np.float32)

    # Generic avg / sum
    combined = zero.copy()
    for key, arr in sums.items():
        combined = combined + arr

    if agg == "avg":
        return (combined / n_total).astype(np.float32)
    return combined.astype(np.float32)


def load_raster_state(
    out_dir: str,
    var_id: str,
    window_label: str,
    suffix: str = "",
) -> tuple[dict[str, np.ndarray] | None, dict | None]:
    """Load existing sums and meta for a var+window combination.

    Returns (sums, meta) or (None, None) if either file is missing.
    """
    base = Path(out_dir) / f"{var_id}_{window_label}{suffix}"
    meta_path = base.with_suffix(".meta.json")
    sums_path = Path(str(base) + ".sums.npz")
    if not meta_path.exists() or not sums_path.exists():
        return None, None
    with open(meta_path) as fh:
        meta = json.load(fh)
    raw = dict(np.load(sums_path))
    sums: dict = {}
    for k, v in raw.items():
        try:
            sums[int(k)] = v
        except ValueError:
            sums[k] = v
    return sums, meta


def save_raster_state(
    out_dir: str,
    var_id: str,
    window_label: str,
    agg: str,
    sums: dict[str, np.ndarray],
    meta: dict,
    suffix: str = "",
) -> None:
    """Write final .npy, .meta.json, and .sums.npz for a var+window combination."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = Path(out_dir) / f"{var_id}_{window_label}{suffix}"
    npy_path = base.with_suffix(".npy")
    meta_path = base.with_suffix(".meta.json")
    sums_path = Path(str(base) + ".sums.npz")

    final = compute_raster_final(var_id, agg, sums, meta["n_era5"], meta["n_gfs"])
    np.save(npy_path, final)
    meta_out = dict(meta)
    if agg != "mode":
        meta_out["render_min"] = float(np.nanmin(final)) if np.any(np.isfinite(final)) else 0.0
        meta_out["render_max"] = float(np.nanmax(final)) if np.any(np.isfinite(final)) else 1.0
    with open(meta_path, "w") as fh:
        json.dump(meta_out, fh, indent=2)
    np.savez(sums_path, **{str(k): v for k, v in sums.items()})

