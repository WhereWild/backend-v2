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
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omfiles import OmFileReader

from config.config import load_config
from util.taxa import get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

_LAT_COL = "decimalLatitude"
_LON_COL = "decimalLongitude"
_TIME_COL = "eventTimestamp"

_S3_BASE_URL = "https://openmeteo.s3.amazonaws.com/data"
_PREFETCH_WORKERS = 8
_PREFETCH_DISK_LIMIT_GB = 1000

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


def _download_chunk(
    chunk_entry: ChunkRange,
    model: str,
    variable: str,
    cache_dir: str,
) -> Path:
    """Download a single .om chunk via aria2c and return the local path."""
    filename = _chunk_filename(chunk_entry)
    url = f"{_S3_BASE_URL}/{model}/{variable}/{filename}"

    dest_dir = Path(cache_dir) / "chunks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{model}_{variable}_{filename}"

    if target.exists():
        return target

    print(f"[download] {url}", flush=True)
    tmp_name = target.name + ".tmp"
    tmp = dest_dir / tmp_name
    try:
        subprocess.run(
            [
                "aria2c",
                "--split=8",
                "--max-connection-per-server=8",
                "--continue=true",
                "--max-tries=12",
                "--retry-wait=15",
                "--connect-timeout=60",
                f"--dir={dest_dir}",
                f"--out={tmp_name}",
                url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[download] done {target.name} ({target.stat().st_size // 1024 // 1024}MB)", flush=True)
    return target


def _download_layer_chunk(
    chunk_entry: ChunkRange,
    model: str,
    variables: list[str],
    cache_dir: str,
) -> ChunkRange:
    """Download all .om files for one chunk/layer combination; return the entry."""
    for var in variables:
        _download_chunk(chunk_entry, model, var, cache_dir)
    return chunk_entry


def prefetch_chunks(
    chunk_entries: list[ChunkRange],
    model: str,
    variables: list[str],
    cache_dir: str,
) -> None:
    """Download all needed chunks in parallel, respecting a disk space limit."""
    dest_dir = Path(cache_dir) / "chunks"
    dest_dir.mkdir(parents=True, exist_ok=True)

    limit_bytes = _PREFETCH_DISK_LIMIT_GB * 1024 ** 3
    tasks: list[tuple[ChunkRange, str, str]] = []
    for entry in chunk_entries:
        for var in variables:
            target = dest_dir / f"{model}_{var}_{_chunk_filename(entry)}"
            if not target.exists():
                tasks.append((entry, model, var))

    if not tasks:
        return

    used = sum(f.stat().st_size for f in dest_dir.glob("*.om") if f.exists())
    print(f"[prefetch] {len(tasks)} files to download, cache={used // 1024 // 1024}MB used", flush=True)

    def _fetch(args: tuple[ChunkRange, str, str]) -> Path:
        entry, mdl, var = args
        # Re-check disk usage before each download
        current = sum(f.stat().st_size for f in dest_dir.glob("*.om") if f.exists())
        if current >= limit_bytes:
            raise RuntimeError(f"Prefetch disk limit ({_PREFETCH_DISK_LIMIT_GB}GB) reached")
        return _download_chunk(entry, mdl, var, cache_dir)

    with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                entry, _, var = futures[fut]
                print(f"[prefetch] warning: {var} chunk={entry.chunk_num} failed — {exc}", flush=True)
            done += 1
            if done % 10 == 0:
                print(f"[prefetch] {done}/{len(tasks)} done", flush=True)
    print("[prefetch] complete", flush=True)


# ---------------------------------------------------------------------------
# Chunk index (S3 metadata only — no .om downloads)
# ---------------------------------------------------------------------------

_CHUNK_INDEX_CACHE: dict[tuple[str, str], ChunkIndex] = {}


def build_chunk_index(
    model: str,
    variable: str,
    *,
    min_year: int | None = None,
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

    ranges.sort(key=lambda r: r.start)

    if min_year is not None:
        cutoff = datetime(min_year, 1, 1, tzinfo=UTC).timestamp()
        ranges = [r for r in ranges if r.end >= cutoff]

    result = ChunkIndex(
        latest_end_time=float(end_time),
        resolution=resolution,
        ranges=ranges,
    )
    _CHUNK_INDEX_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Occurrence index
# ---------------------------------------------------------------------------

def build_occ_index(
    root_taxon_id: str,
    data_root: str,
    occ_filename: str,
    min_year: int | None = None,
) -> pa.Table:
    """Scan all descendant occurrence parquets and return a flat index table.

    Columns: taxon_path (str), row_idx (int64), latitude (float64),
             longitude (float64), timestamp (float64), elevation (float64).

    The `elevation` column is NaN when the DEM pipeline has not yet written an
    elevation column to the parquets; all downstream elevation corrections
    degrade gracefully to no-ops in that case.
    """
    root = get_taxon_by_id(root_taxon_id)
    if root is None:
        raise RuntimeError(f"Unknown root taxon {root_taxon_id}")

    cutoff = (
        datetime(min_year, 1, 1, tzinfo=UTC).timestamp()
        if min_year is not None
        else None
    )

    all_paths: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    all_lats: list[np.ndarray] = []
    all_lons: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    all_elevs: list[np.ndarray] = []

    tree_root = Path(data_root) / "taxonomy" / "tree"
    for node in iter_descendants(root, include_self=True):
        occ_path = tree_root / node["path"] / occ_filename
        if not occ_path.exists():
            continue
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

        all_paths.append(np.full(len(row_idx), str(occ_path), dtype=object))
        all_rows.append(row_idx)
        all_lats.append(lats)
        all_lons.append(lons)
        all_times.append(times)
        all_elevs.append(elevs)

    if not all_paths:
        return pa.table({
            "taxon_path": pa.array([], type=pa.string()),
            "row_idx": pa.array([], type=pa.int64()),
            "latitude": pa.array([], type=pa.float64()),
            "longitude": pa.array([], type=pa.float64()),
            "timestamp": pa.array([], type=pa.float64()),
            "elevation": pa.array([], type=pa.float64()),
        })

    return pa.table({
        "taxon_path": np.concatenate(all_paths),
        "row_idx": np.concatenate(all_rows),
        "latitude": np.concatenate(all_lats),
        "longitude": np.concatenate(all_lons),
        "timestamp": np.concatenate(all_times),
        "elevation": np.concatenate(all_elevs),
    })


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
    data = occ_table.to_pydict()
    times = np.asarray(data["timestamp"], dtype=np.float64)
    lats = np.asarray(data["latitude"], dtype=np.float64)
    lons = np.asarray(data["longitude"], dtype=np.float64)
    row_idx = np.asarray(data["row_idx"], dtype=np.int64)
    taxon_path = np.asarray(data["taxon_path"])
    elevation = np.asarray(
        data.get("elevation", np.full(len(times), np.nan)), dtype=np.float64
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
        "taxon_path": taxon_path,
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

    local_path = _download_chunk(chunk_entry, model, variable, cache_dir)
    reader = OmFileReader(str(local_path))
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

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    new_tail: TailBuffer = {}

    for s, e in zip(group_starts, group_ends):
        li = int(lat[s])
        lo = int(lon[s])

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

    local_paths: list[Path] = [
        _download_chunk(chunk_entry, model, var, cache_dir) for var in source_variables
    ]
    readers = [OmFileReader(str(p)) for p in local_paths]
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

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    new_tail: TailBuffer = {}

    for s, e in zip(group_starts, group_ends):
        li, lo = int(lat[s]), int(lon[s])

        try:
            raw = [np.asarray(r[li, lo, :], dtype=np.float64) for r in readers]
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


# ---------------------------------------------------------------------------
# Parquet write-back
# ---------------------------------------------------------------------------

def _atomic_write(parquet_path: Path, table: pa.Table) -> None:
    parquet_path = parquet_path.resolve()
    with tempfile.NamedTemporaryFile(
        dir=parquet_path.parent, suffix=".parquet", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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


def write_back(updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]) -> None:
    """Write accumulated column updates back to occurrence parquets atomically."""
    for tpath, colmap in updates.items():
        parquet_path = Path(tpath)
        table = pq.read_table(parquet_path).combine_chunks()
        updated = _apply_updates_arrow(table, colmap)
        _atomic_write(parquet_path, updated)


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

