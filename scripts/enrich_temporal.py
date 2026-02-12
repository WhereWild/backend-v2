from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple
import math
import tempfile
import json
import time
import threading
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import fsspec
from omfiles import OmFileReader

from util.config import load_config
import util.gis_lookup as gis_lookup
from util import taxa_navigation


CONFIG = load_config("global")

WINDOW_HOURS = (1, 8, 24, 72)
DEFAULT_CACHE_DIR = "/tmp/wherewild_temporal_cache"
# Larger blocks cut S3 GET count substantially
REMOTE_BLOCK_SIZE = 64 * 1024 * 1024
YEAR_REMOTE_BLOCK_SIZE = 8 * 1024 * 1024
# Cap per-variable chunk workers; 12 pushes harder on 8c/16t boxes without going too wide.
MAX_CHUNK_WORKERS = 1
PROGRESS_TOTAL_ROWS = 0
PROGRESS_DONE_ROWS = 0
PROGRESS_LOCK = threading.Lock()
PROGRESS_STOP = threading.Event()
CHUNK_PREFETCH_LOCKS: dict[str, threading.Lock] = {}
_AXIS_CACHE: dict[tuple[str, str], "AxisInfo"] = {}
# Prefetch year files only when they’re recent (dense and smaller)
YEAR_PREFETCH_CUTOFF = 2018
# Ignore observations before this year (skip corresponding year files)
MIN_YEAR = 2010

# Runtime-configurable (set in main loop)
MODEL = "copernicus_era5"
VARIABLE = "precipitation"
CACHE_DIR = "/tmp/wherewild_temporal_cache"
WORKLIST_PATH: Path | None = None
OCC_INDEX_PATH: Path | None = None
OCC_TABLE: pa.Table | None = None
_GRID_MODE_CACHE: dict[tuple[str, str], tuple[str, int, int, float]] = {}
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_MODEL_ELEV_LOCK = threading.Lock()
_MODEL_ELEV_LOGGED: set[str] = set()
_MODEL_ELEV_GRID_CACHE: dict[str, np.ndarray] = {}
_ELEVATION_WARNED = False


def _read_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        return kb / 1024.0
    except Exception:
        return None
    return None


def _log_status(prefix: str) -> None:
    with PROGRESS_LOCK:
        done = PROGRESS_DONE_ROWS
        total = PROGRESS_TOTAL_ROWS
    rss = _read_rss_mb()
    rss_text = f" rss={rss:.1f}MB" if rss is not None else ""
    print(f"{prefix} rows {done}/{total}{rss_text}")
    sys.stdout.flush()


def _add_progress(rows: int) -> None:
    global PROGRESS_DONE_ROWS
    with PROGRESS_LOCK:
        PROGRESS_DONE_ROWS += rows

LAT_COL = "decimalLatitude"
LON_COL = "decimalLongitude"
TIME_COL = "eventTimestamp"


@dataclass
class ChunkRange:
    chunk_num: int
    start: float
    end: float
    time_len: int
    source: str  # "chunk" or "year"


@dataclass
class AxisInfo:
    lat_start: float
    lat_step: float
    lon_start: float
    lon_step: float
    ny: int
    nx: int

    @property
    def lat_desc(self) -> bool:
        return self.lat_step < 0

    @property
    def lon_360(self) -> bool:
        # Treat grids that start at 0 (or > -1) and span past 180 as 0..360
        lon_end = self.lon_start + self.lon_step * (self.nx - 1)
        return self.lon_start >= -1.0 and lon_end > 180.0


@dataclass
class ChunkIndex:
    latest_end_time: float
    resolution: float
    ranges: list[ChunkRange]
    range_by_chunk: dict[int, ChunkRange]


def _open_s3_json(uri: str) -> dict[str, Any] | None:
    try:
        with fsspec.open(uri, mode="rb", s3={"anon": True}) as handle:
            return json.loads(handle.read())
    except Exception:
        return None


def _parse_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
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


def _open_reader(uri: str, *, block_size: int | None = None) -> OmFileReader:
    """Open an OM file with layered fallbacks to avoid truncated reads."""
    bs = block_size or REMOTE_BLOCK_SIZE
    # Fast path for already-local files (prefetched or pre-downloaded under data/gis/temporal)
    local_path = Path(uri)
    if not local_path.exists() and uri.startswith("s3://openmeteo/data/"):
        # Map s3://openmeteo/data/{model}/{variable}/{file} -> {gis_root}/temporal/{model}/{variable}/{file}
        parts = uri.split("/")
        # parts example: ['s3:', '', 'openmeteo', 'data', '{model}', '{variable}', '{file}']
        if len(parts) >= 7:
            model = parts[4]
            variable = parts[5]
            filename = parts[6]
            candidate = CONFIG.gis_root / "temporal" / model / variable / filename
            if candidate.exists():
                local_path = candidate
    if local_path.exists():
        backend = fsspec.open(local_path.as_posix(), mode="rb")
        return OmFileReader(backend)

    attempts = [
        (
            "simplecache",
            f"simplecache::{uri}",
            {
                "mode": "rb",
                "s3": {"anon": True, "default_block_size": bs},
                "simplecache": {"cache_storage": CACHE_DIR, "same_names": True},
            },
        ),
        (
            "filecache",
            f"filecache::{uri}",
            {
                "mode": "rb",
                "s3": {"anon": True, "default_block_size": bs},
                "filecache": {"cache_storage": CACHE_DIR, "same_names": True},
            },
        ),
        (
            "blockcache",
            f"blockcache::{uri}",
            {
                "mode": "rb",
                "s3": {"anon": True, "default_block_size": bs},
                "blockcache": {"cache_storage": CACHE_DIR, "block_size": bs},
            },
        ),
        (
            "direct",
            uri,
            {"mode": "rb", "s3": {"anon": True, "default_block_size": bs}},
        ),
    ]
    last_exc: Exception | None = None
    for label, target, kwargs in attempts:
        try:
            backend = fsspec.open(target, **kwargs)
            return OmFileReader(backend)
        except Exception as exc:
            last_exc = exc
            continue
    # If all attempts failed, raise the last exception
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to open reader; no attempts executed")


def _read_model_elevation(model: str, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    with _MODEL_ELEV_LOCK:
        grid = _MODEL_ELEV_GRID_CACHE.get(model)
        if grid is None:
            uri = f"s3://openmeteo/data/{model}/static/HSURF.om"
            reader = _open_reader(uri)
            try:
                grid = np.asarray(reader[:, :], dtype=float)
            finally:
                reader.close()
            _MODEL_ELEV_GRID_CACHE[model] = grid
    return grid[lat_idx, lon_idx]


def _prefetch_chunk(data_uri: str) -> Path:
    """Download a chunk/year file once to local cache and return the local path."""
    filename = Path(data_uri).name  # preserves chunk_XXX.om or year_YYYY.om

    # If already pre-downloaded under data/gis/temporal, use that directly.
    if data_uri.startswith("s3://openmeteo/data/"):
        parts = data_uri.split("/")
        if len(parts) >= 7:
            model = parts[4]
            variable = parts[5]
            candidate = CONFIG.gis_root / "temporal" / model / variable / filename
            if candidate.exists():
                return candidate

    chunk_dir = Path(CACHE_DIR) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    target = chunk_dir / f"{MODEL}_{VARIABLE}_{filename}"
    if target.exists():
        return target
    lock = CHUNK_PREFETCH_LOCKS.setdefault(str(target), threading.Lock())
    with lock:
        if target.exists():
            return target
        tmp_path = target.with_suffix(".tmp")
        with fsspec.open(data_uri, mode="rb", s3={"anon": True}) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, REMOTE_BLOCK_SIZE)
        tmp_path.replace(target)
    return target


_CHUNK_INDEX_CACHE: dict[tuple[str, str], ChunkIndex] = {}


def _build_chunk_index(model: str, variable: str) -> ChunkIndex:
    cache_key = (model, variable)
    cached = _CHUNK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    static_meta_uri = f"s3://openmeteo/data/{model}/static/meta.json"
    static_meta = _open_s3_json(static_meta_uri) or {}
    end_time = _parse_time(static_meta.get("data_end_time"))
    resolution = _parse_time(static_meta.get("temporal_resolution_seconds")) or 3600.0
    chunk_time_len = static_meta.get("chunk_time_length")
    if not isinstance(chunk_time_len, (int, float)):
        chunk_time_len = None
    if end_time is None:
        raise RuntimeError(
            f"Missing chunk metadata in static/meta.json for {model}"
        )

    fs = fsspec.filesystem("s3", anon=True)
    base = f"s3://openmeteo/data/{model}/{variable}"
    listing = fs.ls(base)

    chunk_nums: list[int] = []
    year_files: list[int] = []
    for item in listing:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str):
            continue
        leaf = name.split("/")[-1]
        if leaf.startswith("chunk_") and leaf.endswith(".om"):
            try:
                chunk_nums.append(int(leaf.replace("chunk_", "").replace(".om", "")))
            except ValueError:
                continue
        elif leaf.startswith("year_") and leaf.endswith(".om"):
            try:
                year_files.append(int(leaf.replace("year_", "").replace(".om", "")))
            except ValueError:
                continue

    ranges: list[ChunkRange] = []
    range_by_chunk: dict[int, ChunkRange] = {}

    def _year_time_len(year: int, res: float) -> int:
        from datetime import datetime, timezone
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        seconds = (end - start).total_seconds()
        return int(round(seconds / res))

    # Handle chunk_* files (typically recent data). Use running_end to avoid assuming contiguous numbering.
    if chunk_nums:
        chunk_nums_sorted = sorted(chunk_nums, reverse=True)
        running_end = float(end_time)
        for chunk_num in chunk_nums_sorted:
            if chunk_time_len is not None:
                time_len = int(chunk_time_len)
            else:
                data_uri = f"{base}/chunk_{chunk_num}.om"
                reader = _open_reader(data_uri)
                try:
                    time_len = reader.shape[2]
                finally:
                    reader.close()
            start = running_end - (time_len - 1) * resolution
            entry = ChunkRange(chunk_num=chunk_num, start=start, end=running_end, time_len=time_len, source="chunk")
            ranges.append(entry)
            range_by_chunk[chunk_num] = entry
            running_end = start - resolution

    # Handle year_* files (older data stored per year)
    if year_files:
        for year in sorted(year_files):
            time_len = _year_time_len(year, resolution)
            # Jan 1 UTC of that year
            from datetime import datetime, timezone
            start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
            start = start_dt.timestamp()
            end = start + (time_len - 1) * resolution
            entry = ChunkRange(chunk_num=year, start=start, end=end, time_len=time_len, source="year")
            ranges.append(entry)
            range_by_chunk[year] = entry

    if not ranges:
        raise RuntimeError("No OM files found for model/variable")

    # Filter out ranges earlier than MIN_YEAR if configured
    if MIN_YEAR is not None:
        cutoff = datetime(MIN_YEAR, 1, 1, tzinfo=timezone.utc).timestamp()
        ranges = [r for r in ranges if r.end >= cutoff]
        range_by_chunk = {r.chunk_num: r for r in ranges}

    # Sort ranges by start time ascending for consistent searchsorted use
    ranges.sort(key=lambda r: r.start)
    # Populate range_by_chunk for chunk_* entries only
    if not range_by_chunk:
        range_by_chunk = {entry.chunk_num: entry for entry in ranges}

    result = ChunkIndex(
        latest_end_time=float(end_time),
        resolution=float(resolution),
        ranges=ranges,
        range_by_chunk=range_by_chunk,
    )
    _CHUNK_INDEX_CACHE[cache_key] = result
    return result


def _chunk_for_timestamp(ts: float, index: ChunkIndex) -> ChunkRange:
    if ts > index.latest_end_time:
        return index.ranges[0]
    for entry in index.ranges:
        if entry.start <= ts <= entry.end:
            return entry
    return index.ranges[-1]


def _grid_indices(lat: float, lon: float, ny: int, nx: int) -> Tuple[int, int]:
    # ERA5 0.25° grid, lat ascending, lon -180..180
    lat_idx = int(round((lat + 90.0) / 0.25))
    lon_idx = int(round((lon + 180.0) / 0.25))
    lat_idx = max(0, min(lat_idx, ny - 1))
    lon_idx = max(0, min(lon_idx, nx - 1))
    return lat_idx, lon_idx


def _grid_indices_mode(lat: float, lon: float, ny: int, nx: int, mode: str, step: float) -> Tuple[int, int]:
    if mode == "lat_asc_lon_360":
        lat_idx = int(round((lat + 90.0) / step))
        lon_idx = int(round((lon % 360.0) / step))
    elif mode == "lat_asc_lon_pm180":
        lat_idx = int(round((lat + 90.0) / step))
        lon_idx = int(round((lon + 180.0) / step))
    elif mode == "lat_desc_lon_360":
        lat_idx = int(round((90.0 - lat) / step))
        lon_idx = int(round((lon % 360.0) / step))
    else:
        lat_idx = int(round((90.0 - lat) / step))
        lon_idx = int(round((lon + 180.0) / step))
    lat_idx = max(0, min(lat_idx, ny - 1))
    lon_idx = max(0, min(lon_idx, nx - 1))
    return lat_idx, lon_idx


def _axis_from_meta(meta: dict[str, Any], axis_keys: tuple[str, ...]) -> tuple[float, float, int] | None:
    """Return (start, step, count) for a given axis if present."""
    def _extract(payload: Any) -> tuple[float, float, int] | None:
        if isinstance(payload, dict):
            start = payload.get("start")
            step = payload.get("step")
            count = payload.get("count")
            if all(isinstance(v, (int, float)) for v in (start, step, count)):
                return float(start), float(step), int(count)
            if "values" in payload and isinstance(payload["values"], list) and len(payload["values"]) >= 2:
                values = payload["values"]
                return float(values[0]), float(values[1] - values[0]), len(values)
        if isinstance(payload, list) and len(payload) >= 2:
            return float(payload[0]), float(payload[1] - payload[0]), len(payload)
        return None

    for key in axis_keys:
        if key in meta:
            res = _extract(meta[key])
            if res:
                return res
    # search nested
    for value in meta.values():
        if isinstance(value, dict):
            res = _axis_from_meta(value, axis_keys)
            if res:
                return res
    return None


def _axis_info(reader: OmFileReader, model: str, variable: str) -> AxisInfo:
    cache_key = (model, variable)
    cached = _AXIS_CACHE.get(cache_key)
    if cached:
        return cached

    meta = getattr(reader, "meta", None) or getattr(reader, "metadata", None) or {}
    grid = meta.get("grid") or meta.get("dims") or {}

    lat_meta = _axis_from_meta(grid if isinstance(grid, dict) else meta, ("lat", "latitude", "y"))
    lon_meta = _axis_from_meta(grid if isinstance(grid, dict) else meta, ("lon", "longitude", "x"))
    ny, nx, _ = reader.shape

    if lat_meta and lon_meta:
        lat_start, lat_step, _ = lat_meta
        lon_start, lon_step, _ = lon_meta
    else:
        # Fallback to heuristic based on shape
        if ny in (1801, 1800) and nx in (3600, 3601):
            lat_start, lat_step = 90.0, -0.1
            lon_start, lon_step = 0.0, 0.1  # 0..360 grid
        else:
            lat_start, lat_step = 90.0, -0.25
            lon_start, lon_step = -180.0, 0.25

    info = AxisInfo(
        lat_start=lat_start,
        lat_step=lat_step,
        lon_start=lon_start,
        lon_step=lon_step,
        ny=ny,
        nx=nx,
    )
    _AXIS_CACHE[cache_key] = info
    return info


def _grid_indices_axis(lat: float, lon: float, axis: AxisInfo) -> Tuple[int, int]:
    lon_val = lon
    if axis.lon_360 and lon_val < 0:
        lon_val += 360.0
    lat_idx = int(round((lat - axis.lat_start) / axis.lat_step))
    lon_idx = int(round((lon_val - axis.lon_start) / axis.lon_step))
    lat_idx = max(0, min(lat_idx, axis.ny - 1))
    lon_idx = max(0, min(lon_idx, axis.nx - 1))
    return lat_idx, lon_idx


def _resolve_grid_mode(reader: OmFileReader) -> str:
    ny, nx, _ = reader.shape
    # Deprecated: prefer _axis_info; keep for fallback callers
    step = 0.25 if not (ny in (1801, 1800) and nx in (3600, 3601)) else 0.1
    test_lat = 40.8
    test_lon = -111.9
    modes = (
        "lat_asc_lon_pm180",
        "lat_asc_lon_360",
        "lat_desc_lon_pm180",
        "lat_desc_lon_360",
    )
    for mode in modes:
        li, lo = _grid_indices_mode(test_lat, test_lon, ny, nx, mode, step)
        try:
            value = reader[li, lo, 0]
        except Exception:
            continue
        try:
            if not np.isnan(value):
                return mode
        except Exception:
            return mode
    return "lat_asc_lon_pm180"


def _resolve_grid_mode_from_samples(
    reader: OmFileReader,
    lats: np.ndarray,
    lons: np.ndarray,
    step: float,
    max_samples: int = 20,
) -> str:
    """Pick the grid mode that yields the most finite values for sample points."""
    ny, nx, _ = reader.shape
    modes = (
        "lat_asc_lon_pm180",
        "lat_asc_lon_360",
        "lat_desc_lon_pm180",
        "lat_desc_lon_360",
    )
    if len(lats) == 0:
        return _resolve_grid_mode(reader)

    # Sample a few points from observations
    stride = max(1, len(lats) // max_samples)
    sample_idx = np.arange(0, len(lats), stride)[:max_samples]
    sample_lats = lats[sample_idx]
    sample_lons = lons[sample_idx]

    best_mode = modes[0]
    best_score = -1
    for mode in modes:
        score = 0
        for lat, lon in zip(sample_lats, sample_lons):
            li, lo = _grid_indices_mode(float(lat), float(lon), ny, nx, mode, step)
            try:
                value = reader[li, lo, 0]
            except Exception:
                continue
            try:
                if np.isfinite(value):
                    score += 1
            except Exception:
                score += 1
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode


def _atomic_write(parquet_path: Path, table: pa.Table) -> None:
    parquet_path = parquet_path.resolve()
    with tempfile.NamedTemporaryFile(
        dir=parquet_path.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _ensure_column_array(table: pa.Table, column: str, length: int, dtype: pa.DataType | None = None) -> pa.Array:
    if column in table.column_names:
        combined = table[column].combine_chunks()
        if isinstance(combined, pa.ChunkedArray):
            if combined.num_chunks == 0:
                return pa.array([], type=combined.type)
            return combined.chunk(0)
        # Already a single Array
        if len(combined) == 0:
            return pa.array([], type=combined.type)
        return combined
    if dtype is None:
        dtype = pa.float64()
    return pa.nulls(length, type=dtype)


def _apply_updates_arrow(table: pa.Table, updates: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> pa.Table:
    """Apply row-wise updates without pandas; only touched columns are materialized."""
    cols: list[pa.ChunkedArray] = []
    names: list[str] = []
    length = table.num_rows

    # Preload untouched columns for zero-copy reuse
    untouched = {name: col for name, col in zip(table.column_names, table.columns) if name not in updates}

    for name in table.column_names:
        if name in updates:
            base_arr = _ensure_column_array(table, name, length)
            # Ensure writable array; pyarrow may return read-only views
            np_arr = np.array(base_arr.to_numpy(zero_copy_only=False), copy=True)
            for row_ids, vals in updates[name]:
                np_arr[row_ids] = vals
            cols.append(pa.array(np_arr, type=base_arr.type))
            names.append(name)
        else:
            cols.append(untouched[name])
            names.append(name)

    # Append any new columns not previously in table
    for name, chunks in updates.items():
        if name in table.column_names:
            continue
        np_arr = np.full(length, np.nan, dtype=float)
        for row_ids, vals in chunks:
            np_arr[row_ids] = vals
        cols.append(pa.array(np_arr, type=pa.float64()))
        names.append(name)

    return pa.table(cols, names=names)


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan


def _window_steps(resolution: float, window_hours: tuple[int, ...]) -> dict[int, int]:
    """Precompute window lengths (in steps) for each configured hour span."""
    return {hours: int(round((hours * 3600) / resolution)) for hours in window_hours}


def _window_stats_batch(
    series: np.ndarray,
    time_indices: np.ndarray,
    steps: dict[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute window sums and counts for many timestamps against one time series.

    Uses cumulative sums so each window is O(1) instead of slicing per row.
    """
    finite = np.isfinite(series)
    clean_series = np.where(finite, series, 0.0)
    cumsum = np.cumsum(clean_series, dtype=np.float64)
    ccount = np.cumsum(finite.astype(np.int64))
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}

    for hours, window_len in steps.items():
        if window_len <= 0:
            sums[hours] = np.full_like(time_indices, np.nan, dtype=float)
            counts[hours] = np.zeros_like(time_indices, dtype=np.int64)
            continue

        end_idx = time_indices
        start_idx = end_idx - (window_len - 1)
        start_idx = np.clip(start_idx, 0, len(clean_series) - 1)

        prefix_sum = np.where(start_idx > 0, cumsum[start_idx - 1], 0.0)
        prefix_cnt = np.where(start_idx > 0, ccount[start_idx - 1], 0)
        window_sum = cumsum[end_idx] - prefix_sum
        window_cnt = ccount[end_idx] - prefix_cnt
        sums[hours] = window_sum.astype(float)
        counts[hours] = window_cnt.astype(np.int64)

    return sums, counts


def _weather_code_simple_row(
    cloudcover: float | None,
    precipitation: float | None,
    snowfall_water_equivalent: float | None,
    model_dt_seconds: float,
) -> int | None:
    """Derive simple WMO weather code from 1h aggregates (logic from omfiles_sample_parquet)."""
    if not all(v is not None and np.isfinite(v) for v in (cloudcover, precipitation, snowfall_water_equivalent)):
        return None
    model_dt_hours = model_dt_seconds / 3600.0
    snowfall_cm = snowfall_water_equivalent / 10.0

    rate_snow = snowfall_cm / model_dt_hours
    if 0.01 <= rate_snow < 0.2:
        return 71
    if 0.2 <= rate_snow < 0.8:
        return 73
    if rate_snow >= 0.8:
        return 75

    rate_rain = precipitation / model_dt_hours
    if 0.01 <= rate_rain < 0.5:
        return 51
    if 0.5 <= rate_rain < 1.0:
        return 53
    if 1.0 <= rate_rain < 1.3:
        return 55
    if 1.3 <= rate_rain < 2.5:
        return 61
    if 2.5 <= rate_rain < 7.6:
        return 63
    if rate_rain >= 7.6:
        return 65

    if cloudcover < 20:
        return 0
    if cloudcover < 50:
        return 1
    if cloudcover < 80:
        return 2
    return 3


def _build_occ_index(index_path: Path) -> int:
    """Scan all taxa once and write base occurrence index with lat/lon/time."""
    if index_path.exists():
        table = pq.ParquetFile(index_path)
        if "target_elevation" in (table.schema.names or []):
            return table.metadata.num_rows
        index_path.unlink()

    schema = pa.schema(
        [
            ("taxon_path", pa.string()),
            ("row_idx", pa.int64()),
            ("timestamp", pa.float64()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("target_elevation", pa.float64()),
        ]
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(index_path, schema)

    total_rows = 0
    root = CONFIG.root_taxon_id
    root_record = taxa_navigation.get_taxon_by_id(root)
    if root_record is None:
        raise RuntimeError(f"Unknown root taxon {root}")

    nodes = list(taxa_navigation.iter_descendants(root_record, include_self=True))
    total_nodes = len(nodes)
    start_time = time.time()
    last_log = start_time
    for idx, node in enumerate(nodes, 1):
        occ_path = Path(node["path"]) / CONFIG.occurrence_parquet_filename
        if not occ_path.exists():
            continue
        table = pq.read_table(occ_path).combine_chunks()
        df = table.to_pandas()
        mask = (
            df[TIME_COL].notna()
            & df[LAT_COL].notna()
            & df[LON_COL].notna()
        )
        if not mask.any():
            continue

        pending = df[mask]
        pending_index = pending.index.to_numpy()
        # Convert timestamps to POSIX seconds. If they are already numeric seconds,
        # leave them; if they look like pandas datetime64[ns], scale down.
        times_raw = pending[TIME_COL].to_numpy()
        if hasattr(times_raw, "dtype") and str(times_raw.dtype).startswith("datetime64"):
            times = times_raw.astype("datetime64[ns]").astype("int64") / 1e9
        else:
            times = times_raw.astype(float)
            # Heuristic: values >> 1e12 are likely ns; scale down to seconds
            if times.size and float(times[0]) > 1e12:
                times = times / 1e9
        lats = pending[LAT_COL].to_numpy(dtype=float)
        lons = pending[LON_COL].to_numpy(dtype=float)
        elev_cols = getattr(CONFIG, "temporal_elevation_columns", ())
        elev_col = next((c for c in elev_cols if c in pending.columns), None)
        if elev_col is None:
            target_elev = np.full(len(pending_index), np.nan, dtype=float)
        else:
            target_elev = pending[elev_col].to_numpy(dtype=float)

        base_table = pa.Table.from_pydict(
            {
                "taxon_path": np.array([str(occ_path)] * len(pending_index)),
                "row_idx": pending_index.astype(np.int64),
                "timestamp": times.astype(np.float64),
                "latitude": lats.astype(np.float64),
                "longitude": lons.astype(np.float64),
                "target_elevation": target_elev.astype(np.float64),
            }
        )
        writer.write_table(base_table)
        total_rows += len(pending_index)

        now = time.time()
        if (idx == 1) or (idx % 250 == 0) or (now - last_log) >= 10:
            elapsed = now - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0.0
            remaining_nodes = max(0, total_nodes - idx)
            eta = (elapsed / idx) * remaining_nodes if idx > 0 else float("inf")
            eta_text = "--" if not math.isfinite(eta) else f"{int(eta//60)}m{int(eta%60):02d}s"
            print(
                f"[occ_index] taxa {idx}/{total_nodes} rows={total_rows} "
                f"avg {rate:.1f} rows/s eta {eta_text}"
            )
            last_log = now

    writer.close()
    return total_rows


def _iter_occ_index_batches(index_path: Path, batch_rows: int) -> Iterable[pa.Table]:
    if batch_rows <= 0:
        yield pq.read_table(index_path).combine_chunks()
        return
    parquet = pq.ParquetFile(index_path)
    for batch in parquet.iter_batches(batch_size=batch_rows):
        yield pa.Table.from_batches([batch]).combine_chunks()


def _build_worklist_from_index(chunk_index: ChunkIndex, occ_table: pa.Table, model: str, variable: str) -> pa.Table:
    """Project base occurrence index onto a model/variable chunk grid (in-memory)."""
    # Use ascending-by-start ordering for searchsorted
    asc_ranges = sorted(chunk_index.ranges, key=lambda r: r.start)
    asc_starts = np.array([r.start for r in asc_ranges], dtype=float)
    asc_chunk_nums = np.array([r.chunk_num for r in asc_ranges], dtype=int)
    asc_time_lens = np.array([r.time_len for r in asc_ranges], dtype=int)
    data = occ_table.to_pydict()
    times = np.array(data["timestamp"], dtype=float)
    lats = np.array(data["latitude"], dtype=float)
    lons = np.array(data["longitude"], dtype=float)
    row_idx = np.array(data["row_idx"], dtype=int)
    taxon_path = np.array(data["taxon_path"])
    target_elev = np.array(data.get("target_elevation", np.full(len(times), np.nan)), dtype=float)

    if MIN_YEAR is not None:
        cutoff = datetime(MIN_YEAR, 1, 1, tzinfo=timezone.utc).timestamp()
        valid_mask = times >= cutoff
        if not valid_mask.any():
            return pa.Table.from_pydict(
                {
                    "taxon_path": np.array([], dtype=object),
                    "row_idx": np.array([], dtype=np.int64),
                    "chunk_num": np.array([], dtype=np.int32),
                    "lat_idx": np.array([], dtype=np.int32),
                    "lon_idx": np.array([], dtype=np.int32),
                    "time_idx": np.array([], dtype=np.int32),
                }
            )
        times = times[valid_mask]
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        row_idx = row_idx[valid_mask]
        taxon_path = taxon_path[valid_mask]
        target_elev = target_elev[valid_mask]

    # Resolve grid mode by probing sample points against actual data (more reliable than meta for era5_land)
    cache_key = (model, variable)
    grid_meta = _GRID_MODE_CACHE.get(cache_key)
    if grid_meta is None:
        override_mode = CONFIG.temporal_grid_mode_by_model.get(model)
        allowed_modes = {
            "lat_asc_lon_pm180",
            "lat_asc_lon_360",
            "lat_desc_lon_pm180",
            "lat_desc_lon_360",
        }
        if override_mode not in allowed_modes:
            raise RuntimeError(
                f"Missing/invalid temporal_grid_mode_by_model for {model}: {override_mode}"
            )
        # Pick a representative range (latest in time) and open the right file (chunk/year)
        sample_range = chunk_index.ranges[-1]
        if sample_range.source == "year":
            sample_uri = f"s3://openmeteo/data/{model}/{variable}/year_{sample_range.chunk_num}.om"
        else:
            sample_uri = f"s3://openmeteo/data/{model}/{variable}/chunk_{sample_range.chunk_num}.om"
        reader = _open_reader(sample_uri)
        try:
            ny, nx, _ = reader.shape
            step = 0.1 if (ny in (1801, 1800) and nx in (3600, 3601)) else 0.25
            grid_mode = override_mode
            print(f"[grid] variable={variable} model={model} mode={grid_mode} (override)")
        finally:
            reader.close()
        grid_meta = (grid_mode, ny, nx, step)
        _GRID_MODE_CACHE[cache_key] = grid_meta

    # Compute grid indices per observation using chosen grid mode
    grid_mode, ny, nx, step = grid_meta
    if grid_mode == "lat_asc_lon_360":
        lat_idx = np.rint((lats + 90.0) / step).astype(int)
        lon_idx = np.rint((np.mod(lons, 360.0)) / step).astype(int)
    elif grid_mode == "lat_asc_lon_pm180":
        lat_idx = np.rint((lats + 90.0) / step).astype(int)
        lon_idx = np.rint((lons + 180.0) / step).astype(int)
    elif grid_mode == "lat_desc_lon_360":
        lat_idx = np.rint((90.0 - lats) / step).astype(int)
        lon_idx = np.rint((np.mod(lons, 360.0)) / step).astype(int)
    else:
        lat_idx = np.rint((90.0 - lats) / step).astype(int)
        lon_idx = np.rint((lons + 180.0) / step).astype(int)
    lat_idx = np.clip(lat_idx, 0, ny - 1)
    lon_idx = np.clip(lon_idx, 0, nx - 1)

    chunk_lookup = np.searchsorted(asc_starts, times, side="right") - 1
    chunk_lookup = np.clip(chunk_lookup, 0, len(asc_ranges) - 1)
    chunk_nums = asc_chunk_nums[chunk_lookup]
    chunk_starts = asc_starts[chunk_lookup]
    chunk_time_lens = asc_time_lens[chunk_lookup]
    # Align to the start of the hour (floor), matching Open-Meteo + OM 24h window logic.
    time_indices = np.floor((times - chunk_starts) / chunk_index.resolution).astype(int)
    time_indices = np.clip(time_indices, 0, chunk_time_lens - 1)

    total_rows = len(row_idx)
    print(f"[worklist] projected {total_rows}/{occ_table.num_rows} rows onto model grid")
    if getattr(CONFIG, "temporal_debug_model_elevation", False) and model not in _MODEL_ELEV_LOGGED:
        try:
            elev = _read_model_elevation(model, lat_idx, lon_idx)
            elev = np.where(elev <= -900, np.nan, elev)
            finite = np.isfinite(elev)
            if finite.any():
                elev_min = float(np.nanmin(elev))
                elev_med = float(np.nanmedian(elev))
                elev_max = float(np.nanmax(elev))
                min_i = int(np.nanargmin(elev))
                max_i = int(np.nanargmax(elev))
                min_lat = float(lats[min_i])
                min_lon = float(lons[min_i])
                max_lat = float(lats[max_i])
                max_lon = float(lons[max_i])
                missing = 1.0 - (finite.sum() / len(elev))
                print(
                    f"[model_elev] model={model} rows={len(elev)} "
                    f"min={elev_min:.2f} @({min_lat:.5f},{min_lon:.5f}) "
                    f"med={elev_med:.2f} "
                    f"max={elev_max:.2f} @({max_lat:.5f},{max_lon:.5f}) "
                    f"missing={missing:.3%}"
                )
            else:
                print(f"[model_elev] model={model} rows={len(elev)} all_missing")
        except Exception as exc:
            print(f"[model_elev] model={model} failed: {exc}")
        _MODEL_ELEV_LOGGED.add(model)
    return pa.Table.from_pydict(
        {
            "taxon_path": taxon_path,
            "row_idx": row_idx.astype(np.int64),
            "chunk_num": chunk_nums.astype(np.int32),
            "lat_idx": lat_idx.astype(np.int32),
            "lon_idx": lon_idx.astype(np.int32),
            "time_idx": time_indices.astype(np.int32),
            "target_elevation": target_elev.astype(np.float32),
        }
    )


def _apply_pre_cutoff_nans(
    occ_table: pa.Table,
    variable: str,
    window_hours: tuple[int, ...],
    agg_mode: str,
) -> None:
    if MIN_YEAR is None:
        return
    cutoff = datetime(MIN_YEAR, 1, 1, tzinfo=timezone.utc).timestamp()
    data = occ_table.to_pydict()
    times = np.array(data["timestamp"], dtype=float)
    pre_mask = times < cutoff
    if not pre_mask.any():
        return
    taxon_path = np.array(data["taxon_path"])[pre_mask]
    row_idx = np.array(data["row_idx"], dtype=int)[pre_mask]

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for tpath in np.unique(taxon_path):
        mask = taxon_path == tpath
        rows = row_idx[mask]
        if rows.size == 0:
            continue
        colmap = updates.setdefault(tpath, {})
        for hours in window_hours:
            col = f"{variable}_{agg_mode}_{hours}h"
            colmap.setdefault(col, []).append((rows, np.full(rows.shape, np.nan, dtype=float)))

    for tpath, colmap in updates.items():
        lock = _WRITE_LOCKS.setdefault(tpath, threading.Lock())
        with lock:
            parquet_path = Path(tpath)
            table = pq.read_table(parquet_path).combine_chunks()
            updated = _apply_updates_arrow(table, colmap)
            _atomic_write(parquet_path, updated)


def _process_worklist(
    chunk_index: ChunkIndex,
    worklist: pa.Table,
    model: str,
    variable: str,
    window_hours: tuple[int, ...],
    agg_mode: str,
) -> None:
    """Process global worklist: read each chunk once, write results back to parquets."""
    steps = _window_steps(chunk_index.resolution, window_hours)
    ranges_by_start = sorted(chunk_index.ranges, key=lambda r: r.start)
    prev_by_chunk: dict[int, ChunkRange | None] = {}
    for idx, entry in enumerate(ranges_by_start):
        prev_by_chunk[entry.chunk_num] = ranges_by_start[idx - 1] if idx > 0 else None
    # Collect per-taxon updates; merged after chunk parallelism
    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    updates_lock = threading.Lock()

    def _open_chunk_reader(entry: ChunkRange) -> OmFileReader:
        if entry.source == "year":
            data_uri = f"s3://openmeteo/data/{model}/{variable}/year_{entry.chunk_num}.om"
            if entry.chunk_num >= YEAR_PREFETCH_CUTOFF:
                local_chunk = _prefetch_chunk(data_uri)
                return _open_reader(local_chunk.as_posix(), block_size=YEAR_REMOTE_BLOCK_SIZE)
            return _open_reader(data_uri, block_size=YEAR_REMOTE_BLOCK_SIZE)
        data_uri = f"s3://openmeteo/data/{model}/{variable}/chunk_{entry.chunk_num}.om"
        local_chunk = _prefetch_chunk(data_uri)
        return _open_reader(local_chunk.as_posix())

    def _process_chunk(chunk_entry: ChunkRange, chunk_table: pa.Table) -> tuple[int, int, bool, str]:
        # verbose chunk start suppressed for normal runs
        data = chunk_table.to_pydict()
        lat = np.array(data["lat_idx"], dtype=int)
        lon = np.array(data["lon_idx"], dtype=int)
        time_idx = np.array(data["time_idx"], dtype=int)
        taxon_path = np.array(data["taxon_path"])
        row_idx = np.array(data["row_idx"], dtype=int)
        target_elev = np.array(data.get("target_elevation", np.full(len(lat), np.nan)), dtype=float)
        do_elev = variable in getattr(CONFIG, "temporal_elevation_correctable_vars", ())
        correction = None

        order = np.lexsort((lon, lat))
        lat = lat[order]
        lon = lon[order]
        time_idx = time_idx[order]
        taxon_path = taxon_path[order]
        row_idx = row_idx[order]
        target_elev = target_elev[order]

        if do_elev:
            if not np.isfinite(target_elev).any():
                global _ELEVATION_WARNED
                if not _ELEVATION_WARNED:
                    print("[elevation] missing target elevation column; skipping correction")
                    _ELEVATION_WARNED = True
                do_elev = False
            else:
                model_elev = _read_model_elevation(model, lat, lon)
                # Treat nodata/invalid model elevations as missing
                model_elev = np.where(model_elev <= -900, np.nan, model_elev)
                correction = (model_elev - target_elev) * 0.0065
                correction_mask = np.isfinite(correction)

        change = np.empty_like(lat, dtype=bool)
        change[0] = True
        change[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
        group_starts = np.flatnonzero(change)
        group_ends = np.append(group_starts[1:], len(lat))

        reader = _open_chunk_reader(chunk_entry)
        ny, nx, _ = reader.shape

        local_updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}

        max_window_steps = max(steps.values()) if steps else 0
        prev_entry = prev_by_chunk.get(chunk_entry.chunk_num)
        prev_reader: OmFileReader | None = None

        for s, e in zip(group_starts, group_ends):
            li = int(lat[s])
            lo = int(lon[s])
            if li >= ny or lo >= nx:
                continue
            try:
                series = np.array(reader[li, lo, :], dtype=float)
            except Exception:
                series = np.array([reader[li, lo, i] for i in range(reader.shape[2])], dtype=float)

            if series.size == 0:
                continue
            time_slice = time_idx[s:e]
            min_idx = int(time_slice.min())
            max_idx = int(time_slice.max())
            prev_len = 0
            if prev_entry is not None and max_window_steps > 1 and min_idx < (max_window_steps - 1):
                try:
                    if prev_reader is None:
                        prev_reader = _open_chunk_reader(prev_entry)
                    prev_series = np.array(prev_reader[li, lo, :], dtype=float)
                    need = (max_window_steps - 1) - min_idx
                    if prev_series.size > 0 and need > 0:
                        prev_len = min(int(need), int(prev_series.size))
                        series = np.concatenate([prev_series[-prev_len:], series])
                except Exception:
                    prev_len = 0
            slice_start = max(0, (min_idx + prev_len) - (max_window_steps - 1))
            slice_end = min(series.size - 1, max_idx + prev_len)
            series_slice = series[slice_start : slice_end + 1]
            local_time = np.clip((time_slice + prev_len) - slice_start, 0, series_slice.size - 1)
            window_sums, window_counts = _window_stats_batch(series_slice, local_time, steps)

            paths_slice = taxon_path[s:e]
            rows_slice = row_idx[s:e]
            for tpath in np.unique(paths_slice):
                mask = paths_slice == tpath
                if not mask.any():
                    continue
                row_ids = rows_slice[mask]
                for hours, sums in window_sums.items():
                    counts = window_counts[hours]
                    if agg_mode == "sum":
                        values = np.where(counts > 0, sums, np.nan)
                    else:
                        values = np.full_like(sums, np.nan, dtype=float)
                        np.divide(sums, counts, out=values, where=counts > 0)
                        if do_elev and correction is not None:
                            corr_slice = correction[s:e]
                            mask_slice = correction_mask[s:e]
                            if mask_slice.any():
                                values = values.copy()
                                values[mask_slice] += corr_slice[mask_slice]
                    col = f"{variable}_{agg_mode}_{hours}h"
                    local_updates.setdefault(tpath, {}).setdefault(col, []).append((row_ids, values[mask]))

            _add_progress(e - s)

        reader.close()
        if prev_reader is not None:
            prev_reader.close()

        with updates_lock:
            for tpath, colmap in local_updates.items():
                for col, chunks in colmap.items():
                    updates.setdefault(tpath, {}).setdefault(col, []).extend(chunks)

    chunk_tables: list[tuple[ChunkRange, pa.Table]] = []
    for chunk_entry in chunk_index.ranges:
        table = worklist.filter(pa.compute.equal(worklist["chunk_num"], chunk_entry.chunk_num))
        if table.num_rows == 0:
            continue
        chunk_tables.append((chunk_entry, table))

    if not chunk_tables:
        return

    worker_count = min(MAX_CHUNK_WORKERS, len(chunk_tables) or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_process_chunk, ce, tbl) for ce, tbl in chunk_tables]
        for future in as_completed(futures):
            future.result()

    for tpath, colmap in updates.items():
        lock = _WRITE_LOCKS.setdefault(tpath, threading.Lock())
        with lock:
            parquet_path = Path(tpath)
            table = pq.read_table(parquet_path).combine_chunks()
            updated = _apply_updates_arrow(table, colmap)
            _atomic_write(parquet_path, updated)


def _process_vpd_worklist(
    chunk_index: ChunkIndex,
    worklist: pa.Table,
    model: str,
    window_hours: tuple[int, ...],
) -> None:
    """Compute VPD per hour from temperature_2m and dew_point_2m, then aggregate."""
    steps = _window_steps(chunk_index.resolution, window_hours)
    ranges_by_start = sorted(chunk_index.ranges, key=lambda r: r.start)
    prev_by_chunk: dict[int, ChunkRange | None] = {}
    for idx, entry in enumerate(ranges_by_start):
        prev_by_chunk[entry.chunk_num] = ranges_by_start[idx - 1] if idx > 0 else None

    updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    updates_lock = threading.Lock()

    def _open_chunk_reader(entry: ChunkRange, variable: str) -> OmFileReader:
        if entry.source == "year":
            data_uri = f"s3://openmeteo/data/{model}/{variable}/year_{entry.chunk_num}.om"
            if entry.chunk_num >= YEAR_PREFETCH_CUTOFF:
                local_chunk = _prefetch_chunk(data_uri)
                return _open_reader(local_chunk.as_posix(), block_size=YEAR_REMOTE_BLOCK_SIZE)
            return _open_reader(data_uri, block_size=YEAR_REMOTE_BLOCK_SIZE)
        data_uri = f"s3://openmeteo/data/{model}/{variable}/chunk_{entry.chunk_num}.om"
        local_chunk = _prefetch_chunk(data_uri)
        return _open_reader(local_chunk.as_posix())

    def _process_chunk(chunk_entry: ChunkRange, chunk_table: pa.Table) -> None:
        data = chunk_table.to_pydict()
        lat = np.array(data["lat_idx"], dtype=int)
        lon = np.array(data["lon_idx"], dtype=int)
        time_idx = np.array(data["time_idx"], dtype=int)
        taxon_path = np.array(data["taxon_path"])
        row_idx = np.array(data["row_idx"], dtype=int)
        target_elev = np.array(data.get("target_elevation", np.full(len(lat), np.nan)), dtype=float)
        do_elev = "temperature_2m" in getattr(CONFIG, "temporal_elevation_correctable_vars", ())
        correction = None

        order = np.lexsort((lon, lat))
        lat = lat[order]
        lon = lon[order]
        time_idx = time_idx[order]
        taxon_path = taxon_path[order]
        row_idx = row_idx[order]
        target_elev = target_elev[order]

        if do_elev:
            if not np.isfinite(target_elev).any():
                global _ELEVATION_WARNED
                if not _ELEVATION_WARNED:
                    print("[elevation] missing target elevation column; skipping correction")
                    _ELEVATION_WARNED = True
                do_elev = False
            else:
                model_elev = _read_model_elevation(model, lat, lon)
                model_elev = np.where(model_elev <= -900, np.nan, model_elev)
                correction = (model_elev - target_elev) * 0.0065

        change = np.empty_like(lat, dtype=bool)
        change[0] = True
        change[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
        group_starts = np.flatnonzero(change)
        group_ends = np.append(group_starts[1:], len(lat))

        temp_reader = _open_chunk_reader(chunk_entry, "temperature_2m")
        dew_reader = _open_chunk_reader(chunk_entry, "dew_point_2m")
        ny, nx, _ = temp_reader.shape

        max_window_steps = max(steps.values()) if steps else 0
        prev_entry = prev_by_chunk.get(chunk_entry.chunk_num)
        prev_temp_reader: OmFileReader | None = None
        prev_dew_reader: OmFileReader | None = None

        local_updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}

        for s, e in zip(group_starts, group_ends):
            li = int(lat[s])
            lo = int(lon[s])
            if li >= ny or lo >= nx:
                continue
            try:
                temp_series = np.array(temp_reader[li, lo, :], dtype=float)
            except Exception:
                temp_series = np.array([temp_reader[li, lo, i] for i in range(temp_reader.shape[2])], dtype=float)
            try:
                dew_series = np.array(dew_reader[li, lo, :], dtype=float)
            except Exception:
                dew_series = np.array([dew_reader[li, lo, i] for i in range(dew_reader.shape[2])], dtype=float)

            if temp_series.size == 0 or dew_series.size == 0:
                continue

            time_slice = time_idx[s:e]
            min_idx = int(time_slice.min())
            max_idx = int(time_slice.max())
            prev_len = 0
            if prev_entry is not None and max_window_steps > 1 and min_idx < (max_window_steps - 1):
                need = (max_window_steps - 1) - min_idx
                if need > 0:
                    if prev_temp_reader is None:
                        prev_temp_reader = _open_chunk_reader(prev_entry, "temperature_2m")
                    if prev_dew_reader is None:
                        prev_dew_reader = _open_chunk_reader(prev_entry, "dew_point_2m")
                    try:
                        prev_temp = np.array(prev_temp_reader[li, lo, :], dtype=float)
                        prev_dew = np.array(prev_dew_reader[li, lo, :], dtype=float)
                        prev_len = min(int(need), int(prev_temp.size), int(prev_dew.size))
                        if prev_len > 0:
                            temp_series = np.concatenate([prev_temp[-prev_len:], temp_series])
                            dew_series = np.concatenate([prev_dew[-prev_len:], dew_series])
                    except Exception:
                        prev_len = 0

            slice_start = max(0, (min_idx + prev_len) - (max_window_steps - 1))
            slice_end = min(temp_series.size - 1, max_idx + prev_len)
            temp_slice = temp_series[slice_start : slice_end + 1]
            dew_slice = dew_series[slice_start : slice_end + 1]
            local_time = np.clip((time_slice + prev_len) - slice_start, 0, temp_slice.size - 1)

            if do_elev and correction is not None:
                corr = correction[s]
                if np.isfinite(corr):
                    temp_slice = temp_slice + corr
                    dew_slice = dew_slice + corr

            # Compute VPD per hour (kPa)
            es = 0.6108 * np.exp((17.27 * temp_slice) / (temp_slice + 237.3))
            ea = 0.6108 * np.exp((17.27 * dew_slice) / (dew_slice + 237.3))
            vpd_series = es - ea

            window_sums, window_counts = _window_stats_batch(vpd_series, local_time, steps)

            paths_slice = taxon_path[s:e]
            rows_slice = row_idx[s:e]
            for tpath in np.unique(paths_slice):
                mask = paths_slice == tpath
                if not mask.any():
                    continue
                row_ids = rows_slice[mask]
                for hours, sums in window_sums.items():
                    counts = window_counts[hours]
                    values = np.full_like(sums, np.nan, dtype=float)
                    np.divide(sums, counts, out=values, where=counts > 0)
                    col = f"vapor_pressure_deficit_avg_{hours}h"
                    local_updates.setdefault(tpath, {}).setdefault(col, []).append((row_ids, values[mask]))

        temp_reader.close()
        dew_reader.close()
        if prev_temp_reader is not None:
            prev_temp_reader.close()
        if prev_dew_reader is not None:
            prev_dew_reader.close()

        with updates_lock:
            for tpath, colmap in local_updates.items():
                for col, chunks in colmap.items():
                    updates.setdefault(tpath, {}).setdefault(col, []).extend(chunks)

    chunk_tables: list[tuple[ChunkRange, pa.Table]] = []
    for chunk_entry in chunk_index.ranges:
        table = worklist.filter(pa.compute.equal(worklist["chunk_num"], chunk_entry.chunk_num))
        if table.num_rows == 0:
            continue
        chunk_tables.append((chunk_entry, table))

    if not chunk_tables:
        return

    worker_count = min(MAX_CHUNK_WORKERS, len(chunk_tables) or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_process_chunk, ce, tbl) for ce, tbl in chunk_tables]
        for future in as_completed(futures):
            future.result()

    for tpath, colmap in updates.items():
        lock = _WRITE_LOCKS.setdefault(tpath, threading.Lock())
        with lock:
            parquet_path = Path(tpath)
            table = pq.read_table(parquet_path).combine_chunks()
            updated = _apply_updates_arrow(table, colmap)
            _atomic_write(parquet_path, updated)

def _process_taxon(taxon: taxa_navigation.TaxonRecord, chunk_index: ChunkIndex) -> None:
    path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    if not path.exists():
        return

    table = pq.read_table(path).combine_chunks()
    df = table.to_pandas()
    if df.empty:
        return

    target_cols = [f"{VARIABLE}_sum_{h}h" for h in WINDOW_HOURS]
    _ensure_columns(df, target_cols)

    mask = (
        df[TIME_COL].notna()
        & df[LAT_COL].notna()
        & df[LON_COL].notna()
    )
    pending = df[mask].copy()
    if pending.empty:
        return

    # Pre-bucket rows by chunk + grid cell using vectorized ops
    pending_index = pending.index.to_numpy()
    times_raw = pending[TIME_COL].to_numpy()
    if hasattr(times_raw, "dtype") and str(times_raw.dtype).startswith("datetime64"):
        times = times_raw.astype("datetime64[ns]").astype("int64") / 1e9
    else:
        times = times_raw.astype(float)
        if times.size and float(times[0]) > 1e12:
            times = times / 1e9
    lats = pending[LAT_COL].to_numpy(dtype=float)
    lons = pending[LON_COL].to_numpy(dtype=float)

    target_arrays: dict[str, np.ndarray] = {
        col: df[col].to_numpy(copy=True) for col in target_cols
    }

    # Chunk lookup arrays (ascending by start time for searchsorted)
    asc_ranges = sorted(chunk_index.ranges, key=lambda r: r.start)
    asc_starts = np.array([r.start for r in asc_ranges], dtype=float)
    asc_chunk_nums = np.array([r.chunk_num for r in asc_ranges], dtype=int)
    asc_time_lens = np.array([r.time_len for r in asc_ranges], dtype=int)

    chunk_lookup = np.searchsorted(asc_starts, times, side="right") - 1
    chunk_lookup = np.clip(chunk_lookup, 0, len(asc_ranges) - 1)

    chunk_nums = asc_chunk_nums[chunk_lookup]
    chunk_starts = asc_starts[chunk_lookup]
    chunk_time_lens = asc_time_lens[chunk_lookup]

    time_indices = np.rint((times - chunk_starts) / chunk_index.resolution).astype(int)
    time_indices = np.clip(time_indices, 0, chunk_time_lens - 1)

    # Use axis info for correct grid mapping
    sample_uri = f"s3://openmeteo/data/{MODEL}/{VARIABLE}/chunk_{chunk_index.ranges[0].chunk_num}.om"
    reader = _open_reader(sample_uri)
    try:
        axis_info = _axis_info(reader, MODEL, VARIABLE)
    finally:
        reader.close()

    lons_adj = np.where((axis_info.lon_360) & (lons < 0), lons + 360.0, lons)
    lat_indices = np.rint((lats - axis_info.lat_start) / axis_info.lat_step).astype(int)
    lon_indices = np.rint((lons_adj - axis_info.lon_start) / axis_info.lon_step).astype(int)
    lat_indices = np.clip(lat_indices, 0, axis_info.ny - 1)
    lon_indices = np.clip(lon_indices, 0, axis_info.nx - 1)

    # Sort so identical groups are contiguous
    order = np.lexsort((lon_indices, lat_indices, chunk_nums))
    chunk_nums = chunk_nums[order]
    lat_indices = lat_indices[order]
    lon_indices = lon_indices[order]
    time_indices = time_indices[order]
    row_positions = pending_index[order]

    # Identify group boundaries
    if len(order) == 0:
        return
    change = np.empty_like(chunk_nums, dtype=bool)
    change[0] = True
    change[1:] = (
        (chunk_nums[1:] != chunk_nums[:-1])
        | (lat_indices[1:] != lat_indices[:-1])
        | (lon_indices[1:] != lon_indices[:-1])
    )
    group_starts = np.flatnonzero(change)
    group_ends = np.append(group_starts[1:], len(order))

    print(f"[taxon {taxon['taxon_key']}] groups: {len(group_starts)} rows: {len(pending)}")
    reader_cache: dict[int, OmFileReader] = {}

    total_groups = len(group_starts)
    total_rows = len(pending)
    groups_done = 0
    rows_done = 0
    start = time.time()
    last_progress = start

    steps = _window_steps(chunk_index.resolution)

    for start_idx, end_idx in zip(group_starts, group_ends):
        chunk_num = int(chunk_nums[start_idx])
        lat_idx = int(lat_indices[start_idx])
        lon_idx = int(lon_indices[start_idx])
        row_slice = slice(start_idx, end_idx)
        row_positions_group = row_positions[row_slice]
        time_indices_group = time_indices[row_slice]

        chunk = chunk_index.range_by_chunk[chunk_num]
        if chunk.source == "year":
            data_uri = f"s3://openmeteo/data/{MODEL}/{VARIABLE}/year_{chunk_num}.om"
            if chunk_num >= YEAR_PREFETCH_CUTOFF:
                local_chunk = _prefetch_chunk(data_uri)
                reader = _open_reader(local_chunk.as_posix(), block_size=YEAR_REMOTE_BLOCK_SIZE)
            else:
                reader = _open_reader(data_uri, block_size=YEAR_REMOTE_BLOCK_SIZE)
        else:
            data_uri = f"s3://openmeteo/data/{MODEL}/{VARIABLE}/chunk_{chunk_num}.om"
            local_chunk = _prefetch_chunk(data_uri)
            reader = _open_reader(local_chunk.as_posix())
        rows_in_group = len(row_positions_group)
        print(f"  [chunk {chunk_num}] source={chunk.source} cells: ({lat_idx},{lon_idx}) rows: {rows_in_group}")
        if chunk_num not in reader_cache:
            reader_cache[chunk_num] = reader
        else:
            # Reuse cached reader if already opened
            reader = reader_cache[chunk_num]
        ny, nx, time_len = reader.shape
        if lat_idx >= ny or lon_idx >= nx:
            continue

        # Read full time series for this cell once
        try:
            series = np.array(reader[lat_idx, lon_idx, :], dtype=float)
        except Exception:
            # Fallback to legacy loop if slicing is unsupported
            series = np.array([reader[lat_idx, lon_idx, i] for i in range(time_len)], dtype=float)

        # Ensure time indices fit actual series length
        if series.size:
            time_indices_group = np.clip(time_indices_group, 0, series.size - 1)

        window_sums, _ = _window_stats_batch(series, time_indices_group, steps)
        for hours, values in window_sums.items():
            col = f"{VARIABLE}_sum_{hours}h"
            target_arrays[col][row_positions_group] = values

        groups_done += 1
        rows_done += rows_in_group
        now = time.time()
        elapsed = now - start
        if groups_done == 1 or groups_done == total_groups or (now - last_progress) >= 5:
            per_group = elapsed / max(1, groups_done)
            per_row = elapsed / max(1, rows_done)
            remaining_groups = total_groups - groups_done
            eta = remaining_groups * per_group
            print(
                f"    [progress] groups {groups_done}/{total_groups} "
                f"rows {rows_done}/{total_rows} "
                f"avg {per_group:.3f}s/group {per_row:.5f}s/row "
                f"eta {eta/60:.1f}m"
            )
            last_progress = now

    for reader in reader_cache.values():
        reader.close()

    # Write back filled columns
    for col, arr in target_arrays.items():
        df[col] = arr

    updated = pa.Table.from_pandas(df, preserve_index=False)
    _atomic_write(path, updated)


def _run_variable(model: str, variable: str, window_hours: tuple[int, ...]) -> None:
    """Run enrichment for a single variable/model combo using a global worklist."""

    if not window_hours:
        print(f"[skip] variable={variable} model={model} (no windows configured)")
        return

    # Special case: weather_code_simple is derived locally from existing 1h aggregates
    if variable == "weather_code_simple":
        _run_weather_code_simple(model)
        return

    global MODEL, VARIABLE, WINDOW_HOURS, CACHE_DIR, WORKLIST_PATH, OCC_INDEX_PATH
    MODEL = model
    VARIABLE = variable
    WINDOW_HOURS = window_hours

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    WORKLIST_PATH = Path(CACHE_DIR) / f"worklist_{VARIABLE}_{MODEL}.parquet"
    if WORKLIST_PATH.exists():
        WORKLIST_PATH.unlink()

    if OCC_INDEX_PATH is None:
        OCC_INDEX_PATH = Path(CACHE_DIR) / "occ_index.parquet"
    # Always rebuild for clarity; Pediocactus subset is small and avoids stale caches.
    if OCC_INDEX_PATH.exists():
        OCC_INDEX_PATH.unlink()
    occ_rows = _build_occ_index(OCC_INDEX_PATH)
    print(f"[occ_index] built rows={occ_rows} file={OCC_INDEX_PATH} root={CONFIG.root_taxon_id}")

    print(f"[init] model={MODEL} variable={VARIABLE} windows={WINDOW_HOURS} cache_dir={CACHE_DIR}")
    try:
        chunk_index = _build_chunk_index(MODEL, VARIABLE)
    except FileNotFoundError:
        print(f"[skip] variable={VARIABLE} model={MODEL} missing on s3; skipping")
        return
    print(f"[init] chunks={len(chunk_index.ranges)} latest_end={datetime.fromtimestamp(chunk_index.latest_end_time, tz=timezone.utc).isoformat()}")

    batch_rows = CONFIG.temporal_worklist_batch_rows
    agg_mode = CONFIG.temporal_agg_by_variable.get(variable, "avg")
    for occ_batch in _iter_occ_index_batches(OCC_INDEX_PATH, batch_rows):
        _apply_pre_cutoff_nans(occ_batch, variable, window_hours, agg_mode)
        worklist_table = _build_worklist_from_index(chunk_index, occ_batch, model, variable)
        if worklist_table.num_rows == 0:
            continue
        _process_worklist(chunk_index, worklist_table, model, variable, window_hours, agg_mode)
    WORKLIST_PATH.unlink(missing_ok=True)


def _run_weather_code_simple(model: str) -> None:
    """Derive weather_code_simple locally using 1h aggregates."""
    # We need 1h sums already computed for precipitation, snowfall_water_equivalent, cloud_cover
    model_dt_seconds = 3600.0  # 1h timestep assumption for simple code
    target_col = "weather_code_simple"

    root = CONFIG.root_taxon_id
    root_record = taxa_navigation.get_taxon_by_id(root)
    if root_record is None:
        raise RuntimeError(f"Unknown root taxon {root}")

    for node in taxa_navigation.iter_descendants(root_record, include_self=True):
        path = Path(node["path"]) / CONFIG.occurrence_parquet_filename
        if not path.exists():
            continue
        table = pq.read_table(path).combine_chunks()
        df = table.to_pandas()
        if df.empty:
            continue

        if target_col not in df.columns:
            df[target_col] = np.nan

        cc = df.get("cloud_cover_avg_1h")
        pr = df.get("precipitation_sum_1h")
        sw = df.get("snowfall_water_equivalent_sum_1h")
        if cc is None or pr is None or sw is None:
            print(f"[weather_code] missing source cols in {path}, skipping")
            continue

        values = []
        for a, b, c in zip(cc.to_numpy(), pr.to_numpy(), sw.to_numpy()):
            values.append(_weather_code_simple_row(a, b, c, model_dt_seconds))
        df[target_col] = values

        updated = pa.Table.from_pandas(df, preserve_index=False)
        _atomic_write(path, updated)

    print(f"[weather_code] derived {target_col} for model={model}")


def _run_vapor_pressure_deficit() -> None:
    """Derive vapor_pressure_deficit from temperature_2m and dew_point_2m aggregates."""
    root = CONFIG.root_taxon_id
    root_record = taxa_navigation.get_taxon_by_id(root)
    if root_record is None:
        raise RuntimeError(f"Unknown root taxon {root}")

    temporal_registry = gis_lookup.load_temporal_registry()
    registry_windows = temporal_registry.get("windows", [])
    windows_by_var = {
        str(entry.get("id")): tuple(entry.get("windows") or registry_windows or ())
        for entry in temporal_registry.get("layers", [])
        if entry.get("id")
    }
    windows = windows_by_var.get("vapor_pressure_deficit", tuple(registry_windows))
    if not windows:
        print("[vpd] no windows configured; skipping")
        return

    for node in taxa_navigation.iter_descendants(root_record, include_self=True):
        path = Path(node["path"]) / CONFIG.occurrence_parquet_filename
        if not path.exists():
            continue
        table = pq.read_table(path).combine_chunks()
        df = table.to_pandas()
        if df.empty:
            continue

        updated_any = False
        for hours in windows:
            t_col = f"temperature_2m_avg_{hours}h"
            td_col = f"dew_point_2m_avg_{hours}h"
            vpd_col = f"vapor_pressure_deficit_avg_{hours}h"
            if t_col not in df.columns or td_col not in df.columns:
                continue

            t = df[t_col].to_numpy(dtype=float)
            td = df[td_col].to_numpy(dtype=float)
            # Saturation vapor pressure (kPa)
            es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
            ea = 0.6108 * np.exp((17.27 * td) / (td + 237.3))
            vpd = es - ea
            vpd[~np.isfinite(vpd)] = np.nan
            df[vpd_col] = vpd
            updated_any = True

        if updated_any:
            updated = pa.Table.from_pandas(df, preserve_index=False)
            _atomic_write(path, updated)

    print("[vpd] derived vapor_pressure_deficit")


def main() -> None:
    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"[signal] received {signum}")
        _log_status("[signal]")
        PROGRESS_STOP.set()
        sys.stdout.flush()
        sys.exit(1)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    cache_dir = DEFAULT_CACHE_DIR
    global CACHE_DIR
    CACHE_DIR = cache_dir
    # Always start clean (no reuse between runs)
    try:
        cache_root = Path(CACHE_DIR)
        if cache_root.exists():
            for path in cache_root.iterdir():
                try:
                    if path.is_file():
                        path.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    models_by_var = CONFIG.temporal_models_by_variable
    model_preference = CONFIG.temporal_model_preference
    temporal_registry = gis_lookup.load_temporal_registry()
    registry_windows = temporal_registry.get("windows", [])
    registry_vars = temporal_registry.get("layers", [])
    windows_by_var = {
        str(entry.get("id")): tuple(entry.get("windows") or registry_windows or ())
        for entry in registry_vars
        if entry.get("id")
    }
    agg_by_var = {
        str(entry.get("id")): str(entry.get("agg"))
        for entry in registry_vars
        if entry.get("id") and entry.get("agg")
    }
    derived_by_var = {
        str(entry.get("id")): bool(entry.get("derived"))
        for entry in registry_vars
        if entry.get("id")
    }
    default_windows = tuple(registry_windows or CONFIG.temporal_window_hours_default)

    # Build occurrence index once for the chosen root
    if OCC_INDEX_PATH is None:
        occ_index_path = Path(CACHE_DIR) / "occ_index.parquet"
    else:
        occ_index_path = OCC_INDEX_PATH
    if occ_index_path.exists():
        occ_index_path.unlink()
    occ_rows = _build_occ_index(occ_index_path)
    print(f"[occ_index] built rows={occ_rows} file={occ_index_path} root={CONFIG.root_taxon_id}")

    # Phase 1: build processing plan per variable
    work_plan: list[tuple[str, str, tuple[int, ...], str, ChunkIndex]] = []
    global PROGRESS_TOTAL_ROWS, PROGRESS_DONE_ROWS
    PROGRESS_DONE_ROWS = 0
    PROGRESS_STOP.clear()
    total_rows_all = 0
    root = CONFIG.root_taxon_id
    root_record = taxa_navigation.get_taxon_by_id(root)
    if root_record is None:
        raise RuntimeError(f"Unknown root taxon {root}")
    # Collect all occurrence parquets under root for skip checks
    occurrence_paths: list[Path] = []
    for node in taxa_navigation.iter_descendants(root_record, include_self=True):
        path = Path(node["path"]) / CONFIG.occurrence_parquet_filename
        if path.exists():
            occurrence_paths.append(path)

    def _all_columns_present(target_cols: list[str]) -> bool:
        if not occurrence_paths:
            return False
        for path in occurrence_paths:
            try:
                cols = pq.read_schema(path).names
            except Exception:
                return False
            if not all(col in cols for col in target_cols):
                return False
        return True

    if registry_vars:
        temporal_vars = [str(entry.get("id")) for entry in registry_vars if entry.get("id")]
    else:
        temporal_vars = list(models_by_var.keys())

    for variable in temporal_vars:
        if derived_by_var.get(variable) and variable != "vapor_pressure_deficit":
            window_hours = windows_by_var.get(variable, default_windows)
            if not window_hours:
                print(f"[skip] variable={variable} (no windows configured)")
                continue
            work_plan.append((variable, "derived", window_hours, "derived", None))
            continue

        models = models_by_var.get(variable, ())
        if not models:
            print(f"[skip] variable={variable} (no models configured)")
            continue
        window_hours = windows_by_var.get(variable, default_windows)
        if not window_hours:
            print(f"[skip] variable={variable} (no windows configured)")
            continue
        if variable == "weather_code_simple":
            work_plan.append((variable, "derived", window_hours, "snapshot", None))  # handled later
            continue
        chosen_model = next((m for m in model_preference if m in models), models[0])
        try:
            chunk_var = "temperature_2m" if variable == "vapor_pressure_deficit" else variable
            chunk_index = _build_chunk_index(chosen_model, chunk_var)
        except FileNotFoundError:
            print(f"[skip] variable={variable} model={chosen_model} missing on s3; skipping")
            continue
        agg_mode = agg_by_var.get(variable, CONFIG.temporal_agg_by_variable.get(variable, "avg"))
        target_cols = [f"{variable}_{agg_mode}_{h}h" for h in window_hours]
        overwrite_all = CONFIG.temporal_overwrite_columns is None
        overwrite_cols = set(CONFIG.temporal_overwrite_columns or ())
        # Skip only if all descendant occurrence parquets already have these columns
        if _all_columns_present(target_cols) and not overwrite_all and not any(col in overwrite_cols for col in target_cols):
            print(f"[skip] variable={variable} (columns already present for all taxa)")
            continue
        work_plan.append((variable, chosen_model, window_hours, agg_mode, chunk_index))

    # Approximate total rows for progress
    total_rows_all = occ_rows * sum(1 for v, *_ in work_plan if v != "weather_code_simple")
    PROGRESS_TOTAL_ROWS = total_rows_all

    # Phase 2: process worklists (parallel across variables)
    def _process_task(variable: str, model: str, window_hours: tuple[int, ...], agg_mode: str, chunk_index: ChunkIndex) -> None:
        try:
            if variable == "weather_code_simple":
                # weather code is derived after all other variables complete
                return
            if variable == "vapor_pressure_deficit":
                batch_rows = CONFIG.temporal_worklist_batch_rows
                print(f"[process] variable={variable} model={model} agg=avg batch_rows={batch_rows}")
                for occ_batch in _iter_occ_index_batches(occ_index_path, batch_rows):
                    if occ_batch.num_rows == 0:
                        continue
                    _apply_pre_cutoff_nans(occ_batch, variable, window_hours, "avg")
                    worklist_table = _build_worklist_from_index(chunk_index, occ_batch, model, "temperature_2m")
                    if worklist_table.num_rows == 0:
                        continue
                    _process_vpd_worklist(chunk_index, worklist_table, model, window_hours)
                return
            if model == "derived" or agg_mode == "derived":
                return
            batch_rows = CONFIG.temporal_worklist_batch_rows
            print(f"[process] variable={variable} model={model} agg={agg_mode} batch_rows={batch_rows}")
            for occ_batch in _iter_occ_index_batches(occ_index_path, batch_rows):
                if occ_batch.num_rows == 0:
                    continue
                _apply_pre_cutoff_nans(occ_batch, variable, window_hours, agg_mode)
                worklist_table = _build_worklist_from_index(chunk_index, occ_batch, model, variable)
                if worklist_table.num_rows == 0:
                    continue
                _process_worklist(chunk_index, worklist_table, model, variable, window_hours, agg_mode)
        except Exception:
            print(f"[error] variable={variable} model={model} agg={agg_mode}")
            traceback.print_exc()
            _log_status("[error]")
            raise

    def _print_global_progress(start_time: float) -> None:
        if PROGRESS_TOTAL_ROWS == 0:
            return
        with PROGRESS_LOCK:
            done = PROGRESS_DONE_ROWS
        elapsed = time.time() - start_time
        throughput = done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, PROGRESS_TOTAL_ROWS - done)
        eta_seconds = remaining / throughput if throughput > 0 else float("inf")
        eta_text = "--" if not math.isfinite(eta_seconds) else f"{int(eta_seconds//60)}m{int(eta_seconds%60):02d}s"
        rss = _read_rss_mb() if getattr(CONFIG, "temporal_log_memory", False) else None
        rss_text = f" rss={rss:.1f}MB" if rss is not None else ""
        print(f"[overall] rows {done}/{PROGRESS_TOTAL_ROWS} avg {throughput:.1f} rows/s eta {eta_text}{rss_text}")

    def _global_progress_printer() -> None:
        start_time = time.time()
        while not PROGRESS_STOP.is_set():
            time.sleep(5)
            _print_global_progress(start_time)
        _print_global_progress(start_time)

    if work_plan:
        # Allow more variables in flight; inner pools still capped to avoid runaway threads
        max_workers = 1
        progress_thread = threading.Thread(target=_global_progress_printer, daemon=True)
        progress_thread.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for variable, model, window_hours, agg_mode, chunk_index in work_plan:
                futures.append(
                    executor.submit(_process_task, variable, model, window_hours, agg_mode, chunk_index)
                )
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    _log_status("[error]")
                    raise
        PROGRESS_STOP.set()
        progress_thread.join()

    # Run derived variables after all other variables complete
    if any(v == "weather_code_simple" for v, *_ in work_plan):
        _run_weather_code_simple("derived")
    # VPD is derived during processing using hourly temperature/dew point.

    # Cleanup temp cache artifacts
    try:
        if occ_index_path.exists():
            occ_index_path.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[fatal] enrich_temporal failed")
        traceback.print_exc()
        _log_status("[fatal]")
        raise
