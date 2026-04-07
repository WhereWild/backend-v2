"""Live weather tile rendering from Open-Meteo ncep_gfs013 spatial data on S3."""
from __future__ import annotations

import io
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import fsspec
import numpy as np
from omfiles import OmFileReader
from PIL import Image

from rasterio.crs import CRS
from rasterio.transform import from_bounds as rasterio_from_bounds
from rasterio.warp import reproject, Resampling as RasterioResampling

from util.tiles import TileSpec, tile_bounds_mercator, WEB_MERCATOR

log = logging.getLogger(__name__)

S3_SPATIAL = "s3://openmeteo/data_spatial"

_TEMPORAL_RASTER_DIR = Path(__file__).parent.parent / "data" / "gis" / "temporal" / "rasters"
_TEMPORAL_WINDOW_LABELS: dict[int, str] = {
    1: "1h", 8: "8h", 24: "24h", 72: "3d", 168: "7d", 720: "30d", 2160: "90d",
}

# Per-model grid configuration
MODEL_CONFIGS: dict[str, dict] = {
    "ncep_gfs013": {
        "lat_min": -89.912125, "lat_max":  89.912125,
        "lon_min": -180.0,     "lon_max":  179.88281,
        "flipud": True,   # stored south-up
    },
}

# --- Color stops ---
_BLUE_RED = np.array([
    [  0,   0, 200],
    [  0, 150, 255],
    [255, 255, 100],
    [255, 120,   0],
    [200,   0,   0],
], dtype=np.float32)

_CLOUD = np.array([
    [ 30,  30,  60],
    [100, 120, 160],
    [180, 190, 210],
    [230, 235, 245],
    [255, 255, 255],
], dtype=np.float32)

_PRECIP = np.array([
    [240, 248, 255],
    [100, 180, 255],
    [ 30, 100, 220],
    [  0,  50, 160],
    [  0,  20,  80],
], dtype=np.float32)

_SNOWFALL = np.array([
    [230, 240, 255],
    [160, 200, 255],
    [ 80, 140, 255],
    [ 20,  60, 200],
    [  0,  10, 100],
], dtype=np.float32)

_SOIL_MOIST = np.array([
    [210, 170, 100],
    [160, 130,  60],
    [ 80, 160,  80],
    [ 30, 120, 180],
    [  0,  60, 160],
], dtype=np.float32)

_VPD = np.array([
    [  0, 120, 200],  # low VPD: blue
    [ 80, 200, 120],
    [255, 240,  80],
    [255, 140,   0],
    [200,   0,   0],  # high VPD: red
], dtype=np.float32)

# Categorical color lookup for weather_code_simple (RGBA tuples)
_WEATHER_CODE_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (255, 240,  80),  # clear
    1:  (220, 230, 120),  # mainly clear
    2:  (180, 190, 180),  # partly cloudy
    3:  (120, 120, 130),  # overcast
    51: (160, 210, 255),  # light drizzle
    53: (100, 170, 255),  # moderate drizzle
    55: ( 60, 130, 240),  # dense drizzle
    61: ( 30, 100, 220),  # light rain
    63: (  0,  60, 180),  # moderate rain
    65: (  0,  20, 120),  # heavy rain
    71: (220, 245, 255),  # slight snow — near white/icy
    73: (160, 220, 240),  # moderate snow — cyan
    75: ( 80, 170, 210),  # heavy snow — teal
}

# "derived": True  → computed from other vars, not fetched from S3
_EXTRA_RAW = {"relative_humidity_2m"}  # fetched but not rendered directly

LIVE_WEATHER_VARIABLES: dict[str, dict] = {
    # --- direct ---
    "temperature_2m":            {"model": "ncep_gfs013", "lo": -50.0, "hi":  50.0, "stops": _BLUE_RED},
    "cloud_cover":               {"model": "ncep_gfs013", "lo":   0.0, "hi": 100.0, "stops": _CLOUD},
    "precipitation":             {"model": "ncep_gfs013", "lo":   0.0, "hi":   5.0, "stops": _PRECIP},
    "snowfall_water_equivalent": {"model": "ncep_gfs013", "lo":   0.0, "hi":  10.0, "stops": _SNOWFALL},
    "soil_moisture_0_to_7cm":    {"model": "ncep_gfs013", "lo":   0.0, "hi":   0.5, "stops": _SOIL_MOIST, "fetch_as": "soil_moisture_0_to_10cm"},
    "soil_temperature_0_to_7cm": {"model": "ncep_gfs013", "lo": -10.0, "hi":  40.0, "stops": _BLUE_RED,   "fetch_as": "soil_temperature_0_to_10cm"},
    # --- derived from temperature_2m + relative_humidity_2m ---
    "dew_point_2m":           {"model": "ncep_gfs013", "lo": -40.0, "hi":  35.0, "stops": _BLUE_RED, "derived": True},
    "vapor_pressure_deficit":  {"model": "ncep_gfs013", "lo":   0.0, "hi":   5.0, "stops": _VPD,     "derived": True},
    # --- derived from cloud_cover + precipitation + snowfall_water_equivalent ---
    "weather_code_simple":    {"model": "ncep_gfs013", "categorical": True, "derived": True},
}

# --- Disk cache ---
_DISK_CACHE_DIR = Path(__file__).parent.parent / "data" / "gis" / "temporal" / "cache"


def _disk_path(model: str, ref_time: str, var_id: str, forecast_hours: int = 0) -> Path:
    safe = ref_time.replace(":", "-").replace(" ", "_")
    suffix = f"__f{forecast_hours:03d}h" if forecast_hours else ""
    return _DISK_CACHE_DIR / f"{model}__{safe}__{var_id}{suffix}.npy"


def _save_to_disk(model: str, ref_time: str, var_id: str, arr: np.ndarray,
                  forecast_hours: int = 0) -> None:
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_disk_path(model, ref_time, var_id, forecast_hours), arr)


def _load_from_disk(model: str, ref_time: str, var_id: str,
                    forecast_hours: int = 0) -> np.ndarray | None:
    path = _disk_path(model, ref_time, var_id, forecast_hours)
    if path.exists():
        return np.load(path)
    return None


# --- Memory cache ---
_cache: dict[str, np.ndarray] = {}                       # var_id → arr  (forecast_hours=0)
_forecast_cache: dict[int, dict[str, np.ndarray]] = {}   # forecast_hours → var_id → arr
_cache_ref_times: dict[str, str] = {}                    # model → ref_time (current)
_forecast_ref_times: dict[int, dict[str, str]] = {}      # forecast_hours → model → ref_time
_cache_mtimes: dict[str, float] = {}                     # "fh:var_id" → mtime of current_*.npy
_cache_lock = threading.Lock()
_forecast_load_locks: dict[int, threading.Lock] = {}     # one lock per forecast offset
_forecast_load_locks_lock = threading.Lock()             # protects the dict above

FORECAST_HOURS_OPTIONS = [1, 8, 24, 72, 168]  # supported forecast offsets


def _get_forecast_lock(forecast_hours: int) -> threading.Lock:
    with _forecast_load_locks_lock:
        if forecast_hours not in _forecast_load_locks:
            _forecast_load_locks[forecast_hours] = threading.Lock()
        return _forecast_load_locks[forecast_hours]


def _s3_path(model: str, ref: datetime, valid: datetime) -> str:
    run_dir = f"{ref.year:04d}/{ref.month:02d}/{ref.day:02d}/{ref.hour:02d}{ref.minute:02d}Z"
    fname = valid.strftime("%Y-%m-%dT%H%M") + ".om"
    return f"{S3_SPATIAL}/{model}/{run_dir}/{fname}"


def _populate_cache_from_hits(model: str, ref_time: str, disk_hits: dict[str, np.ndarray],
                               forecast_hours: int) -> None:
    """Compute derived variables and store everything in the in-memory cache."""
    cc   = disk_hits.get("cloud_cover")
    prec = disk_hits.get("precipitation")
    swe  = disk_hits.get("snowfall_water_equivalent")
    if cc is not None and prec is not None and swe is not None:
        snow_rate = swe / 10.0
        rain_rate = prec
        code = np.full(cc.shape, 3, dtype=np.float32)
        code[cc < 80] = 2
        code[cc < 50] = 1
        code[cc < 20] = 0
        code[rain_rate >= 0.01] = 51
        code[rain_rate >= 0.5] = 53
        code[rain_rate >= 1.0] = 55
        code[rain_rate >= 1.3] = 61
        code[rain_rate >= 2.5] = 63
        code[rain_rate >= 7.6] = 65
        code[snow_rate >= 0.01] = 71
        code[snow_rate >= 0.2] = 73
        code[snow_rate >= 0.8]  = 75
        disk_hits["weather_code_simple"] = code

    T  = disk_hits.get("temperature_2m")
    RH = disk_hits.get("relative_humidity_2m")
    if T is not None and RH is not None:
        RH_c  = np.clip(RH, 1.0, 100.0)
        gamma = np.log(RH_c / 100.0) + 17.625 * T / (243.04 + T)
        disk_hits["dew_point_2m"] = (243.04 * gamma / (17.625 - gamma)).astype(np.float32)
        es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
        disk_hits["vapor_pressure_deficit"] = (es * (1.0 - RH_c / 100.0)).astype(np.float32)

    with _cache_lock:
        target = _cache if forecast_hours == 0 else _forecast_cache.setdefault(forecast_hours, {})
        for var_id, cfg in LIVE_WEATHER_VARIABLES.items():
            if cfg["model"] != model:
                continue
            arr = disk_hits.get(cfg.get("fetch_as", var_id))
            if arr is not None:
                target[var_id] = arr
                _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                suffix = f"__f{forecast_hours:03d}h" if forecast_hours else ""
                p = _DISK_CACHE_DIR / f"current_{model}_{var_id}{suffix}.npy"
                np.save(p, arr)
                _cache_mtimes[f"{forecast_hours}:{var_id}"] = p.stat().st_mtime
        if forecast_hours == 0:
            _cache_ref_times[model] = ref_time
        else:
            _forecast_ref_times.setdefault(forecast_hours, {})[model] = ref_time


def _load_model(model: str, forecast_hours: int = 0) -> None:
    """Fetch all variables for one model at a given forecast offset (0 = current)."""
    fs = fsspec.filesystem("s3", anon=True)
    with fs.open(f"{S3_SPATIAL}/{model}/latest.json") as f:
        meta = json.load(f)
    ref_time = meta["reference_time"]

    with _cache_lock:
        if forecast_hours == 0:
            current_ref = _cache_ref_times.get(model)
            vars_for_model = {vid for vid, cfg in LIVE_WEATHER_VARIABLES.items() if cfg["model"] == model}
            already_done = current_ref == ref_time and all(v in _cache for v in vars_for_model)
        else:
            current_ref = _forecast_ref_times.get(forecast_hours, {}).get(model)
            fc = _forecast_cache.get(forecast_hours, {})
            vars_for_model = {vid for vid, cfg in LIVE_WEATHER_VARIABLES.items() if cfg["model"] == model}
            already_done = current_ref == ref_time and all(v in fc for v in vars_for_model)
    if already_done:
        print(f"[weather_tiles] {model}+{forecast_hours}h already current ({ref_time})", flush=True)
        return

    raw_needs = (
        {cfg.get("fetch_as", var_id) for var_id, cfg in LIVE_WEATHER_VARIABLES.items()
         if cfg["model"] == model and not cfg.get("derived")}
        | _EXTRA_RAW
    )

    # Pick the valid_time closest to ref + forecast_hours
    ref_dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
    target_dt = ref_dt + timedelta(hours=forecast_hours) if forecast_hours else None
    valid_times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in meta["valid_times"]]
    valid = valid_times[1] if target_dt is None else min(
        valid_times, key=lambda t: abs((t - target_dt).total_seconds()))

    tag = f"+{forecast_hours}h" if forecast_hours else "current"
    print(f"[weather_tiles] {model} {tag}: valid={valid.strftime('%Y-%m-%dT%HZ')}", flush=True)

    # Try disk cache first
    disk_hits: dict[str, np.ndarray] = {}
    for var_id in raw_needs:
        arr = _load_from_disk(model, ref_time, var_id, forecast_hours)
        if arr is not None:
            disk_hits[var_id] = arr
            print(f"[weather_tiles] {model}/{var_id}+{forecast_hours}h from disk", flush=True)

    # Fetch missing vars from S3
    need = [v for v in raw_needs if v not in disk_hits]
    if need:
        s3_path = _s3_path(model, ref_dt, valid)
        safe_valid = valid.strftime("%Y%m%dT%HZ")
        local_om = _DISK_CACHE_DIR / f"{model}__{safe_valid}.om"
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not local_om.exists():
            print(f"[weather_tiles] S3 fetch {tag}: {need} from {model}", flush=True)
            fs_anon = fsspec.filesystem("s3", anon=True)
            fs_anon.get(s3_path, str(local_om))
        else:
            print(f"[weather_tiles] S3 om file from disk {tag} from {model}", flush=True)
        root = OmFileReader(str(local_om))
        available = {root.get_child_by_index(i).name for i in range(root.num_children)}
        for var_id in need:
            if var_id not in available:
                print(f"[weather_tiles] {var_id} not in {model}, skipping", flush=True)
                continue
            node = root.get_child_by_name(var_id)
            ny, nx = node.shape
            arr = node.read_array((slice(0, ny), slice(0, nx)))
            _save_to_disk(model, ref_time, var_id, arr, forecast_hours)
            disk_hits[var_id] = arr
            print(f"[weather_tiles] {model}/{var_id}+{forecast_hours}h  "
                  f"range=[{float(arr.min()):.1f}, {float(arr.max()):.1f}]", flush=True)

    _populate_cache_from_hits(model, ref_time, disk_hits, forecast_hours)


def load_cache() -> None:
    """Populate memory cache for all models (current snapshot)."""
    models = {cfg["model"] for cfg in LIVE_WEATHER_VARIABLES.values()}
    for model in sorted(models):
        try:
            _load_model(model, forecast_hours=0)
        except Exception as exc:
            print(f"[weather_tiles] ERROR loading {model}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    print(f"[weather_tiles] cache ready. vars={list(_cache.keys())}", flush=True)


def preload_all_forecasts() -> None:
    """Pre-fetch and disk-cache all forecast offsets. Safe to call from build scripts."""
    print("[weather_tiles] preloading current snapshot ...", flush=True)
    load_cache()
    for hours in FORECAST_HOURS_OPTIONS:
        print(f"[weather_tiles] preloading +{hours}h forecast ...", flush=True)
        ensure_forecast_loaded(hours)
    print("[weather_tiles] all forecast offsets cached to disk.", flush=True)


def load_from_disk() -> None:
    """Populate in-memory cache from disk without any S3 access.

    Scans the disk cache directory for the newest reference time per model and
    loads all variables (current + all forecast offsets) into memory.
    Called on API startup so the API never needs to touch S3.
    """
    if not _DISK_CACHE_DIR.exists():
        print("[weather_tiles] disk cache dir not found — run build script first", flush=True)
        return

    raw_needs_by_model: dict[str, set[str]] = {}
    for var_id, cfg in LIVE_WEATHER_VARIABLES.items():
        model = cfg["model"]
        raw_needs_by_model.setdefault(model, set())
        if not cfg.get("derived"):
            raw_needs_by_model[model].add(cfg.get("fetch_as", var_id))
    for model in raw_needs_by_model:
        raw_needs_by_model[model] |= _EXTRA_RAW

    for model, raw_needs in sorted(raw_needs_by_model.items()):
        # Find the newest ref_time by scanning filenames (format: model__safe_ref__var.npy)
        files = list(_DISK_CACHE_DIR.glob(f"{model}__*__*.npy"))
        if not files:
            print(f"[weather_tiles] no disk cache for {model} — run build script first", flush=True)
            continue
        ref_times: set[str] = set()
        for f in files:
            parts = f.stem.split("__")
            if len(parts) >= 2:
                ref_times.add(parts[1])
        newest_ref = max(ref_times)  # ISO timestamps sort lexicographically

        for forecast_hours in [0] + FORECAST_HOURS_OPTIONS:
            disk_hits: dict[str, np.ndarray] = {}
            for var_id in raw_needs:
                arr = _load_from_disk(model, newest_ref, var_id, forecast_hours)
                if arr is not None:
                    disk_hits[var_id] = arr
            if disk_hits:
                _populate_cache_from_hits(model, newest_ref, disk_hits, forecast_hours)
            else:
                tag = f"+{forecast_hours}h" if forecast_hours else "current"
                print(f"[weather_tiles] {model} {tag} not on disk — run build script", flush=True)


def cleanup_weather_disk_cache(now_ts: float, keep_hours: int = 24) -> None:
    """Delete weather disk cache files older than keep_hours. Keeps the current run's files."""
    if not _DISK_CACHE_DIR.exists():
        return
    cutoff = now_ts - keep_hours * 3600
    removed = 0
    for path in list(_DISK_CACHE_DIR.glob("*.npy")) + list(_DISK_CACHE_DIR.glob("*.om")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"[weather_tiles] cleaned {removed} stale weather cache file(s)", flush=True)


def ensure_forecast_loaded(forecast_hours: int) -> None:
    """Lazily load forecast cache for a given offset. Serialized per offset — callers wait."""
    with _get_forecast_lock(forecast_hours):
        # Re-check inside the lock — another thread may have loaded while we waited
        with _cache_lock:
            fc = _forecast_cache.get(forecast_hours, {})
            vars_needed = {vid for vid in LIVE_WEATHER_VARIABLES}
            already = all(v in fc for v in vars_needed)
        if already:
            return
        models = {cfg["model"] for cfg in LIVE_WEATHER_VARIABLES.values()}
        for model in sorted(models):
            try:
                _load_model(model, forecast_hours=forecast_hours)
            except Exception as exc:
                print(f"[weather_tiles] ERROR loading {model}+{forecast_hours}h: {exc}", flush=True)


def _colorize(arr: np.ndarray, stops: np.ndarray, lo: float, hi: float) -> np.ndarray:
    norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, stops.shape[0], dtype=np.float32)
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.interp(norm, positions, stops[:, 0]).astype(np.uint8)
    rgba[..., 1] = np.interp(norm, positions, stops[:, 1]).astype(np.uint8)
    rgba[..., 2] = np.interp(norm, positions, stops[:, 2]).astype(np.uint8)
    rgba[..., 3] = 210
    return rgba


def render_weather_tile_bytes(variable_id: str, z: int, x: int, y: int,
                               tile_size: int = 256,
                               forecast_hours: int = 0) -> bytes | None:
    """Render a tile PNG for the current snapshot or a forecast offset.
    Reads from in-memory cache if warm, falls back to the current-snapshot disk file.
    """
    cfg = LIVE_WEATHER_VARIABLES.get(variable_id)
    if cfg is None:
        return None
    suffix = f"__f{forecast_hours:03d}h" if forecast_hours else ""
    disk_p = _DISK_CACHE_DIR / f"current_{cfg['model']}_{variable_id}{suffix}.npy"
    mtime_key = f"{forecast_hours}:{variable_id}"
    try:
        disk_mtime = disk_p.stat().st_mtime
    except FileNotFoundError:
        disk_mtime = None

    with _cache_lock:
        arr = (_forecast_cache.get(forecast_hours, {}) if forecast_hours else _cache).get(variable_id)
        cached_mtime = _cache_mtimes.get(mtime_key)

    if disk_mtime is not None and cached_mtime != disk_mtime:
        arr = np.load(disk_p)
        with _cache_lock:
            target = _cache if forecast_hours == 0 else _forecast_cache.setdefault(forecast_hours, {})
            target[variable_id] = arr
            _cache_mtimes[mtime_key] = disk_mtime

    if arr is None:
        return None

    model_cfg = MODEL_CONFIGS[cfg["model"]]
    ny, nx = arr.shape

    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)
    src = np.flipud(arr) if model_cfg["flipud"] else arr
    src_transform = rasterio_from_bounds(
        model_cfg["lon_min"], model_cfg["lat_min"],
        model_cfg["lon_max"], model_cfg["lat_max"],
        nx, ny,
    )
    src_crs = CRS.from_epsg(4326)

    minx, miny, maxx, maxy = tile_bounds_mercator(spec)
    dst_transform = rasterio_from_bounds(minx, miny, maxx, maxy, tile_size, tile_size)
    dst_crs = CRS.from_string(WEB_MERCATOR)

    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=RasterioResampling.nearest if cfg.get("categorical") else RasterioResampling.bilinear,
    )

    if cfg.get("categorical"):
        rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        for code, rgb in _WEATHER_CODE_COLORS.items():
            mask = (dest == code)
            rgba[mask, 0] = rgb[0]
            rgba[mask, 1] = rgb[1]
            rgba[mask, 2] = rgb[2]
            rgba[mask, 3] = 210
    else:
        rgba = _colorize(dest, cfg["stops"], cfg["lo"], cfg["hi"])
    rgba[~np.isfinite(dest), 3] = 0
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Aggregate range overrides — sum over window is larger than snapshot range
_AGG_RANGE_OVERRIDES: dict[str, dict[str, dict[str, float]]] = {
    "precipitation": {
        "24h": {"lo": 0.0, "hi":   50.0},
        "7d":  {"lo": 0.0, "hi":  200.0},
        "30d": {"lo": 0.0, "hi":  500.0},
    },
    "snowfall_water_equivalent": {
        "24h": {"lo": 0.0, "hi":   20.0},
        "7d":  {"lo": 0.0, "hi":   80.0},
        "30d": {"lo": 0.0, "hi":  200.0},
    },
}

# ERA5 grid (lat ascending, south→north, no flipud needed)
_ERA5_LAT_MIN, _ERA5_LAT_MAX = -90.0,  90.0
_ERA5_LON_MIN, _ERA5_LON_MAX = -180.0, 180.0


def render_aggregate_tile_bytes(variable_id: str, window: str, z: int, x: int, y: int,
                                 tile_size: int = 256, forecast_hours: int = 0) -> bytes | None:
    """Render a tile from a pre-computed aggregate raster (.npy on disk)."""
    suffix = f"__f{forecast_hours:03d}h" if forecast_hours else ""
    npy_path = Path(__file__).parent.parent / "data" / "gis" / "temporal" / "rasters" / f"{variable_id}_{window}{suffix}.npy"
    if not npy_path.exists():
        return None

    arr = np.load(npy_path)  # [721, 1440], ERA5 grid, lat ascending
    cfg = LIVE_WEATHER_VARIABLES.get(variable_id)
    if cfg is None:
        return None

    ny, nx = arr.shape
    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)
    src_transform = rasterio_from_bounds(_ERA5_LON_MIN, _ERA5_LAT_MIN, _ERA5_LON_MAX, _ERA5_LAT_MAX, nx, ny)

    minx, miny, maxx, maxy = tile_bounds_mercator(spec)
    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=dest,
        src_transform=src_transform,
        src_crs=CRS.from_epsg(4326),
        src_nodata=np.nan,
        dst_transform=rasterio_from_bounds(minx, miny, maxx, maxy, tile_size, tile_size),
        dst_crs=CRS.from_string(WEB_MERCATOR),
        dst_nodata=np.nan,
        resampling=RasterioResampling.nearest if cfg.get("categorical") else RasterioResampling.bilinear,
    )

    if cfg.get("categorical"):
        rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        for code, rgb in _WEATHER_CODE_COLORS.items():
            mask = (dest == code)
            rgba[mask, 0] = rgb[0]
            rgba[mask, 1] = rgb[1]
            rgba[mask, 2] = rgb[2]
            rgba[mask, 3] = 210
        rgba[~np.isfinite(dest), 3] = 0
    else:
        overrides = _AGG_RANGE_OVERRIDES.get(variable_id, {}).get(window, {})
        lo = overrides.get("lo", cfg["lo"])
        hi = overrides.get("hi", cfg["hi"])
        rgba = _colorize(dest, cfg["stops"], lo, hi)
        rgba[~np.isfinite(dest), 3] = 0

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Maps temporal window hours to forecast offset used for live prediction.
# Windows beyond the longest available forecast (168 h) use the 168 h frame.
_TEMPORAL_WINDOW_TO_FORECAST_HOURS: dict[int, int] = {
    1: 1,
    8: 8,
    24: 24,
    72: 72,
    168: 168,
    720: 168,   # best available proxy for 30-day window
    2160: 168,  # best available proxy for 90-day window
}


def sample_grid_for_tile(
    variable_id: str,
    window_hours: int,
    forecast_hours: int,
    spec: "TileSpec",
) -> np.ndarray:
    """Return a (tile_size, tile_size) float32 array from pre-built temporal rasters.

    Reads from data/gis/temporal/rasters/{variable_id}_{window_label}[__f{forecast:03d}h].npy.
    Returns all-NaN when the file doesn't exist.
    """
    nan_tile = np.full((spec.tile_size, spec.tile_size), np.nan, dtype=np.float32)
    window_label = _TEMPORAL_WINDOW_LABELS.get(window_hours)
    if window_label is None:
        return nan_tile

    if forecast_hours == 0:
        npy_path = _TEMPORAL_RASTER_DIR / f"{variable_id}_{window_label}.npy"
    else:
        npy_path = _TEMPORAL_RASTER_DIR / f"{variable_id}_{window_label}__f{forecast_hours:03d}h.npy"

    if not npy_path.exists():
        return nan_tile

    arr = np.load(npy_path).astype(np.float32)
    # All temporal rasters are on the ERA5 0.25° grid: 721×1440, lat -90→90, lon -180→180
    ny, nx = arr.shape
    src_transform = rasterio_from_bounds(-180.0, -90.0, 180.0, 90.0, nx, ny)

    minx, miny, maxx, maxy = tile_bounds_mercator(spec)
    dst_transform = rasterio_from_bounds(minx, miny, maxx, maxy, spec.tile_size, spec.tile_size)

    dest = nan_tile.copy()
    reproject(
        source=arr,
        destination=dest,
        src_transform=src_transform,
        src_crs=CRS.from_epsg(4326),
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=CRS.from_string(WEB_MERCATOR),
        dst_nodata=np.nan,
        resampling=RasterioResampling.bilinear,
    )
    return dest
