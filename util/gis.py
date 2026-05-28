from __future__ import annotations

import os

import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows

# Ensure GDAL block cache is generous for the terrain tile reads.
os.environ.setdefault("GDAL_CACHEMAX", "4096")

from util.temporal import (
    _LAPSE_RATE,
    ELEVATION_CORRECTABLE_VARS,
    _read_model_elevation,
    grid_indices,
)
from util.tiles import (
    _MODEL_GRID_PARAMS,
    LAYERS_DIR,
    TEMPORAL_RASTERS_DIR,
    _load_temporal_npy,
)


def sample_point(layer: dict, lat: float, lon: float) -> float | None:
    """Return the raster value for a layer at a lat/lon coordinate.

    For static COG layers: opens the file and reads a single pixel, applying
    scale/offset from the catalog.  For temporal layers: samples the current
    (no-forecast-offset) .npy grid.  Returns None for nodata or out-of-bounds.
    """
    if layer.get("window_hours") is not None:
        return _sample_temporal_point(layer, lat, lon)
    if layer["id"] == "slope":
        return compute_slope_at_point(lat, lon)
    if layer["id"] == "aspect":
        return compute_aspect_at_point(lat, lon)
    return _sample_cog_point(layer, lat, lon)


def _sample_cog_point(layer: dict, lat: float, lon: float) -> float | None:
    path = LAYERS_DIR / layer["filename"]
    scale = layer.get("scale_factor") or 1.0
    offset = layer.get("add_offset") or 0.0
    try:
        with rasterio.open(path) as ds:
            row, col = ds.index(lon, lat)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                return None
            window = rasterio.windows.Window(col, row, 1, 1)
            data = ds.read(1, window=window, masked=True)
            if data.mask.all():
                return None
            raw = float(data.data.flat[0])
            if np.issubdtype(data.dtype, np.integer) and ds.nodata is not None:
                dtype_max = np.iinfo(data.dtype).max
                nd_int = round(ds.nodata)
                if raw == nd_int or raw >= dtype_max - 3:
                    return None
            return raw * scale + offset
    except Exception:
        return None


def _sample_temporal_point(layer: dict, lat: float, lon: float) -> float | None:
    """Sample the current (no-forecast-offset) temporal .npy at a lat/lon."""
    var_id = layer["var_id"]
    window_label = layer["window_label"]
    model = layer.get("model", "copernicus_era5")

    arr = _load_temporal_npy(TEMPORAL_RASTERS_DIR / f"{var_id}_{window_label}.npy")
    if arr is None:
        return None

    shape_to_model = {(721, 1440): "copernicus_era5", (1801, 3600): "copernicus_era5_land"}
    grid = _MODEL_GRID_PARAMS.get(shape_to_model.get(arr.shape, model), _MODEL_GRID_PARAMS["copernicus_era5"])

    row = round((lat - grid["lat_min"]) / (grid["lat_max"] - grid["lat_min"]) * (grid["ny"] - 1))
    col = round((lon - grid["lon_min"]) / (grid["lon_max"] - grid["lon_min"]) * (grid["nx"] - 1))
    if not (0 <= row < grid["ny"] and 0 <= col < grid["nx"]):
        return None

    val = float(arr[row, col])
    if not np.isfinite(val):
        return None

    if var_id in ELEVATION_CORRECTABLE_VARS:
        step = layer.get("grid_step", 0.25)
        mode = layer.get("grid_mode", "lat_asc_lon_pm180")
        val = _apply_point_elevation_correction(val, lat, lon, model, step, mode)

    return val


def _apply_point_elevation_correction(
    val: float, lat: float, lon: float, model: str, step: float, mode: str,
) -> float:
    """Apply lapse-rate correction to a temporal value at a single point.

    Samples the elevation COG for the true surface elevation, then looks up
    the model's smoothed grid elevation (HSURF) at the same grid cell.
    Correction = (model_elev - obs_elev) * LAPSE_RATE.
    Returns val unchanged if either elevation is unavailable.
    """
    elev_layer = LAYERS_DIR / "elevation.tif"
    if not elev_layer.exists():
        return val

    try:
        with rasterio.open(elev_layer) as ds:
            r, c = ds.index(lon, lat)
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                return val
            window = rasterio.windows.Window(c, r, 1, 1)
            data = ds.read(1, window=window, masked=True)
            if data.mask.all():
                return val
            obs_elev = float(data.data.flat[0])
    except Exception:
        return val

    if not np.isfinite(obs_elev) or obs_elev <= -9000:
        return val

    ny = int(round(180.0 / step)) + 1
    nx = int(round(360.0 / step)) + 1
    li, lo = grid_indices(lat, lon, ny, nx, mode, step)
    lat_arr = np.array([li], dtype=np.int32)
    lon_arr = np.array([lo], dtype=np.int32)
    model_elev = float(_read_model_elevation(model, lat_arr, lon_arr)[0])
    if not np.isfinite(model_elev):
        return val

    return val + (model_elev - obs_elev) * _LAPSE_RATE


# ---------------------------------------------------------------------------
# Terrain derivation helpers
# ---------------------------------------------------------------------------

_EARTH_A = 6378137.0        # WGS84 semi-major axis (m)
_EARTH_B = 6356752.3142     # WGS84 semi-minor axis (m)

# Layers derived on-the-fly from elevation.tif (no separate COG stored).
DERIVED_FROM_ELEVATION: frozenset[str] = frozenset({"slope", "aspect"})



def _meters_per_degree(lat: float) -> tuple[float, float]:
    """Return (m_per_deg_lat, m_per_deg_lon) at the given latitude."""
    lat_rad = np.radians(lat)
    m_per_deg_lat = (
        111132.954
        - 559.822 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )
    return float(m_per_deg_lat), float(m_per_deg_lon)


def _horn_slope(patch: np.ndarray, dx_m: float, dy_m: float) -> float:
    z1, z2, z3 = patch[0, 0], patch[0, 1], patch[0, 2]
    z4, z6 = patch[1, 0], patch[1, 2]
    z7, z8, z9 = patch[2, 0], patch[2, 1], patch[2, 2]
    dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * dx_m)
    dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * dy_m)
    return float(np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))))


def _horn_aspect(patch: np.ndarray, dx_m: float, dy_m: float) -> float:
    z1, z2, z3 = patch[0, 0], patch[0, 1], patch[0, 2]
    z4, z6 = patch[1, 0], patch[1, 2]
    z7, z8, z9 = patch[2, 0], patch[2, 1], patch[2, 2]
    dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * dx_m)
    dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * dy_m)
    return float((90.0 - np.degrees(np.arctan2(dzdy, -dzdx))) % 360.0)


def sample_elevation_terrain_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    want_elevation: bool = False,
    want_slope: bool = False,
    want_aspect: bool = False,
) -> dict[str, list[float | None]]:
    """Sample elevation, slope, and/or aspect in a single pass over elevation.tif.

    Opens the file once and reads one 3×3 window per point, computing all
    requested outputs from the same read.  Points should be hilbert-sorted
    for GDAL block-cache locality.

    Returns a dict with keys matching the requested outputs.
    """
    n = len(lats)
    elev_path = LAYERS_DIR / "elevation.tif"
    results: dict[str, list[float | None]] = {}
    if want_elevation:
        results["elevation"] = [None] * n
    if want_slope:
        results["slope"] = [None] * n
    if want_aspect:
        results["aspect"] = [None] * n
    if not results or not elev_path.exists() or n == 0:
        return results
    try:
        with rasterio.open(elev_path) as ds:
            nodata = ds.nodata
            pixel_deg = abs(ds.transform.a)
            h, w = ds.height, ds.width
            need_terrain = want_slope or want_aspect
            for i, (lat, lon) in enumerate(zip(lats.tolist(), lons.tolist())):
                try:
                    row, col = ds.index(lon, lat)
                    if need_terrain:
                        if row < 1 or col < 1 or row >= h - 1 or col >= w - 1:
                            continue
                        win = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                        patch = ds.read(1, window=win).astype(np.float64)
                        if patch.shape != (3, 3):
                            continue
                        if nodata is not None and np.any(patch == nodata):
                            continue
                        if np.any(~np.isfinite(patch)):
                            continue
                        if want_elevation:
                            results["elevation"][i] = float(patch[1, 1])
                        m_lat, m_lon = _meters_per_degree(lat)
                        dx_m = pixel_deg * m_lon
                        dy_m = pixel_deg * m_lat
                        if dx_m == 0 or dy_m == 0:
                            continue
                        if want_slope:
                            results["slope"][i] = _horn_slope(patch, dx_m, dy_m)
                        if want_aspect:
                            results["aspect"][i] = _horn_aspect(patch, dx_m, dy_m)
                    else:
                        # elevation only — single-pixel read is enough
                        if row < 0 or col < 0 or row >= h or col >= w:
                            continue
                        win = rasterio.windows.Window(col, row, 1, 1)
                        val = ds.read(1, window=win)
                        v = float(val.flat[0])
                        if nodata is not None and v == nodata:
                            continue
                        if np.isfinite(v):
                            results["elevation"][i] = v
                except Exception:
                    continue
    except Exception:
        pass
    return results


def sample_slope_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute slope (degrees) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_slope=True).get("slope", [None] * len(lats))


def compute_slope_at_point(lat: float, lon: float) -> float | None:
    """Compute slope (degrees) at a single lat/lon."""
    return sample_slope_batch(np.array([lat]), np.array([lon]))[0]


def sample_aspect_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute aspect (°, N=0 clockwise) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_aspect=True).get("aspect", [None] * len(lats))


def compute_aspect_at_point(lat: float, lon: float) -> float | None:
    """Compute aspect (°, N=0 clockwise) at a single lat/lon."""
    return sample_aspect_batch(np.array([lat]), np.array([lon]))[0]


def derive_slope_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive slope (degrees) from a 2-D DEM array using np.gradient.

    Uses the pixel resolution from `transform` and a latitude-aware unit
    conversion.  NaN cells in `dem` propagate to NaN in the output.
    Returns a float32 array of the same shape as `dem`.
    """
    from rasterio.transform import xy as tf_xy
    finite = np.isfinite(dem)
    if not np.any(finite):
        return np.full(dem.shape, np.nan, dtype=np.float32)

    fill = float(np.nanmedian(dem[finite]))
    filled = np.where(finite, dem, fill).astype(np.float64)

    pixel_deg_x = abs(transform.a)
    pixel_deg_y = abs(transform.e)
    # Use centre-row latitude for the whole tile (good enough for a 256-px tile)
    centre_row = dem.shape[0] // 2
    centre_col = dem.shape[1] // 2
    lon_c, lat_c = tf_xy(transform, centre_row, centre_col)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(float(lat_c))
    dx_m = pixel_deg_x * m_per_deg_lon
    dy_m = pixel_deg_y * m_per_deg_lat

    dz_dy, dz_dx = np.gradient(filled, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    slope[~finite] = np.nan
    return slope


def derive_aspect_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive aspect (°, N=0 clockwise) from a 2-D DEM array using np.gradient.

    Uses the same pixel/latitude-aware unit conversion as derive_slope_array.
    NaN cells and flat pixels (slope < _FLAT_SLOPE_THRESHOLD) are set to NaN.
    Returns a float32 array of the same shape as `dem`.
    """
    from rasterio.transform import xy as tf_xy
    finite = np.isfinite(dem)
    if not np.any(finite):
        return np.full(dem.shape, np.nan, dtype=np.float32)

    fill = float(np.nanmedian(dem[finite]))
    filled = np.where(finite, dem, fill).astype(np.float64)

    pixel_deg_x = abs(transform.a)
    pixel_deg_y = abs(transform.e)
    centre_row = dem.shape[0] // 2
    centre_col = dem.shape[1] // 2
    lon_c, lat_c = tf_xy(transform, centre_row, centre_col)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(float(lat_c))
    dx_m = pixel_deg_x * m_per_deg_lon
    dy_m = pixel_deg_y * m_per_deg_lat

    dz_dy, dz_dx = np.gradient(filled, dy_m, dx_m)
    raw = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = ((90.0 - raw) % 360.0).astype(np.float32)
    aspect[~finite] = np.nan
    return aspect


# ---------------------------------------------------------------------------
# Hilbert curve order for spatial indexing.
# Order 13 → 2^13 × 2^13 grid → ~4.9km cells at equator.
# Smaller than a 256-pixel tile at 30m resolution (~7.68km), so observations
# in the same COG internal tile get consecutive indices. Trivially holds for
# all coarser rasters. Index fits in int32 (max value 2^26 - 1 ≈ 67M).
_HILBERT_ORDER = 13


def hilbert_index(latitude: float, longitude: float) -> int:
    """Return a Hilbert curve index for a coordinate (order 13, ~4.9km cells).

    Sort observations by this value before COG raster sampling to maximise
    spatial cache locality across all raster resolutions ≥ 30m.
    """
    n = 1 << _HILBERT_ORDER
    x = min(max(int((longitude + 180.0) / 360.0 * n), 0), n - 1)
    y = min(max(int((latitude + 90.0) / 180.0 * n), 0), n - 1)

    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d
