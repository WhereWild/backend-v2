from __future__ import annotations

import numpy as np
import rasterio
import rasterio.windows

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
    return None if not np.isfinite(val) else val


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
