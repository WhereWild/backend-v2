"""
Tile renderer for global COG layers.

Renders XYZ slippy-map tiles (Web Mercator, EPSG:3857) from single-file
global COGs stored in data/gis/layers/. All display metadata is read from
config/gis/catalog.json — scale_factor, add_offset, render_min, render_max.
"""

from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject as warp_reproject
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

CATALOG_PATH = Path("config/gis/catalog.json")
LAYERS_DIR   = Path("data/gis/layers")

WEB_MERCATOR      = CRS.from_epsg(3857)
WGS84             = CRS.from_epsg(4326)
_MERCATOR_HALF    = 2 * math.pi * 6378137 / 2.0

HEATMAP_COLOR_STOPS = np.asarray(
    [[28, 38, 102], [34, 94, 168], [59, 170, 165], [246, 190, 0], [230, 57, 70]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def load_layers() -> list[dict]:
    return [
        layer
        for category in _catalog()["categories"]
        for layer in category["layers"]
    ]


def get_layer(layer_id: str) -> dict:
    for layer in load_layers():
        if layer["id"] == layer_id:
            return layer
    raise KeyError(f"Layer '{layer_id}' not found in catalog")


# ---------------------------------------------------------------------------
# Tile bounds
# ---------------------------------------------------------------------------

def tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) in EPSG:3857."""
    res    = (2 * _MERCATOR_HALF) / (256 * (2 ** z))
    x_min  = x       * 256 * res - _MERCATOR_HALF
    x_max  = (x + 1) * 256 * res - _MERCATOR_HALF
    y_max  = _MERCATOR_HALF - y       * 256 * res
    y_min  = _MERCATOR_HALF - (y + 1) * 256 * res
    return x_min, y_min, x_max, y_max


def tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) in WGS84."""
    def _unproject(mx: float, my: float) -> tuple[float, float]:
        lon = (mx / _MERCATOR_HALF) * 180.0
        lat = (my / _MERCATOR_HALF) * 180.0
        lat = math.degrees(2.0 * math.atan(math.exp(math.radians(lat))) - math.pi / 2.0)
        return lon, lat

    mx0, my0, mx1, my1 = tile_bounds_mercator(z, x, y)
    lon0, lat0 = _unproject(mx0, my0)
    lon1, lat1 = _unproject(mx1, my1)
    return min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)


# ---------------------------------------------------------------------------
# Colorization
# ---------------------------------------------------------------------------

def _colorize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    rgba   = np.zeros((*values.shape, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgba

    span        = max(float(vmax) - float(vmin), 1e-6)
    norm        = np.clip((values - vmin) / span, 0.0, 1.0)
    finite_norm = norm[finite]
    positions   = np.linspace(0.0, 1.0, HEATMAP_COLOR_STOPS.shape[0], dtype=np.float32)

    rgba[finite, 0] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 0]).astype(np.uint8)
    rgba[finite, 1] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 1]).astype(np.uint8)
    rgba[finite, 2] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 2]).astype(np.uint8)
    rgba[finite, 3] = np.clip(40.0 + finite_norm * 215.0, 0.0, 255.0).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Tile renderer
# ---------------------------------------------------------------------------

def render_layer_tile_bytes(
    layer_id: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
) -> bytes:
    layer   = get_layer(layer_id)
    path    = LAYERS_DIR / layer["filename"]
    scale   = layer.get("scale_factor") or 1.0
    offset  = layer.get("add_offset")   or 0.0
    nominal = str(layer.get("value_type") or "").lower() == "nominal"
    vmin    = layer.get("render_min")
    vmax    = layer.get("render_max")

    resampling   = Resampling.nearest if nominal else Resampling.bilinear
    lon0, lat0, lon1, lat1 = tile_bounds_wgs84(z, x, y)
    mx0,  my0,  mx1,  my1  = tile_bounds_mercator(z, x, y)
    dst_transform = from_bounds(mx0, my0, mx1, my1, tile_size, tile_size)
    dest          = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

    with rasterio.open(path) as ds:
        db = ds.bounds
        rl0 = max(lon0, db.left)
        rl1 = min(lon1, db.right)
        rb0 = max(lat0, db.bottom)
        rb1 = min(lat1, db.top)

        if rl0 < rl1 and rb0 < rb1:
            src_window = window_from_bounds(rl0, rb0, rl1, rb1, ds.transform)

            # Pick read resolution: how many source pixels cover this tile?
            src_px_w = ds.width  * (rl1 - rl0) / (db.right - db.left)
            src_px_h = ds.height * (rb1 - rb0) / (db.top   - db.bottom)
            overviews = ds.overviews(1) or []

            if overviews and src_px_w > tile_size:
                desired  = src_px_w / tile_size
                factor   = min(overviews, key=lambda f: abs(f - desired))
                read_w   = max(1, round(src_px_w / factor))
                read_h   = max(1, round(src_px_h / factor))
            else:
                read_w = max(1, round(src_px_w))
                read_h = max(1, round(src_px_h))

            raw = ds.read(
                1,
                window=src_window,
                out_shape=(read_h, read_w),
                resampling=resampling,
            ).astype(np.float32)

            if ds.nodata is not None:
                raw[raw == ds.nodata] = np.nan

            raw = raw * scale + offset

            if vmin is None:
                vmin = float(np.nanmin(raw)) if np.any(np.isfinite(raw)) else 0.0
            if vmax is None:
                vmax = float(np.nanmax(raw)) if np.any(np.isfinite(raw)) else 1.0

            src_tf = window_transform(src_window, ds.transform) * Affine.scale(
                src_window.width  / read_w,
                src_window.height / read_h,
            )

            warp_reproject(
                source=raw,
                destination=dest,
                src_transform=src_tf,
                src_crs=ds.crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=WEB_MERCATOR,
                dst_nodata=np.nan,
                resampling=resampling,
            )

    vmin = vmin if vmin is not None else 0.0
    vmax = vmax if vmax is not None else 1.0

    rgba = _colorize(dest, vmin, vmax)
    img  = Image.fromarray(rgba, mode="RGBA")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
