"""
Tile renderer for global COG layers.

Renders XYZ slippy-map tiles (Web Mercator, EPSG:3857) from single-file
global COGs stored in data/gis/layers/. All display metadata is read from
config/gis/catalog.json — scale_factor, add_offset, render_min, render_max.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject as warp_reproject
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

from util.storage import ParquetStorageProxy

CATALOG_PATH = Path("config/gis/catalog.json")
LAYERS_DIR   = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "gis" / "layers"

_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent.parent,
)


@contextlib.contextmanager
def _open_raster(path: Path):
    """Open a raster file, using GDAL /vsis3/ when running against remote B2 storage.

    Sets GDAL credentials directly in os.environ to bypass rasterio.Env's
    boto3 guard, which rejects AWS_* keys when boto3 is installed.
    """
    storage = _storage.current()
    if storage.is_remote and not path.exists():
        gdal_env = storage.gdal_env()
        prev = {k: os.environ.get(k) for k in gdal_env}
        os.environ.update(gdal_env)
        try:
            with rasterio.open(storage.vsis3_path(path)) as ds:
                yield ds
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        with rasterio.open(path) as ds:
            yield ds

WEB_MERCATOR      = CRS.from_epsg(3857)
WGS84             = CRS.from_epsg(4326)
_MERCATOR_HALF    = 2 * math.pi * 6378137 / 2.0

SUPPORTED_COLORMAPS = frozenset({"viridis", "plasma", "inferno", "magma", "cividis"})
_DEFAULT_COLORMAP = "viridis"

SUPPORTED_CIRCULAR_COLORMAPS = frozenset({"twilight", "twilight_90", "twilight_180", "twilight_270"})
_DEFAULT_CIRCULAR_COLORMAP = "twilight_90"

# Phase offsets (in LUT entries out of 256) for each twilight variant
_TWILIGHT_PHASE_OFFSETS: dict[str, int] = {
    "twilight":     0,
    "twilight_90":  64,
    "twilight_180": 128,
    "twilight_270": 192,
}

@lru_cache(maxsize=16)
def _get_cmap_lut(name: str) -> np.ndarray:
    from matplotlib import colormaps
    cmap = colormaps[name]
    xs = np.linspace(0.0, 1.0, 256)
    rgba = cmap(xs)  # (256, 4) float64 in [0,1]
    return (rgba[:, :3] * 255.0).astype(np.float32)  # (256, 3)


@lru_cache(maxsize=8)
def _get_circular_cmap_lut(name: str) -> np.ndarray:
    """Return a 256-entry RGB LUT for the named circular (twilight) colormap."""
    from matplotlib import colormaps
    phase = _TWILIGHT_PHASE_OFFSETS.get(name, 0)
    cmap = colormaps["twilight"]
    xs = np.linspace(0.0, 1.0, 256, endpoint=False)
    rgba = cmap(xs)
    lut = (rgba[:, :3] * 255.0).astype(np.float32)  # (256, 3)
    return np.roll(lut, phase, axis=0)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


TEMPORAL_RASTERS_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "gis" / "temporal" / "rasters"

_WINDOW_LABELS = {1: "1h", 8: "8h", 24: "24h", 72: "3d", 168: "7d", 720: "30d", 2160: "90d"}

_MODEL_GRID_PARAMS: dict[str, dict] = {
    "copernicus_era5":      {"ny": 721,  "nx": 1440, "lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "copernicus_era5_land": {"ny": 1801, "nx": 3600, "lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
}

_npy_cache: dict[Path, tuple[float, np.ndarray]] = {}


def _load_temporal_npy(path: Path) -> np.ndarray | None:
    storage = _storage.current()
    cached = _npy_cache.get(path)
    if cached is not None:
        if not storage.is_remote and path.exists() and cached[0] != path.stat().st_mtime:
            pass  # local file changed — fall through to reload
        else:
            return cached[1]

    if path.exists():
        arr = np.load(path).astype(np.float32)
        mtime = path.stat().st_mtime
    elif storage.is_remote:
        try:
            with storage.open_input_file(path) as f:
                arr = np.load(f).astype(np.float32)
        except Exception:
            return None
        mtime = 0.0
    else:
        return None

    _npy_cache[path] = (mtime, arr)
    return arr


def _expand_temporal_layers(category: dict) -> list[dict]:
    """Expand temporal layers into one entry per window with synthesized ids."""
    cat_windows = category.get("windows", [])
    expanded = []
    for layer in category["layers"]:
        agg = layer.get("agg", "avg")
        windows = layer.get("windows", cat_windows)
        for w in windows:
            label = _WINDOW_LABELS.get(w, f"{w}h")
            expanded.append({
                **layer,
                "id": f"{layer['id']}_{agg}_{w}h",
                "var_id": layer["id"],
                "display_name": layer.get("display_name", layer["id"]),
                "window_hours": w,
                "window_label": label,
            })
    return expanded


def load_layers() -> list[dict]:
    layers = []
    for category in _catalog()["categories"]:
        if category.get("id") == "temporal":
            layers.extend(_expand_temporal_layers(category))
        else:
            layers.extend(category["layers"])
    return layers


def load_layers_with_category() -> list[tuple[dict, dict]]:
    """Return (layer, category) pairs for every layer in the catalog."""
    result = []
    for category in _catalog()["categories"]:
        if category.get("id") == "temporal":
            for layer in _expand_temporal_layers(category):
                result.append((layer, category))
        else:
            for layer in category["layers"]:
                result.append((layer, category))
    return result


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

def _colorize_circular(values: np.ndarray, colormap: str = _DEFAULT_CIRCULAR_COLORMAP) -> np.ndarray:
    """Colorize angular values (0–360°) using a twilight circular colormap.

    Maps degrees cyclically to the LUT so 0° and 360° return the same color.
    NaN pixels (nodata) are fully transparent.
    """
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgba

    name = colormap if colormap in SUPPORTED_CIRCULAR_COLORMAPS else _DEFAULT_CIRCULAR_COLORMAP
    lut = _get_circular_cmap_lut(name)
    indices = np.clip(
        ((values[finite] % 360.0) / 360.0 * len(lut)).astype(np.int32) % len(lut),
        0, len(lut) - 1,
    )
    rgba[finite, 0] = lut[indices, 0].astype(np.uint8)
    rgba[finite, 1] = lut[indices, 1].astype(np.uint8)
    rgba[finite, 2] = lut[indices, 2].astype(np.uint8)
    rgba[finite, 3] = 200
    return rgba


_LEGEND_DIR = Path("config/gis/legends")


@lru_cache(maxsize=16)
def _load_nominal_colormap(layer_id: str) -> dict[int, tuple[int, int, int]]:
    """Return {class_id: (R, G, B)} from the layer's legend file."""
    path = _LEGEND_DIR / f"{layer_id}_legend.json"
    if not path.exists():
        import re
        base_id = re.sub(r'_(avg|sum|mode|snapshot)_\d+h$', '', layer_id, flags=re.IGNORECASE)
        if base_id != layer_id:
            path = _LEGEND_DIR / f"{base_id}_legend.json"
    if not path.exists():
        return {}
    colormap: dict[int, tuple[int, int, int]] = {}
    for cls in json.loads(path.read_text()).get("classes", []):
        hex_color = (cls.get("traits") or {}).get("color", "")
        if hex_color.startswith("#") and len(hex_color) == 7:
            colormap[int(cls["id"])] = (
                int(hex_color[1:3], 16),
                int(hex_color[3:5], 16),
                int(hex_color[5:7], 16),
            )
    return colormap


def _nominal_fallback_color(class_id: int) -> tuple[int, int, int]:
    """Generate a stable color for a class ID not present in the legend."""
    import colorsys
    hue = ((class_id * 137) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


def _colorize_nominal(values: np.ndarray, colormap: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Map integer class IDs to RGBA using legend colors (fully opaque)."""
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    if not colormap:
        return rgba
    max_id = max(colormap.keys()) + 1
    lut = np.zeros((max_id, 4), dtype=np.uint8)
    for cid, (r, g, b) in colormap.items():
        if 0 <= cid < max_id:
            lut[cid] = [r, g, b, 255]
    finite = np.isfinite(values)
    ids    = np.where(finite, np.round(values).astype(np.int32), -1)
    known  = finite & (ids >= 0) & (ids < max_id)
    rgba[known] = lut[ids[known]]
    # Fall back to generated colors for any finite class ID not in the legend
    unknown = finite & ~known
    if np.any(unknown):
        for cid in np.unique(ids[unknown]):
            r, g, b = _nominal_fallback_color(int(cid))
            mask = unknown & (ids == cid)
            rgba[mask] = [r, g, b, 255]
    return rgba


def _colorize(values: np.ndarray, vmin: float, vmax: float, colormap: str = _DEFAULT_COLORMAP) -> np.ndarray:
    rgba   = np.zeros((*values.shape, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgba

    lut         = _get_cmap_lut(colormap if colormap in SUPPORTED_COLORMAPS else _DEFAULT_COLORMAP)
    span        = max(float(vmax) - float(vmin), 1e-6)
    norm        = np.clip((values - vmin) / span, 0.0, 1.0)
    finite_norm = norm[finite]
    indices     = np.clip((finite_norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)

    rgba[finite, 0] = lut[indices, 0].astype(np.uint8)
    rgba[finite, 1] = lut[indices, 1].astype(np.uint8)
    rgba[finite, 2] = lut[indices, 2].astype(np.uint8)
    rgba[finite, 3] = np.clip(40.0 + finite_norm * 215.0, 0.0, 255.0).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Tile renderer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _load_temporal_meta(var_id: str, window_label: str) -> dict:
    path = TEMPORAL_RASTERS_DIR / f"{var_id}_{window_label}.meta.json"
    try:
        with _storage.open_input_file(path) as f:
            return json.loads(f.read())
    except Exception:
        return {}


def get_layer_render_range(layer: dict) -> tuple[float | None, float | None]:
    """Return (render_min, render_max) for a layer, falling back to meta.json for temporal layers."""
    rmin = layer.get("render_min")
    rmax = layer.get("render_max")
    if (rmin is None or rmax is None) and layer.get("window_hours") is not None:
        meta = _load_temporal_meta(layer["var_id"], layer["window_label"])
        if rmin is None:
            rmin = meta.get("render_min")
        if rmax is None:
            rmax = meta.get("render_max")
    return rmin, rmax


def render_temporal_tile_bytes(
    layer_id: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    colormap: str = _DEFAULT_COLORMAP,
) -> bytes:
    layer = get_layer(layer_id)
    var_id = layer["var_id"]
    window_label = layer["window_label"]
    model = layer.get("model", "copernicus_era5")
    nominal = str(layer.get("value_type") or "").lower() == "nominal"

    npy_path = TEMPORAL_RASTERS_DIR / f"{var_id}_{window_label}.npy"
    arr = _load_temporal_npy(npy_path)

    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    vmin = layer.get("render_min")
    vmax = layer.get("render_max")

    if arr is not None:
        # Load pre-computed render range from meta.json if catalog doesn't have it
        if not nominal and (vmin is None or vmax is None):
            meta = _load_temporal_meta(var_id, window_label)
            if vmin is None:
                vmin = meta.get("render_min")
            if vmax is None:
                vmax = meta.get("render_max")
        # Last-resort: compute from the array
        if not nominal:
            finite = np.isfinite(arr)
            if vmin is None:
                vmin = float(np.nanpercentile(arr[finite], 2)) if finite.any() else 0.0
            if vmax is None:
                vmax = float(np.nanpercentile(arr[finite], 98)) if finite.any() else 1.0

        # Detect grid from array shape (more reliable than catalog model field)
        shape_to_model = {(721, 1440): "copernicus_era5", (1801, 3600): "copernicus_era5_land"}
        detected = shape_to_model.get(arr.shape, model)
        grid = _MODEL_GRID_PARAMS.get(detected, _MODEL_GRID_PARAMS["copernicus_era5"])
        # arr is lat-ascending (row 0 = south); flipud for rasterio north-up convention
        arr_nu = np.flipud(arr)
        src_transform = from_bounds(
            grid["lon_min"], grid["lat_min"], grid["lon_max"], grid["lat_max"],
            grid["nx"], grid["ny"],
        )
        mx0, my0, mx1, my1 = tile_bounds_mercator(z, x, y)
        dst_transform = from_bounds(mx0, my0, mx1, my1, tile_size, tile_size)
        resample = Resampling.nearest if nominal else Resampling.bilinear
        warp_reproject(
            source=arr_nu,
            destination=dest,
            src_transform=src_transform,
            src_crs=WGS84,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=WEB_MERCATOR,
            dst_nodata=np.nan,
            resampling=resample,
        )
    else:
        vmin = vmin if vmin is not None else 0.0
        vmax = vmax if vmax is not None else 1.0

    if nominal:
        nominal_cmap = _load_nominal_colormap(layer_id)
        rgba = _colorize_nominal(dest, nominal_cmap) if nominal_cmap else _colorize(dest, vmin or 0.0, vmax or 1.0, colormap)
    else:
        rgba = _colorize(dest, vmin, vmax, colormap)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _nominal_tile_range_classes_temporal(
    layer: dict, z: int, x0: int, y0: int, x1: int, y1: int
) -> dict[int, int]:
    """Class-count variant for temporal .npy nominal layers (e.g. weather_code_simple)."""
    var_id = layer["var_id"]
    window_label = layer["window_label"]
    model = layer.get("model", "copernicus_era5")

    arr = _load_temporal_npy(TEMPORAL_RASTERS_DIR / f"{var_id}_{window_label}.npy")
    if arr is None:
        return {}

    shape_to_model = {(721, 1440): "copernicus_era5", (1801, 3600): "copernicus_era5_land"}
    detected = shape_to_model.get(arr.shape, model)
    grid = _MODEL_GRID_PARAMS.get(detected, _MODEL_GRID_PARAMS["copernicus_era5"])
    arr_nu = np.flipud(arr)
    src_transform = from_bounds(
        grid["lon_min"], grid["lat_min"], grid["lon_max"], grid["lat_max"],
        grid["nx"], grid["ny"],
    )
    counts: dict[int, int] = {}
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            mx0, my0, mx1, my1 = tile_bounds_mercator(z, tx, ty)
            dst_transform = from_bounds(mx0, my0, mx1, my1, 256, 256)
            dest = np.full((256, 256), np.nan, dtype=np.float32)
            warp_reproject(
                source=arr_nu,
                destination=dest,
                src_transform=src_transform,
                src_crs=WGS84,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=WEB_MERCATOR,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            finite = np.isfinite(dest)
            if not finite.any():
                continue
            vals, tile_counts = np.unique(np.round(dest[finite]).astype(np.int32), return_counts=True)
            for v, c in zip(vals.tolist(), tile_counts.tolist()):
                counts[int(v)] = counts.get(int(v), 0) + int(c)
    return counts


def nominal_tile_range_classes(
    layer_id: str, z: int, x0: int, y0: int, x1: int, y1: int
) -> dict[int, int]:
    """Return nominal class pixel counts visible across a viewport tile range.

    Uses the same per-tile overview-selection logic as render_layer_tile_bytes so
    the read scale exactly matches what was rendered.  Returns {} for non-nominal layers.
    Keys are class IDs; values are total pixel counts across all tiles in the range.
    """
    layer = get_layer(layer_id)
    if str(layer.get("value_type") or "").lower() != "nominal":
        return {}

    if layer.get("window_hours") is not None:
        return _nominal_tile_range_classes_temporal(layer, z, x0, y0, x1, y1)

    path = LAYERS_DIR / layer["filename"]
    counts: dict[int, int] = {}

    with _open_raster(path) as ds:
        db = ds.bounds
        overviews = ds.overviews(1) or []

        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                lon0, lat0, lon1, lat1 = tile_bounds_wgs84(z, tx, ty)
                rl0 = max(lon0, db.left)
                rl1 = min(lon1, db.right)
                rb0 = max(lat0, db.bottom)
                rb1 = min(lat1, db.top)
                if rl0 >= rl1 or rb0 >= rb1:
                    continue

                src_window = window_from_bounds(rl0, rb0, rl1, rb1, ds.transform)
                src_px_w = ds.width  * (rl1 - rl0) / (db.right - db.left)
                src_px_h = ds.height * (rb1 - rb0) / (db.top   - db.bottom)

                if overviews and src_px_w > 256:
                    desired = src_px_w / 256
                    factor  = min(overviews, key=lambda f: abs(f - desired))
                    read_w  = max(1, round(src_px_w / factor))
                    read_h  = max(1, round(src_px_h / factor))
                else:
                    read_w = max(1, round(src_px_w))
                    read_h = max(1, round(src_px_h))

                raw = ds.read(
                    1,
                    window=src_window,
                    out_shape=(read_h, read_w),
                    resampling=Resampling.nearest,
                )

                if np.issubdtype(raw.dtype, np.integer):
                    dtype_max = int(np.iinfo(raw.dtype).max)
                    nd_int = round(ds.nodata) if ds.nodata is not None else dtype_max
                    mask = (raw != nd_int) & (raw < dtype_max - 3)
                else:
                    mask = np.isfinite(raw)
                    if ds.nodata is not None:
                        mask &= raw != ds.nodata

                vals, tile_counts = np.unique(raw[mask], return_counts=True)
                for v, c in zip(vals.tolist(), tile_counts.tolist()):
                    k = int(v)
                    counts[k] = counts.get(k, 0) + int(c)

    return counts


def _render_derived_elevation_tile_bytes(
    layer: dict,
    z: int,
    x: int,
    y: int,
    tile_size: int,
    derive_fn,
    colormap: str = _DEFAULT_COLORMAP,
) -> bytes:
    """Render a tile for a layer derived on-the-fly from elevation.tif."""
    elev_path = LAYERS_DIR / "elevation.tif"
    vmin = layer.get("render_min", 0.0)
    vmax = layer.get("render_max", 90.0)

    lon0, lat0, lon1, lat1 = tile_bounds_wgs84(z, x, y)
    mx0,  my0,  mx1,  my1  = tile_bounds_mercator(z, x, y)
    dst_transform = from_bounds(mx0, my0, mx1, my1, tile_size, tile_size)
    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

    try:
        with _open_raster(elev_path) as ds:
            db = ds.bounds
            rl0 = max(lon0, db.left)
            rl1 = min(lon1, db.right)
            rb0 = max(lat0, db.bottom)
            rb1 = min(lat1, db.top)
            if rl0 < rl1 and rb0 < rb1:
                src_window = window_from_bounds(rl0, rb0, rl1, rb1, ds.transform)
                src_px_w = ds.width  * (rl1 - rl0) / (db.right - db.left)
                src_px_h = ds.height * (rb1 - rb0) / (db.top   - db.bottom)
                overviews = ds.overviews(1) or []
                if overviews and src_px_w > tile_size:
                    desired = src_px_w / tile_size
                    factor  = min(overviews, key=lambda f: abs(f - desired))
                    read_w  = max(1, round(src_px_w / factor))
                    read_h  = max(1, round(src_px_h / factor))
                else:
                    read_w = max(1, round(src_px_w))
                    read_h = max(1, round(src_px_h))
                raw = ds.read(1, window=src_window, out_shape=(read_h, read_w),
                              resampling=Resampling.bilinear).astype(np.float32)
                if ds.nodata is not None:
                    raw[raw == ds.nodata] = np.nan

                src_tf = window_transform(src_window, ds.transform) * Affine.scale(
                    src_window.width / read_w, src_window.height / read_h,
                )
                derived = derive_fn(raw, src_tf)
                warp_reproject(
                    source=derived, destination=dest,
                    src_transform=src_tf, src_crs=ds.crs,
                    src_nodata=np.nan,
                    dst_transform=dst_transform, dst_crs=WEB_MERCATOR,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
    except Exception:
        pass

    if layer["id"] == "aspect":
        rgba = _colorize_circular(dest, colormap)
    else:
        rgba = _colorize(dest, vmin or 0.0, vmax or 90.0, colormap)
    img  = Image.fromarray(rgba, mode="RGBA")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_layer_tile_bytes(
    layer_id: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    colormap: str = _DEFAULT_COLORMAP,
) -> bytes:
    from util.gis import DERIVED_FROM_ELEVATION, derive_aspect_array, derive_slope_array
    layer = get_layer(layer_id)
    if layer.get("window_hours") is not None:
        return render_temporal_tile_bytes(layer_id, z, x, y, tile_size, colormap)
    if layer_id in DERIVED_FROM_ELEVATION:
        derive_fn = derive_aspect_array if layer_id == "aspect" else derive_slope_array
        return _render_derived_elevation_tile_bytes(layer, z, x, y, tile_size, derive_fn, colormap)
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

    with _open_raster(path) as ds:
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

            raw_native = ds.read(
                1,
                window=src_window,
                out_shape=(read_h, read_w),
                resampling=resampling,
            )

            if np.issubdtype(raw_native.dtype, np.integer):
                iinfo = np.iinfo(raw_native.dtype)
                dtype_max = iinfo.max
                # Compare in native dtype to avoid float32 precision loss on uint32.
                nd_int = round(ds.nodata) if ds.nodata is not None else dtype_max
                if nd_int == dtype_max:
                    nodata_mask = raw_native >= dtype_max - 3
                else:
                    nodata_mask = (raw_native == nd_int) | (raw_native >= dtype_max - 3)
                raw = raw_native.astype(np.float32)
                raw[nodata_mask] = np.nan
            else:
                raw = raw_native.astype(np.float32)
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

    if nominal:
        nominal_cmap = _load_nominal_colormap(layer_id)
        rgba = _colorize_nominal(dest, nominal_cmap) if nominal_cmap else _colorize(dest, vmin, vmax, colormap)
    else:
        rgba = _colorize(dest, vmin, vmax, colormap)
    img  = Image.fromarray(rgba, mode="RGBA")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
