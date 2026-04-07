from __future__ import annotations

import colorsys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import io
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import pyarrow.fs as pafs
import rasterio
from rasterio.crs import CRS as _CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds as window_from_bounds, transform as window_transform

from util.config import load_config
from util import gis_lookup, models, units


CONFIG = load_config("global")
WEB_MERCATOR = "EPSG:3857"


def _is_temporal_column(col: str) -> bool:
    return gis_lookup.is_temporal_layer_id(col)


WGS84_CRS = "EPSG:4326"

LANDCOVER_ID = "landcover"
LITHOLOGY_ID = "lithology"
WRB_ID = "wrb"
CATEGORICAL_VALUE_TYPE = "categorical"
NUMERIC_VALUE_TYPE = "numeric"
DERIVED_ASPECT_ID = "aspect"
DERIVED_ASPECT_DEG_ID = "aspect_deg"
DERIVED_SLOPE_ID = "slope"
DERIVED_FROM_DEM_IDS = frozenset({DERIVED_SLOPE_ID, DERIVED_ASPECT_ID, DERIVED_ASPECT_DEG_ID})

# Keep existing landcover palette for continuity with prior branch behavior.
LANDCOVER_COLORS: dict[int, tuple[int, int, int]] = {
    10: (255, 255, 100),
    11: (255, 255, 100),
    12: (255, 255, 0),
    20: (170, 240, 240),
    51: (76, 115, 0),
    52: (0, 100, 0),
    61: (170, 200, 0),
    62: (0, 160, 0),
    71: (0, 80, 0),
    72: (0, 60, 0),
    81: (40, 100, 0),
    82: (40, 80, 0),
    91: (160, 180, 50),
    92: (120, 130, 0),
    120: (150, 100, 0),
    121: (150, 75, 0),
    122: (150, 100, 0),
    130: (255, 180, 50),
    140: (255, 220, 210),
    150: (255, 235, 175),
    152: (255, 210, 120),
    153: (255, 235, 175),
    180: (0, 220, 130),
    190: (195, 20, 0),
    200: (255, 245, 215),
    201: (220, 220, 220),
    202: (255, 245, 215),
    210: (0, 70, 200),
    220: (255, 255, 255),
    250: (255, 255, 255),
}

# Fixed categorical palettes so class colors stay stable across runs/tiles.
LITHOLOGY_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 220, 220),
    1: (166, 206, 227),
    2: (31, 120, 180),
    3: (178, 223, 138),
    4: (51, 160, 44),
    5: (251, 154, 153),
    6: (227, 26, 28),
    7: (253, 191, 111),
    8: (255, 127, 0),
    9: (202, 178, 214),
    10: (106, 61, 154),
    11: (255, 255, 153),
    12: (177, 89, 40),
    13: (141, 211, 199),
    14: (255, 255, 179),
    15: (190, 186, 218),
}

WRB_COLORS: dict[int, tuple[int, int, int]] = {
    0: (166, 206, 227),
    1: (31, 120, 180),
    2: (178, 223, 138),
    3: (51, 160, 44),
    4: (251, 154, 153),
    5: (227, 26, 28),
    6: (253, 191, 111),
    7: (255, 127, 0),
    8: (202, 178, 214),
    9: (106, 61, 154),
    10: (255, 255, 153),
    11: (177, 89, 40),
    12: (141, 211, 199),
    13: (255, 255, 179),
    14: (190, 186, 218),
    15: (251, 128, 114),
    16: (128, 177, 211),
    17: (253, 180, 98),
    18: (179, 222, 105),
    19: (252, 205, 229),
    20: (217, 217, 217),
    21: (188, 128, 189),
    22: (204, 235, 197),
    23: (255, 237, 111),
    24: (140, 150, 198),
    25: (252, 141, 98),
    26: (102, 194, 165),
    27: (141, 160, 203),
    28: (231, 138, 195),
    29: (166, 216, 84),
}

# Viridis-like stops used for numeric layer rendering.
NUMERIC_COLOR_STOPS = np.asarray(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ],
    dtype=np.float32,
)

# Deterministic numeric ranges so adjacent tiles use the same color scale.
NUMERIC_RANGE_OVERRIDES: dict[str, tuple[float, float]] = {
    "bio_1": (-20.0, 40.0),
    "bio_2": (0.0, 25.0),
    "bio_3": (0.0, 100.0),
    "bio_4": (0.0, 2000.0),
    "bio_5": (-10.0, 55.0),
    "bio_6": (-50.0, 30.0),
    "bio_7": (0.0, 70.0),
    "bio_8": (-30.0, 40.0),
    "bio_9": (-30.0, 40.0),
    "bio_10": (-30.0, 40.0),
    "bio_11": (-50.0, 30.0),
    "bio_12": (0.0, 4000.0),
    "bio_13": (0.0, 1500.0),
    "bio_14": (0.0, 300.0),
    "bio_15": (0.0, 200.0),
    "bio_16": (0.0, 2500.0),
    "bio_17": (0.0, 1000.0),
    "bio_18": (0.0, 2500.0),
    "bio_19": (0.0, 1500.0),
    "elevation": (-500.0, 6000.0),
    "slope": (0.0, 90.0),
    "aspect_deg": (0.0, 360.0),
}


@dataclass(frozen=True)
class TileSpec:
    z: int
    x: int
    y: int
    tile_size: int


def tile_bounds_mercator(spec: TileSpec) -> tuple[float, float, float, float]:
    origin_shift = 2 * math.pi * 6378137 / 2.0
    res = (2 * origin_shift) / (spec.tile_size * (2**spec.z))
    minx = spec.x * spec.tile_size * res - origin_shift
    maxx = (spec.x + 1) * spec.tile_size * res - origin_shift
    maxy = origin_shift - spec.y * spec.tile_size * res
    miny = origin_shift - (spec.y + 1) * spec.tile_size * res
    return minx, miny, maxx, maxy


def _mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    origin_shift = 2 * math.pi * 6378137 / 2.0
    lon = (x / origin_shift) * 180.0
    lat = (y / origin_shift) * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat


def tile_bounds_wgs84(spec: TileSpec) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = tile_bounds_mercator(spec)
    lon_w, lat_s = _mercator_to_lonlat(minx, miny)
    lon_e, lat_n = _mercator_to_lonlat(maxx, maxy)
    return min(lon_w, lon_e), min(lat_s, lat_n), max(lon_w, lon_e), max(lat_s, lat_n)


def _iter_region_origins(
    bounds_wgs84: tuple[float, float, float, float],
    region_size: float,
) -> Iterable[tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    start_lat = math.floor(min_lat / region_size) * region_size
    end_lat = math.floor(max_lat / region_size) * region_size
    start_lon = math.floor(min_lon / region_size) * region_size
    end_lon = math.floor(max_lon / region_size) * region_size
    lat = start_lat
    while lat <= end_lat:
        lon = start_lon
        while lon <= end_lon:
            yield int(lat), int(lon)
            lon += region_size
        lat += region_size


def _region_id_from_origin(lat0: int, lon0: int) -> str:
    return f"lat{lat0}_lon{lon0}"


def _estimate_overview_factor(
    ds: rasterio.DatasetReader,
    bounds_wgs84: tuple[float, float, float, float],
    tile_size: int,
) -> tuple[list[int], float]:
    overviews = ds.overviews(1) or []
    src_res_x = abs(ds.transform.a)
    src_res_y = abs(ds.transform.e)
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    dst_res_x = abs(max_lon - min_lon) / tile_size
    dst_res_y = abs(max_lat - min_lat) / tile_size
    desired = max(dst_res_x / src_res_x, dst_res_y / src_res_y) if src_res_x and src_res_y else 1.0
    return overviews, desired


def _choose_overview_level(overviews: list[int], desired: float) -> tuple[int | None, int | None]:
    if not overviews:
        return None, None
    for idx, factor in enumerate(overviews):
        if factor >= desired:
            return idx, factor
    return len(overviews) - 1, overviews[-1]


def _intersect_bounds(
    left: float,
    bottom: float,
    right: float,
    top: float,
    *,
    ds: rasterio.DatasetReader,
) -> tuple[float, float, float, float] | None:
    ds_left, ds_bottom, ds_right, ds_top = ds.bounds
    x0 = max(left, ds_left)
    y0 = max(bottom, ds_bottom)
    x1 = min(right, ds_right)
    y1 = min(top, ds_top)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _clamp_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_end = min(width, int(math.ceil(window.col_off + window.width)))
    row_end = min(height, int(math.ceil(window.row_off + window.height)))
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(0, col_end - col_off),
        height=max(0, row_end - row_off),
    )


def _window_shape(window: Window) -> tuple[int, int]:
    return int(window.height), int(window.width)


def _is_wgs84(ds: rasterio.DatasetReader) -> bool:
    if ds.crs is None:
        return False
    crs_text = str(ds.crs).upper()
    return crs_text in {"EPSG:4326", "OGC:CRS84"}


def _layer_metadata(layer_id: str) -> dict[str, Any]:
    layer = gis_lookup.load_layer_metadata().get(layer_id)
    if layer is None:
        raise ValueError(f"Layer '{layer_id}' not found in GIS catalog.")
    region_root = str(layer.get("region_root") or "").strip()
    filename_template = str(layer.get("filename_template") or "").strip()
    if not region_root or not filename_template:
        raise ValueError(f"Layer '{layer_id}' is missing region raster metadata.")
    if bool(layer.get("derived")) and layer_id not in DERIVED_FROM_DEM_IDS:
        raise ValueError(f"Layer '{layer_id}' is derived and not currently tile-renderable.")
    return layer


def _layer_value_type(layer_id: str, layer_meta: dict[str, Any]) -> str:
    value_type = str(layer_meta.get("value_type") or "").strip().lower()
    if layer_id in {DERIVED_SLOPE_ID, DERIVED_ASPECT_DEG_ID}:
        return NUMERIC_VALUE_TYPE
    if layer_id == DERIVED_ASPECT_ID:
        return CATEGORICAL_VALUE_TYPE
    if layer_id == LANDCOVER_ID:
        return CATEGORICAL_VALUE_TYPE
    if value_type == CATEGORICAL_VALUE_TYPE:
        return CATEGORICAL_VALUE_TYPE
    return NUMERIC_VALUE_TYPE


def _get_region_source(layer_id: str, layer_meta: dict[str, Any], region_id: str) -> gis_lookup.RasterSource | None:
    region_root = str(layer_meta.get("region_root") or "").strip()
    filename_template = str(layer_meta.get("filename_template") or "").strip()
    filename = filename_template.format(id=layer_id)
    cog_path = CONFIG.gis_root / region_root / region_id / filename
    return gis_lookup.resolve_raster_source(cog_path)


def _resolution_meters(
    transform: rasterio.Affine,
    shape: tuple[int, int],
    crs: Any,
) -> tuple[float, float]:
    xres = abs(transform.a)
    yres = abs(transform.e)
    if not xres or not yres:
        return 1.0, 1.0

    crs_text = str(crs).upper() if crs is not None else ""
    if crs_text in {"EPSG:4326", "OGC:CRS84"}:
        center_row = shape[0] / 2.0
        center_lat = transform.f + (transform.e * center_row)
        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = meters_per_degree_lat * max(0.01, math.cos(math.radians(center_lat)))
        return max(1e-6, xres * meters_per_degree_lon), max(1e-6, yres * meters_per_degree_lat)
    return max(1e-6, xres), max(1e-6, yres)


def _derive_slope_aspect(
    dem: np.ndarray,
    *,
    transform: rasterio.Affine,
    crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(dem)
    if not np.any(finite):
        empty = np.full(dem.shape, np.nan, dtype=np.float32)
        return empty, empty

    fill_value = float(np.nanmedian(dem[finite]))
    filled = np.where(finite, dem, fill_value).astype(np.float32, copy=False)
    xres_m, yres_m = _resolution_meters(transform, dem.shape, crs)
    dz_dy, dz_dx = np.gradient(filled, yres_m, xres_m)

    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    raw_aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (90.0 - raw_aspect) % 360.0
    aspect = aspect.astype(np.float32)

    slope[~finite] = np.nan
    aspect[~finite] = np.nan
    aspect[slope <= 0.001] = np.nan
    return slope, aspect


def _aspect_degrees_to_bins(aspect_degrees: np.ndarray) -> np.ndarray:
    out = np.full(aspect_degrees.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(aspect_degrees)
    if not np.any(finite):
        return out
    normalized = np.mod(aspect_degrees[finite], 360.0)
    # 1..8 bins for N, NE, E, SE, S, SW, W, NW (matching legend ids).
    bins = (np.floor((normalized + 22.5) / 45.0) % 8.0) + 1.0
    out[finite] = bins.astype(np.float32)
    return out


def _finalize_layer_values(
    layer_id: str,
    layer_values: np.ndarray,
    *,
    transform: rasterio.Affine,
    crs: Any,
) -> np.ndarray:
    output_values = layer_values

    if layer_id in DERIVED_FROM_DEM_IDS:
        slope_degrees, aspect_degrees = _derive_slope_aspect(
            layer_values,
            transform=transform,
            crs=crs,
        )
        if layer_id == DERIVED_SLOPE_ID:
            output_values = slope_degrees
        elif layer_id == DERIVED_ASPECT_DEG_ID:
            output_values = aspect_degrees
        else:
            output_values = _aspect_degrees_to_bins(aspect_degrees)

    scale = units.variable_display_scale(layer_id)
    if scale != 1.0:
        finite = np.isfinite(output_values)
        if np.any(finite):
            output_values = output_values.copy()
            output_values[finite] = output_values[finite] * scale
    return output_values


def _render_layer_values(
    layer_id: str,
    spec: TileSpec,
    *,
    reproject_to_mercator: bool,
) -> np.ndarray:
    layer_meta = _layer_metadata(layer_id)

    bounds_wgs84 = tile_bounds_wgs84(spec)
    bounds_mercator = tile_bounds_mercator(spec)
    region_size = float(layer_meta.get("region_size") or 10.0)
    value_type = _layer_value_type(layer_id, layer_meta)
    resampling = Resampling.nearest if value_type == CATEGORICAL_VALUE_TYPE else Resampling.bilinear
    dest = np.full((spec.tile_size, spec.tile_size), np.nan, dtype=np.float32)

    if reproject_to_mercator:
        minx, miny, maxx, maxy = bounds_mercator
        dst_transform = from_bounds(minx, miny, maxx, maxy, spec.tile_size, spec.tile_size)
        dst_crs = WEB_MERCATOR
    else:
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, spec.tile_size, spec.tile_size)
        dst_crs = WGS84_CRS

    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    for lat0, lon0 in _iter_region_origins(bounds_wgs84, region_size):
        region_id = _region_id_from_origin(lat0, lon0)
        source = _get_region_source(layer_id, layer_meta, region_id)
        if source is None:
            continue

        with gis_lookup.open_raster(source) as ds:
            overlap = _intersect_bounds(min_lon, min_lat, max_lon, max_lat, ds=ds)
            if overlap is None:
                continue

            src_window = _clamp_window(
                window_from_bounds(*overlap, transform=ds.transform),
                ds.width,
                ds.height,
            )
            src_h, src_w = _window_shape(src_window)
            if src_h <= 0 or src_w <= 0:
                continue

            overviews, desired = _estimate_overview_factor(ds, bounds_wgs84, spec.tile_size)
            chosen_level, chosen_factor = _choose_overview_level(overviews, desired)
            overview_factor = max(1, int(chosen_factor or 1))

            if not reproject_to_mercator and _is_wgs84(ds):
                raw_dst_window = window_from_bounds(*overlap, transform=dst_transform)
                dst_window = _clamp_window(raw_dst_window, spec.tile_size, spec.tile_size)
                dst_h, dst_w = _window_shape(dst_window)
                if dst_h <= 0 or dst_w <= 0:
                    continue
                env_ctx = rasterio.Env(OVR_LEVEL=str(chosen_level)) if chosen_level is not None else nullcontext()
                with env_ctx:
                    tile = ds.read(
                        1,
                        window=src_window,
                        out_shape=(dst_h, dst_w),
                        resampling=resampling,
                    ).astype(np.float32, copy=False)
                if ds.nodata is not None:
                    tile[tile == ds.nodata] = np.nan
                dst_local_transform = window_transform(dst_window, dst_transform)
                tile = _finalize_layer_values(
                    layer_id,
                    tile,
                    transform=dst_local_transform,
                    crs=dst_crs,
                )
                row0 = int(dst_window.row_off)
                col0 = int(dst_window.col_off)
                row1 = row0 + dst_h
                col1 = col0 + dst_w
                section = dest[row0:row1, col0:col1]
                mask = np.isfinite(tile)
                section[mask] = tile[mask]
                continue

            read_h = max(1, int(math.ceil(src_h / overview_factor)))
            read_w = max(1, int(math.ceil(src_w / overview_factor)))
            env_ctx = rasterio.Env(OVR_LEVEL=str(chosen_level)) if chosen_level is not None else nullcontext()
            with env_ctx:
                source_tile = ds.read(
                    1,
                    window=src_window,
                    out_shape=(read_h, read_w),
                    resampling=resampling,
                ).astype(np.float32, copy=False)

            if ds.nodata is not None:
                source_tile[source_tile == ds.nodata] = np.nan
            src_transform = window_transform(src_window, ds.transform) * rasterio.Affine.scale(
                src_w / read_w,
                src_h / read_h,
            )
            source_tile = _finalize_layer_values(
                layer_id,
                source_tile,
                transform=src_transform,
                crs=ds.crs,
            )
            temp = np.full_like(dest, np.nan)
            reproject(
                source=source_tile,
                destination=temp,
                src_transform=src_transform,
                src_crs=ds.crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
            mask = np.isfinite(temp)
            dest[mask] = temp[mask]

    return dest


def _coerce_int_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number))


def _fallback_categorical_color(class_id: int) -> tuple[int, int, int]:
    hue = ((class_id * 137) % 360) / 360.0
    saturation = 0.65
    value = 0.92
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


@lru_cache(maxsize=64)
def _categorical_palette(layer_id: str) -> dict[int, tuple[int, int, int]]:
    if layer_id == LANDCOVER_ID:
        return dict(LANDCOVER_COLORS)
    if layer_id == LITHOLOGY_ID:
        return dict(LITHOLOGY_COLORS)
    if layer_id == WRB_ID:
        return dict(WRB_COLORS)

    legend = gis_lookup.load_layer_legend(layer_id)
    class_ids = sorted(
        {
            class_id
            for class_id in (_coerce_int_id(entry.get("id")) for entry in legend.values())
            if class_id is not None
        }
    )
    if not class_ids:
        return {}

    palette: dict[int, tuple[int, int, int]] = {}
    total = max(1, len(class_ids))
    for index, class_id in enumerate(class_ids):
        hue = index / total
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.88)
        palette[class_id] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def _colorize_categorical(class_ids: np.ndarray, layer_id: str) -> np.ndarray:
    nan_mask = ~np.isfinite(class_ids)
    rgba = np.zeros((*class_ids.shape, 4), dtype=np.uint8)
    int_ids = np.zeros(class_ids.shape, dtype=np.int32)
    finite = ~nan_mask
    int_ids[finite] = np.rint(class_ids[finite]).astype(np.int32)

    palette = _categorical_palette(layer_id)
    for class_id, color in palette.items():
        mask = int_ids == class_id
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = 255

    unknown_mask = (rgba[..., 3] == 0) & finite
    if np.any(unknown_mask):
        unknown_ids = np.unique(int_ids[unknown_mask])
        for class_id in unknown_ids:
            color = _fallback_categorical_color(int(class_id))
            mask = unknown_mask & (int_ids == class_id)
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
            rgba[mask, 3] = 255

    rgba[nan_mask, 3] = 0
    return rgba


def _normalize_numeric_range(lo: float, hi: float) -> tuple[float, float]:
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi):
        hi = lo + 1.0
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _units_default_numeric_range(units: str) -> tuple[float, float] | None:
    normalized = (
        units.strip()
        .lower()
        .replace(" ", "")
        .replace("⁻", "-")
        .replace("−", "-")
        .replace("×", "x")
        .replace("°", "deg")
        .replace("¹", "1")
        .replace("²", "2")
        .replace("³", "3")
    )
    if normalized in {"m", "meter", "meters"}:
        return (-500.0, 9000.0)
    if normalized in {"degc", "c", "degreescelsius"}:
        return (-50.0, 50.0)
    if normalized in {"mm", "millimeter", "millimeters"}:
        return (0.0, 5000.0)
    if normalized in {"%", "percent"}:
        return (0.0, 100.0)
    if normalized in {"degrees", "degree"}:
        return (0.0, 360.0)
    if normalized in {"pa", "pascal", "pascals"}:
        return (0.0, 8000.0)
    if normalized in {"w/m2", "wm-2", "w/m^2", "wattsperm2"}:
        return (0.0, 400.0)
    if normalized in {"m/s", "ms⁻¹", "ms-1"}:
        return (0.0, 30.0)
    if normalized in {"kpa"}:
        return (0.0, 10.0)
    if normalized in {"m3/m3"}:
        return (0.0, 1.0)
    if normalized in {"cm3/dm3"}:
        return (0.0, 1000.0)
    if normalized in {"ph*10"}:
        return (0.0, 140.0)
    if normalized in {"ratiox100"}:
        return (0.0, 100.0)
    if normalized in {"standarddeviationx100"}:
        return (0.0, 3000.0)
    if normalized in {"coefficient"}:
        return (0.0, 200.0)
    if normalized in {"mjm-2d-1", "mjm-2day-1"}:
        return (0.0, 40.0)
    if "kgm" in normalized and "yr" in normalized:
        return (0.0, 5000.0)
    if "days" in normalized and "yr" in normalized:
        return (0.0, 365.0)
    if normalized in {"g/kg", "gkg"}:
        return (0.0, 1000.0)
    if normalized in {"cg/kg", "cgkg"}:
        return (0.0, 10000.0)
    if normalized in {"dg/kg", "dgkg"}:
        return (0.0, 1000.0)
    return None


def _iter_layer_region_sources(
    layer_id: str,
    layer_meta: dict[str, Any],
) -> Iterable[gis_lookup.RasterSource]:
    region_root = CONFIG.gis_root / str(layer_meta.get("region_root") or "").strip()
    filename = str(layer_meta.get("filename_template") or "").strip().format(id=layer_id)
    yielded: set[str] = set()

    if region_root.exists():
        for region_dir in sorted(region_root.iterdir()):
            if not region_dir.is_dir():
                continue
            source = gis_lookup.resolve_raster_source(region_dir / filename)
            if source is None:
                continue
            if source.uri in yielded:
                continue
            yielded.add(source.uri)
            yield source
        return

    storage_getter = getattr(gis_lookup, "_raster_storage", None)
    if not callable(storage_getter):
        return
    try:
        storage = storage_getter()
    except Exception:
        return
    if not getattr(storage, "is_remote", False) or getattr(storage, "filesystem", None) is None:
        return
    try:
        selector = pafs.FileSelector(storage.resolve(region_root), recursive=True)
        for info in storage.filesystem.get_file_info(selector):
            if info.type != pafs.FileType.File:
                continue
            path = str(info.path)
            if not path.endswith(f"/{filename}"):
                continue
            uri = f"/vsis3/{path}"
            if uri in yielded:
                continue
            yielded.add(uri)
            yield gis_lookup.RasterSource(
                uri=uri,
                gdal_env=storage.gdal_env(),
                is_remote=True,
            )
    except Exception:
        return


@lru_cache(maxsize=128)
def _global_numeric_range(layer_id: str) -> tuple[float, float] | None:
    scale = units.variable_display_scale(layer_id)
    override = NUMERIC_RANGE_OVERRIDES.get(layer_id)
    if override is not None:
        lo, hi = _normalize_numeric_range(float(override[0]), float(override[1]))
        if scale != 1.0:
            lo, hi = _normalize_numeric_range(lo * scale, hi * scale)
        return lo, hi

    layer_meta = _layer_metadata(layer_id)
    explicit_min = layer_meta.get("render_min")
    explicit_max = layer_meta.get("render_max")
    if explicit_min is not None and explicit_max is not None:
        try:
            lo, hi = _normalize_numeric_range(float(explicit_min), float(explicit_max))
            if scale != 1.0:
                lo, hi = _normalize_numeric_range(lo * scale, hi * scale)
            return lo, hi
        except (TypeError, ValueError):
            pass

    def _tag_float(tags: dict[str, str], *keys: str) -> float | None:
        if not tags:
            return None
        lower = {str(k).lower(): v for k, v in tags.items()}
        for key in keys:
            raw = lower.get(key.lower())
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    range_min: float | None = None
    range_max: float | None = None
    max_sources = 4096
    for index, source in enumerate(_iter_layer_region_sources(layer_id, layer_meta)):
        if index >= max_sources:
            break
        try:
            with gis_lookup.open_raster(source) as ds:
                band_tags = ds.tags(1) if ds.count >= 1 else {}
                if not band_tags:
                    band_tags = ds.tags()
                lo = _tag_float(
                    band_tags,
                    "STATISTICS_MINIMUM",
                    "STATISTICS_MIN",
                    "minimum",
                    "min",
                )
                hi = _tag_float(
                    band_tags,
                    "STATISTICS_MAXIMUM",
                    "STATISTICS_MAX",
                    "maximum",
                    "max",
                )
                if lo is None or hi is None:
                    continue
                if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                    continue
                range_min = lo if range_min is None else min(range_min, lo)
                range_max = hi if range_max is None else max(range_max, hi)
        except Exception:
            continue
    if range_min is not None and range_max is not None:
        lo, hi = _normalize_numeric_range(range_min, range_max)
        if scale != 1.0:
            lo, hi = _normalize_numeric_range(lo * scale, hi * scale)
        return lo, hi

    fallback = _units_default_numeric_range(str(layer_meta.get("units") or ""))
    if fallback is None:
        return None
    lo, hi = _normalize_numeric_range(float(fallback[0]), float(fallback[1]))
    return lo, hi


def _colorize_numeric(values: np.ndarray, layer_id: str) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgba

    value_range = _global_numeric_range(layer_id)
    if value_range is None:
        value_range = (0.0, 1.0)
    if value_range is None:
        return rgba
    lo, hi = value_range

    norm = np.clip((values - lo) / max(1e-9, (hi - lo)), 0.0, 1.0)
    finite_norm = norm[finite]
    positions = np.linspace(0.0, 1.0, NUMERIC_COLOR_STOPS.shape[0], dtype=np.float32)
    rgba[finite, 0] = np.interp(finite_norm, positions, NUMERIC_COLOR_STOPS[:, 0]).astype(np.uint8)
    rgba[finite, 1] = np.interp(finite_norm, positions, NUMERIC_COLOR_STOPS[:, 1]).astype(np.uint8)
    rgba[finite, 2] = np.interp(finite_norm, positions, NUMERIC_COLOR_STOPS[:, 2]).astype(np.uint8)
    rgba[finite, 3] = 255
    return rgba


def _colorize_layer(values: np.ndarray, layer_id: str, value_type: str) -> np.ndarray:
    if value_type == CATEGORICAL_VALUE_TYPE:
        return _colorize_categorical(values, layer_id)
    return _colorize_numeric(values, layer_id)


def render_variable_tile_bytes(
    variable_id: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    reproject: bool = True,
) -> bytes:
    layer_id = str(variable_id or "").strip().lower()
    if not layer_id:
        raise ValueError("variable_id is required.")

    layer_meta = _layer_metadata(layer_id)
    value_type = _layer_value_type(layer_id, layer_meta)

    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)
    values = _render_layer_values(
        layer_id,
        spec,
        reproject_to_mercator=reproject,
    )
    rgba = _colorize_layer(values, layer_id, value_type)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


HEATMAP_COLOR_STOPS = np.asarray(
    [
        [28, 38, 102],
        [34, 94, 168],
        [59, 170, 165],
        [246, 190, 0],
        [230, 57, 70],
    ],
    dtype=np.float32,
)


def _colorize_heatmap(values: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgba

    span = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    finite_norm = norm[finite]
    positions = np.linspace(0.0, 1.0, HEATMAP_COLOR_STOPS.shape[0], dtype=np.float32)
    rgba[finite, 0] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 0]).astype(np.uint8)
    rgba[finite, 1] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 1]).astype(np.uint8)
    rgba[finite, 2] = np.interp(finite_norm, positions, HEATMAP_COLOR_STOPS[:, 2]).astype(np.uint8)
    rgba[finite, 3] = np.clip(40.0 + (finite_norm * 215.0), 0.0, 255.0).astype(np.uint8)
    return rgba


def _load_model_layers(
    taxon_id: int,
    model_id: str | None,
    layers: Sequence[str] | None,
) -> list[str]:
    if layers:
        layer_list = [str(layer).strip() for layer in layers if str(layer).strip()]
    else:
        layer_list = models.model_feature_columns(model_id, taxon_id=taxon_id)

    if not layer_list:
        requested = (model_id or "").strip() or models.DEFAULT_MODEL_ID
        raise ValueError(f"No feature columns available for taxon {taxon_id} and model '{requested}'.")

    # Temporal columns don't live in the GIS catalog — only validate non-temporal ones.
    layer_meta = gis_lookup.load_layer_metadata()
    unknown_gis = [layer for layer in layer_list if not _is_temporal_column(layer) and layer not in layer_meta]
    if unknown_gis:
        raise ValueError(
            "Model feature columns are not available in the GIS catalog: " + ", ".join(sorted(unknown_gis))
        )
    return layer_list


def _render_feature_stack(
    layer_list: list[str],
    spec: "TileSpec",
    reproject: bool,
    forecast_hours: int,
    *,
    layer_cache: "dict[str, np.ndarray] | None" = None,
) -> np.ndarray:
    """Render a (tile_size, tile_size, C) feature tensor for the given layer list.

    If *layer_cache* is provided, already-rendered layers are read from it and
    newly rendered layers are stored back into it so subsequent calls sharing the
    same cache dict skip redundant I/O.
    """
    from util import weather_tiles as _wt

    tile_size = spec.tile_size
    stack = np.empty((tile_size, tile_size, len(layer_list)), dtype=np.float32)
    for idx, layer_id in enumerate(layer_list):
        if layer_cache is not None and layer_id in layer_cache:
            stack[:, :, idx] = layer_cache[layer_id]
            continue
        parsed_temporal = gis_lookup.parse_temporal_layer_id(layer_id)
        if parsed_temporal is not None:
            variable_id, _agg, window_hours = parsed_temporal
            try:
                arr = _wt.sample_grid_for_tile(variable_id, window_hours, forecast_hours, spec)
            except Exception as exc:
                arr = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
                print(f"[model-tile] WARNING: temporal layer {layer_id} failed: {exc}")
        else:
            try:
                arr = _render_layer_values(
                    layer_id,
                    spec,
                    reproject_to_mercator=reproject,
                )
            except Exception as exc:
                arr = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
                print(
                    f"[model-tile] WARNING: layer {layer_id} read failed for tile "
                    f"z={spec.z} x={spec.x} y={spec.y} — filling NaN. Error: {exc}"
                )
        stack[:, :, idx] = arr
        if layer_cache is not None:
            layer_cache[layer_id] = arr
        if idx == 0 or idx == len(layer_list) - 1 or (idx + 1) % 10 == 0:
            print(f"[model-tile] rendered layers {idx + 1}/{len(layer_list)} current_layer={layer_id}")
    return stack


def _compute_model_probs(
    taxon_id: int,
    spec: TileSpec,
    *,
    model_id: str | None = None,
    layers: Sequence[str] | None = None,
    reproject: bool = True,
    forecast_hours: int = 0,
    apply_phenology: bool = True,
    phenology_only: bool = False,
    layer_cache: "dict[str, np.ndarray] | None" = None,
) -> np.ndarray:
    """Compute per-pixel probabilities for a single taxon. Returns (H, W) float array."""
    has_phenology = models.has_phenology_model(taxon_id)
    has_full = models.has_full_model(taxon_id)

    if phenology_only:
        active_model_id = (
            models.AUTO_PHENOLOGY_MODEL_ID if has_phenology else models.AUTO_FULL_MODEL_ID if has_full else model_id
        )
    elif apply_phenology and has_full and not has_phenology:
        active_model_id = models.AUTO_FULL_MODEL_ID
    else:
        active_model_id = model_id

    active_layers = _load_model_layers(taxon_id, active_model_id, layers if active_model_id == model_id else None)
    print(
        f"[model-tile] taxon={taxon_id} model={active_model_id} "
        f"features={len(active_layers)} forecast_hours={forecast_hours}"
    )
    stack = _render_feature_stack(active_layers, spec, reproject, forecast_hours, layer_cache=layer_cache)
    probs = models.predict(active_model_id, stack, feature_ids=active_layers, taxon_id=taxon_id)

    # Plants with both SDM + phenology: multiply for combined view
    if apply_phenology and not phenology_only and has_phenology and active_model_id == model_id:
        try:
            pheno_layers = _load_model_layers(taxon_id, models.AUTO_PHENOLOGY_MODEL_ID, None)
            all_layers = list(dict.fromkeys(active_layers + pheno_layers))
            if all_layers != active_layers:
                stack = _render_feature_stack(all_layers, spec, reproject, forecast_hours, layer_cache=layer_cache)
                probs = models.predict(model_id, stack, feature_ids=all_layers, taxon_id=taxon_id)
            pheno_probs = models.predict(
                models.AUTO_PHENOLOGY_MODEL_ID,
                stack,
                feature_ids=all_layers,
                taxon_id=taxon_id,
            )
            probs = probs * pheno_probs
        except ValueError:
            pass

    return probs


def render_model_tile_bytes(
    taxon_id: int,
    z: int,
    x: int,
    y: int,
    *,
    model_id: str | None = None,
    layers: Sequence[str] | None = None,
    tile_size: int = 256,
    reproject: bool = True,
    forecast_hours: int = 0,
    apply_phenology: bool = True,
    phenology_only: bool = False,
) -> bytes:
    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)
    probs = _compute_model_probs(
        taxon_id,
        spec,
        model_id=model_id,
        layers=layers,
        reproject=reproject,
        forecast_hours=forecast_hours,
        apply_phenology=apply_phenology,
        phenology_only=phenology_only,
    )
    rgba = _colorize_heatmap(probs)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_aggregate_tile_bytes(
    z: int,
    x: int,
    y: int,
    *,
    tile_size: int = 256,
    reproject: bool = True,
    forecast_hours: int = 0,
) -> bytes:
    """Render an aggregate heatmap averaged across all species with available models.

    Each species contributes its best combined score (SDM × phenology for plants,
    full model for non-plants, SDM alone as fallback). Scores are averaged pixel-wise.
    """
    taxon_ids = models.get_all_sdm_taxon_ids()
    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)

    # Shared layer cache: GIS rasters and weather grids are sampled once and
    # reused across all taxon model forward passes.
    layer_cache: dict[str, np.ndarray] = {}

    acc: np.ndarray | None = None
    count = 0
    for taxon_id in taxon_ids:
        try:
            probs = _compute_model_probs(
                int(taxon_id),
                spec,
                reproject=reproject,
                forecast_hours=forecast_hours,
                apply_phenology=True,
                layer_cache=layer_cache,
            )
            acc = probs if acc is None else acc + probs
            count += 1
        except Exception as exc:
            print(f"[aggregate-tile] skipping taxon={taxon_id}: {exc}")

    if acc is None or count == 0:
        # No models — return transparent tile
        empty = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        image = Image.fromarray(empty, mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    avg = acc / count
    print(f"[aggregate-tile] z={z} x={x} y={y} species={count} avg_max={float(avg.max()):.3f}")
    rgba = _colorize_heatmap(avg)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _warp_array_to_tile_png(
    src_array: "np.ndarray",
    src_transform: "Affine",
    src_crs: "rasterio.crs.CRS",
    spec: "TileSpec",
    tile_size: int,
    vmin: float,
    vmax: float,
) -> bytes:
    """Warp an in-memory WGS84 float32 array to a Web Mercator tile and return PNG bytes."""
    merc_minx, merc_miny, merc_maxx, merc_maxy = tile_bounds_mercator(spec)
    dst_crs = _CRS.from_epsg(3857)
    dst_transform = from_bounds(merc_minx, merc_miny, merc_maxx, merc_maxy, tile_size, tile_size)
    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    try:
        reproject(
            source=src_array,
            destination=dest,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    except Exception:
        pass
    rgba = _colorize_heatmap(dest, vmin=vmin, vmax=vmax)
    image = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_homepage_tile_bytes(
    z: int,
    x: int,
    y: int,
    *,
    raster_path: "Path | str",
    tile_size: int = 256,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> bytes:
    """Serve a tile from the pre-built aggregate SDM GeoTIFF. Fast — no model inference."""
    from pathlib import Path as _Path

    raster_path = _Path(raster_path)
    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)

    dest = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    if raster_path.exists():
        with rasterio.open(raster_path) as ds:
            merc_minx, merc_miny, merc_maxx, merc_maxy = tile_bounds_mercator(spec)
            dst_transform = from_bounds(merc_minx, merc_miny, merc_maxx, merc_maxy, tile_size, tile_size)
            try:
                reproject(
                    source=rasterio.band(ds, 1),
                    destination=dest,
                    dst_transform=dst_transform,
                    dst_crs=_CRS.from_epsg(3857),
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            except Exception:
                pass

    rgba = _colorize_heatmap(dest, vmin=vmin, vmax=vmax)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_homepage_tile_from_arrays(
    z: int,
    x: int,
    y: int,
    *,
    taxon_arrays: "dict[str, np.ndarray]",
    taxon_ids: "list[str]",
    src_transform: "Affine",
    src_crs: "rasterio.crs.CRS",
    gis_arrays: "dict[str, np.ndarray] | None" = None,
    tile_size: int = 256,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> bytes:
    """Average the given per-taxon WGS84 arrays and warp to a Mercator tile PNG."""
    spec = TileSpec(z=z, x=x, y=y, tile_size=tile_size)

    # Average selected taxa
    first = next((taxon_arrays[tid] for tid in taxon_ids if tid in taxon_arrays), None)
    if first is None:
        return _warp_array_to_tile_png(
            np.full((1, 1), np.nan, dtype=np.float32),
            src_transform,
            src_crs,
            spec,
            tile_size,
            vmin,
            vmax,
        )
    acc = np.zeros(first.shape, dtype=np.float64)
    count = np.zeros(first.shape, dtype=np.int32)
    for tid in taxon_ids:
        arr = taxon_arrays.get(tid)
        if arr is None:
            continue
        finite = np.isfinite(arr)
        acc[finite] += arr[finite]
        count[finite] += 1
    avg = np.full(first.shape, np.nan, dtype=np.float32)
    has = count > 0
    avg[has] = (acc[has] / count[has]).astype(np.float32)

    # Ocean mask in-situ
    if gis_arrays:
        lc = gis_arrays.get("landcover")
        elev = gis_arrays.get("elevation")
        ocean = np.zeros(avg.shape, dtype=bool)
        if lc is not None:
            is_water = np.isfinite(lc) & (lc == 210)
            no_lc = ~np.isfinite(lc)
            if elev is not None:
                ocean |= is_water & ((elev < 1.0) | ~np.isfinite(elev))
                ocean |= no_lc & ((elev < 1.0) | ~np.isfinite(elev))
            else:
                ocean |= is_water | no_lc
        avg[ocean] = np.nan

    return _warp_array_to_tile_png(avg, src_transform, src_crs, spec, tile_size, vmin, vmax)
