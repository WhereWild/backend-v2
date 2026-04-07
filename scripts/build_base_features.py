from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.config import load_config
from util import gis_lookup, models

CONFIG = load_config("global")

# ---------------------------------------------------------------------------
# Tweak these to control the run
# ---------------------------------------------------------------------------

# GIS layer used to discover region coverage and native CRS
REFERENCE_LAYER = "bio_1"

# Output resolution in degrees.
#   0.008333° ≈ 1 km  (bio1 native — very slow, large file)
#   0.1°      ≈ 10 km (reasonable detail)
#   0.25°     ≈ 25 km (matches ERA5 temporal resolution — fast, small)
OUTPUT_RESOLUTION_DEGREES = 0.03

# Where to write the output files
OUTPUT_DIR = Path("data/gis/temporal/homepage")

# Bounding box — clamp output grid to this region (set to None for global)
# Matches the front-end CONUS_MAX_BOUNDS in LocalMapSection.tsx
BBOX_MIN_LON: float | None = -135.0
BBOX_MAX_LON: float | None = -60.0
BBOX_MIN_LAT: float | None = 22.0
BBOX_MAX_LAT: float | None = 55.0

# ---------------------------------------------------------------------------

WGS84 = CRS.from_epsg(4326)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_meta(layer_id: str) -> dict[str, Any]:
    meta = gis_lookup.load_layer_metadata().get(layer_id)
    if meta is None:
        raise KeyError(f"Layer '{layer_id}' not found in GIS catalog")
    return meta


def _region_tile_path(layer_id: str, meta: dict, lat0: int, lon0: int) -> Path:
    region_root = CONFIG.gis_root / str(meta.get("region_root", "regions"))
    filename = str(meta.get("filename_template", "{id}.tif")).format(id=layer_id)
    return region_root / f"lat{lat0}_lon{lon0}" / filename


def _available_regions(reference_layer_id: str) -> list[tuple[int, int]]:
    meta = _layer_meta(reference_layer_id)
    region_root = CONFIG.gis_root / str(meta.get("region_root", "regions"))
    filename = str(meta.get("filename_template", "{id}.tif")).format(id=reference_layer_id)
    origins: list[tuple[int, int]] = []
    if not region_root.exists():
        return origins
    for region_dir in sorted(region_root.iterdir()):
        if not region_dir.is_dir():
            continue
        if not (region_dir / filename).exists():
            continue
        m = re.match(r"lat(-?\d+)_lon(-?\d+)$", region_dir.name)
        if m:
            origins.append((int(m.group(1)), int(m.group(2))))
    return origins


def _reference_crs(reference_layer_id: str, regions: list[tuple[int, int]]) -> CRS:
    meta = _layer_meta(reference_layer_id)
    for lat0, lon0 in regions:
        p = _region_tile_path(reference_layer_id, meta, lat0, lon0)
        if p.exists():
            src = gis_lookup.resolve_raster_source(p)
            with gis_lookup.open_raster(src) as ds:
                return ds.crs
    return WGS84


def _read_gis_layer_global(
    layer_id: str,
    regions: list[tuple[int, int]],
    out: np.ndarray,  # (H, W) global output — written in-place
    out_transform: rasterio.Affine,
    out_bounds: tuple[float, float, float, float],  # min_lon, min_lat, max_lon, max_lat
) -> None:
    """Read a GIS layer into the global output array, one region tile at a time.

    Each tile is read via ds.read(out=small_array) which lets rasterio pick the
    correct overview level automatically — no reprojection, no GDAL warp overhead.
    This works because all region tiles are already WGS84 and axis-aligned.
    """
    meta = _layer_meta(layer_id)
    value_type = meta.get("value_type", "continuous")
    resampling = Resampling.nearest if value_type == "categorical" else Resampling.bilinear
    res = OUTPUT_RESOLUTION_DEGREES
    min_lon_out, min_lat_out, max_lon_out, max_lat_out = out_bounds
    out_h, out_w = out.shape

    for lat0, lon0 in regions:
        p = _region_tile_path(layer_id, meta, lat0, lon0)
        if not p.exists():
            continue
        src = gis_lookup.resolve_raster_source(p)
        if src is None:
            continue

        # Destination slice in the global array for this 10°×10° region
        region_size = 10
        lon_min_r = float(lon0)
        lon_max_r = float(lon0 + region_size)
        lat_min_r = float(lat0)
        lat_max_r = float(lat0 + region_size)

        col_off = round((lon_min_r - min_lon_out) / res)
        row_off = round((max_lat_out - lat_max_r) / res)
        # Compute end from max coordinate — avoids 1-px seams from independent round() calls
        col_end = min(round((lon_max_r - min_lon_out) / res), out_w)
        row_end = min(round((max_lat_out - lat_min_r) / res), out_h)
        col_off = max(col_off, 0)
        row_off = max(row_off, 0)
        slice_w = col_end - col_off
        slice_h = row_end - row_off

        if slice_w <= 0 or slice_h <= 0:
            continue

        with gis_lookup.open_raster(src) as ds:
            # Same overview selection strategy as tiles.py:
            # desired = dst_res / src_res, pick smallest overview >= desired
            src_res = abs(ds.transform.a)
            desired = (res / src_res) if src_res else 1.0
            overviews = ds.overviews(1) or []
            chosen_level = None
            for idx, factor in enumerate(overviews):
                if factor >= desired:
                    chosen_level = idx
                    break
            if chosen_level is None and overviews:
                chosen_level = len(overviews) - 1

            src_window = window_from_bounds(
                lon_min_r,
                lat_min_r,
                lon_max_r,
                lat_max_r,
                transform=ds.transform,
            )
            env_ctx = rasterio.Env(OVR_LEVEL=str(chosen_level)) if chosen_level is not None else nullcontext()
            with env_ctx:
                buf = ds.read(
                    1,
                    window=src_window,
                    out_shape=(slice_h, slice_w),
                    resampling=resampling,
                ).astype(np.float32, copy=False)

            if ds.nodata is not None:
                buf[buf == ds.nodata] = np.nan

        out[row_off:row_end, col_off:col_end] = buf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    regions = _available_regions(REFERENCE_LAYER)
    if not regions:
        raise RuntimeError(f"No region tiles found for layer '{REFERENCE_LAYER}'")
    print(f"[base-features] {len(regions)} regions  resolution={OUTPUT_RESOLUTION_DEGREES}°")

    native_crs = _reference_crs(REFERENCE_LAYER, regions)

    # Collect all GIS layer IDs needed by any model (no temporal)
    taxon_ids = models.get_all_sdm_taxon_ids()
    layer_catalog = gis_lookup.load_layer_metadata()
    temporal_registry = gis_lookup.load_temporal_registry() or {}
    temporal_variable_ids: set[str] = {layer["id"] for layer in temporal_registry.get("layers", []) if layer.get("id")}

    gis_layer_set: set[str] = set()
    for taxon_id in taxon_ids:
        for model_id in (models.AUTO_MODEL_ID, models.AUTO_PHENOLOGY_MODEL_ID, models.AUTO_FULL_MODEL_ID):
            cols = models.model_feature_columns(model_id, taxon_id=taxon_id)
            for col in cols:
                # Must be in the GIS catalog and not a temporal variable
                if col not in layer_catalog:
                    continue
                parsed_temporal = gis_lookup.parse_temporal_layer_id(col)
                base = parsed_temporal[0] if parsed_temporal else col
                if base in temporal_variable_ids or col in temporal_variable_ids:
                    continue
                gis_layer_set.add(col)

    gis_layer_ids = sorted(gis_layer_set)
    print(f"[base-features] {len(gis_layer_ids)} unique GIS layers across {len(taxon_ids)} taxa")

    region_size = 10
    min_lon = min(lon0 for _, lon0 in regions)
    max_lon = max(lon0 for _, lon0 in regions) + region_size
    min_lat = min(lat0 for lat0, _ in regions)
    max_lat = max(lat0 for lat0, _ in regions) + region_size
    if BBOX_MIN_LON is not None:
        min_lon = max(min_lon, BBOX_MIN_LON)
    if BBOX_MAX_LON is not None:
        max_lon = min(max_lon, BBOX_MAX_LON)
    if BBOX_MIN_LAT is not None:
        min_lat = max(min_lat, BBOX_MIN_LAT)
    if BBOX_MAX_LAT is not None:
        max_lat = min(max_lat, BBOX_MAX_LAT)
    res = OUTPUT_RESOLUTION_DEGREES
    out_width = round((max_lon - min_lon) / res)
    out_height = round((max_lat - min_lat) / res)
    out_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, out_width, out_height)
    print(
        f"[base-features] global grid: {out_width}×{out_height} px  lon[{min_lon},{max_lon}] lat[{min_lat},{max_lat}]"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for i, layer_id in enumerate(gis_layer_ids):
        print(f"  [{i + 1}/{len(gis_layer_ids)}] {layer_id}")
        arr = np.full((out_height, out_width), np.nan, dtype=np.float32)
        _read_gis_layer_global(layer_id, regions, arr, out_transform, (min_lon, min_lat, max_lon, max_lat))
        arrays[layer_id] = arr

    # Save uncompressed so infer_aggregate_raster can memory-map it (mmap_mode='r')
    # and slice chunks without loading the full file into RAM.
    npz_path = OUTPUT_DIR / "base_features.npz"
    np.savez(npz_path, **arrays)
    print(f"[base-features] saved {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    meta_path = OUTPUT_DIR / "base_features_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "layers": gis_layer_ids,
                "shape": [out_height, out_width],
                "transform": [
                    out_transform.a,
                    out_transform.b,
                    out_transform.c,
                    out_transform.d,
                    out_transform.e,
                    out_transform.f,
                ],
                "crs_wkt": native_crs.to_wkt(),
                "resolution_degrees": res,
                "bounds": {"min_lon": min_lon, "max_lon": max_lon, "min_lat": min_lat, "max_lat": max_lat},
            },
            indent=2,
        )
    )
    print(f"[base-features] saved {meta_path}")


if __name__ == "__main__":
    main()
