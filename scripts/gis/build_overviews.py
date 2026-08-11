# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build internal overviews for all GeoTIFFs in data/gis/layers/, converting
each to a Cloud Optimized GeoTIFF (COG) with appropriate overview levels.

Overview resampling is chosen by value_type:
  interval / ratio → average
  nominal          → nearest  (preserves discrete class values)

Also the one place two config.py-driven corrections actually get applied,
for the same reason in both cases: acting on every layer in data/gis/layers/
from one place means every consumer benefits, instead of each download
script needing its own special-casing.

  - ZERO_NODATA_LAYERS: nodata pixels are burned in as a real 0 (and the
    nodata flag cleared) before overviews are built from the corrected
    data — map tiles render a real "absent"-equivalent color with no
    transparent gaps, /gis/point returns 0 instead of "No data", and
    enrich_tree.py's own nodata check becomes a harmless no-op.

  - Every continuous (interval/ratio) layer's render_min/render_max is
    recomputed as the 1st/99th percentile of valid pixel values
    (PERCENTILE_RENDER_BOUNDS) instead of true min/max — avoids a long tail
    (e.g. precipitation) compressing the bulk of "normal" values into a
    narrow slice of the color range under a linear scale. Never applied to
    nominal/ordinal layers.

Also builds the equivalent of "overviews" for native vector-source layers
(catalog entries with a "vector_field", e.g. ecoregions/biome — see
scripts/gis/download_ecoregions.py): a pre-simplified sibling file per zoom
level 0-VECTOR_PYRAMID_MAX_ZOOM, via GeoSeries.simplify_coverage(). Without
this, util/tiles.py's per-tile vector rasterizer would have to simplify (or
worse, burn full-detail geometry) on every request — full-detail RESOLVE
ecoregion polygons can carry tens of thousands of vertices each, and a
zoomed-out tile's bbox typically overlaps dozens of them, so paying that
cost live made low-zoom tiles painfully slow. simplify_coverage (rather
than simplifying each polygon independently) treats the layer as a
coverage and simplifies its shared boundary network jointly, so adjacent
polygons — ecoregions share a lot of borders — don't drift apart into
gaps/overlaps.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import rasterio

from config.config import PERCENTILE_RENDER_BOUNDS, ZERO_NODATA_LAYERS

CATALOG_PATH        = Path("config/gis/catalog.json")
LAYERS_DIR          = Path("data/gis/layers")
TARGET_MIN_ZOOM     = 3
TARGET_TILE_SIZE    = 256
MAX_OVERVIEW_FACTOR = 2048

OVERVIEW_FACTOR_TOLERANCE_RATIO = 0.03
OVERVIEW_FACTOR_TOLERANCE_MIN   = 2

# Must match util.tiles.VECTOR_PYRAMID_MAX_ZOOM (which picks these files at
# request time) — zoom levels 0..N are precomputed here; above N the
# full-detail source is used directly, since tolerance is sub-meter by then
# and RESOLVE's own source data isn't that precise to begin with.
VECTOR_PYRAMID_MAX_ZOOM = 10
_WEB_MERCATOR_HALF_CIRCUMFERENCE_M = 2 * math.pi * 6378137 / 2.0


def _load_layer_meta() -> dict[str, dict]:
    """Return {filename: layer_entry} for every layer in the catalog."""
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    return {
        layer["filename"]: layer
        for category in catalog["categories"]
        for layer in category["layers"]
    }


def _is_class_based(layer: dict | None) -> bool:
    """True for discrete-class layers (nominal/ordinal) that must use mode resampling."""
    if not layer:
        return False
    return str(layer.get("value_type") or "").lower() in ("nominal", "ordinal")


def _target_dst_res_degrees() -> float:
    return 360.0 / ((2 ** TARGET_MIN_ZOOM) * TARGET_TILE_SIZE)


def _next_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def _overview_factors_for_dataset(ds: rasterio.DatasetReader) -> list[int]:
    src_res_x = abs(ds.transform.a) if ds.transform else 0.0
    src_res_y = abs(ds.transform.e) if ds.transform else 0.0
    if not src_res_x or not src_res_y or not math.isfinite(src_res_x) or not math.isfinite(src_res_y):
        return []
    dst_res = _target_dst_res_degrees()
    desired = max(dst_res / src_res_x, dst_res / src_res_y)
    if not math.isfinite(desired) or desired <= 1:
        return []
    target = min(_next_power_of_two(desired), MAX_OVERVIEW_FACTOR)
    min_dim = int(min(ds.width, ds.height))
    factors: list[int] = []
    factor = 2
    while factor <= target and factor < min_dim:
        factors.append(factor)
        factor *= 2
    return factors


def _overview_factor_close(actual: int, target: int) -> bool:
    tolerance = max(
        OVERVIEW_FACTOR_TOLERANCE_MIN,
        int(round(target * OVERVIEW_FACTOR_TOLERANCE_RATIO)),
    )
    return abs(actual - target) <= tolerance


def _has_required_overviews(existing: list[int], desired: list[int]) -> bool:
    if not desired:
        return True
    if not existing:
        return False
    return all(
        any(_overview_factor_close(actual, target) for actual in existing)
        for target in desired
    )


def _fill_nodata_with_zero(path: Path) -> bool:
    """Burn nodata pixels in as a real 0 and clear the nodata marker.

    Mutates path in place (r+) — cheap read/modify/write, not a full COG
    rebuild; _build_cog rebuilds the actual COG (with fresh overviews) from
    this corrected file right after, so this doesn't itself need to produce
    a valid COG. Returns True if anything changed, so the caller can force
    an overview rebuild even when existing overviews would otherwise look
    sufficient (old overviews built from the un-filled data could have
    transparent-gap artifacts baked into their downsampled pixels).
    """
    # IGNORE_COG_LAYOUT_BREAK: a file already processed by a previous
    # build_overviews run is a strict-layout COG, and GDAL's COG driver
    # refuses in-place r+ writes against that layout by default. Harmless
    # here — _build_cog rebuilds a fresh, correctly-laid-out COG from this
    # file immediately after, so whatever layout optimization this write
    # "breaks" is being discarded anyway.
    with rasterio.open(path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as ds:
        nodata = ds.nodata
        if nodata is None:
            return False
        data = ds.read(1)
        mask = np.isnan(data) if np.isnan(nodata) else (data == nodata)
        changed = bool(np.any(mask))
        if changed:
            data[mask] = 0
            ds.write(data, 1)
        ds.nodata = None
        return changed


def _compute_percentile_bounds(path: Path, layer: dict) -> tuple[float, float]:
    """Return (render_min, render_max) as the PERCENTILE_RENDER_BOUNDS
    percentiles of valid pixel values, in display units (scale_factor/
    add_offset applied) — same nodata-masking logic as download_chelsa.py's
    _compute_stats, just np.percentile instead of nanmin/nanmax.
    """
    scale = layer.get("scale_factor") or 1.0
    offset = layer.get("add_offset") or 0.0
    with rasterio.open(path) as ds:
        dtype_str = ds.dtypes[0]
        nodata = ds.nodata
        raw_native = ds.read(1)
    if np.issubdtype(np.dtype(dtype_str), np.integer):
        iinfo = np.iinfo(dtype_str)
        dtype_max = iinfo.max
        nd_int = round(nodata) if nodata is not None else dtype_max
        if nd_int == dtype_max:
            nodata_mask = raw_native >= dtype_max - 3
        else:
            nodata_mask = (raw_native == nd_int) | (raw_native >= dtype_max - 3)
        raw = raw_native.astype(np.float32)
        raw[nodata_mask] = np.nan
    else:
        raw = raw_native.astype(np.float32)
        if nodata is not None:
            raw[raw == nodata] = np.nan
    raw = raw * scale + offset
    valid = raw[np.isfinite(raw)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, list(PERCENTILE_RENDER_BOUNDS))
    return round(float(lo), 6), round(float(hi), 6)


def _build_cog(src_path: Path, dst_path: Path, *, nominal: bool, overview_factors: list[int]) -> None:
    resampling = "mode" if nominal else "average"
    base_tif = dst_path.with_suffix(".base.tif")
    try:
        subprocess.run(
            [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "TILED=YES",
                "-co", "COMPRESS=DEFLATE",
                "-co", "BIGTIFF=IF_SAFER",
                str(src_path), str(base_tif),
            ],
            check=True,
        )
        if overview_factors:
            subprocess.run(
                ["gdaladdo", "-r", resampling, str(base_tif), *[str(f) for f in overview_factors]],
                check=True,
            )
        subprocess.run(
            [
                "gdal_translate",
                "-of", "COG",
                "-co", "COMPRESS=DEFLATE",
                "-co", "BIGTIFF=IF_SAFER",
                "-co", "OVERVIEWS=FORCE_USE_EXISTING",
                "-co", f"OVERVIEW_RESAMPLING={resampling.upper()}",
                str(base_tif), str(dst_path),
            ],
            check=True,
        )
    finally:
        base_tif.unlink(missing_ok=True)


def _vector_layer_filenames(layer_meta: dict[str, dict]) -> set[str]:
    return {fname for fname, layer in layer_meta.items() if layer.get("vector_field")}


def _tolerance_for_zoom(z: int) -> float:
    """Web Mercator meters-per-pixel at zoom `z` (256px tiles) — the same
    "don't remove detail finer than one screen pixel" tolerance Mapnik/
    GeoServer/Tippecanoe use for scale-dependent generalization.
    """
    return (2 * _WEB_MERCATOR_HALF_CIRCUMFERENCE_M) / (256 * (2 ** z))


def _build_vector_pyramid(path: Path) -> bool:
    """Precompute simplify_coverage()'d siblings of a vector layer source at
    each zoom in [0, VECTOR_PYRAMID_MAX_ZOOM] — see util.tiles._vector_pyramid_path
    for how the tile renderer picks between them.
    """
    import geopandas as gpd

    gdf = gpd.read_parquet(path)
    src_mtime = path.stat().st_mtime
    built_any = False
    for z in range(VECTOR_PYRAMID_MAX_ZOOM + 1):
        out_path = path.with_name(f"{path.stem}.z{z:02d}{path.suffix}")
        if out_path.exists() and out_path.stat().st_mtime >= src_mtime:
            continue
        simplified = gdf.copy()
        simplified["geometry"] = gdf.geometry.simplify_coverage(_tolerance_for_zoom(z))
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        simplified.to_parquet(tmp, compression="zstd")
        os.replace(tmp, out_path)
        built_any = True
    return built_any


def main(fill: bool = False) -> None:
    if not LAYERS_DIR.exists():
        raise FileNotFoundError(f"Layers directory not found: {LAYERS_DIR}")

    layer_meta = _load_layer_meta()
    total = updated = skipped = 0
    catalog_updates: dict[str, tuple[float, float]] = {}

    for path in sorted(LAYERS_DIR.glob("*.tif")):
        total += 1
        layer = layer_meta.get(path.name)
        nominal = _is_class_based(layer)
        layer_id = layer.get("id") if layer else None
        value_type = str(layer.get("value_type") or "").lower() if layer else ""

        try:
            force_rebuild = False
            if fill and layer_id in ZERO_NODATA_LAYERS and _fill_nodata_with_zero(path):
                print(f"[overview] zero-filled nodata -> {path.name}")
                force_rebuild = True

            has_render_bounds = bool(layer) and layer.get("render_min") is not None and layer.get("render_max") is not None
            if layer_id and value_type not in ("nominal", "ordinal") and not has_render_bounds:
                new_min, new_max = _compute_percentile_bounds(path, layer)
                if (layer.get("render_min"), layer.get("render_max")) != (new_min, new_max):
                    print(
                        f"[overview] percentile render bounds -> {path.name}: "
                        f"{layer.get('render_min')}/{layer.get('render_max')} -> {new_min}/{new_max}"
                    )
                    catalog_updates[layer_id] = (new_min, new_max)

            with rasterio.open(path) as ds:
                existing = ds.overviews(1) or []
                desired = _overview_factors_for_dataset(ds)

            if not force_rebuild and existing and _has_required_overviews(existing, desired):
                skipped += 1
                continue

            if existing:
                print(f"[overview] upgrading {path.name}  existing={existing}  target={desired}")
            else:
                print(f"[overview] building  {path.name}  target={desired}")

            tmp = path.with_suffix(".tif.tmp")
            _build_cog(path, tmp, nominal=nominal, overview_factors=desired)
            os.replace(tmp, path)
            updated += 1

        except Exception as exc:
            print(f"[overview] failed {path.name}: {exc}")
            path.with_suffix(".tif.tmp").unlink(missing_ok=True)
            path.with_suffix(".base.tif").unlink(missing_ok=True)

    for filename in sorted(_vector_layer_filenames(layer_meta)):
        path = LAYERS_DIR / filename
        if not path.exists():
            continue
        total += 1
        try:
            if _build_vector_pyramid(path):
                print(f"[overview] vector pyramid built -> {path.name}")
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"[overview] failed vector pyramid {path.name}: {exc}")

    if catalog_updates:
        # Re-read from disk before writing so external edits made while this
        # ran aren't clobbered — same pattern as download_chelsa.py.
        with open(CATALOG_PATH) as f:
            on_disk = json.load(f)
        for cat in on_disk["categories"]:
            for layer in cat["layers"]:
                if layer["id"] in catalog_updates:
                    render_min, render_max = catalog_updates[layer["id"]]
                    layer["render_min"] = render_min
                    layer["render_max"] = render_max
        with open(CATALOG_PATH, "w") as f:
            json.dump(on_disk, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[overview] catalog updated: {len(catalog_updates)} layer(s) -> {CATALOG_PATH}")

    print(f"[overview] done  total={total}  updated={updated}  skipped={skipped}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fill", action="store_true",
        help="Also zero-fill nodata pixels for ZERO_NODATA_LAYERS (reads each "
             "matching layer's full band every run until its nodata flag is "
             "cleared — off by default, opt in when actually needed).",
    )
    args = parser.parse_args()
    main(fill=args.fill)
