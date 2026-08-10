# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build internal overviews for all GeoTIFFs in data/gis/layers/, converting
each to a Cloud Optimized GeoTIFF (COG) with appropriate overview levels.

Overview resampling is chosen by value_type:
  interval / ratio → average
  nominal          → nearest  (preserves discrete class values)

Also the one place config.ZERO_NODATA_LAYERS actually gets acted on: for
those layers, nodata pixels are burned in as a real 0 (and the nodata flag
cleared) before overviews are built from the corrected data. Doing it here
rather than in each download script means every consumer benefits from one
fix — map tiles render a real "no snow"-equivalent color with no
transparent gaps, the /gis/point background-point endpoint returns 0
instead of "No data", and enrich_tree.py's own nodata check becomes a
harmless no-op for these layers (ds.nodata reads back as None).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import rasterio

from config.config import ZERO_NODATA_LAYERS

CATALOG_PATH        = Path("config/gis/catalog.json")
LAYERS_DIR          = Path("data/gis/layers")
TARGET_MIN_ZOOM     = 3
TARGET_TILE_SIZE    = 256
MAX_OVERVIEW_FACTOR = 2048

OVERVIEW_FACTOR_TOLERANCE_RATIO = 0.03
OVERVIEW_FACTOR_TOLERANCE_MIN   = 2


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


def main() -> None:
    if not LAYERS_DIR.exists():
        raise FileNotFoundError(f"Layers directory not found: {LAYERS_DIR}")

    layer_meta = _load_layer_meta()
    total = updated = skipped = 0

    for path in sorted(LAYERS_DIR.glob("*.tif")):
        total += 1
        layer = layer_meta.get(path.name)
        nominal = _is_class_based(layer)
        layer_id = layer.get("id") if layer else None

        try:
            force_rebuild = False
            if layer_id in ZERO_NODATA_LAYERS and _fill_nodata_with_zero(path):
                print(f"[overview] zero-filled nodata -> {path.name}")
                force_rebuild = True

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

    print(f"[overview] done  total={total}  updated={updated}  skipped={skipped}")


if __name__ == "__main__":  # pragma: no cover
    main()
