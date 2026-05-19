"""
Build internal overviews for all GeoTIFFs in data/gis/layers/, converting
each to a Cloud Optimized GeoTIFF (COG) with appropriate overview levels.

Overview resampling is chosen by value_type:
  interval / ratio → average
  nominal          → nearest  (preserves discrete class values)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import rasterio

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


def _is_nominal(layer: dict | None) -> bool:
    if not layer:
        return False
    return str(layer.get("value_type") or "").lower() == "nominal"


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


def _build_cog(src_path: Path, dst_path: Path, *, nominal: bool, overview_factors: list[int]) -> None:
    resampling = "nearest" if nominal else "average"
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
        nominal = _is_nominal(layer)

        try:
            with rasterio.open(path) as ds:
                existing = ds.overviews(1) or []
                desired = _overview_factors_for_dataset(ds)

            if existing and _has_required_overviews(existing, desired):
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
