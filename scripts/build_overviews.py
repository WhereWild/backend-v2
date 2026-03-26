"""
Build internal overviews for all region GeoTIFFs.

Usage (inside container or with rasterio installed):
  python scripts/build_overviews.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os
import subprocess
import math

import rasterio

from util.config import load_config
from util import gis_lookup


CONFIG = load_config("global")

FORCED_CATEGORICAL = {"landcover", "koppen_geiger"}
TARGET_MIN_ZOOM = 3
TARGET_TILE_SIZE = CONFIG.sdm_tile_size
MAX_OVERVIEW_FACTOR = 2048
OVERVIEW_FACTOR_TOLERANCE_RATIO = 0.03
OVERVIEW_FACTOR_TOLERANCE_MIN = 2


def _iter_tifs(root: Path) -> Iterable[Path]:
    return root.rglob("*.tif")


def _resolve_layer_id(path: Path, meta: dict[str, dict]) -> str | None:
    filename = path.name
    stem = path.stem

    # Direct stem match for templates like "{id}.tif"
    if stem in meta:
        return stem

    # Match fixed filename templates like "landcover.tif" / "dem.tif"
    for layer_id, entry in meta.items():
        if entry.get("derived"):
            continue
        template = entry.get("filename_template") or ""
        if "{id}" in template:
            continue
        if template == filename:
            return layer_id
    return None


def _is_categorical(layer_id: str | None, meta: dict[str, dict]) -> bool:
    if not layer_id:
        return False
    if layer_id in FORCED_CATEGORICAL:
        return True
    value_type = str(meta.get(layer_id, {}).get("value_type") or "").lower()
    return value_type == "categorical"


def _build_cog(
    src_path: Path,
    dst_path: Path,
    *,
    categorical: bool,
    overview_factors: list[int],
) -> None:
    resampling = "nearest" if categorical else "average"
    base_tif = dst_path.with_suffix(dst_path.suffix + ".base.tif")
    try:
        # Build a regular tiled GTiff first so gdaladdo can safely write overviews.
        base_cmd = [
            "gdal_translate",
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
            str(src_path),
            str(base_tif),
        ]
        subprocess.run(base_cmd, check=True)

        if overview_factors:
            addo_cmd = [
                "gdaladdo",
                "-r",
                resampling,
                str(base_tif),
                *[str(value) for value in overview_factors],
            ]
            subprocess.run(addo_cmd, check=True)

        # Convert to final COG and force using existing overviews from base GTiff.
        cog_cmd = [
            "gdal_translate",
            "-of",
            "COG",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            "OVERVIEWS=FORCE_USE_EXISTING",
            "-co",
            f"OVERVIEW_RESAMPLING={resampling.upper()}",
            str(base_tif),
            str(dst_path),
        ]
        subprocess.run(cog_cmd, check=True)
    finally:
        if base_tif.exists():
            base_tif.unlink()


def _target_dst_res_degrees() -> float:
    return 360.0 / ((2**TARGET_MIN_ZOOM) * TARGET_TILE_SIZE)


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
    """Return True when existing overview factors sufficiently match desired factors."""
    if not desired:
        return True
    if not existing:
        return False
    for target in desired:
        if not any(_overview_factor_close(actual, target) for actual in existing):
            return False
    return True


def main() -> None:
    regions_root = CONFIG.gis_regions_root
    if not regions_root.exists():
        raise FileNotFoundError(f"Regions root not found: {regions_root}")

    meta = gis_lookup.load_layer_metadata()

    total = 0
    updated = 0
    skipped = 0

    for path in _iter_tifs(regions_root):
        total += 1
        try:
            layer_id = _resolve_layer_id(path, meta)
            categorical = _is_categorical(layer_id, meta)
            with rasterio.open(path) as ds:
                existing = ds.overviews(1) or []
                desired_factors = _overview_factors_for_dataset(ds)
                needs_more = not _has_required_overviews(existing, desired_factors)
                if existing and not needs_more:
                    skipped += 1
                    if skipped % 500 == 0:
                        print(f"[overview] skipped {skipped} files (already have overviews)")
                    continue
                if needs_more:
                    print(
                        f"[overview] upgrading {path.name} existing={existing} target={desired_factors}",
                    )

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            _build_cog(
                path,
                tmp_path,
                categorical=categorical,
                overview_factors=desired_factors,
            )
            os.replace(tmp_path, path)
            updated += 1
            if updated % 100 == 0:
                print(f"[overview] rebuilt {updated} files (last: {path.name})")
        except Exception as exc:
            print(f"[overview] failed {path}: {exc}")
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

    print(f"[overview] done total={total} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
