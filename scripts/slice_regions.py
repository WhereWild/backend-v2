'''
A simple script to "slice" a tif into regions (10x10 deg chunks) and write them into the regions folder.
'''

import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from util.config import load_config

CONFIG = load_config("global")

gis_global_inputs = [
            ("landcover_global.tif", "landcover.tif"),
            ("koppen_geiger.tif", "koppen_geiger.tif"),
            ("landform.tif", "landform.tif"),
            ("lithology.tif", "lithology.tif"),
            ("clay.tif", "clay.tif"),
            ("sand.tif", "sand.tif"),
            ("silt.tif", "silt.tif"),
            ("cfvo.tif", "cfvo.tif"),
            ("wrb.tif", "wrb.tif"),
            # CHELSA variables
            ("clt.tif", "clt.tif"),
            ("rsds.tif", "rsds.tif"),
            ("scd.tif", "scd.tif"),
            ("sfc.tif", "sfc.tif"),
            ("swe.tif", "swe.tif"),
            ("vpd.tif", "vpd.tif"),
            ("soc.tif", "soc.tif"),
            ("phh20.tif", "phh20.tif"),
            ("nitrogen.tif", "nitrogen.tif"),            
        ]

slice_region_lat_end = 90

slice_region_lat_start = -90

slice_region_lon_end = 180

slice_region_lon_start = -180

slice_region_step = 10

slice_sentinel_threshold = 1e30


regions = [
    (south, west)
    for south in range(
        slice_region_lat_start,
        slice_region_lat_end,
        slice_region_step,
    )
    for west in range(
        slice_region_lon_start,
        slice_region_lon_end,
        slice_region_step,
    )
]

os.makedirs(CONFIG.gis_regions_root, exist_ok=True)

bioclim_tifs = []
if CONFIG.bioclim_root.exists():
    bioclim_tifs = [
        CONFIG.bioclim_root / f for f in os.listdir(CONFIG.bioclim_root) if f.lower().endswith(".tif")
    ]
sources: list[tuple[Path, str]] = [(path, path.name) for path in bioclim_tifs]

for item in gis_global_inputs:
    filename, out_name = item if isinstance(item, tuple) else (item, item)
    src_path = CONFIG.gis_root / filename
    if src_path.exists():
        sources.append((src_path, out_name))
    else:
        print(f"warning: {src_path} missing; skipping {out_name} slicing")

total_regions = len(regions)
region_done = 0


def ensure_nodata(src_label, south, west, nodata_value, sample):
    has_extreme = np.any(np.abs(sample) > slice_sentinel_threshold)
    if nodata_value is None and has_extreme:
        raise ValueError(
            f"{src_label} is missing nodata metadata but tile "
            f"lat{south}_lon{west} contains extreme sentinel-like values."
        )

def aligned_block_size(size: int, default: int = 512) -> int:
    # Find a good block (tile) size for writing that is aligned.
    target = min(size, default)
    if target <= 0:
        return 1
    if target < 16:
        return target
    aligned = (target // 16) * 16
    if aligned == 0:
        aligned = 16
    if aligned > size:
        aligned = size
    return aligned


def slice_into_region(src_path: Path, out_path: Path, south: int, west: int):
    '''Simply slice a COG into a 10 by 10 degree region'''
    north = south + 10
    east = west + 10

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".tmp", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(src_path) as src:
            window = from_bounds(west, south, east, north, src.transform)

            profile = src.profile.copy()
            block_width = aligned_block_size(window.width)
            block_height = aligned_block_size(window.height)
            profile.update(
                width=window.width,
                height=window.height,
                transform=src.window_transform(window),
                nodata=src.nodata,
                compress="deflate",
                tiled=True,
                blockxsize=block_width,
                blockysize=block_height,
            )
            nodata_value = profile.get("nodata")

            with rasterio.open(tmp_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    data = src.read(i, window=window)
                    if i == 1:
                        ensure_nodata(
                            src_path.name, south, west, nodata_value, data
                        )
                    dst.write(data, i)

        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def cleanup_tmp_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".tmp"):
                os.remove(os.path.join(dirpath, name))

cleanup_tmp_files(CONFIG.gis_regions_root)

for south, west in regions:
    # Iter over and slice
    region_dir = CONFIG.gis_regions_root / f"lat{south}_lon{west}"
    region_dir.mkdir(parents=True, exist_ok=True)

    all_done = True

    for src_path, out_name in sources:
        out_path = region_dir / out_name

        if not out_path.exists():
            all_done = False
            slice_into_region(src_path, out_path, south, west)

    region_done += 1
    status = "skipped" if all_done else "written"
    print(f"[{region_done}/{total_regions}] {status} region lat{south}_lon{west}")
