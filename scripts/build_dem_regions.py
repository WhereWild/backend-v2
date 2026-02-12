#!/usr/bin/env python3
"""
Build 10x10 degree DEM region COGs from 1x1 FABDEM tiles.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from util.config import load_config

CONFIG = load_config("global")

SOURCE_DIR = CONFIG.project_root / "data" / "dem_all_raw_tiles"
OUT_ROOT = CONFIG.gis_regions_root
OUTPUT_NAME = "dem.tif"
TILE_GLOB = "*_FABDEM_V1-2.tif"
NODATA_OVERRIDE: float | None = None
RESAMPLING = "near"
OVERVIEW_RESAMPLING = "average"
OVERWRITE = False
REGION_LIMIT: int | None = None

TILE_NAME_PATTERN = re.compile(
    r"^([NS])(\d{1,2})([EW])(\d{1,3})_FABDEM_V1-2\.tif$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Tile:
    path: Path
    lat: int
    lon: int
    region_lat: int
    region_lon: int


def _run_cmd(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with code {completed.returncode}:\n{completed.stderr}"
        )


def _region_origin(value: int) -> int:
    return math.floor(value / 10) * 10


def _parse_tile(path: Path) -> Tile | None:
    match = TILE_NAME_PATTERN.match(path.name)
    if not match:
        return None
    lat_dir, lat_val, lon_dir, lon_val = match.groups()
    lat = int(lat_val) * (1 if lat_dir.upper() == "N" else -1)
    lon = int(lon_val) * (1 if lon_dir.upper() == "E" else -1)
    region_lat = _region_origin(lat)
    region_lon = _region_origin(lon)
    return Tile(path=path, lat=lat, lon=lon, region_lat=region_lat, region_lon=region_lon)


def _iter_tiles(source_dir: Path, pattern: str) -> List[Tile]:
    tiles: List[Tile] = []
    for path in sorted(source_dir.glob(pattern)):
        if path.is_dir():
            continue
        tile = _parse_tile(path)
        if tile is None:
            continue
        tiles.append(tile)
    return tiles


def _detect_nodata(sample_path: Path) -> float | None:
    try:
        import rasterio  # type: ignore
    except Exception:
        return None
    with rasterio.open(sample_path) as ds:
        return ds.nodata


def _build_region_cog(
    tiles: Iterable[Path],
    out_path: Path,
    bounds: Tuple[int, int, int, int],
    nodata: float | None,
    resampling: str,
    overview_resampling: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    west, south, east, north = bounds

    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        vrt_path = tmp_dir_path / "region.vrt"
        tmp_warp = tmp_dir_path / "region.tif"

        _run_cmd(
            [
                "gdalbuildvrt",
                "-overwrite",
                "-resolution",
                "highest",
                str(vrt_path),
                *[str(p) for p in tiles],
            ]
        )

        warp_cmd = [
            "gdalwarp",
            "-overwrite",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-r",
            resampling,
            "-te",
            str(west),
            str(south),
            str(east),
            str(north),
            "-te_srs",
            "EPSG:4326",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
            "-co",
            "BIGTIFF=YES",
            str(vrt_path),
            str(tmp_warp),
        ]
        if nodata is not None:
            warp_cmd[1:1] = ["-srcnodata", str(nodata), "-dstnodata", str(nodata)]

        _run_cmd(warp_cmd)

        translate_cmd = [
            "gdal_translate",
            "-of",
            "COG",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BLOCKSIZE=512",
            "-co",
            "BIGTIFF=YES",
            "-r",
            overview_resampling,
            str(tmp_warp),
            str(tmp_out),
        ]
        _run_cmd(translate_cmd)
        if out_path.exists():
            out_path.unlink()
        tmp_out.replace(out_path)


def main() -> None:
    source_dir = SOURCE_DIR.expanduser().resolve()
    out_root = OUT_ROOT.expanduser().resolve()

    if not source_dir.exists():
        raise SystemExit(f"source dir {source_dir} missing")

    tiles = _iter_tiles(source_dir, TILE_GLOB)
    if not tiles:
        raise SystemExit(f"no tiles found in {source_dir} (pattern {TILE_GLOB})")

    nodata = NODATA_OVERRIDE
    if nodata is None:
        nodata = _detect_nodata(tiles[0].path)

    tiles_by_region: Dict[Tuple[int, int], List[Path]] = {}
    for tile in tiles:
        key = (tile.region_lat, tile.region_lon)
        tiles_by_region.setdefault(key, []).append(tile.path)

    region_keys = sorted(tiles_by_region.keys())
    if REGION_LIMIT:
        region_keys = region_keys[: REGION_LIMIT]
        print(f"Limiting to first {len(region_keys)} regions for debugging.")

    total = len(region_keys)
    for idx, (lat0, lon0) in enumerate(region_keys, 1):
        out_dir = out_root / f"lat{lat0}_lon{lon0}"
        out_path = out_dir / OUTPUT_NAME
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_out.exists() and not out_path.exists():
            print(
                f"[{idx}/{total}] stale tmp for lat{lat0}_lon{lon0}; removing"
            )
            tmp_out.unlink()
        if out_path.exists() and not OVERWRITE:
            print(f"[{idx}/{total}] skip lat{lat0}_lon{lon0} (exists)")
            continue

        bounds = (lon0, lat0, lon0 + 10, lat0 + 10)
        tile_paths = sorted(tiles_by_region[(lat0, lon0)])
        print(f"[{idx}/{total}] build lat{lat0}_lon{lon0} ({len(tile_paths)} tiles)")
        _build_region_cog(
            tile_paths,
            out_path,
            bounds,
            nodata,
            RESAMPLING,
            OVERVIEW_RESAMPLING,
        )
        print(f"[{idx}/{total}] wrote lat{lat0}_lon{lon0}")


if __name__ == "__main__":
    main()
