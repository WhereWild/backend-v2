#!/usr/bin/env python3
"""
Mosaic GLC_FCS30 tiles into a single global COG.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from util.config import load_config

CONFIG = load_config("global")

landcover_clip_prefix = "landcover_clip_"

landcover_coord_pattern = r"([EW])(\d+)([NS])(\d+)"

landcover_global_filename = "landcover_global.tif"

landcover_source_pattern = "GLC_FCS30_2020_*.tif"

landcover_tile_limit = None


COORD_PATTERN = re.compile(landcover_coord_pattern, re.IGNORECASE)


@dataclass(frozen=True)
class Tile:
    path: Path
    west: float
    south: float
    east: float
    north: float


def _run_cmd(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with code {completed.returncode}:\n{completed.stderr}"
        )


def _parse_bounds(path: Path) -> tuple[float, float, float, float]:
    match = COORD_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Unable to parse bounds for {path}")
    lon_dir, lon_val, lat_dir, lat_val = match.groups()
    lon = int(lon_val)
    lat = int(lat_val)
    west = -lon if lon_dir.upper() == "W" else lon
    north = -lat if lat_dir.upper() == "S" else lat
    south = north - 5
    return west, south, west + 5, north


def iter_tiles(source_dir: Path, pattern: str) -> List[Tile]:
    tiles: List[Tile] = []
    for path in sorted(source_dir.glob(pattern)):
        if path.is_dir():
            continue
        west, south, east, north = _parse_bounds(path)
        tiles.append(Tile(path=path, west=west, south=south, east=east, north=north))
    return tiles


def _clip_tile(tile: Tile, clip_dir: Path) -> Path:
    clip_dir.mkdir(parents=True, exist_ok=True)
    target = clip_dir / tile.path.name
    if target.exists():
        return target
    _run_cmd(
        [
            "gdalwarp",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-r",
            "near",
            "-srcnodata",
            "0",
            "-dstnodata",
            "0",
            "-te",
            str(tile.west),
            str(tile.south),
            str(tile.east),
            str(tile.north),
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
            str(tile.path),
            str(target),
        ]
    )
    return target


def _build_global_cog(tile_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        vrt_path = tmp_dir_path / "mosaic.vrt"
        tmp_warp = tmp_dir_path / "mosaic.tif"

        _run_cmd(
            [
                "gdalbuildvrt",
                "-overwrite",
                "-resolution",
                "highest",
                str(vrt_path),
                *[str(p) for p in tile_paths],
            ]
        )

        _run_cmd(
            [
                "gdalwarp",
                "-overwrite",
                "-multi",
                "-wo",
                "NUM_THREADS=ALL_CPUS",
                "-r",
                "near",
                "-srcnodata",
                "0",
                "-dstnodata",
                "0",
                "-te",
                "-180",
                "-90",
                "180",
                "90",
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
        )

        if out_path.exists():
            out_path.unlink()
        shutil.move(tmp_warp, out_path)


def main() -> None:
    source_dir = CONFIG.gis_landcover_root.expanduser().resolve()
    if not source_dir.exists():
        print(f"source dir {source_dir} missing")
        return

    tiles = iter_tiles(source_dir, landcover_source_pattern)
    if not tiles:
        print(f"no tiles found in {source_dir}")
        return

    if landcover_tile_limit:
        tiles = tiles[: landcover_tile_limit]
        print(f"Limiting to first {len(tiles)} tiles for debugging.")

    clip_dir = Path(
        tempfile.mkdtemp(prefix=landcover_clip_prefix, dir=str(CONFIG.gis_root))
    )
    try:
        clipped_paths = [_clip_tile(tile, clip_dir) for tile in tiles]
        print(f"Building global mosaic with {len(clipped_paths)} tiles…")
        output_path = CONFIG.gis_root / landcover_global_filename
        _build_global_cog([Path(p) for p in clipped_paths], output_path)
        print(f"Wrote {output_path}")
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
