"""
Download FABDEM V1-2 tiles from the University of Bristol data repository,
extract 1°×1° GeoTIFFs, and stitch them into 10°×10° region COGs.

Steps:
  1. Download the single FABDEM ZIP from the Bristol dataset page
  2. Extract 1°×1° FABDEM TIFs to data/gis/dem_raw_tiles/
  3. Group tiles into 10°×10° regions
  4. Build region COG: VRT → gdalwarp (clip + resample) → gdal_translate (COG)

Output: data/gis/regions/lat{lat}_lon{lon}/elevation.tif

Re-running is safe: already-downloaded ZIPs, already-extracted tiles, and
already-built region COGs are all skipped unless OVERWRITE = True.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

DATASET_URL = "https://data.bris.ac.uk/datasets/tar/s5hqmjcdj8yo2ibzi9b4ew3sn.zip"
ZIP_FILENAME = "FABDEM_V1-2.zip"

RAW_ZIPS_DIR  = Path("data/gis/dem_raw_zips")
RAW_TILES_DIR = Path("data/gis/dem_raw_tiles")
REGIONS_DIR   = Path("data/gis/regions")
REGION_FILENAME = "elevation.tif"

TILE_PATTERN = re.compile(
    r"^([NS])(\d{1,2})([EW])(\d{1,3})_FABDEM_V1-2\.tif$",
    re.IGNORECASE,
)

OVERWRITE: bool = False
REGION_LIMIT: int | None = None


@dataclass(frozen=True)
class Tile:
    path: Path
    lat: int
    lon: int


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _region_origin(v: int) -> int:
    return math.floor(v / 10) * 10


def _parse_tile(path: Path) -> Tile | None:
    m = TILE_PATTERN.match(path.name)
    if not m:
        return None
    lat_dir, lat_val, lon_dir, lon_val = m.groups()
    lat = int(lat_val) * (1 if lat_dir.upper() == "N" else -1)
    lon = int(lon_val) * (1 if lon_dir.upper() == "E" else -1)
    return Tile(path=path, lat=lat, lon=lon)




def _download_zip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=16",
            "--max-connection-per-server=16",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ],
        check=True,
    )


def _extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not TILE_PATTERN.match(name):
                continue
            dest = out_dir / name
            if not dest.exists():
                dest.write_bytes(zf.read(member))
            extracted.append(dest)
    return extracted


def _detect_nodata(path: Path) -> float | None:
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as ds:
            return ds.nodata
    except Exception:
        return None


def _build_region_cog(
    tile_paths: list[Path],
    out_path: Path,
    bounds: tuple[int, int, int, int],
    nodata: float | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tif.tmp")
    west, south, east, north = bounds

    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        vrt_path = tmp_dir_path / "region.vrt"
        tmp_warp = tmp_dir_path / "region.tif"

        _run([
            "gdalbuildvrt", "-overwrite", "-resolution", "highest",
            str(vrt_path), *[str(p) for p in tile_paths],
        ])

        warp_cmd = [
            "gdalwarp", "-overwrite",
            "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
            "-r", "near",
            "-te", str(west), str(south), str(east), str(north),
            "-te_srs", "EPSG:4326",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=512",
            "-co", "BLOCKYSIZE=512",
            "-co", "BIGTIFF=YES",
            str(vrt_path), str(tmp_warp),
        ]
        if nodata is not None:
            warp_cmd[1:1] = ["-srcnodata", str(nodata), "-dstnodata", str(nodata)]
        _run(warp_cmd)

        _run([
            "gdal_translate", "-of", "COG",
            "-co", "COMPRESS=DEFLATE",
            "-co", "BLOCKSIZE=512",
            "-co", "BIGTIFF=YES",
            "-r", "average",
            str(tmp_warp), str(tmp_out),
        ])

    if out_path.exists():
        out_path.unlink()
    tmp_out.replace(out_path)


def main() -> None:
    if VARS_TO_DOWNLOAD is not None and "elevation" not in VARS_TO_DOWNLOAD:
        print("[download_dem] skipped (elevation not in VARS_TO_DOWNLOAD)")
        return

    zip_dest = RAW_ZIPS_DIR / ZIP_FILENAME
    aria2_control = zip_dest.with_suffix(zip_dest.suffix + ".aria2")
    if not zip_dest.exists() or aria2_control.exists():
        print(f"Downloading {DATASET_URL} ...")
        _download_zip(DATASET_URL, zip_dest)
    else:
        print(f"ZIP already downloaded: {zip_dest}")

    print("Extracting tiles...")
    extracted = _extract_zip(zip_dest, RAW_TILES_DIR)

    all_tiles: list[Tile] = []
    for path in extracted:
        tile = _parse_tile(path)
        if tile:
            all_tiles.append(tile)

    print(f"\nCollected {len(all_tiles)} 1°×1° tiles")

    tiles_by_region: dict[tuple[int, int], list[Path]] = {}
    for tile in all_tiles:
        key = (_region_origin(tile.lat), _region_origin(tile.lon))
        tiles_by_region.setdefault(key, []).append(tile.path)

    region_keys = sorted(tiles_by_region)
    if REGION_LIMIT:
        region_keys = region_keys[:REGION_LIMIT]
        print(f"Limited to first {len(region_keys)} regions for debugging")

    nodata: float | None = None
    if all_tiles:
        nodata = _detect_nodata(all_tiles[0].path)
        if nodata is not None:
            print(f"Detected nodata value: {nodata}")

    total = len(region_keys)
    print(f"\nBuilding {total} region COGs...\n")

    for idx, (lat0, lon0) in enumerate(region_keys, 1):
        out_dir  = REGIONS_DIR / f"lat{lat0}_lon{lon0}"
        out_path = out_dir / REGION_FILENAME
        tmp_out  = out_path.with_suffix(".tif.tmp")

        if tmp_out.exists() and not out_path.exists():
            print(f"[{idx}/{total}] removing stale tmp for lat{lat0}_lon{lon0}")
            tmp_out.unlink()

        if out_path.exists() and not OVERWRITE:
            print(f"[{idx}/{total}] skip lat{lat0}_lon{lon0} (exists)")
            continue

        tile_paths = sorted(tiles_by_region[(lat0, lon0)])
        bounds = (lon0, lat0, lon0 + 10, lat0 + 10)
        print(f"[{idx}/{total}] building lat{lat0}_lon{lon0} ({len(tile_paths)} tiles)...")
        _build_region_cog(tile_paths, out_path, bounds, nodata)
        print(f"[{idx}/{total}] wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
