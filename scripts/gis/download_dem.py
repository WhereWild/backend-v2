"""
Download FABDEM V1-2 tiles from the University of Bristol data repository,
extract 1°×1° GeoTIFFs, and stitch them into a single global COG.

Steps:
  1. Download the single FABDEM ZIP from the Bristol dataset page
  2. Extract 1°×1° FABDEM TIFs to data/gis/dem_raw_tiles/  (atomic: tmp→rename)
  3. Build a VRT spanning all tiles
  4. gdal_translate -of COG → data/gis/layers/elevation.tif

Re-running is safe: already-extracted tiles and the final COG are skipped
unless --force is passed.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

DATASET_URL = "https://data.bris.ac.uk/datasets/tar/s5hqmjcdj8yo2ibzi9b4ew3sn.zip"
ZIP_FILENAME = "FABDEM_V1-2.zip"

RAW_ZIPS_DIR  = Path("data/gis/dem_raw_zips")
RAW_TILES_DIR = Path("data/gis/dem_raw_tiles")
LAYERS_DIR    = Path("data/gis/layers")
OUT_PATH      = LAYERS_DIR / "elevation.tif"

TILE_PATTERN = re.compile(
    r"^([NS])(\d{1,2})([EW])(\d{1,3})_FABDEM_V1-2\.tif$",
    re.IGNORECASE,
)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


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


def _extract_tiles(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract 1°×1° TIF tiles from the outer zip-of-zips.

    Checks disk first — if tiles already exist in out_dir, skips opening the
    outer zip entirely. Writes atomically (tmp → rename) so a killed process
    never leaves a partial tile that would be silently skipped on re-run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in out_dir.iterdir() if TILE_PATTERN.match(p.name)]
    if existing:
        print(f"  Found {len(existing)} tiles already on disk, skipping extraction.")
        return existing

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as outer:
        inner_zips = [m for m in outer.namelist() if m.endswith(".zip")]
        total = len(inner_zips)
        for i, member in enumerate(inner_zips, 1):
            print(f"  [{i}/{total}] {Path(member).name}", flush=True)
            inner_data = outer.read(member)
            with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
                for tile_member in inner.namelist():
                    name = Path(tile_member).name
                    if not TILE_PATTERN.match(name):
                        continue
                    dest = out_dir / name
                    if dest.exists():
                        extracted.append(dest)
                        continue
                    tmp = dest.with_suffix(".tif.tmp")
                    tmp.write_bytes(inner.read(tile_member))
                    tmp.replace(dest)
                    extracted.append(dest)
    return extracted


def _detect_nodata(path: Path) -> float | None:
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as ds:
            return ds.nodata
    except Exception:
        return None


def _build_global_cog(tile_paths: list[Path], out_path: Path, nodata: float | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tif.tmp")

    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt_path = Path(tmp_dir) / "global.vrt"

        print(f"Building VRT from {len(tile_paths)} tiles...", flush=True)
        _run([
            "gdalbuildvrt", "-overwrite", "-resolution", "highest",
            str(vrt_path), *[str(p) for p in sorted(tile_paths)],
        ])

        print("Translating to GeoTIFF...", flush=True)
        translate_cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=4",
        ]
        if nodata is not None:
            translate_cmd += ["-a_nodata", str(nodata)]
        translate_cmd += [str(vrt_path), str(tmp_out)]
        _run(translate_cmd)

    if out_path.exists():
        out_path.unlink()
    tmp_out.replace(out_path)


def main(force: bool = False) -> None:
    if not force and VARS_TO_DOWNLOAD is not None and "elevation" not in VARS_TO_DOWNLOAD:
        print("[download_dem] skipped (elevation not in VARS_TO_DOWNLOAD)")
        return

    zip_dest = RAW_ZIPS_DIR / ZIP_FILENAME
    aria2_control = zip_dest.with_suffix(zip_dest.suffix + ".aria2")
    if not zip_dest.exists() or aria2_control.exists():
        print(f"Downloading {DATASET_URL} ...")
        _download_zip(DATASET_URL, zip_dest)
    else:
        print(f"ZIP already downloaded: {zip_dest}")

    if OUT_PATH.exists() and not force:
        print(f"COG already exists: {OUT_PATH} (use --force to rebuild)")
        return

    print("Extracting tiles...")
    tile_paths = _extract_tiles(zip_dest, RAW_TILES_DIR)
    print(f"Collected {len(tile_paths)} 1°×1° tiles")

    if not tile_paths:
        print("No tiles found — aborting.")
        return

    nodata = _detect_nodata(tile_paths[0])
    if nodata is not None:
        print(f"Detected nodata value: {nodata}")

    print(f"\nBuilding global COG → {OUT_PATH}")
    _build_global_cog(tile_paths, OUT_PATH, nodata)
    print(f"\nDone. Wrote {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download FABDEM elevation tiles")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output already exists")
    args = parser.parse_args()
    main(force=args.force)
