# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download two grids from GLWD v2 (Lehner et al. 2025) and recompress each as
a global COG:
  - glwd.tif      nominal dominant wetland class per pixel (0=dryland, 1-33)
  - glwd_pct.tif  ratio, total wetland cover per pixel in percent (0-100)

GLWD v2's "combined_classes" product bundles these (plus two grids we don't
need — absolute area in hectares, and a 50%-dominance variant of the class
grid) as a single ~925MB zip. Both source grids are already global,
15 arc-second, EPSG:4326 GeoTIFFs (see
https://www.hydrosheds.org/products/glwd) — no tiling/mosaicking needed,
unlike landcover/DEM. We deliberately skip the dataset's other product
(33 individual per-class sub-cell-fraction grids, one per wetland type) —
"dominant class" + "% wetland cover" is enough signal without carrying 33
near-always-mostly-zero layers through the whole per-taxon stats pipeline.

Uses remotezip (HTTP range requests) to pull only the two needed TIFs
out of the remote zip rather than downloading the full archive.

Usage (inside the gdal container):
    uv run python scripts/gis/download_glwd.py [--force]
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from remotezip import RemoteZip

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

# figshare's redirect/presigned-URL flow expires in ~10s, too short-lived to
# use directly with remotezip's multi-request central-directory scan — this
# stable ndownloader URL is what remotezip needs to be pointed at instead.
ZIP_URL = "https://ndownloader.figshare.com/files/54001814"  # GLWD_v2_0_combined_classes_tif.zip
CLASS_MEMBER = "GLWD_v2_0_combined_classes/GLWD_v2_0_main_class.tif"
PCT_MEMBER   = "GLWD_v2_0_combined_classes/GLWD_v2_0_area_pct.tif"

RAW_DIR    = Path("data/gis/glwd_raw")
LAYERS_DIR = Path("data/gis/layers")
CLASS_OUT  = LAYERS_DIR / "glwd.tif"
PCT_OUT    = LAYERS_DIR / "glwd_pct.tif"


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _fetch_member(member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  Fetching {member} ...", flush=True)
    with RemoteZip(ZIP_URL) as rz:
        data = rz.read(member)
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _detect_nodata(path: Path) -> float | None:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.nodata


def _build_cog(src: Path, out_path: Path) -> None:
    # gdal_translate writes into a local tempdir (fast, reliable seek-back
    # for tiled BigTIFF output), then the finished file is landed in
    # data/gis/layers/ via a plain byte copy rather than os.rename/replace —
    # the tempdir and data/gis/layers/ can be different filesystems (e.g. a
    # WSL2 bind mount), and cross-device rename raises EXDEV. See the
    # matching comment in download_ecoregions.py._rasterize for the same
    # constraint on this host.
    import shutil

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nodata = _detect_nodata(src)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / out_path.name
        cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=ALL_CPUS",
        ]
        if nodata is not None:
            cmd += ["-a_nodata", str(nodata)]
        cmd += [str(src), str(tmp_out)]
        _run(cmd)

        dest_tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        dest_tmp.unlink(missing_ok=True)
        with tmp_out.open("rb") as fsrc, dest_tmp.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=64 * 1024 * 1024)
    out_path.unlink(missing_ok=True)
    dest_tmp.replace(out_path)


def main(force: bool = False) -> None:
    target_ids = ("glwd", "glwd_pct")
    if VARS_TO_DOWNLOAD is not None and not any(v in VARS_TO_DOWNLOAD for v in target_ids):
        print("[download_glwd] skipped (glwd/glwd_pct not in VARS_TO_DOWNLOAD)")
        return

    outputs_done = CLASS_OUT.exists() and PCT_OUT.exists()
    if outputs_done and not force:
        print(f"[skip] COGs already exist: {CLASS_OUT}, {PCT_OUT} (--force to rebuild)")
        return

    class_raw = RAW_DIR / "GLWD_v2_0_main_class.tif"
    pct_raw   = RAW_DIR / "GLWD_v2_0_area_pct.tif"

    print("Downloading GLWD v2 combined-classes grids (remotezip range requests)...")
    _fetch_member(CLASS_MEMBER, class_raw)
    _fetch_member(PCT_MEMBER, pct_raw)

    print(f"Building COG → {CLASS_OUT}")
    _build_cog(class_raw, CLASS_OUT)

    print(f"Building COG → {PCT_OUT}")
    _build_cog(pct_raw, PCT_OUT)

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GLWD v2 dominant-class + wetland-percent grids")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs already exist")
    args = parser.parse_args()
    main(force=args.force)
