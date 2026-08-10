# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download soil layers defined in config/gis/catalog.json.

Supports two sources:

  soilgrids_2_0 — SoilGrids 2.0 VRTs at files.isric.org. Native projection is
    Interrupted Goode's Homolosine, so each layer is warped to EPSG:4326 via
    gdalwarp in one streaming pass. No raw tiles are stored locally.

  isric_salinity_2016 — Global Soil Salinity Map (Ivushkin et al. 2019) at
    files.isric.org. Already in EPSG:4326, so no reprojection is needed —
    translated directly to a COG via gdal_translate.

render_min/render_max are NOT computed here — scripts/gis/build_overviews.py
computes those for every continuous layer as a percentile of valid pixel
values (config.PERCENTILE_RENDER_BOUNDS) and runs right after this stage in
the rebuild pipeline, so anything set here would be immediately overwritten.

Re-running skips files that already exist (use --force to rebuild).

Usage (inside the gdal container):
    uv run python scripts/gis/download_soil.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

CATALOG_PATH     = Path("config/gis/catalog.json")
LAYERS_DIR       = Path("data/gis/layers")
SOURCE_ID_SG     = "soilgrids_2_0"
SOURCE_ID_SAL    = "isric_salinity"
_SOURCES         = {SOURCE_ID_SG, SOURCE_ID_SAL}

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

# Pixel size in degrees (~250 m at equator), used for SoilGrids warp only
_TR = str(1 / 480)

_HTTP_CONFIG = [
    "--config", "GDAL_HTTP_UNSAFESSL", "YES",
    "--config", "GDAL_HTTP_MAX_RETRY", "5",
    "--config", "GDAL_HTTP_RETRY_DELAY", "15",
    "--config", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.vrt",
]


# ── Catalog ───────────────────────────────────────────────────────────────────

def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _soil_layers(catalog: dict) -> list[dict]:
    return [
        layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") in _SOURCES and layer.get("filename")
    ]


# ── GDAL ──────────────────────────────────────────────────────────────────────

def _gdalwarp(vrt_url: str, dest: Path, resampling: str) -> None:
    tmp = dest.with_suffix(".tif.tmp")
    cmd = [
        "gdalwarp",
        *_HTTP_CONFIG,
        "-t_srs", "EPSG:4326",
        "-tr", _TR, _TR,
        "-r", resampling,
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=256",
        "-co", "BLOCKYSIZE=256",
        "-co", "BIGTIFF=YES",
        "-co", "NUM_THREADS=ALL_CPUS",
        f"/vsicurl/{vrt_url}",
        str(tmp),
    ]
    result = subprocess.run(cmd, check=False, capture_output=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"gdalwarp failed (exit {result.returncode})")
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)


def _gdal_translate(vrt_url: str, dest: Path) -> None:
    """COG-ify a VRT that is already in EPSG:4326 (no reprojection)."""
    tmp = dest.with_suffix(".tif.tmp")
    cmd = [
        "gdal_translate",
        *_HTTP_CONFIG,
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=256",
        "-co", "BLOCKYSIZE=256",
        "-co", "BIGTIFF=YES",
        "-co", "NUM_THREADS=ALL_CPUS",
        f"/vsicurl/{vrt_url}",
        str(tmp),
    ]
    result = subprocess.run(cmd, check=False, capture_output=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"gdal_translate failed (exit {result.returncode})")
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    catalog = _load_catalog()
    layers  = _soil_layers(catalog)
    if not layers:
        raise SystemExit(f"No soil layers ({', '.join(sorted(_SOURCES))}) found in catalog.json")

    LAYERS_DIR.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        layer_id = layer["id"]
        source   = layer.get("source", "")
        out_path = LAYERS_DIR / layer["filename"]

        if VARS_TO_DOWNLOAD is not None and layer_id not in VARS_TO_DOWNLOAD:
            continue

        if out_path.exists() and not force:
            print(f"[skip] {layer_id} already exists: {out_path}  (--force to rebuild)")
            continue

        vrt_url = layer["vrt_url"]

        print(f"\nDownloading {layer_id} → {out_path}")
        print(f"  VRT: {vrt_url}")
        if source == SOURCE_ID_SAL:
            _gdal_translate(vrt_url, out_path)
        else:
            resampling = layer.get("resampling", "bilinear")
            _gdalwarp(vrt_url, out_path, resampling)
        print(f"  Done: {out_path}")

    print("\nAll soil layers complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SoilGrids 2.0 and ISRIC salinity layers")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output already exists")
    args = parser.parse_args()
    main(force=args.force)
