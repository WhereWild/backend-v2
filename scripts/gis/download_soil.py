# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download SoilGrids 2.0 layers defined in config/gis/catalog.json.

SoilGrids exposes global VRT files (referencing COG tiles via vsicurl internally)
at files.isric.org. GDAL can warp directly from those remote VRTs in one pass,
reprojecting from the native Interrupted Goode's Homolosine to EPSG:4326.

No raw tiles are stored locally — the warp streams from the remote VRT straight
to the output COG. Re-running skips files that already exist.

Usage (inside the gdal container):
    uv run python scripts/gis/download_soil.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

CATALOG_PATH  = Path("config/gis/catalog.json")
LAYERS_DIR    = Path("data/gis/layers")
SOURCE_ID     = "soilgrids_2_0"
SG_BASE       = "https://files.isric.org/soilgrids/latest/data"

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

# Pixel size in degrees (~250 m at equator)
_TR = str(1 / 480)


# ── Catalog ───────────────────────────────────────────────────────────────────

def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _soil_layers(catalog: dict) -> list[dict]:
    return [
        layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") == SOURCE_ID
    ]


# ── GDAL ──────────────────────────────────────────────────────────────────────

def _gdalwarp(vrt_url: str, dest: Path, resampling: str) -> None:
    tmp = dest.with_suffix(".tif.tmp")
    cmd = [
        "gdalwarp",
        "--config", "GDAL_HTTP_UNSAFESSL", "YES",
        "--config", "GDAL_HTTP_MAX_RETRY", "5",
        "--config", "GDAL_HTTP_RETRY_DELAY", "15",
        "--config", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.vrt",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    catalog = _load_catalog()
    layers  = _soil_layers(catalog)
    if not layers:
        raise SystemExit("No 'soilgrids_2_0' layers found in catalog.json")

    LAYERS_DIR.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        layer_id = layer["id"]
        out_path = LAYERS_DIR / layer["filename"]

        if VARS_TO_DOWNLOAD is not None and layer_id not in VARS_TO_DOWNLOAD:
            continue
        if out_path.exists() and not force:
            print(f"[skip] {layer_id} already exists: {out_path}  (--force to rebuild)")
            continue

        vrt_url    = layer["vrt_url"]
        resampling = layer.get("resampling", "bilinear")
        print(f"\nDownloading {layer_id} → {out_path}")
        print(f"  VRT: {vrt_url}")
        _gdalwarp(vrt_url, out_path, resampling)
        print(f"  Done: {out_path}")

    print("\nAll soil layers complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SoilGrids 2.0 layers via remote VRT warp")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output already exists")
    args = parser.parse_args()
    main(force=args.force)
