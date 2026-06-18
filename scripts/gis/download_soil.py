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
    translated directly to a COG via gdal_translate. render_min/render_max are
    computed from approximate pixel statistics and patched into catalog.json if
    they are currently null.

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

import rasterio

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
        if layer.get("source") in _SOURCES
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


def _compute_render_range(path: Path, scale: float, offset: float) -> tuple[float, float]:
    """Return (min, max) in display units (scale+offset applied) from approx statistics."""
    print("  Computing render range (approx stats)...", flush=True)
    with rasterio.open(path) as ds:
        stats = ds.statistics(bidx=1, approx=True)
    return float(stats.min) * scale + offset, float(stats.max) * scale + offset


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    catalog = _load_catalog()
    layers  = _soil_layers(catalog)
    if not layers:
        raise SystemExit(f"No soil layers ({', '.join(sorted(_SOURCES))}) found in catalog.json")

    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    catalog_dirty = False

    for layer in layers:
        layer_id = layer["id"]
        source   = layer.get("source", "")
        out_path = LAYERS_DIR / layer["filename"]

        if VARS_TO_DOWNLOAD is not None and layer_id not in VARS_TO_DOWNLOAD:
            continue

        needs_stats = layer.get("value_type") not in ("nominal", "ordinal") and (
            layer.get("render_min") is None or layer.get("render_max") is None
        )
        needs_download = not out_path.exists() or force

        if not needs_download and not needs_stats:
            print(f"[skip] {layer_id} already exists: {out_path}  (--force to rebuild)")
            continue

        vrt_url = layer["vrt_url"]

        if needs_download:
            print(f"\nDownloading {layer_id} → {out_path}")
            print(f"  VRT: {vrt_url}")
            if source == SOURCE_ID_SAL:
                _gdal_translate(vrt_url, out_path)
            else:
                resampling = layer.get("resampling", "bilinear")
                _gdalwarp(vrt_url, out_path, resampling)
            print(f"  Done: {out_path}")
        else:
            print(f"\n[stats only] {layer_id} — {out_path}")

        if needs_stats:
            scale  = layer.get("scale_factor") or 1.0
            offset = layer.get("add_offset")   or 0.0
            rmin, rmax = _compute_render_range(out_path, scale, offset)
            out_path.with_name(out_path.name + ".aux.xml").unlink(missing_ok=True)
            for key, val in [("render_min", round(rmin, 4)), ("render_max", round(rmax, 4))]:
                if layer.get(key) is None:
                    print(f"  {key}: null → {val}")
                    layer[key] = val
                    catalog_dirty = True

    if catalog_dirty:
        updates = {
            layer["id"]: {k: layer[k] for k in ("render_min", "render_max")}
            for cat in catalog["categories"]
            for layer in cat["layers"]
            if layer.get("source") in _SOURCES
        }
        with open(CATALOG_PATH) as f:
            on_disk = json.load(f)
        for cat in on_disk["categories"]:
            for layer in cat["layers"]:
                if layer["id"] in updates:
                    layer.update(updates[layer["id"]])
        with open(CATALOG_PATH, "w") as f:
            json.dump(on_disk, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Catalog updated: {CATALOG_PATH}")

    print("\nAll soil layers complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SoilGrids 2.0 and ISRIC salinity layers")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output already exists")
    args = parser.parse_args()
    main(force=args.force)
