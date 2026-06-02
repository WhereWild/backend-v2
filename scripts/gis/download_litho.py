# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download Hengl 2018 global landform and lithology classified TIFs from Zenodo
and convert each to a Cloud-Optimised GeoTIFF.

Source: https://zenodo.org/records/1464846
Two files are fetched — the composite classified rasters (suffix _c_):
  dtm_landform_usgs.ecotapestry_c_250m_s0..0cm_2014_v1.0.tif   (~683 MB)
  dtm_lithology_usgs.ecotapestry_c_250m_s0..0cm_2014_v1.0.tif  (~232 MB)

Steps:
  1. Query Zenodo API for per-file download URLs (cached in raw_dir)
  2. Download each file via aria2c (skips if already present)
  3. Convert to COG via gdal_translate

Zenodo public records need no credentials. Set ZENODO_TOKEN in .env for
access-restricted records.

Usage (inside the gdal container):
    uv run python scripts/gis/download_hengl.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG_PATH = Path("config/gis/catalog.json")
LAYERS_DIR   = Path("data/gis/layers")
RAW_DIR      = Path("data/gis/hengl_raw")
ZENODO_API   = "https://zenodo.org/api/records"
RECORD_ID    = "1464846"

_TARGET_LAYERS = ("landform", "lithology")
_COMPOSITE_STEM = "_c_250m_s0..0cm_2014_v1.0.tif"
_CSV_STEM       = "_c_250m_s0..0cm_2014_v1.0.tif.csv"  # RAT: class id → name mapping
_UA = "wherewild-download-hengl/1.0"


# ── Catalog ───────────────────────────────────────────────────────────────────

def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _hengl_layers(catalog: dict) -> list[dict]:
    return [
        layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") == "hengl_landform_2018"
    ]


# ── Zenodo API ────────────────────────────────────────────────────────────────

def _zenodo_files(record_id: str, token: str | None) -> dict[str, str]:
    """Return {filename: download_url} for all files in the record."""
    url = f"{ZENODO_API}/{record_id}"
    if token:
        url += f"?access_token={token}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": _UA})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise SystemExit(
                f"Zenodo record {record_id} is access-restricted. "
                "Set ZENODO_TOKEN in your .env and re-run."
            ) from exc
        raise

    result: dict[str, str] = {}
    for entry in data.get("files", []):
        key  = entry.get("key", "")
        links = entry.get("links", {})
        dl   = links.get("content") or links.get("download") or links.get("self", "")
        if dl:
            result[key] = dl
    if not result:
        raise SystemExit(f"No files found in Zenodo record {record_id}")
    return result


# ── GDAL helpers ──────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _aria2c(url: str, dest: Path, token: str | None) -> None:
    cmd = [
        "aria2c",
        "--split=4",
        "--max-connection-per-server=4",
        "--continue=true",
        "--max-tries=12",
        "--retry-wait=15",
        "--connect-timeout=60",
        f"--dir={dest.parent}",
        f"--out={dest.name}",
    ]
    if token:
        cmd += [f"--header=Authorization: Bearer {token}"]
    cmd.append(url)
    subprocess.run(cmd, check=True)


def _to_cog(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(".tif.tmp")
    _run([
        "gdal_translate",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=256",
        "-co", "BLOCKYSIZE=256",
        "-co", "BIGTIFF=YES",
        "-co", "NUM_THREADS=ALL_CPUS",
        str(src), str(tmp),
    ])
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
    vars_filter: list[str] | None = [v.strip() for v in raw_vars.split(",") if v.strip()] or None
    if vars_filter is not None and not any(v in vars_filter for v in _TARGET_LAYERS):
        print("[download_hengl] skipped (landform/lithology not in VARS_TO_DOWNLOAD)")
        return

    catalog = _load_catalog()
    layers  = _hengl_layers(catalog)
    if not layers:
        raise SystemExit("No 'hengl_landform_2018' layers found in catalog.json")

    token = os.environ.get("ZENODO_TOKEN")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if all outputs already exist
    all_done = all(
        (LAYERS_DIR / layer["filename"]).exists()
        for layer in layers
        if vars_filter is None or layer["id"] in vars_filter
    )
    if all_done and not force:
        for layer in layers:
            print(f"[skip] {layer['id']} COG already exists: {LAYERS_DIR / layer['filename']}  (--force to rebuild)")
        return

    # Query Zenodo once for all file URLs
    print(f"Querying Zenodo record {RECORD_ID}...")
    zenodo_files = _zenodo_files(RECORD_ID, token)
    print(f"  Found {len(zenodo_files)} file(s) in record")

    for layer in layers:
        layer_id = layer["id"]
        out_path = LAYERS_DIR / layer["filename"]

        if vars_filter is not None and layer_id not in vars_filter:
            continue
        if out_path.exists() and not force:
            print(f"[skip] {layer_id} COG already exists: {out_path}")
            continue

        # Find matching file in Zenodo listing
        stem_key = f"dtm_{layer_id}_usgs.ecotapestry{_COMPOSITE_STEM}"
        dl_url = zenodo_files.get(stem_key)
        if not dl_url:
            # Fallback: search by layer_id and _c_ pattern
            matches = [
                (k, v) for k, v in zenodo_files.items()
                if layer_id in k and "_c_" in k and k.endswith(".tif")
            ]
            if not matches:
                raise SystemExit(
                    f"Could not find a composite TIF for '{layer_id}' in Zenodo record {RECORD_ID}. "
                    f"Available files: {list(zenodo_files.keys())[:10]}"
                )
            stem_key, dl_url = matches[0]
            print(f"  Resolved {layer_id} → {stem_key}")

        raw_path = RAW_DIR / stem_key

        # Download CSV (class mapping / RAT) alongside the TIF — use it to verify legend
        csv_key  = f"dtm_{layer_id}_usgs.ecotapestry{_CSV_STEM}"
        csv_path = RAW_DIR / csv_key
        csv_url  = zenodo_files.get(csv_key)
        if csv_url and (not csv_path.exists() or force):
            print(f"\nDownloading {layer_id} class CSV ({csv_key})...")
            _aria2c(csv_url, csv_path, token)
        elif csv_path.exists():
            print(f"  CSV on disk: {csv_path}")
        else:
            print(f"  WARNING: no CSV found for {layer_id} — cannot verify legend class IDs")

        # Download
        if not raw_path.exists() or force:
            print(f"\nDownloading {layer_id} ({stem_key})...")
            _aria2c(dl_url, raw_path, token)
        else:
            print(f"  Raw file on disk: {raw_path}")

        # Convert to COG
        print(f"  Converting {layer_id} to COG → {out_path}")
        _to_cog(raw_path, out_path)
        print(f"  Done: {out_path}")
        if csv_path.exists():
            print(f"  → Verify legend class IDs against: {csv_path}")

    print("\nAll Hengl layers complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hengl 2018 landform and lithology")
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild even if outputs exist")
    args = parser.parse_args()
    main(force=args.force)
