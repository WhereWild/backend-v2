# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download GLC_FCS30D v2 land cover tiles from Zenodo and build a global COG
for the most recent year (2022).

Dataset structure inside each longitude-band ZIP:
  - GLC_FCS30D_19852000_<tile>_5years_V1.1.tif  — 5-year composites, ~4 bands
  - GLC_FCS30D_20002022_<tile>_Annual_V1.1.tif  — annual maps 2000-2022, 23 bands

Uses remotezip (HTTP range requests) to extract only the *_Annual_* TIFs from
each remote ZIP without downloading the full archives (~135 GB total).

Steps:
  1. Query Zenodo API for the longitude-band ZIP URLs (cached in manifest.json)
  2. For each ZIP: list its contents and fetch only the Annual TIF entries
  3. Extract band 23 (2022) and clip each tile to its nominal bounds in one pass
  4. Stitch clipped tiles into a global VRT → COG

Zenodo public records need no credentials. Set ZENODO_TOKEN in .env for
access-restricted records.

Usage (inside the gdal container):
    uv run python scripts/gis/download_landcover.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from remotezip import RemoteZip

CATALOG_PATH  = Path("config/gis/catalog.json")
LAYERS_DIR    = Path("data/gis/layers")
RAW_TILES_DIR = Path("data/gis/landcover_raw_tiles")
CLIP_DIR      = Path("data/gis/landcover_2022_tiles")
MANIFEST_PATH = RAW_TILES_DIR / "manifest.json"
ZENODO_API    = "https://zenodo.org/api/records"

# Annual TIFs span 2000–2022 inclusive → 23 bands. Last band = 2022.
ANNUAL_BAND_COUNT = 23
TARGET_YEAR       = 2022
DOWNLOAD_WORKERS  = 8

_UA      = "wherewild-download-landcover/1.0"
_print_lock = threading.Lock()

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = (
    [v.strip() for v in _raw_vars.split(",") if v.strip()] or None
)


# ── Catalog ───────────────────────────────────────────────────────────────────

def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _find_layer(catalog: dict) -> dict:
    for category in catalog["categories"]:
        for layer in category["layers"]:
            if layer.get("id") == "landcover":
                return layer
    raise SystemExit("No 'landcover' layer found in catalog.json")


# ── Zenodo API ────────────────────────────────────────────────────────────────

def _zenodo_zip_files(record_id: str, token: str | None) -> list[dict]:
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
    files = data.get("files", [])
    zips  = [f for f in files if f["key"].endswith(".zip")]
    if not zips:
        raise SystemExit(f"No ZIP files found in Zenodo record {record_id}")
    return zips


def _download_url_for(entry: dict, token: str | None) -> str:
    links = entry.get("links", {})
    url   = links.get("self") or links.get("download") or links.get("content")
    if not url:
        url = f"https://zenodo.org/records/{entry.get('record_id', '')}/files/{entry['key']}"
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}access_token={token}"
    return url


# ── Tile bounds ───────────────────────────────────────────────────────────────

_COORD_RE = re.compile(r"([EW])(\d+)([NS])(\d+)", re.IGNORECASE)


def _parse_bounds(path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) from a tile filename like W175N55."""
    m = _COORD_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot parse tile bounds from filename: {path.name}")
    lon_dir, lon_val, lat_dir, lat_val = m.groups()
    west  = -int(lon_val) if lon_dir.upper() == "W" else int(lon_val)
    north = -int(lat_val) if lat_dir.upper() == "S" else int(lat_val)
    return west, north - 5, west + 5, north


# ── GDAL helpers ──────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _band_count(path: Path) -> int:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.count


def _extract_and_clip(src: Path, band: int, clip_dir: Path) -> Path:
    """Extract one band from a multi-band tile and clip to nominal bounds in one pass."""
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
    from rasterio.windows import transform as window_transform

    west, south, east, north = _parse_bounds(src)
    dest_name = src.name.replace("_Annual_", f"_{TARGET_YEAR}_")
    dest = clip_dir / dest_name
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".tif.tmp")
    if tmp.exists():
        tmp.unlink()

    with rasterio.open(src) as ds:
        window = window_from_bounds(west, south, east, north, ds.transform)
        data = ds.read(band, window=window)
        profile = ds.profile.copy()
        profile.update(
            count=1,
            width=data.shape[1],
            height=data.shape[0],
            transform=window_transform(window, ds.transform),
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(data, 1)
    tmp.replace(dest)
    return dest


def _build_global_cog(tile_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tif.tmp")
    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt_path = Path(tmp_dir) / "global.vrt"
        print(f"  Building VRT from {len(tile_paths)} tiles...", flush=True)
        _run([
            "gdalbuildvrt", "-overwrite", "-resolution", "highest",
            str(vrt_path), *[str(p) for p in sorted(tile_paths)],
        ])
        print("  Translating to GeoTIFF...", flush=True)
        _run([
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=ALL_CPUS",
            str(vrt_path), str(tmp_out),
        ])
    if out_path.exists():
        out_path.unlink()
    tmp_out.replace(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    if VARS_TO_DOWNLOAD is not None and "landcover" not in VARS_TO_DOWNLOAD:
        print("[download_landcover] skipped (landcover not in VARS_TO_DOWNLOAD)")
        return

    catalog   = _load_catalog()
    layer     = _find_layer(catalog)
    record_id = layer.get("zenodo_record_id")
    out_path  = LAYERS_DIR / layer.get("filename", "landcover.tif")
    token     = os.environ.get("ZENODO_TOKEN")

    if not record_id:
        raise SystemExit("catalog.json landcover entry is missing 'zenodo_record_id'")

    if out_path.exists() and not force:
        print(f"[skip] landcover COG already exists: {out_path}  (--force to rebuild)")
        return

    RAW_TILES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build or load tile manifest (skips Zenodo API + ZIP scanning on re-runs)
    # work_items: list of (zip_url, member_name, dest, compress_size_mb)
    work_items: list[tuple[str, str, Path, float]] = []
    all_annual: list[Path] = []

    if MANIFEST_PATH.exists():
        print(f"Loading tile manifest ({MANIFEST_PATH})...")
        for entry in json.loads(MANIFEST_PATH.read_text()):
            dest = RAW_TILES_DIR / entry["dest_name"]
            if dest.exists():
                all_annual.append(dest)
            else:
                work_items.append((entry["zip_url"], entry["member_name"], dest, entry["size_mb"]))
        print(f"  {len(all_annual)} tiles on disk, {len(work_items)} to download")
    else:
        print(f"Querying Zenodo record {record_id}...")
        zip_entries = _zenodo_zip_files(record_id, token)
        print(f"  Found {len(zip_entries)} ZIP(s)\nScanning ZIP central directories...")
        manifest_rows = []
        for i, entry in enumerate(zip_entries, 1):
            zip_url  = _download_url_for(entry, token)
            zip_name = entry["key"]
            with RemoteZip(zip_url) as rz:
                annual = [n for n in rz.namelist() if "_Annual_" in n and n.endswith(".tif")]
                for name in annual:
                    dest     = RAW_TILES_DIR / Path(name).name
                    size_mb  = rz.getinfo(name).compress_size / 1_048_576
                    manifest_rows.append({"zip_url": zip_url, "member_name": name,
                                          "dest_name": dest.name, "size_mb": size_mb})
                    if dest.exists():
                        all_annual.append(dest)
                    else:
                        work_items.append((zip_url, name, dest, size_mb))
            print(f"  [{i}/{len(zip_entries)}] {zip_name}: {len(annual)} Annual TIF(s)", flush=True)
        MANIFEST_PATH.write_text(json.dumps(manifest_rows, indent=2))
        total_mb = sum(w[3] for w in work_items)
        print(f"\n{len(all_annual)} tiles on disk, {len(work_items)} to download ({total_mb:.0f} MB compressed)")

    # 2. Download missing tiles in parallel ────────────────────────────────────
    if work_items:
        done = 0
        total = len(work_items)

        def _fetch(zip_url: str, member_name: str, dest: Path, size_mb: float) -> Path:
            tmp = dest.with_suffix(".tif.tmp")
            for attempt in range(1, 6):
                if tmp.exists():
                    tmp.unlink()
                try:
                    with RemoteZip(zip_url) as rz:
                        data = rz.read(member_name)
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    return dest
                except Exception as exc:
                    if attempt == 5:
                        raise
                    wait = 15 * attempt
                    with _print_lock:
                        print(f"  retry {attempt}/5 for {dest.name} in {wait}s ({exc})", flush=True)
                    time.sleep(wait)
            raise RuntimeError("unreachable")

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {
                pool.submit(_fetch, url, name, dest, mb): dest
                for url, name, dest, mb in work_items
            }
            for future in as_completed(futures):
                dest = futures[future]
                done += 1
                try:
                    future.result()
                    with _print_lock:
                        print(f"  [{done}/{total}] {dest.name}", flush=True)
                except Exception as exc:
                    with _print_lock:
                        print(f"  [{done}/{total}] ERROR {dest.name}: {exc}", flush=True)
                    raise
                all_annual.append(dest)

    if not all_annual:
        raise SystemExit("No Annual TIFs found across all ZIPs.")

    # 3. Extract band 23 (2022) + clip each tile to its nominal bounds ─────────
    sample_count = _band_count(all_annual[0])
    if sample_count != ANNUAL_BAND_COUNT:
        print(
            f"WARNING: expected {ANNUAL_BAND_COUNT} bands (2000–{TARGET_YEAR}) "
            f"but {all_annual[0].name} has {sample_count}. Extracting last band."
        )
    target_band = sample_count
    print(f"\nExtracting band {target_band} ({TARGET_YEAR}) + clipping {len(all_annual)} tiles...")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clipped_tiles: list[Path] = []
    for j, src in enumerate(all_annual, 1):
        dest_name = src.name.replace("_Annual_", f"_{TARGET_YEAR}_")
        dest = CLIP_DIR / dest_name
        if not dest.exists():
            print(f"  [{j}/{len(all_annual)}] {src.name}", flush=True)
        clipped_tiles.append(_extract_and_clip(src, target_band, CLIP_DIR))

    # 4. Build global COG ──────────────────────────────────────────────────────
    print(f"\nBuilding global COG → {out_path}")
    _build_global_cog(clipped_tiles, out_path)
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GLC_FCS30D v2 land cover")
    parser.add_argument("--force", action="store_true", help="Rebuild even if COG already exists")
    args = parser.parse_args()
    main(force=args.force)
