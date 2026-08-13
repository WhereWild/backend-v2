# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download CHELSA layers defined in config/gis/catalog.json into data/gis/layers/.

Only processes layers whose source is "chelsa_v2_1". Re-running skips files
that already exist. After each download, reads scale/offset from the file and
patches them into the catalog if they are currently null, or warns if they
differ from what the catalog already records.
"""

import json
import os
import subprocess
from pathlib import Path

import rasterio

CATALOG_PATH = Path("config/gis/catalog.json")
LAYERS_DIR   = Path("data/gis/layers")
SOURCE_ID    = "chelsa_v2_1"

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None


def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _chelsa_layers(catalog: dict) -> list[dict]:
    return [
        layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") == SOURCE_ID
    ]


def _download(url: str, dest: Path) -> None:
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
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


def _read_file_metadata(path: Path) -> dict:
    """Read scale, offset, and statistics from the file. Returns only values that are meaningfully present."""
    with rasterio.open(path) as ds:
        scale  = ds.scales[0]  if ds.scales  else None
        offset = ds.offsets[0] if ds.offsets else None
        if scale == 1.0 and offset == 0.0:
            scale, offset = None, None

        tags = ds.tags(1) or ds.tags() or {}
        lower = {k.lower(): v for k, v in tags.items()}

        def _stat(key: str) -> float | None:
            raw = lower.get(key)
            try:
                return float(raw) if raw is not None else None
            except (ValueError, TypeError):
                return None

        stat_min = _stat("statistics_minimum") or _stat("minimum") or _stat("min")
        stat_max = _stat("statistics_maximum") or _stat("maximum") or _stat("max")

        return {
            "scale_factor": scale,
            "add_offset":   offset,
            "stat_min":     stat_min,
            "stat_max":     stat_max,
            "crs":          str(ds.crs),
            "shape":        (ds.height, ds.width),
            "dtype":        ds.dtypes[0],
            "nodata":       ds.nodata,
            "bounds":       ds.bounds,
        }


def _inspect(path: Path, meta: dict) -> None:
    print(f"  CRS        : {meta['crs']}")
    print(f"  Shape      : {meta['shape'][0]} x {meta['shape'][1]}")
    print(f"  Dtype      : {meta['dtype']}")
    print(f"  Nodata     : {meta['nodata']}")
    print(f"  Bounds     : {meta['bounds']}")
    print(f"  Scale      : {meta['scale_factor']}")
    print(f"  Offset     : {meta['add_offset']}")
    print(f"  Stat min   : {meta['stat_min']}")
    print(f"  Stat max   : {meta['stat_max']}")


def _sync_catalog(layer: dict, meta: dict) -> bool:
    """Patch null catalog fields from file metadata. Returns True if anything changed.

    render_min/render_max are deliberately NOT synced here — scripts/gis/
    build_overviews.py now computes those for every continuous layer as a
    percentile of valid pixel values (see PERCENTILE_RENDER_BOUNDS), and
    runs unconditionally as the next rebuild stage after this one, so
    anything set here would just be immediately overwritten. Leaving it out
    avoids a second full-raster read for the exact same fields.
    """
    changed = False

    # scale_factor / add_offset — file is informational, catalog is authoritative
    for key in ("scale_factor", "add_offset"):
        file_val    = meta[key]
        catalog_val = layer.get(key)
        if catalog_val is None and file_val is not None:
            print(f"  {key}: null → {file_val} (read from file)")
            layer[key] = file_val
            changed = True
        elif catalog_val is not None and file_val is not None and catalog_val != file_val:
            print(f"  WARNING: catalog {key}={catalog_val} differs from file {file_val} — keeping catalog value")

    return changed


def main() -> None:
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    layers = _chelsa_layers(catalog)
    if VARS_TO_DOWNLOAD is not None:
        layers = [layer for layer in layers if layer["id"] in VARS_TO_DOWNLOAD]
    catalog_dirty = False

    for layer in layers:
        dest = LAYERS_DIR / layer["filename"]
        if dest.exists():
            print(f"[skip] {layer['id']} — already at {dest}")
        else:
            print(f"[download] {layer['id']} ({layer['display_name']})")
            _download(layer["download_url"], dest)
            print(f"  Saved to {dest}")

        meta = _read_file_metadata(dest)
        print(f"[inspect] {layer['id']}")
        _inspect(dest, meta)

        if _sync_catalog(layer, meta):
            catalog_dirty = True
        print()

    if catalog_dirty:
        # Re-read from disk before writing so external edits (e.g. value_type) aren't clobbered.
        updates = {
            layer["id"]: {k: layer[k] for k in ("scale_factor", "add_offset")}
            for cat in catalog["categories"]
            for layer in cat["layers"]
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


if __name__ == "__main__":
    main()
