# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download GEDTM30 (Ho et al. 2025, CC-BY-4.0) from OpenGeoHub: the terrain
model itself, plus the published land surface parameters that can't be
derived on the fly from it (slope, aspect, curvature, heat load etc. are
computed per request from elevation.tif — see util/gis.py).

Layers (VARS_TO_DOWNLOAD key → data/gis/layers/<file>):
  elevation → elevation.tif  GEDTM30 v1.2 DTM, Float32 metres, 1" (85°N–65°S)
  twi       → twi.tif        GEDTM30 Topographic Wetness Index v1.2.0, Int16
                             ×0.01 (catalog scale_factor), 1" (84°N–57°S)

Both are already COGs with overviews, so they're stored as published —
lossless, no re-encode. The DTM's own scale tag says 0.1, but its values are
metres (1300.0 at Salt Lake City, where FABDEM has 1298.3), so the tag is
cleared: anything honouring it would read heights 10× too low.

The TWI layer stops at 57°S while the DTM reaches 65°S; the land in between
(South Shetlands/Orkneys/Sandwich, the tip of the Antarctic Peninsula) has
no published TWI.

Full run: aria2c each file into data/gis/dem_raw/, fix it up there, then
rename it into place. --bbox W S E N clips every layer from the remote COGs
into --out-dir instead (for testing on a region without the ~770 GB).

Re-running is safe: finished layers are skipped unless --force is passed,
and aria2c resumes a partial download.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

S3_BASE = "https://s3.opengeohub.org/global/dtm"
LAYERS: dict[str, tuple[str, str]] = {
    # key: (published file under S3_BASE, file in data/gis/layers)
    "elevation": ("v1.2/gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif", "elevation.tif"),
    "twi": ("v1.2.0/twi_gedtm_m_30m_s_20060101_20151231_go_epsg.4326_v1.2.0.tif", "twi.tif"),
}
# Layers whose scale tag is wrong and has to be cleared (see docstring).
CLEAR_SCALE = {"elevation"}

RAW_DIR    = Path("data/gis/dem_raw")
LAYERS_DIR = Path("data/gis/layers")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=16",
            "--max-connection-per-server=16",
            "--min-split-size=64M",
            "--file-allocation=none",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            "--summary-interval=300",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ],
        check=True,
    )


def _clip(url: str, dest: Path, bbox: tuple[float, float, float, float], clear_scale: bool) -> None:
    """Clip a remote COG to bbox (W, S, E, N) as a lossless local COG."""
    w, s, e, n = bbox
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tif.tmp")
    _run([
        "gdal_translate", "-q",
        "-projwin", str(w), str(n), str(e), str(s),
        *(["-a_scale", "1", "-a_offset", "0"] if clear_scale else []),
        "-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
        "-co", "OVERVIEW_RESAMPLING=AVERAGE", "-co", "NUM_THREADS=4",
        f"/vsicurl/{url}", str(tmp),
    ])
    tmp.replace(dest)


def _clear_scale(path: Path) -> None:
    """Rewrites only the metadata tag; GDAL calls that a COG layout break,
    but the IFDs, tiles and overviews stay where they are."""
    _run(["gdal_edit", "-oo", "IGNORE_COG_LAYOUT_BREAK=YES", "-scale", "1", "-offset", "0", str(path)])


def main(force: bool = False, bbox: tuple[float, float, float, float] | None = None,
         out_dir: Path = LAYERS_DIR) -> None:
    for key, (remote, filename) in LAYERS.items():
        if not force and VARS_TO_DOWNLOAD is not None and key not in VARS_TO_DOWNLOAD:
            print(f"[download_dem] {key} skipped (not in VARS_TO_DOWNLOAD)")
            continue
        out_path = out_dir / filename
        if out_path.exists() and not force:
            print(f"[download_dem] {key}: already exists: {out_path} (use --force to replace)")
            continue
        url = f"{S3_BASE}/{remote}"

        if bbox is not None:
            print(f"[download_dem] {key}: clipping {bbox} → {out_path}", flush=True)
            _clip(url, out_path, bbox, clear_scale=key in CLEAR_SCALE)
            continue

        raw = RAW_DIR / Path(remote).name
        aria2_control = raw.with_suffix(raw.suffix + ".aria2")
        if not raw.exists() or aria2_control.exists():
            print(f"[download_dem] {key}: downloading {url}", flush=True)
            _download(url, raw)
        if key in CLEAR_SCALE:
            _clear_scale(raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw.replace(out_path)
        print(f"[download_dem] {key}: wrote {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GEDTM30 elevation and terrain layers")
    parser.add_argument("--force", action="store_true", help="Replace layers that already exist")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                        help="Clip to this lon/lat box from the remote COGs instead of downloading")
    parser.add_argument("--out-dir", type=Path, default=LAYERS_DIR, help="Where to write (default data/gis/layers)")
    args = parser.parse_args()
    main(force=args.force, bbox=tuple(args.bbox) if args.bbox else None, out_dir=args.out_dir)
