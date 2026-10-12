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

The published TWI stops at 57°S while the DTM reaches 65°S, and skips some
islands; with "twi_fill" in VARS_TO_DOWNLOAD, stage 2 below computes the
southern ones with GEDTM30's own recipe right after the download (off by
default — exhaustive, but a ~340 GB rewrite for ~25,000 km²).

Full run: aria2c each file into data/gis/dem_raw/, fix it up there, then
rename it into place. --bbox W S E N clips every layer from the remote COGs
into --out-dir instead (for testing on a region without the ~770 GB).

Re-running is safe: finished layers are skipped unless --force is passed,
and aria2c resumes a partial download.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject

from scripts.gis import build_overviews
from scripts.gis import download_geomorphons as geo

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
            "--max-tries=0",          # retry forever: a Wi-Fi or DNS blip only pauses it
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


# ── Stage 2 (opt-in): fill the published TWI's southern gaps ──────────────────
#
# GEDTM30's published TWI v1.2.0 stops at 57°S (the DTM reaches 65°S) and
# skips whole Southern Ocean islands — Kerguelen, South Georgia, Heard, Prince
# Edward, the Chathams and everything south of 57°S: ~25,000 km² of land with
# elevation but no TWI. This computes them with GEDTM30's own recipe
# (parametrization/prod_lsp_v1.2.0.py, at 30 m, in WhiteboxTools — the
# open-source tools behind their Whitebox Workflows calls):
#   DTM warped to a buffered Equi7 window (cubic spline) → Gaussian filter
#   (sigma 0.75) → BreachDepressionsLeastCost (dist 400 cells, flat increment
#   0.001, fill remaining) → D8 specific catchment area; slope in degrees from
#   the filtered DTM → WetnessIndex; back to lat/lon (cubic spline), ×100 as
#   Int16 on the published file's own grid. Their water lowering (3 m + 0.17 ×
#   JRC surface-water occurrence) is skipped: the constant part changes
#   nothing, and JRC has no data south of ~60°S.
# Water stays empty as GEDTM30 leaves it (flat gap patches = lakes; the
# Caspian and Lake Victoria, which they mask). Steps: find land gap patches on
# ~2 km overviews → per Equi7 tile, the recipe over the gaps' window →
# 1° cells inside the gap mask → one pass to a new COG with the published
# values on top, extended to the DTM's southern edge. Resumable (.done
# markers); the published file is kept as data/tmp/twi_published.tif.

ELEV_PATH   = LAYERS_DIR / "elevation.tif"
TWI_PATH    = LAYERS_DIR / "twi.tif"
FILL_DIR    = Path("data/gis/twi_fill")
CELLS_DIR   = FILL_DIR / "cells"
DONE_DIR    = FILL_DIR / "done"
SCRATCH_DIR = FILL_DIR / "scratch"
MASK_PATH   = FILL_DIR / "mask.tif"
PUBLISHED_BACKUP = Path("data/tmp/twi_published.tif")

DETECT_OVERVIEW_LEVEL = 5     # 64x: ~2 km
MIN_PATCH_KM2 = 20.0   # smaller patches are coastline specks; their windows are mostly mainland (slow breaching)
# Only the Southern Ocean islands and the land south of the published 57°S
# edge — the gaps that show. Further north the misses are thin tile-edge
# strips (Greenland, Ellesmere, Severnaya Zemlya) and a few antimeridian
# tiles; raise this to 90 to fill those too (~12+ h more on GamBase).
MAX_FILL_LAT = -40.0
MIN_ELEV_STD_M = 2.0          # flatter gap patches are water surfaces
# GEDTM30 masks these two outright (their get_mask_layer); keep them empty.
MASKED_LAKES = (
    (46.5, 36.5, 55.5, 47.5),   # Caspian Sea (W, S, E, N)
    (31.4, -3.2, 35.0, 0.7),    # Lake Victoria
)

BUFFER_M = 101_760            # GEDTM30's Equi7 tile overlap
RES_M = 30
TWI_SCALE = 100               # published TWI is Int16 × 100
TWI_NODATA = 32767
WBT_PROCS = 4                 # see download_geomorphons.WARP_THREADS


def _wbt():
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_max_procs(WBT_PROCS)
    return wbt


# ── Step 1: where the published TWI is missing land ───────────────────────────

def _find_gaps():
    """Land gap patches at the elevation overview's ~2 km resolution: the
    (dilated) mask, its transform, and per (zone, tile name) the gap area,
    the gap points' lon/lat and the 1° cells they touch."""
    from equi7grid import get_standard_equi7grid
    from scipy import ndimage
    from shapely.geometry import Point

    with rasterio.open(ELEV_PATH, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as d:
        dem = d.read(1)
        dtf = d.transform
        dnd = d.nodata
    with rasterio.open(TWI_PATH, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as t:
        twi = t.read(1)
        ttf = t.transform
        tnd = t.nodata

    lat = dtf.f + (np.arange(dem.shape[0]) + 0.5) * dtf.e
    lon = dtf.c + (np.arange(dem.shape[1]) + 0.5) * dtf.a
    tr = np.floor((ttf.f - lat) / -ttf.e).astype(int)
    tc = np.floor((lon - ttf.c) / ttf.a).astype(int)
    in_rows = (tr >= 0) & (tr < twi.shape[0])
    twi_at = twi[np.ix_(np.clip(tr, 0, twi.shape[0] - 1), np.clip(tc, 0, twi.shape[1] - 1))]
    land = np.isfinite(dem) & (dem != dnd)
    gap = land & ~((twi_at != tnd) & in_rows[:, None])
    for w, s, e, n in MASKED_LAKES:
        gap[np.ix_((lat >= s) & (lat <= n), (lon >= w) & (lon <= e))] = False
    gap[lat > MAX_FILL_LAT, :] = False

    cell_km2 = abs(dtf.a) * 111.32 * np.cos(np.radians(lat)) * abs(dtf.e) * 110.57
    lab, n = ndimage.label(gap, structure=np.ones((3, 3)))
    ids = np.arange(1, n + 1)
    z = np.where(land, dem, 0).astype(np.float64)
    mean = ndimage.mean(z, lab, ids)
    std = np.sqrt(np.maximum(ndimage.mean(z ** 2, lab, ids) - mean ** 2, 0))
    km2 = ndimage.sum(np.broadcast_to(cell_km2[:, None], gap.shape), lab, ids)
    keep = ids[(std > MIN_ELEV_STD_M) & (km2 >= MIN_PATCH_KM2)]
    mask = np.isin(lab, keep)
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))  # cover patch edges at 30 m

    grid = get_standard_equi7grid({geo.EQUI7_TILING: geo.EQUI7_RES_M})
    fps = geo._zone_footprints()
    tiles: dict[tuple[str, str], dict] = {}
    for r, c in zip(*np.nonzero(mask & gap)):
        pt = Point(float(lon[c]), float(lat[r]))
        for zone in geo.EQUI7_ZONES:
            if fps[zone].contains(pt):
                name = getattr(grid, zone).get_tile_from_lonlat(pt.x, pt.y, geo.EQUI7_TILING).name
                t = tiles.setdefault((zone, name), {"km2": 0.0, "pts": [], "cells": set()})
                t["km2"] += cell_km2[r]
                t["pts"].append((pt.x, pt.y))
                t["cells"].add((math.floor(pt.x), math.floor(pt.y)))
                break
    total = sum(cell_km2[r] for r in np.nonzero(mask & gap)[0])
    print(f"[fill_twi] {len(keep)} land gap patches, {total:,.0f} km², in {len(tiles)} Equi7 tiles", flush=True)
    return mask, dtf, tiles


def _write_mask(mask, tf) -> None:
    FILL_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(MASK_PATH, "w", driver="GTiff", width=mask.shape[1], height=mask.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=tf, compress="deflate") as ds:
        ds.write(mask.astype(np.uint8), 1)


# ── Step 2: TWI for one tile, GEDTM30's recipe ────────────────────────────────

def _data_window(path: Path, nodata: float, margin: int = 2):
    """Row/col bounds of the valid data in a (mostly empty) raster."""
    with rasterio.open(path) as ds:
        rows = np.zeros(ds.height, bool)
        cols = np.zeros(ds.width, bool)
        for _, win in ds.block_windows(1):
            a = ds.read(1, window=win)
            v = a != nodata
            if v.any():
                rows[win.row_off: win.row_off + win.height] |= v.any(axis=1)
                cols[win.col_off: win.col_off + win.width] |= v.any(axis=0)
        if not rows.any():
            return None
        r = np.nonzero(rows)[0]
        c = np.nonzero(cols)[0]
        r0, r1 = max(r[0] - margin, 0), min(r[-1] + margin + 1, ds.height)
        c0, c1 = max(c[0] - margin, 0), min(c[-1] + margin + 1, ds.width)
        return c0, r0, c1 - c0, r1 - r0


def _tile_twi(zone: str, epsg: int, tile, pts, scratch: Path) -> Path | None:
    """GEDTM30's recipe over the gap points' bounding box (+ their padding,
    within the padded tile) — not the whole tile, which for a few km² of
    coastline would mean breaching depressions across ~800 km of land."""
    from pyproj import Transformer
    w, s, e, n = tile.outer_boundary_extent
    xs, ys = Transformer.from_crs(4326, epsg, always_xy=True).transform(*zip(*pts))
    half = 2 * 111_320 / 3600 * 2 ** DETECT_OVERVIEW_LEVEL   # two detection pixels, in metres
    bw = max(w - BUFFER_M, min(xs) - BUFFER_M - half)
    bs = max(s - BUFFER_M, min(ys) - BUFFER_M - half)
    be = min(e + BUFFER_M, max(xs) + BUFFER_M + half)
    bn = min(n + BUFFER_M, max(ys) + BUFFER_M + half)
    def snap(v, f):
        return f(v / RES_M) * RES_M

    full = scratch / "dem_full.tif"
    geo._run([
        "gdalwarp", "-overwrite", "-q", "-t_srs", f"EPSG:{epsg}",
        "-te", str(snap(bw, math.floor)), str(snap(bs, math.floor)), str(snap(be, math.ceil)), str(snap(bn, math.ceil)),
        "-tr", str(RES_M), str(RES_M), "-r", "cubicspline", "-et", "0", "-ovr", "NONE",
        "-srcnodata", repr(geo._elev_nodata()), "-dstnodata", "-9999", "-ot", "Float32",
        "-wm", "2048", "-multi", "-wo", f"NUM_THREADS={geo.WARP_THREADS}",
        "-co", "TILED=YES", "-co", "SPARSE_OK=TRUE", "-co", "BIGTIFF=YES",
        str(ELEV_PATH), str(full),
    ])
    win = _data_window(full, -9999.0)
    if win is None:
        return None
    dem = scratch / "dem.tif"
    geo._run(["gdal_translate", "-q", "-srcwin", *map(str, win), str(full), str(dem)])
    full.unlink()

    wbt = _wbt()
    wbt.set_working_dir(str(scratch.resolve()))
    steps = [
        ("gaussian_filter", dict(i="dem.tif", output="dem_f.tif", sigma=0.75)),
        ("breach_depressions_least_cost", dict(dem="dem_f.tif", output="dem_b.tif", dist=400, flat_increment=0.001, fill=True)),
        ("slope", dict(dem="dem_f.tif", output="slope.tif", units="degrees")),
        ("d8_flow_accumulation", dict(i="dem_b.tif", output="sca.tif", out_type="specific contributing area")),
        ("wetness_index", dict(sca="sca.tif", slope="slope.tif", output="twi.tif")),
    ]
    for name, kw in steps:
        if getattr(wbt, name)(**kw) != 0:
            raise RuntimeError(f"WhiteboxTools {name} failed on {tile.name}")
    return scratch / "twi.tif"


def _publish_grid():
    """The published TWI's transform; its grid continues south unchanged."""
    with rasterio.open(TWI_PATH) as t:
        return t.transform


def _write_cells(zone: str, tile, cells, twi_path: Path, grid_tf, fps, mask_src) -> int:
    px = grid_tf.a
    written = 0
    with rasterio.open(twi_path) as src:
        for lon, lat in sorted(cells):
            if lat < -65:
                continue
            col0 = math.floor((lon - grid_tf.c) / px + 1e-6)
            row0 = math.floor((grid_tf.f - (lat + 1)) / px + 1e-6)
            size = round(1.0 / px) + 1
            dst_tf = from_origin(grid_tf.c + col0 * px, grid_tf.f - row0 * px, px, px)
            out = np.full((size, size), np.nan, np.float32)
            reproject(source=rasterio.band(src, 1), destination=out, src_nodata=src.nodata,
                      dst_transform=dst_tf, dst_crs="EPSG:4326", dst_nodata=np.nan,
                      resampling=Resampling.cubic_spline)
            # only this zone's own footprint, only inside the gap mask
            inzone = rasterize([(fps[zone], 1)], out_shape=out.shape, transform=dst_tf, fill=0, dtype="uint8").astype(bool)
            m = np.zeros(out.shape, np.uint8)
            reproject(source=rasterio.band(mask_src, 1), destination=m, dst_transform=dst_tf,
                      dst_crs="EPSG:4326", resampling=Resampling.nearest)
            keep = inzone & (m > 0) & np.isfinite(out)
            if not keep.any():
                continue
            vals = np.full(out.shape, TWI_NODATA, np.int16)
            vals[keep] = np.clip(np.round(out[keep] * TWI_SCALE), -32767, 32766).astype(np.int16)
            dest = CELLS_DIR / f"{tile.name}_{geo._cell_name(lon, lat)}.tif"
            tmp = dest.with_suffix(".tif.tmp")
            with rasterio.open(tmp, "w", driver="GTiff", width=size, height=size, count=1, dtype="int16",
                               crs="EPSG:4326", transform=dst_tf, nodata=TWI_NODATA, compress="deflate",
                               tiled=True, blockxsize=512, blockysize=512) as ds:
                ds.write(vals, 1)
                # same scale tag as the published file: gdalbuildvrt skips any
                # source whose band scale differs from the first one's
                ds.scales = (1 / TWI_SCALE,)
            geo._commit(tmp, dest)
            written += 1
    return written


# ── Step 3: merged COG ────────────────────────────────────────────────────────

def _merge(grid_tf) -> None:
    with rasterio.open(ELEV_PATH) as e:
        south = e.bounds.bottom
    with rasterio.open(TWI_PATH) as t:
        w, n, e_ = t.bounds.left, t.bounds.top, t.bounds.right
    px = grid_tf.a
    s = grid_tf.f - math.ceil((grid_tf.f - south) / px) * px
    cells = sorted(str(p) for p in CELLS_DIR.glob("*.tif"))
    out_tmp = TWI_PATH.with_suffix(".tif.filled.tmp")
    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt = Path(tmp_dir) / "merged.vrt"
        scaled = Path(tmp_dir) / "scaled.vrt"
        # later sources draw on top: the published TWI last, so its values always win
        geo._run(["gdalbuildvrt", "-te", str(w), str(s), str(e_), str(n), "-tr", str(px), str(px),
                  "-srcnodata", str(TWI_NODATA), "-vrtnodata", str(TWI_NODATA), str(vrt), *cells, str(TWI_PATH)])
        geo._run(["gdal_translate", "-q", "-of", "VRT", "-a_scale", str(1 / TWI_SCALE), str(vrt), str(scaled)])
        with rasterio.open(scaled) as ds:
            factors = build_overviews._overview_factors_for_dataset(ds)
        print(f"[fill_twi] writing merged COG ({len(cells)} fill cells, overviews {factors})...", flush=True)
        build_overviews._build_cog(scaled, out_tmp, nominal=False, overview_factors=factors)
    # The merge may only add data: wherever the published TWI has a value, the
    # merged file must too (checked on ~2 km overviews). A source silently
    # dropped from the VRT would otherwise ship a near-empty layer.
    with rasterio.open(TWI_PATH, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as pub, \
            rasterio.open(out_tmp, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as new:
        pv = pub.read(1) != pub.nodata
        nv = new.read(1, window=rasterio.windows.Window(0, 0, pub.width, pub.height)) != new.nodata
    lost = int((pv & ~nv).sum())
    if lost > pv.sum() * 1e-4:
        out_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"[fill_twi] merged TWI lost {lost:,} of {int(pv.sum()):,} published overview cells — not installed")
    PUBLISHED_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    TWI_PATH.replace(PUBLISHED_BACKUP)
    geo._commit(out_tmp, TWI_PATH)


def fill_twi_gaps() -> None:
    """Stage 2 (opt-in, \"twi_fill\"): see the module docstring."""
    mask, mask_tf, tiles = _find_gaps()
    if not tiles:
        print("[fill_twi] no land gaps — nothing to do")
        return
    _write_mask(mask, mask_tf)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    grid_tf = _publish_grid()
    fps = geo._zone_footprints()
    by_name = {t.name: (z, e, t) for z, e, t in geo._land_tiles()}
    todo = sorted(tiles, key=lambda k: -tiles[k]["km2"])
    with rasterio.open(MASK_PATH) as mask_src:
        for i, (zone, name) in enumerate(todo, 1):
            marker = DONE_DIR / f"{name}.done"
            if marker.exists():
                continue
            if name not in by_name:
                print(f"[fill_twi]   {name}: not a land tile, skipped", flush=True)
                continue
            z, epsg, tile = by_name[name]
            t0 = time.time()
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
            SCRATCH_DIR.mkdir(parents=True)
            try:
                info = tiles[(zone, name)]
                twi = _tile_twi(z, epsg, tile, info["pts"], SCRATCH_DIR)
                n = 0 if twi is None else _write_cells(z, tile, info["cells"], twi, grid_tf, fps, mask_src)
            finally:
                shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
            marker.touch()
            geo._log(f"[fill_twi] [{i}/{len(todo)}] {name} ({tiles[(zone, name)]['km2']:,.0f} km² gap): {n} cells in {time.time() - t0:.0f}s")
    _merge(grid_tf)
    print(f"[fill_twi] done: {TWI_PATH} (published file kept at {PUBLISHED_BACKUP})", flush=True)


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

    # Opt-in: the published TWI misses ~25,000 km² of Southern Ocean islands
    # (stage 2 below) — exhaustive but rarely worth a ~340 GB rewrite,
    # so it only runs when "twi_fill" is listed explicitly.
    if bbox is None and VARS_TO_DOWNLOAD is not None and "twi_fill" in VARS_TO_DOWNLOAD \
            and (out_dir / "twi.tif").exists() and (out_dir / "elevation.tif").exists():
        fill_twi_gaps()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GEDTM30 elevation and terrain layers")
    parser.add_argument("--force", action="store_true", help="Replace layers that already exist")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                        help="Clip to this lon/lat box from the remote COGs instead of downloading")
    parser.add_argument("--out-dir", type=Path, default=LAYERS_DIR, help="Where to write (default data/gis/layers)")
    args = parser.parse_args()
    main(force=args.force, bbox=tuple(args.bbox) if args.bbox else None, out_dir=args.out_dir)
