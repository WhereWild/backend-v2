# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fill the land GEDTM30's published TWI leaves empty, with GEDTM30's own
recipe, so twi.tif covers all the land elevation.tif does.

The published TWI v1.2.0 stops at 57°S (the DTM reaches 65°S) and skips some
whole Equi7 tiles and tile-edge strips — Kerguelen, South Georgia, Heard,
Prince Edward, the Chathams, Fiji's and the Aleutians' antimeridian tiles,
strips of north Greenland/Ellesmere/Severnaya Zemlya: ~50,000 km² of land
with elevation but no TWI. Water stays empty, as GEDTM30 leaves it: lakes
and sea fringes (flat surfaces — told apart from land by their elevation
spread) and the Caspian Sea and Lake Victoria, which GEDTM30 masks outright.

Recipe (GEDTM30 parametrization/prod_lsp_v1.2.0.py, at 30 m, in WhiteboxTools
— the open-source tools behind their Whitebox Workflows calls):
  DTM warped to the buffered Equi7 tile (cubic spline) → Gaussian filter
  (sigma 0.75) → BreachDepressionsLeastCost (dist 400 cells, flat increment
  0.001, fill remaining) → D8 specific catchment area; slope in degrees from
  the filtered DTM → WetnessIndex. Back to lat/lon with cubic spline, ×100
  as Int16, on the published file's own grid. Their water lowering
  (3 m + 0.17 × JRC surface-water occurrence) is skipped: the constant part
  changes nothing, and JRC has no data south of ~60°S where most of the gap is.

Steps:
  1. Find gap patches on ~2 km overviews: land in elevation.tif, no TWI,
     elevation spread > 2 m (not a lake), ≥ MIN_PATCH_KM2; map them to the
     Equi7 T6 tiles that hold them → data/gis/twi_fill/mask.tif
  2. Per tile: TWI by the recipe, cut into 1° cells on the published grid,
     kept only inside the gap mask → data/gis/twi_fill/cells/
  3. One pass to a new COG: the cells under the published TWI (published
     values always win), extended to the DTM's southern edge → twi.tif.
     The published file is kept as data/tmp/twi_published.tif until
     replaced on the next download.

Resumable: finished tiles leave a .done marker. Runs from download_dem.py
right after the published TWI is downloaded.
"""

from __future__ import annotations

import math
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

LAYERS_DIR  = Path("data/gis/layers")
ELEV_PATH   = LAYERS_DIR / "elevation.tif"
TWI_PATH    = LAYERS_DIR / "twi.tif"
WORK_DIR    = Path("data/gis/twi_fill")
CELLS_DIR   = WORK_DIR / "cells"
DONE_DIR    = WORK_DIR / "done"
SCRATCH_DIR = WORK_DIR / "scratch"
MASK_PATH   = WORK_DIR / "mask.tif"
PUBLISHED_BACKUP = Path("data/tmp/twi_published.tif")

DETECT_OVERVIEW_LEVEL = 5     # 64x: ~2 km
MIN_PATCH_KM2 = 5.0
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
    from scipy import ndimage
    from shapely.geometry import Point
    from equi7grid import get_standard_equi7grid

    with rasterio.open(ELEV_PATH, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as d:
        dem = d.read(1); dtf = d.transform; dnd = d.nodata
    with rasterio.open(TWI_PATH, OVERVIEW_LEVEL=DETECT_OVERVIEW_LEVEL) as t:
        twi = t.read(1); ttf = t.transform; tnd = t.nodata

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
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(MASK_PATH, "w", driver="GTiff", width=mask.shape[1], height=mask.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=tf, compress="deflate") as ds:
        ds.write(mask.astype(np.uint8), 1)


# ── Step 2: TWI for one tile, GEDTM30's recipe ────────────────────────────────

def _data_window(path: Path, nodata: float, margin: int = 2):
    """Row/col bounds of the valid data in a (mostly empty) raster."""
    with rasterio.open(path) as ds:
        rows = np.zeros(ds.height, bool); cols = np.zeros(ds.width, bool)
        for _, win in ds.block_windows(1):
            a = ds.read(1, window=win)
            v = a != nodata
            if v.any():
                rows[win.row_off: win.row_off + win.height] |= v.any(axis=1)
                cols[win.col_off: win.col_off + win.width] |= v.any(axis=0)
        if not rows.any():
            return None
        r = np.nonzero(rows)[0]; c = np.nonzero(cols)[0]
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
    bw = max(w - BUFFER_M, min(xs) - BUFFER_M - half); bs = max(s - BUFFER_M, min(ys) - BUFFER_M - half)
    be = min(e + BUFFER_M, max(xs) + BUFFER_M + half); bn = min(n + BUFFER_M, max(ys) + BUFFER_M + half)
    snap = lambda v, f: f(v / RES_M) * RES_M
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
        geo._run(["gdalbuildvrt", "-q", "-te", str(w), str(s), str(e_), str(n), "-tr", str(px), str(px),
                  "-srcnodata", str(TWI_NODATA), "-vrtnodata", str(TWI_NODATA), str(vrt), *cells, str(TWI_PATH)])
        geo._run(["gdal_translate", "-q", "-of", "VRT", "-a_scale", str(1 / TWI_SCALE), str(vrt), str(scaled)])
        with rasterio.open(scaled) as ds:
            factors = build_overviews._overview_factors_for_dataset(ds)
        print(f"[fill_twi] writing merged COG ({len(cells)} fill cells, overviews {factors})...", flush=True)
        build_overviews._build_cog(scaled, out_tmp, nominal=False, overview_factors=factors)
    PUBLISHED_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    TWI_PATH.replace(PUBLISHED_BACKUP)
    geo._commit(out_tmp, TWI_PATH)


def main() -> None:
    if not (ELEV_PATH.exists() and TWI_PATH.exists()):
        raise RuntimeError("fill_twi_gaps needs elevation.tif and twi.tif (download_dem.py)")
    mask, mask_tf, tiles = _find_gaps()
    if not tiles:
        print("[fill_twi] no land gaps — nothing to do")
        return
    _write_mask(mask, mask_tf)
    CELLS_DIR.mkdir(parents=True, exist_ok=True); DONE_DIR.mkdir(parents=True, exist_ok=True)
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
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True); SCRATCH_DIR.mkdir(parents=True)
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


if __name__ == "__main__":
    main()
