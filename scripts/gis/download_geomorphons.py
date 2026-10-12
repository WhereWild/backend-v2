# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Derive geomorphon landform classes from GEDTM30 (data/gis/layers/elevation.tif)
and build a single global COG, replicating Geomorpho90m's method (Amatulli et
al. 2020, Sci. Data 7:162) at GEDTM30's 30 m instead of MERIT's 90 m.

Geomorpho90m's recipe, per the paper's methods / code availability sections:
reproject the DEM to the Equi7 grid (seven azimuthal-equidistant continental
zones, EPSG:27701-27707) with gdalwarp bilinear, process it in buffered Equi7
"T6" tiles (600 × 600 km) with GRASS r.geomorphon (Jasiewicz & Stepinski 2013),
and reproject back to WGS84 with nearest neighbour. Geomorphons have to be
computed on a projected grid: r.geomorphon's 8-direction line-of-sight search
assumes square cells, which 1" cells aren't away from the equator.

The paper never lists r.geomorphon's arguments, so they are GRASS's defaults
(search=3 cells, skip=0, flat=1°, dist=0, the original "anglev1" comparison) —
a 300 m search radius at their 100 m cells. The search radius is held at the
same 300 m on the ground here (10 cells at 30 m), so the classes describe
landforms of the same size.

Their 401 km tile buffer was sized for their multiscale roughness layers; a
geomorphon only looks SEARCH_CELLS out, so BUFFER_CELLS a little past that
gives output bit-identical to processing a whole continent at once — provided
the warp is extent-independent too (see the gdalwarp flags below). Across
Equi7 zone borders cells come from different projections, so classes can
change there, as in Geomorpho90m; those borders mostly run through oceans.

Output classes (r.geomorphon forms, same codes as Geomorpho90m's geom layer):
  1 flat, 2 summit, 3 ridge, 4 shoulder, 5 spur, 6 slope, 7 hollow,
  8 footslope, 9 valley, 10 depression; 0 = nodata.

Steps:
  1. For each Equi7 T6 tile covering land: gdalwarp elevation.tif → buffered
     30 m Equi7 tile, run r.geomorphon in a throwaway GRASS location, export
     the unbuffered core to data/gis/geomorphons_equi7_tiles/<zone>/
  2. Nearest-neighbour each 1° cell back onto elevation.tif's own 1" grid,
     taking each pixel from the Equi7 zone whose footprint contains it →
     data/gis/geomorphons_1deg_tiles/
  3. Build a VRT spanning all 1° tiles
  4. gdal_translate → data/gis/layers/geomorphons.tif

A full run is days of CPU (≈ 13 min per mountainous Equi7 tile, 805 tiles).
Re-running is safe and resumes where it stopped: finished tiles (written
tmp → rename) and the final COG are skipped; --force rebuilds the COG from
the existing tiles.

Needs GRASS GIS (grass-core, see Dockerfile) and elevation.tif from
download_dem.py, which rebuild.py runs first (download_*.py run in name order).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

LAYERS_DIR       = Path("data/gis/layers")
ELEV_PATH        = LAYERS_DIR / "elevation.tif"
OUT_PATH         = LAYERS_DIR / "geomorphons.tif"
EQUI7_TILES_DIR  = Path("data/gis/geomorphons_equi7_tiles")  # <zone>/<tile>.tif, unbuffered core
WGS84_TILES_DIR  = Path("data/gis/geomorphons_1deg_tiles")   # <N00E000>.tif, elevation.tif grid
SCRATCH_DIR      = Path("data/gis/geomorphons_scratch")      # buffered DEM of the tile in progress

DEM_BOUNDS = (-180.0, -65.0, 180.0, 85.0)  # GEDTM30 v1.2
EQUI7_ZONES   = ("AF", "AN", "AS", "EU", "NA", "OC", "SA")
EQUI7_TILING  = "T6"
EQUI7_RES_M   = 30

# r.geomorphon — GRASS defaults as used by Geomorpho90m, search held at 300 m.
SEARCH_CELLS = 10
SKIP_CELLS   = 0
FLAT_DEG     = 1.0
DIST_CELLS   = 0
COMPARISON   = "anglev1"
BUFFER_CELLS = SEARCH_CELLS + 6

# elevation.tif overview (~240 m) used to find land equi7grid's mask misses.
LAND_CHECK_OVERVIEW_FACTOR = 8

# gdalwarp worker threads. All-core warps have twice panicked this box with a
# Machine Check on one physical core (logical CPUs 6/14) while all-core
# summary-stat loads don't — dense vector work on every core at once seems to
# be what trips it. The warp is ~35 s of a ~13 min tile, so few threads cost
# almost nothing.
WARP_THREADS = 4

DEM_NODATA  = -9999.0  # in the warped tiles; elevation.tif's own is read from it
GEOM_NODATA = 0

_GRASS_SCRIPT = """\
set -e
r.external input="$DEM" output=dem --quiet
g.region raster=dem
r.geomorphon elevation=dem forms=geom search=$SEARCH skip=$SKIP flat=$FLAT dist=$DIST comparison=$COMPARISON --quiet
g.region w=$W s=$S e=$E n=$N res=$RES
r.out.gdal -c -f input=geom output="$OUT" format=GTiff type=Byte nodata=0 \\
    createopt="COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=512,BLOCKYSIZE=512" --quiet
"""


def _commit(tmp: Path, dest: Path) -> None:
    """Rename a finished file into place durably. Without the fsyncs, a hard
    reset can keep the rename but not the data, leaving a zero-length file
    that a resumed run would take as done."""
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(dest)
    dir_fd = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _is_done(path: Path) -> bool:
    """A finished output — zero-length ones are debris from a crash, not results."""
    return path.exists() and path.stat().st_size > 0


def _log(msg: str) -> None:
    """print() that survives a kernel panic: a panic reboots before the page
    cache is written back, which otherwise loses the lines showing which step
    the run died in."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
    try:
        os.fsync(sys.stdout.fileno())
    except OSError:
        pass  # a terminal or pipe, nothing to sync


def _run(cmd: list[str], env: dict | None = None) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _land_tiles():
    """Yield (zone_name, epsg, tile) for every Equi7 T6 tile with land in
    elevation.tif inside its own zone.

    equi7grid's cover_land mask is coarser than the DEM: a global run that
    trusted it alone left ~65 km² of real land unprocessed (the Ninigo and
    Hermit Islands, the Recherche Archipelago, Palmerston Atoll, outer Yap
    islands and a strip of the Suriname coast). So every other tile in the
    DEM's bounds is also checked against elevation.tif's overview for land
    (any data: GEDTM30 stores sea as nodata).
    """
    from equi7grid import get_standard_equi7grid

    grid = get_standard_equi7grid({EQUI7_TILING: EQUI7_RES_M})
    footprints = _zone_footprints()
    with rasterio.open(ELEV_PATH) as ds:
        factors = ds.overviews(1)
    level = next((i for i, f in enumerate(factors) if f >= LAND_CHECK_OVERVIEW_FACTOR), None)
    if level is None:
        raise RuntimeError(f"{ELEV_PATH} has no overview ≥ {LAND_CHECK_OVERVIEW_FACTOR}× (it should be a COG)")
    with rasterio.open(ELEV_PATH, OVERVIEW_LEVEL=level) as ov:
        for zone_name in EQUI7_ZONES:
            zone = getattr(grid, zone_name)
            epsg = zone.pyproj_crs.to_epsg()
            land = {t.name for t in zone.get_tiles_in_geog_bbox(DEM_BOUNDS, EQUI7_TILING, cover_land=True)}
            for tile in zone.get_tiles_in_geog_bbox(DEM_BOUNDS, EQUI7_TILING, cover_land=False):
                if tile.name in land or _has_land(
                    ov, epsg, tile.outer_boundary_extent, footprints[zone_name],
                ):
                    yield zone_name, epsg, tile


def _has_land(ov, epsg: int, extent: tuple[float, float, float, float], zone_fp) -> bool:
    """Whether elevation.tif has land where this tile overlaps its zone."""
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    from shapely import affinity
    from shapely.geometry import Polygon, box

    fp = _tile_footprint_lonlat(epsg, extent)
    if fp.bounds[2] - fp.bounds[0] > 180:
        # Crosses the antimeridian: unwrap to 0–360, then fold the part east
        # of 180° back to -180.
        lon, lat = np.asarray(fp.exterior.coords).T
        unwrapped = Polygon(zip(np.where(lon < 0, lon + 360, lon), lat)).buffer(0)
        parts = [
            unwrapped.intersection(box(-180, -90, 180, 90)),
            affinity.translate(unwrapped.intersection(box(180, -90, 540, 90)), xoff=-360),
        ]
    else:
        parts = [fp.buffer(0)]
    for part in parts:
        area = part.intersection(zone_fp).intersection(box(*DEM_BOUNDS))
        if area.is_empty or area.area == 0:
            continue
        win = from_bounds(*area.bounds, ov.transform).round_offsets().round_lengths()
        if win.width < 1 or win.height < 1:
            continue
        elev = ov.read(1, window=win)
        inside = ~geometry_mask([area], out_shape=elev.shape, transform=ov.window_transform(win))
        if np.any(inside & (elev != ov.nodata) & np.isfinite(elev)):
            return True
    return False


def _zone_footprints() -> dict:
    """Unbuffered Equi7 zone polygons in lon/lat (antimeridian already split)."""
    from equi7grid import get_standard_equi7grid

    grid = get_standard_equi7grid({EQUI7_TILING: EQUI7_RES_M})
    return {z: getattr(grid, z).proj_zone_geog.geom for z in EQUI7_ZONES}


# ── Step 1: Equi7 tiles ───────────────────────────────────────────────────────

def _elev_nodata() -> float:
    with rasterio.open(ELEV_PATH) as ds:
        return ds.nodata

def _has_data(path: Path) -> bool:
    """Whether a warped tile has any valid cell — scanned block by block and
    stopping at the first hit, since a decimated read can step right over a
    tile whose only land is a small island."""
    with rasterio.open(path) as ds:
        for _, window in ds.block_windows(1):
            if np.any(ds.read(1, window=window) != DEM_NODATA):
                return True
    return False


def _tile_paths(zone_name: str, tile) -> tuple[Path, Path]:
    dest = EQUI7_TILES_DIR / zone_name / f"{tile.name}.tif"
    return dest, dest.with_suffix(".empty")


def _process_equi7_tile(zone_name: str, epsg: int, tile) -> Path | None:
    """Compute one Equi7 tile's geomorphons; None if it has no DEM data."""
    dest, empty_marker = _tile_paths(zone_name, tile)
    if _is_done(dest):
        return dest
    if empty_marker.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)

    w, s, e, n = tile.outer_boundary_extent
    buf = BUFFER_CELLS * EQUI7_RES_M
    dem_path = SCRATCH_DIR / f"{tile.name}_dem.tif"
    _log(f"    {tile.name}: warping ({WARP_THREADS} threads)")
    _run([
        "gdalwarp", "-overwrite", "-q",
        "-t_srs", f"EPSG:{epsg}",
        "-te", str(w - buf), str(s - buf), str(e + buf), str(n + buf),
        "-tr", str(EQUI7_RES_M), str(EQUI7_RES_M),
        "-r", "bilinear",
        # Make every output cell a function of its own coordinates only, so a
        # cell's elevation is identical in its own tile and in a neighbour's
        # buffer (otherwise tiles disagree by cm–m and seams appear): exact
        # reprojection instead of the per-scanline approximation (-et 0), a
        # plain 2×2 bilinear kernel instead of one rescaled per warp chunk
        # (XSCALE/YSCALE=1), and always the full-resolution source (-ovr NONE).
        # The cost: no low-pass where the DEM's 1" cells are narrower than 30 m
        # (east-west, above ~45° latitude).
        "-et", "0", "-wo", "XSCALE=1", "-wo", "YSCALE=1", "-ovr", "NONE",
        "-srcnodata", repr(_elev_nodata()), "-dstnodata", str(DEM_NODATA),
        "-ot", "Float32",
        "-wm", "2048", "-multi", "-wo", f"NUM_THREADS={WARP_THREADS}",
        "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
        str(ELEV_PATH), str(dem_path),
    ])
    try:
        if not _has_data(dem_path):
            empty_marker.touch()
            return None
        _log(f"    {tile.name}: r.geomorphon")

        tmp_out = dest.with_suffix(".tif.tmp")
        tmp_out.unlink(missing_ok=True)  # partial export from a killed run; r.out.gdal won't overwrite
        env = os.environ | {
            # grass-core brings its own libgdal; keep it from loading the gdal
            # image's plugins, built for a different GDAL (harmless but noisy).
            "GDAL_DRIVER_PATH": "disable",
            "DEM": str(dem_path.resolve()), "OUT": str(tmp_out.resolve()),
            "SEARCH": str(SEARCH_CELLS), "SKIP": str(SKIP_CELLS),
            "FLAT": str(FLAT_DEG), "DIST": str(DIST_CELLS), "COMPARISON": COMPARISON,
            "W": str(w), "S": str(s), "E": str(e), "N": str(n), "RES": str(EQUI7_RES_M),
        }
        _run(["grass", "--tmp-location", f"EPSG:{epsg}", "--exec", "bash", "-c", _GRASS_SCRIPT], env=env)
        _commit(tmp_out, dest)
        return dest
    finally:
        dem_path.unlink(missing_ok=True)


# ── Step 2: back to WGS84 1° tiles ────────────────────────────────────────────

def _cell_name(lon: int, lat: int) -> str:
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def _tile_footprint_lonlat(epsg: int, extent: tuple[float, float, float, float]):
    """Densified lon/lat outline of an Equi7 tile extent."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    w, s, e, n = extent
    steps = np.linspace(0.0, 1.0, 33)
    xs = np.concatenate([w + (e - w) * steps, np.full(33, e), e - (e - w) * steps, np.full(33, w)])
    ys = np.concatenate([np.full(33, s), s + (n - s) * steps, np.full(33, n), n - (n - s) * steps])
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xs, ys)
    return Polygon(zip(lon, lat))


def _cells_for_tiles(done: list[tuple[str, int, tuple]], footprints: dict) -> list[tuple[int, int]]:
    """1° cells (lon, lat of SW corner) covered by any processed Equi7 tile."""
    from shapely.geometry import box

    cells: set[tuple[int, int]] = set()
    for zone_name, epsg, extent in done:
        fp = _tile_footprint_lonlat(epsg, extent)
        if fp.bounds[2] - fp.bounds[0] > 180:
            # Wraps the antimeridian in lon/lat — fall back to the zone's own
            # (already split) footprint for enumeration.
            fp = footprints[zone_name]
        fp = fp.intersection(footprints[zone_name])
        if fp.is_empty:
            continue
        x0, y0, x1, y1 = fp.bounds
        for lat in range(max(math.floor(y0), math.floor(DEM_BOUNDS[1])), min(math.ceil(y1), math.ceil(DEM_BOUNDS[3]))):
            for lon in range(max(math.floor(x0), -180), min(math.ceil(x1), 180)):
                if fp.intersects(box(lon, lat, lon + 1, lat + 1)):
                    cells.add((lon, lat))
    return sorted(cells)


def _write_wgs84_cell(lon: int, lat: int, zone_srcs: dict, footprints: dict, elev_tf) -> Path | None:
    dest = WGS84_TILES_DIR / f"{_cell_name(lon, lat)}.tif"
    if _is_done(dest):
        return dest

    # Same 1" grid as elevation.tif, whose edges sit half a pixel off whole
    # degrees — so these land on k + 0.5, where round() would tip either way
    # on float noise and scatter tiles 1" apart. floor() picks the pixel whose
    # centre is at `lon`/`lat + 1`.
    px = elev_tf.a
    col0 = math.floor((lon - elev_tf.c) / px)
    row0 = math.floor((elev_tf.f - (lat + 1)) / px)
    size = round(1.0 / px)
    dst_tf = from_origin(elev_tf.c + col0 * px, elev_tf.f - row0 * px, px, px)

    shapes = [(footprints[z], i + 1) for i, z in enumerate(EQUI7_ZONES) if z in zone_srcs]
    zone_ids = rasterize(shapes, out_shape=(size, size), transform=dst_tf, fill=0, dtype="uint8")
    out = np.zeros((size, size), dtype=np.uint8)
    for i, zone_name in enumerate(EQUI7_ZONES):
        mask = zone_ids == i + 1
        if zone_name not in zone_srcs or not mask.any():
            continue
        warped = np.zeros((size, size), dtype=np.uint8)
        reproject(
            source=rasterio.band(zone_srcs[zone_name], 1), destination=warped,
            src_nodata=GEOM_NODATA, dst_transform=dst_tf, dst_crs="EPSG:4326",
            dst_nodata=GEOM_NODATA, resampling=Resampling.nearest,
        )
        out[mask] = warped[mask]

    if not out.any():
        return None
    tmp = dest.with_suffix(".tif.tmp")
    with rasterio.open(
        tmp, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint8",
        crs="EPSG:4326", transform=dst_tf, nodata=GEOM_NODATA,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    ) as ds:
        ds.write(out, 1)
    _commit(tmp, dest)
    return dest


def _reproject_to_wgs84(done: list[tuple[str, int, tuple]]) -> list[Path]:
    footprints = _zone_footprints()
    cells = _cells_for_tiles(done, footprints)
    WGS84_TILES_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ELEV_PATH) as elev:
        elev_tf = elev.transform

    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        zone_srcs = {}
        for zone_name in EQUI7_ZONES:
            zone_tiles = sorted((EQUI7_TILES_DIR / zone_name).glob("*.tif"))
            if zone_tiles:
                vrt = Path(tmp_dir) / f"{zone_name}.vrt"
                _run(["gdalbuildvrt", "-overwrite", "-q", str(vrt), *[str(t) for t in zone_tiles]])
                zone_srcs[zone_name] = rasterio.open(vrt)
        try:
            for j, (lon, lat) in enumerate(cells, 1):
                path = _write_wgs84_cell(lon, lat, zone_srcs, footprints, elev_tf)
                if path is not None:
                    written.append(path)
                if j % 500 == 0 or j == len(cells):
                    print(f"  [{j}/{len(cells)}] 1° cells", flush=True)
        finally:
            for ds in zone_srcs.values():
                ds.close()
    return written


# ── Step 3/4: global COG ──────────────────────────────────────────────────────

def _build_global_cog(tile_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tif.tmp")

    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt_path = Path(tmp_dir) / "global.vrt"

        print(f"Building VRT from {len(tile_paths)} tiles...", flush=True)
        _run([
            "gdalbuildvrt", "-overwrite", "-resolution", "highest",
            str(vrt_path), *[str(p) for p in sorted(tile_paths)],
        ])

        print("Translating to GeoTIFF...", flush=True)
        _run([
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=4",
            "-a_nodata", str(GEOM_NODATA),
            str(vrt_path), str(tmp_out),
        ])

    if out_path.exists():
        out_path.unlink()
    _commit(tmp_out, out_path)


def main(force: bool = False) -> None:
    if not force and VARS_TO_DOWNLOAD is not None and "geomorphons" not in VARS_TO_DOWNLOAD:
        print("[download_geomorphons] skipped (geomorphons not in VARS_TO_DOWNLOAD)")
        return

    if OUT_PATH.exists() and not force:
        print(f"COG already exists: {OUT_PATH} (use --force to rebuild)")
        return
    if not ELEV_PATH.exists():
        raise RuntimeError(f"{ELEV_PATH} not found — geomorphons are derived from it (download_dem.py)")
    if shutil.which("grass") is None:
        raise RuntimeError("GRASS GIS (`grass`) not found — install grass-core (see Dockerfile)")

    tiles = list(_land_tiles())
    print(f"Computing geomorphons for {len(tiles)} Equi7 {EQUI7_TILING} tiles at {EQUI7_RES_M} m "
          f"(search={SEARCH_CELLS} cells, skip={SKIP_CELLS}, flat={FLAT_DEG}°, dist={DIST_CELLS})...")

    # Fixed scratch dir wiped on start: a killed run or reboot would otherwise
    # strand the ~1.6 GB buffered DEM of the tile it was on.
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(parents=True)
    done: list[tuple[str, int, tuple]] = []
    computed, compute_s = 0, 0.0
    try:
        for i, (zone_name, epsg, tile) in enumerate(tiles, 1):
            dest, empty_marker = _tile_paths(zone_name, tile)
            resumed = _is_done(dest) or empty_marker.exists()
            t0 = time.perf_counter()
            path = _process_equi7_tile(zone_name, epsg, tile)
            if path is not None:
                done.append((zone_name, epsg, tile.outer_boundary_extent))
            if resumed:
                continue
            # ETA from tiles computed this run; rough, since mostly-ocean tiles
            # take seconds and mountain tiles ~13 min.
            dt = time.perf_counter() - t0
            computed += 1
            compute_s += dt
            eta_h = compute_s / computed * (len(tiles) - i) / 3600
            status = "no data" if path is None else f"{dt:.0f}s"
            _log(f"  [{i}/{len(tiles)}] {tile.name}: {status}  (ETA {eta_h:.1f} h)")
    finally:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    print(f"{len(done)} Equi7 tiles with data ({computed} computed this run)")

    print("\nReprojecting to WGS84 1° tiles...")
    tile_paths = _reproject_to_wgs84(done)
    if not tile_paths:
        print("No tiles produced — aborting.")
        return

    print(f"\nBuilding global COG → {OUT_PATH}")
    _build_global_cog(tile_paths, OUT_PATH)
    print(f"\nDone. Wrote {OUT_PATH}")
    print(
        f"Once {OUT_PATH} checks out, the intermediate tiles can be deleted to free space: "
        f"{EQUI7_TILES_DIR}/ and {WGS84_TILES_DIR}/ (kept so --force can rebuild the COG "
        f"without recomputing them — deleting them means a rebuild starts from scratch)."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Derive geomorphon landform classes from GEDTM30")
    parser.add_argument("--force", action="store_true", help="Rebuild the COG even if it already exists")
    args = parser.parse_args()
    main(force=args.force)
