# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a multi-valued lithology polygon source from Macrostrat's geologic map
vector tiles (https://tiles.macrostrat.org/, CC-BY 4.0).

Every Macrostrat map unit lists the lithology *types* present in it (up to 6
here; carbonate, siliciclastic, plutonic, ...). Nothing says one of them
dominates — the API lists them alphabetically — so this keeps the whole set
(see util/multiclass.py) instead of picking a primary class:

  LITH_MASK   bitmask of every lithology type present   (vector_set_field)
  LITH_SET_ID dense id of the exact combination         (vector_field, for
              tile rendering: one flat color per distinct combination)
  COLOR       Macrostrat's own unit color (informational)
  map_id / source_id   Macrostrat identifiers

How the tiles are used. The carto tileset blends every map in the database,
coarser maps at low zoom and more detailed ones at higher zoom, and a tile is
simply empty where nothing is mapped at that detail. So the tile pyramid is
crawled top-down (z0 -> --max-zoom), descending only into tiles that returned
data. Each location is then taken from the deepest tile covering it: a
parent tile's polygons are cut away wherever one of its child tiles has data.
Tile-edge fragments of one unit are re-merged by map_id.

Steps:
  1. Crawl the tile pyramid (resumable; tiles cached under RAW_DIR)
  2. Per zoom: ogr2ogr the cached tiles to a GeoPackage (-makevalid)
  3. Per zoom: merge fragments by map_id, cut away areas covered by deeper tiles
  4. Concatenate, encode lithology masks, write the parquet + legend

Usage (inside the gdal container):
    uv run python -m scripts.gis.download_macrostrat_lithology [--bbox W,S,E,N]
        [--max-zoom 13] [--threads 8] [--out PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from util.multiclass import MAX_MULTI_CLASSES, class_ids_to_mask

TILE_URL = "https://tiles.macrostrat.org/carto-slim/{z}/{x}/{y}"
LITHOLOGY_DEFS_URL = "https://macrostrat.org/api/v2/defs/lithologies?all&format=json"
UA = "wherewild-download-macrostrat/1.0"

RAW_DIR = Path("data/gis/macrostrat_raw")
WORK_DIR = Path("data/gis/macrostrat_work")
LEGENDS_DIR = Path("config/gis/legends")
DEFAULT_OUT = Path("data/gis/layers/macrostrat_lithology_vector.parquet")

# Below this size a tile is treated as empty (Macrostrat returns a ~0-byte body
# or a tile with an empty layer where nothing is mapped).
EMPTY_TILE_BYTES = 64
TILE_EXTENT = 4096
HALF_WORLD_M = 20037508.342789244

CITATION = (
    "Peters, S.E. et al. Macrostrat: a platform for geological data integration "
    "and deep-time research. https://macrostrat.org"
)


# ── tile math ────────────────────────────────────────────────────────────────

def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    lat = max(min(lat, 85.0511), -85.0511)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _tile_bounds_m(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    size = 2 * HALF_WORLD_M / (2 ** z)
    x0 = -HALF_WORLD_M + x * size
    y1 = HALF_WORLD_M - y * size
    return x0, y1 - size, x0 + size, y1


# ── crawl ────────────────────────────────────────────────────────────────────

def _tile_path(z: int, x: int, y: int) -> Path:
    return RAW_DIR / str(z) / str(x) / f"{y}.pbf"


def _fetch_tile(z: int, x: int, y: int) -> bool:
    """Download one tile into the cache; return True if it has data.

    Empty tiles are cached as zero-byte files so a resumed crawl never
    re-requests them.
    """
    path = _tile_path(z, x, y)
    if path.exists():
        return path.stat().st_size > EMPTY_TILE_BYTES
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": UA})
    body = b""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (204, 404):
                body = b""
                break
            if attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
    if len(body) <= EMPTY_TILE_BYTES:
        body = b""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    return len(body) > 0


def _crawl(bbox: tuple[float, float, float, float] | None, max_zoom: int, threads: int) -> dict[int, set[tuple[int, int]]]:
    """Return {z: {(x, y)}} of every non-empty tile, descending only where data exists."""
    nonempty: dict[int, set[tuple[int, int]]] = {}
    frontier = {(0, 0)}
    lock = threading.Lock()
    for z in range(max_zoom + 1):
        if bbox is not None:
            w, s, e, n = bbox
            x0, y1 = _lonlat_to_tile(w, s, z)
            x1, y0 = _lonlat_to_tile(e, n, z)
            frontier = {(x, y) for x, y in frontier if x0 <= x <= x1 and y0 <= y <= y1}
        todo = sorted(frontier)
        hits: set[tuple[int, int]] = set()

        def work(xy: tuple[int, int], z: int = z, hits: set[tuple[int, int]] = hits) -> None:
            if _fetch_tile(z, xy[0], xy[1]):
                with lock:
                    hits.add(xy)

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(work, todo))
        nonempty[z] = hits
        print(f"  z{z}: {len(hits)}/{len(todo)} tiles with data")
        frontier = {(2 * x + dx, 2 * y + dy) for x, y in hits for dx in (0, 1) for dy in (0, 1)}
        if not hits:
            break
    return nonempty


# ── per-zoom vectorization ───────────────────────────────────────────────────

def _zoom_geopackage(z: int, out: Path) -> Path | None:
    """ogr2ogr every cached tile of one zoom into a single 3857 GeoPackage."""
    src = RAW_DIR / str(z)
    if not src.exists():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    # MVT reads a z/x/y tree of .pbf files georeferenced as Web Mercator.
    # Invalid tile geometry makes GEOS print TopologyException noise; -makevalid
    # repairs those features instead of dropping them.
    result = subprocess.run(
        ["ogr2ogr", "-f", "GPKG", str(out), f"MVT:{RAW_DIR}/{z}", "units",
         "-makevalid", "-nlt", "PROMOTE_TO_MULTI"],
        check=False, capture_output=True, text=True,
    )
    if not out.exists():
        raise RuntimeError(f"ogr2ogr failed for z{z}:\n{result.stderr[-2000:]}")
    return out


def _load_lithology_types() -> tuple[list[str], dict[str, str]]:
    """(sorted lithology type names, type -> Macrostrat lithology class)."""
    with urllib.request.urlopen(
        urllib.request.Request(LITHOLOGY_DEFS_URL, headers={"User-Agent": UA}), timeout=60
    ) as resp:
        data = json.loads(resp.read())["success"]["data"]
    type_class = {d["type"]: d["class"] for d in data if d.get("type")}
    types = sorted(type_class)
    if len(types) > MAX_MULTI_CLASSES:
        raise RuntimeError(f"{len(types)} lithology types exceed the {MAX_MULTI_CLASSES}-bit mask")
    return types, type_class


def _build(nonempty: dict[int, set[tuple[int, int]]], types: list[str],
           type_class: dict[str, str], out_path: Path) -> None:
    import geopandas as gpd
    import pandas as pd
    import shapely
    from shapely.geometry import box

    type_id = {t: i for i, t in enumerate(types)}
    parts = []
    for z in sorted(nonempty):
        if not nonempty[z]:
            continue
        gpkg = _zoom_geopackage(z, WORK_DIR / f"units_z{z:02d}.gpkg")
        if gpkg is None:
            continue
        gdf = gpd.read_file(gpkg)
        if gdf.empty:
            continue
        type_cols = [c for c in gdf.columns if c.startswith("lith_type")]
        gdf["types"] = gdf[type_cols].apply(
            lambda r: tuple(sorted({v for v in r if isinstance(v, str) and v in type_id})), axis=1
        )
        gdf = gdf[gdf["types"].map(len) > 0]
        if gdf.empty:
            continue
        # Re-merge a unit's tile-edge fragments.
        gdf = gdf.dissolve(
            by="map_id", aggfunc={"source_id": "first", "color": "first", "types": "first"}
        ).reset_index()

        # Cut away whatever a deeper tile covers; that tile has the finer data.
        children = {
            (2 * x + dx, 2 * y + dy)
            for x, y in nonempty[z] for dx in (0, 1) for dy in (0, 1)
        } & nonempty.get(z + 1, set())
        if children:
            cover = shapely.union_all([box(*_tile_bounds_m(z + 1, cx, cy)) for cx, cy in children])
            hit = gdf.sindex.query(cover, predicate="intersects")
            geoms = gdf.geometry.values.copy()
            geoms[hit] = shapely.difference(geoms[hit], cover)
            gdf = gdf.set_geometry(gpd.GeoSeries(geoms, crs=gdf.crs))
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
        print(f"  z{z}: {len(gdf)} map units kept")
        parts.append(gdf)

    if not parts:
        raise RuntimeError("No Macrostrat polygons were fetched")
    allp = pd.concat(parts, ignore_index=True)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=3857)
    # Tile seams leave collinear vertices along every cut; tolerance 0 drops
    # those without moving any boundary.
    allp["geometry"] = shapely.simplify(allp.geometry.make_valid().values, 0, preserve_topology=True)

    combos = sorted(set(allp["types"]))
    set_id = {c: i for i, c in enumerate(combos)}
    allp["LITH_SET_ID"] = allp["types"].map(set_id).astype("int32")
    allp["LITH_MASK"] = allp["types"].map(
        lambda ts: float(class_ids_to_mask(type_id[t] for t in ts))
    )
    result = allp.rename(columns={"color": "COLOR"})[
        ["map_id", "source_id", "COLOR", "LITH_SET_ID", "LITH_MASK", "geometry"]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    result.to_parquet(tmp, compression="zstd")
    os.replace(tmp, out_path)
    print(f"  wrote {out_path}: {len(result)} polygons, {len(combos)} distinct lithology combinations")

    _write_legends(types, type_class, combos)


# Macrostrat types that only say the broad rock class ("sedimentary",
# "igneous", "metamorphic"), as opposed to a specific kind within it. Kept in
# the data exactly as Macrostrat reports them; flagged here so consumers can
# treat them as umbrellas of the specific types in the same group.
GENERIC_TYPES = frozenset({"sedimentary", "igneous", "metamorphic"})


def _write_legends(types: list[str], type_class: dict[str, str],
                   combos: list[tuple[str, ...]]) -> None:
    LEGENDS_DIR.mkdir(parents=True, exist_ok=True)
    type_legend = LEGENDS_DIR / "macrostrat_lithology_types.json"
    combo_legend = LEGENDS_DIR / "macrostrat_lithology_combinations.json"
    type_legend.write_text(json.dumps(
        {"layer_id": "macrostrat_lithology", "source": CITATION,
         "classes": [
             {"id": i, "name": t, "group": type_class[t], "group_label": type_class[t].title(),
              "traits": {"generic": t in GENERIC_TYPES}}
             for i, t in enumerate(types)
         ]},
        indent=2) + "\n")
    combo_legend.write_text(json.dumps(
        {"layer_id": "macrostrat_lithology_combination", "source": CITATION,
         "classes": [{"id": i, "name": " + ".join(c)} for i, c in enumerate(combos)]},
        indent=2) + "\n")
    print(f"  wrote {type_legend} and {combo_legend}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(bbox: tuple[float, float, float, float] | None, max_zoom: int, threads: int,
         out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (--force to rebuild)")
        return
    types, type_class = _load_lithology_types()
    print(f"{len(types)} lithology types: {', '.join(types)}")
    print(f"Crawling carto tiles (max zoom {max_zoom}, bbox={bbox}) ...")
    nonempty = _crawl(bbox, max_zoom, threads)
    print("Building polygons ...")
    _build(nonempty, types, type_class, out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Macrostrat multi-valued lithology source")
    parser.add_argument("--bbox", help="W,S,E,N in degrees (default: whole world)")
    parser.add_argument("--max-zoom", type=int, default=13)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--work-root", type=Path, help="put the tile cache, intermediates and legends here instead of data/ and config/ (trial runs)")
    args = parser.parse_args()
    if args.work_root:
        RAW_DIR = args.work_root / "raw"
        WORK_DIR = args.work_root / "work"
        LEGENDS_DIR = args.work_root / "legends"
    bb = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else None
    main(bb, args.max_zoom, args.threads, args.out, args.force)
