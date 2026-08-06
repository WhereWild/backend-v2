# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Rebuild the CANVAS_ROADS and GEONAMES_PLACES datasets embedded in the
frontend's offline map fallback (used when a species-page map can't reach
real basemap tiles). These are NOT part of the taxonomy tree pipeline
(rebuild.py) — this is a standalone, manually-rerun tool for frontend map
assets, kept here because this is where the GIS stack (geopandas/pyogrio,
already in pyproject.toml) lives.

Sources (public domain / CC-BY, matching the attribution already shown in
the offline fallback UI):
  - Natural Earth 1:10m Roads — https://naciscdn.org/naturalearth/10m/cultural/ne_10m_roads.zip
  - GeoNames cities1000       — https://download.geonames.org/export/dump/cities1000.zip

Roads are filtered to "significant" types only (see ROAD_KEEP_TYPES) — the
raw dataset is roughly evenly split between real highways and a huge
"Unknown"/generic "Road" bucket that's too fine-grained for a world-scale
fallback basemap.

Places are pruned with density-aware logic rather than a flat population
cutoff: a flat cutoff would silently erase small, isolated towns that are
the *only* label in their region while barely thinning dense clusters (e.g.
western Europe) where nearby larger cities already provide context. Instead,
each place is kept if either (a) it ranks in the top N by population within
its grid cell, or (b) it has no larger/comparable neighbor within
ISOLATION_DISTANCE_KM (isolation always wins regardless of rank).

Usage:
    uv run -m scripts.build_offline_fallback_data
    uv run -m scripts.build_offline_fallback_data --skip-download   # reuse cached sources

All thresholds below are tunable — re-run after editing to retune without
touching any other code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------

# Natural Earth road "type" values to keep. The other ~46% of the raw dataset
# ("Unknown", generic "Road", "Track", "Ferry Route", "Ferry, seasonal") is
# dropped as too minor for a world-scale offline fallback basemap.
ROAD_KEEP_TYPES = {"Major Highway", "Secondary Highway", "Beltway", "Bypass"}
# Also keep anything NE flags as an expressway, regardless of its `type` bucket.
ROAD_KEEP_EXPRESSWAY = True

# Population floor for a place to be considered at all (cities1000.txt is
# already pre-filtered to population >= 1000 or admin seats).
PLACE_POP_FLOOR = 1000
# Population -> minimum-visible-zoom bucket table, biggest first. A place's
# `z` is the first threshold its population meets or exceeds.
PLACE_Z_BUCKETS = [
    (5_000_000, 0),
    (1_000_000, 1),
    (500_000, 2),
    (200_000, 3),
    (100_000, 4),
    (50_000, 5),
    (20_000, 6),
    (10_000, 7),
    (5_000, 8),
    (2_000, 9),
    (0, 10),
]
# Density-aware pruning: grid the world into DENSE_GRID_DEG x DENSE_GRID_DEG
# cells, keep at most MAX_PLACES_PER_DENSE_CELL by population per cell...
DENSE_GRID_DEG = 1.0
MAX_PLACES_PER_DENSE_CELL = 15
# ...but always additionally keep a place if its nearest neighbor of
# comparable-or-greater population is farther than this, so isolated small
# towns survive even though a naive per-cell top-N would drop them.
ISOLATION_DISTANCE_KM = 150

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WHEREWILD_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = WHEREWILD_ROOT / "data" / "tmp" / "offline_fallback_sources"
ROADS_ZIP = SOURCES_DIR / "ne_10m_roads.zip"
ROADS_SHP = SOURCES_DIR / "ne_10m_roads" / "ne_10m_roads.shp"
PLACES_ZIP = SOURCES_DIR / "cities1000.zip"
PLACES_TXT = SOURCES_DIR / "cities1000.txt"

ROADS_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_roads.zip"
PLACES_URL = "https://download.geonames.org/export/dump/cities1000.zip"

FRONTEND_MAP_DIR = WHEREWILD_ROOT.parent / "frontend" / "components" / "sections" / "speciesOccurrenceMap"
PARTIALS_DIR = FRONTEND_MAP_DIR / "offlineFallbackPartials"

# Each target file + which const(s) it embeds a copy of.
TARGET_FILES = {
    PARTIALS_DIR / "leafletOfflineData.partial.js": ["CANVAS_ROADS", "GEONAMES_PLACES"],
    PARTIALS_DIR / "globeOfflineDataA.partial.js": ["CANVAS_ROADS"],
    PARTIALS_DIR / "globeOfflineDataB.partial.js": ["GEONAMES_PLACES"],
    FRONTEND_MAP_DIR / "SpeciesOccurrenceMapFallback.html": ["CANVAS_ROADS", "GEONAMES_PLACES"],
}

GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)


def ensure_sources(skip_download: bool) -> None:
    if not skip_download or not ROADS_SHP.exists():
        if not ROADS_ZIP.exists():
            _download(ROADS_URL, ROADS_ZIP)
        with zipfile.ZipFile(ROADS_ZIP) as zf:
            zf.extractall(ROADS_SHP.parent)
    if not skip_download or not PLACES_TXT.exists():
        if not PLACES_ZIP.exists():
            _download(PLACES_URL, PLACES_ZIP)
        with zipfile.ZipFile(PLACES_ZIP) as zf:
            zf.extractall(SOURCES_DIR)


# ---------------------------------------------------------------------------
# Web Mercator projection to the [0,1] normalized space already embedded
# ---------------------------------------------------------------------------

_MERCATOR_LAT_LIMIT = 85.05112878


def _project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_clamped = np.clip(lat, -_MERCATOR_LAT_LIMIT, _MERCATOR_LAT_LIMIT)
    x = (lon + 180.0) / 360.0
    lat_rad = np.radians(lat_clamped)
    y = 0.5 - np.log(np.tan(math.pi / 4 + lat_rad / 2)) / (2 * math.pi)
    return x, y


def _round6(x: float) -> float:
    return round(float(x), 6)


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def build_canvas_roads() -> str:
    gdf_all = gpd.read_file(ROADS_SHP)
    keep = gdf_all["type"].isin(ROAD_KEEP_TYPES)
    if ROAD_KEEP_EXPRESSWAY and "expressway" in gdf_all.columns:
        keep = keep | (gdf_all["expressway"] == 1)
    gdf = gdf_all[keep]
    print(f"Roads: kept {len(gdf)} / {len(gdf_all)} features after significance filter")

    entries = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        z_level = round(float(row["min_zoom"]), 1) if pd.notna(row["min_zoom"]) else 7.0
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        rings = []
        minx = miny = math.inf
        maxx = maxy = -math.inf
        for part in parts:
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            lons = np.array([c[0] for c in coords])
            lats = np.array([c[1] for c in coords])
            xs, ys = _project(lons, lats)
            flat: list[float] = []
            for x, y in zip(xs, ys):
                x6, y6 = _round6(x), _round6(y)
                flat.extend([x6, y6])
                minx, maxx = min(minx, x6), max(maxx, x6)
                miny, maxy = min(miny, y6), max(maxy, y6)
            rings.append(flat)
        if not rings:
            continue
        entries.append([[minx, miny, maxx, maxy], z_level, rings, 0])

    print(f"Roads: emitting {len(entries)} entries")
    return "const CANVAS_ROADS = " + json.dumps(entries, separators=(",", ":")) + ";"


# ---------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------


def _place_z(population: int) -> int:
    for threshold, z in PLACE_Z_BUCKETS:
        if population >= threshold:
            return z
    return PLACE_Z_BUCKETS[-1][1]


def build_geonames_places() -> str:
    df = pd.read_csv(
        PLACES_TXT,
        sep="\t",
        header=None,
        names=GEONAMES_COLUMNS,
        usecols=["name", "latitude", "longitude", "population"],
        dtype={"name": str, "latitude": float, "longitude": float, "population": "int64"},
        quoting=3,
        na_filter=False,
    )
    df = df[df["population"] >= PLACE_POP_FLOOR].reset_index(drop=True)
    print(f"Places: {len(df)} candidates after population floor ({PLACE_POP_FLOOR})")

    # Project to radians-on-a-sphere xyz for accurate great-circle nearest-
    # neighbor queries via a KD-tree (cheap, avoids a slow O(n^2) scan over
    # ~150k points).
    lat_rad = np.radians(df["latitude"].to_numpy())
    lon_rad = np.radians(df["longitude"].to_numpy())
    earth_radius_km = 6371.0
    xyz = np.column_stack([
        earth_radius_km * np.cos(lat_rad) * np.cos(lon_rad),
        earth_radius_km * np.cos(lat_rad) * np.sin(lon_rad),
        earth_radius_km * np.sin(lat_rad),
    ])
    tree = cKDTree(xyz)

    population = df["population"].to_numpy()
    # A place is "isolated" if it has no neighbor of >= population within
    # ISOLATION_DISTANCE_KM. Query a generous neighbor count and filter by
    # population in Python (a pure radius query would also count smaller
    # neighbors, which shouldn't count against isolation).
    isolated = np.zeros(len(df), dtype=bool)
    neighbor_idxs = tree.query_ball_point(xyz, r=ISOLATION_DISTANCE_KM)
    for i, neighbors in enumerate(neighbor_idxs):
        has_comparable_neighbor = any(j != i and population[j] >= population[i] for j in neighbors)
        isolated[i] = not has_comparable_neighbor

    # Dense-grid top-N thinning.
    grid_col = np.floor(df["longitude"].to_numpy() / DENSE_GRID_DEG).astype(int)
    grid_row = np.floor(df["latitude"].to_numpy() / DENSE_GRID_DEG).astype(int)
    df = df.assign(_grid=list(zip(grid_row, grid_col)), _isolated=isolated)
    df["_pop_rank_in_cell"] = df.groupby("_grid")["population"].rank(method="first", ascending=False)

    keep_mask = df["_isolated"] | (df["_pop_rank_in_cell"] <= MAX_PLACES_PER_DENSE_CELL)
    kept = df[keep_mask]
    print(
        f"Places: kept {len(kept)} / {len(df)} "
        f"({int(df['_isolated'].sum())} isolated, "
        f"{len(kept) - int((kept['_isolated']).sum())} via dense-cell top-N)"
    )

    entries = []
    for _, row in kept.iterrows():
        lon6 = round(float(row["longitude"]), 4)
        lat6 = round(float(row["latitude"]), 4)
        z = _place_z(int(row["population"]))
        entries.append([lon6, lat6, row["name"], z])

    return "const GEONAMES_PLACES = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";"


# ---------------------------------------------------------------------------
# Writing generated consts back into the repo
# ---------------------------------------------------------------------------

_CONST_LINE_RE = {
    "CANVAS_ROADS": re.compile(r"^\s*const CANVAS_ROADS = .*;\s*$", re.MULTILINE),
    "GEONAMES_PLACES": re.compile(r"^\s*const GEONAMES_PLACES = .*;\s*$", re.MULTILINE),
}


def write_const_into_file(path: Path, const_name: str, new_line: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = _CONST_LINE_RE[const_name]
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"{const_name} declaration not found in {path}")
    # Preserve the original line's leading indentation.
    original_line = match.group(0)
    indent = original_line[: len(original_line) - len(original_line.lstrip())]
    text = text[: match.start()] + indent + new_line + text[match.end() :]
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {const_name} into {path.relative_to(WHEREWILD_ROOT.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="reuse cached sources under data/tmp/offline_fallback_sources")
    args = parser.parse_args()

    ensure_sources(args.skip_download)

    roads_line = build_canvas_roads()
    places_line = build_geonames_places()
    generated = {"CANVAS_ROADS": roads_line, "GEONAMES_PLACES": places_line}

    for path, const_names in TARGET_FILES.items():
        for const_name in const_names:
            write_const_into_file(path, const_name, generated[const_name])


if __name__ == "__main__":
    main()
