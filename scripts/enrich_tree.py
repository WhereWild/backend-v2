# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Enrich the consolidated occurrences.parquet with GIS layer values.

Two sampling paths based on raster size:
- Small rasters (≤ MEMORY_MB_THRESHOLD): loaded fully into RAM on first use,
  then sampled with vectorized numpy indexing — no GDAL overhead per point.
- Large rasters (elevation, landcover, soilgrids, etc.): sampled with
  rasterio ds.sample() on hilbert-sorted coords so GDAL's block cache is
  effective. GDAL_CACHEMAX is set to 4 GB at startup.

Layers are processed in parallel threads.

Rows needing enrichment are streamed directly out of occurrences.parquet via
one DuckDB query (rather than opened file-by-file per taxon), in row_limit-
sized batches ordered by hilbertIdx. Each batch's sampled values are staged
to a small parquet file instead of being written back immediately; once all
batches are processed, one DuckDB pass joins every staged update into
occurrences.parquet and rewrites it — replacing what used to be a
read-modify-atomic-rewrite of every taxon's file, once per batch it
appeared in.
"""

from __future__ import annotations

import functools
import gc
import json
import multiprocessing
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rasterio
import rasterio.transform

from config.config import ZERO_NODATA_LAYERS, load_config
from util.gis import (
    DERIVED_FROM_ELEVATION,
    DERIVED_FROM_SOIL,
    sample_aspect_batch,
    sample_elevation_terrain_batch,
    sample_slope_batch,
    sample_soil_texture_batch,
)
from util.taxa import load_catalog

CONFIG = load_config("global")

OCCURRENCES_FILE = Path("data/taxonomy/occurrences.parquet")
STAGING_DIR = Path("data/taxonomy/.enrich_staging")
LAYERS_DIR = Path("data/gis/layers")
CATALOG_PATH = Path("config/gis/catalog.json")
ROW_LIMIT = 2_500_000

# _finalize_enrichment's LEFT JOIN + ORDER BY over the full occurrences
# table (tens of millions of rows) OOM-killed at ~58GB RSS with a bare
# duckdb.connect() — the same failure mode fixed in scripts/carry_forward.py
# and scripts/populate_tree.py's _duckdb_connect. memory_limit/temp_directory
# let it spill instead of crashing; preserve_insertion_order=false drops the
# default per-thread row-order buffering that isn't needed since the query
# already has its own explicit ORDER BY.
_DUCKDB_SPILL_DIR = Path("data/tmp/duckdb_spill")
_DUCKDB_MEMORY_LIMIT = "28GB"


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    _DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")
    return con

_LAYER_WORKERS = int(os.environ.get("ENRICH_LAYER_WORKERS", "1"))
# Rasters whose uncompressed footprint fits under this limit are loaded fully
# into RAM and sampled with vectorized numpy indexing. The array is held only
# for the duration of the sampling call and freed immediately after — no
# persistent cache, so at most _LAYER_WORKERS rasters live in memory at once.
# Default 24 GB covers SoilGrids (23 GB each). With _LAYER_WORKERS=1 this means
# peak RAM for rasters is one SoilGrids array at a time — safe on 64 GB hosts.
# Raising workers above 1 requires lowering this threshold proportionally.
_MEMORY_MB_THRESHOLD = int(os.environ.get("ENRICH_MEMORY_MB_THRESHOLD", "24000"))


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


_raw_vars = os.environ.get("VARS_TO_ENRICH", "")
VARS_TO_ENRICH: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

# Columns written by populate_tree — everything else is a GIS layer.
_BASE_COLS = frozenset([
    "decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx",
    "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
    "mediaUrl", "mediaAttribution", "mediaLicense",
])
_REQUIRED_COLS = ("decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx")

_LEGENDS_DIR = Path("config/gis/legends")

@functools.cache
def _valid_class_ids(layer_id: str) -> frozenset[int] | None:
    """Return valid class IDs from the legend file, or None if no legend exists."""
    base_id = re.sub(r'_(avg|sum|mode|mean|min|max)_\d+h$', '', layer_id)
    legend_path = _LEGENDS_DIR / f"{base_id}_legend.json"
    if not legend_path.exists():
        return None
    try:
        classes = json.loads(legend_path.read_text()).get("classes", [])
        return frozenset(int(c["id"]) for c in classes if "id" in c)
    except Exception:
        return None


def _load_layers() -> list[dict]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return [
        layer
        for category in cat["categories"]
        for layer in category["layers"]
        # Include raster layers (have a filename) and derived (no-file) layers
        if layer.get("filename")
        or layer.get("id") in DERIVED_FROM_ELEVATION
        or layer.get("id") in DERIVED_FROM_SOIL
    ]


@functools.lru_cache(maxsize=1)
def _temporal_layer_ids() -> frozenset[str]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return frozenset(
        layer["id"]
        for category in cat["categories"]
        if category.get("id") == "temporal"
        for layer in category["layers"]
    )


def _existing_columns() -> set[str]:
    """Column names currently in occurrences.parquet (empty set if it doesn't exist yet)."""
    if not OCCURRENCES_FILE.exists():
        return set()
    return set(pq.read_schema(OCCURRENCES_FILE).names)


def _stale_gis_columns(layer_ids: list[str], existing: set[str]) -> list[str]:
    """GIS/temporal columns present in the file that are no longer in the layer catalog."""
    allowed = _BASE_COLS | {"taxon_key"} | set(layer_ids)
    temporal_ids = _temporal_layer_ids()
    return [
        col for col in existing
        if col not in allowed
        and not any(col.startswith(tid + "_") for tid in temporal_ids)
    ]


def _scope_taxon_keys(root_key: str | int) -> list[str] | None:
    """taxon_keys of root_key and every descendant, resolved from the in-memory catalog.

    Occurrence rows carry only taxon_key (not a path — a taxon's ancestry
    already lives once in the catalog, no reason to duplicate it onto every
    one of its rows), so subtree scoping is a join against this key set
    rather than a stored-path LIKE predicate. Walks the catalog's own path
    field directly (catalog metadata, not per-row data) rather than going
    through util.taxa.iter_descendants, since that helper reads its own
    module-level cached catalog rather than this module's load_catalog.
    """
    catalog = load_catalog()
    root = catalog.get(str(root_key))
    if root is None:
        return None
    prefix = root["path"]
    return [
        str(t["taxon_key"]) for t in catalog.values()
        if t["path"] == prefix or t["path"].startswith(prefix + "/")
    ]


def _iter_worklist_batches(
    layer_ids: list[str],
    root_key: str | int,
    *,
    row_limit: int,
) -> Iterable[pa.Table]:
    """Stream worklist batches (rows missing >=1 target layer), row_limit rows at a time.

    Runs a single DuckDB scan over occurrences.parquet instead of opening one
    file per leaf taxon. Layers with no column yet in the file are treated as
    entirely missing (every in-scope row needs them) — same semantics as the
    old per-taxon "column absent" case, just evaluated once globally instead
    of file-by-file, since the schema is now shared by every row.
    """
    if not layer_ids or not OCCURRENCES_FILE.exists():
        return
    scope_keys = _scope_taxon_keys(root_key)
    if scope_keys is None:
        return

    existing = _existing_columns()
    present_layer_ids = [lid for lid in layer_ids if lid in existing]
    absent_layer_ids = [lid for lid in layer_ids if lid not in existing]

    select_cols = ["catalogNumber", "hilbertIdx", "decimalLatitude", "decimalLongitude", *present_layer_ids]
    col_list = ", ".join(f'"{c}"' for c in select_cols)

    where = '"taxon_key" IN (SELECT "taxon_key" FROM scope_keys)'
    if present_layer_ids and not absent_layer_ids:
        null_check = " OR ".join(f'"{lid}" IS NULL' for lid in present_layer_ids)
        where += f" AND ({null_check})"
    # If any layer is entirely absent, every in-scope row is missing it —
    # no null-check predicate needed, the scope filter alone selects everything.

    sql = (
        f"SELECT {col_list} FROM read_parquet('{OCCURRENCES_FILE.as_posix()}') "
        f"WHERE {where} ORDER BY hilbertIdx"
    )
    con = duckdb.connect()
    try:
        con.register("scope_keys", pa.table({"taxon_key": pa.array(scope_keys, type=pa.string())}))
        reader = con.execute(sql).to_arrow_reader(row_limit)
        batch_count = 0
        for record_batch in reader:
            table = pa.Table.from_batches([record_batch])
            if table.num_rows == 0:
                continue
            batch_count += 1
            print(f"[worklist] batch {batch_count}: {table.num_rows} rows pending GIS lookup")
            yield _build_worklist_chunk(table, present_layer_ids, absent_layer_ids)
    finally:
        con.close()


def _build_worklist_chunk(
    table: pa.Table, present_layer_ids: list[str], absent_layer_ids: list[str]
) -> pa.Table:
    """Attach a per-row missingLayers list, derived from null present-layer values
    plus every absent layer (which is missing for every row by definition)."""
    n = table.num_rows
    all_ids = present_layer_ids + absent_layer_ids
    null_cols = [np.asarray(pc.is_null(table.column(lid))) for lid in present_layer_ids]
    null_cols += [np.ones(n, dtype=bool) for _ in absent_layer_ids]
    layer_arr = np.array(all_ids)
    if null_cols:
        null_matrix = np.column_stack(null_cols)
        missing_layers = [layer_arr[row].tolist() for row in null_matrix]
    else:
        missing_layers = [[] for _ in range(n)]
    return pa.table({
        "catalogNumber":    table.column("catalogNumber"),
        "hilbertIdx":       table.column("hilbertIdx"),
        "decimalLatitude":  table.column("decimalLatitude"),
        "decimalLongitude": table.column("decimalLongitude"),
        "missingLayers":    pa.array(missing_layers, type=pa.list_(pa.large_string())),
    })


def _sample_cog_batch(
    path: Path,
    layer_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Sample a COG at the given coordinates. Returns float64 array (NaN = nodata/missing).

    Small rasters (≤ ENRICH_MEMORY_MB_THRESHOLD) are loaded fully into RAM for
    vectorized numpy indexing. The array lives only for the duration of this call
    so memory is freed as soon as the layer thread exits — at most _LAYER_WORKERS
    rasters occupy RAM simultaneously.

    Large rasters use ds.sample() with hilbert-sorted coords so GDAL's block cache
    stays effective.
    """
    n = len(lats)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    try:
        with rasterio.open(path) as ds:
            itemsize = np.dtype(ds.dtypes[0]).itemsize
            ram_mb = ds.width * ds.height * itemsize // 1024 // 1024
            nodata = ds.nodata
            if ram_mb <= _MEMORY_MB_THRESHOLD:
                # Load fully — vectorized numpy indexing, no per-point GDAL overhead.
                data = ds.read(1)
                h, w = ds.height, ds.width
                rows, cols = rasterio.transform.rowcol(ds.transform, lons, lats)
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)
                valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                if np.any(valid):
                    vals = data[rows[valid], cols[valid]].astype(np.float64)
                    if nodata is not None:
                        nd = vals == float(nodata)
                        vals[nd] = 0.0 if layer_id in ZERO_NODATA_LAYERS else np.nan
                    out[valid] = vals * scale + offset
            else:
                # Too large to load: ds.sample() with hilbert-sorted coords.
                coords = list(zip(lons.tolist(), lats.tolist()))
                for i, point in enumerate(ds.sample(coords)):
                    v = float(point[0])
                    if nodata is not None and v == nodata:
                        out[i] = 0.0 if layer_id in ZERO_NODATA_LAYERS else np.nan
                    else:
                        out[i] = v * scale + offset
    except Exception:
        pass
    return out


def _process_batch(worklist: pa.Table, layers: list[dict]) -> pa.Table | None:
    """Sample all layers for every row in the worklist.

    Returns a staging table of (catalogNumber, <sampled layer columns...>) for
    rows that received at least one value, or None if nothing was sampled.
    Columns are null for any row/layer combination this batch didn't touch,
    so a later ``COALESCE(update, existing)`` join leaves untouched values
    alone instead of clobbering them.
    """
    print(f"[process] batch start  rss={_rss_mb():.0f}MB  rows={worklist.num_rows}")
    df = worklist.to_pandas()
    if df.empty:
        return None
    df.sort_values("hilbertIdx", inplace=True)
    df.reset_index(drop=True, inplace=True)

    lats = df["decimalLatitude"].to_numpy(dtype=float)
    lons = df["decimalLongitude"].to_numpy(dtype=float)
    catalogs = df["catalogNumber"].astype(str).to_numpy()

    # Determine which rows need each layer
    layer_row_lists: dict[str, list[int]] = defaultdict(list)
    for row_idx, missing in enumerate(df["missingLayers"].tolist()):
        if missing is not None and len(missing) > 0:
            for lid in missing:
                layer_row_lists[lid].append(row_idx)
    layer_rows: dict[str, np.ndarray] = {
        lid: np.array(rows, dtype=np.int64) for lid, rows in layer_row_lists.items()
    }

    layer_meta = {layer["id"]: layer for layer in layers}

    elev_layer_id = "elevation"
    _terrain_ids = (DERIVED_FROM_ELEVATION | {elev_layer_id}) & layer_rows.keys()

    def _sample_layer(layer_id: str) -> tuple[str, np.ndarray]:
        """Sample one layer; returns (layer_id, full-length float64 array, NaN=missing)."""
        arr = layer_rows[layer_id]
        layer = layer_meta.get(layer_id)
        if layer is None:
            print(f"[warn] unknown layer {layer_id!r} in worklist; skipping")
            return layer_id, np.full(len(lats), np.nan)

        if layer_id in DERIVED_FROM_ELEVATION:
            elev_path = LAYERS_DIR / "elevation.tif"
            if not elev_path.exists():
                print(f"[skip] elevation.tif not found; cannot derive {layer_id}")
                return layer_id, np.full(len(lats), np.nan)
            if layer_id == "aspect":
                raw = sample_aspect_batch(lats[arr], lons[arr])
            else:
                raw = sample_slope_batch(lats[arr], lons[arr])
            vals = np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)
        elif layer_id in DERIVED_FROM_SOIL:
            raw = sample_soil_texture_batch(lats[arr], lons[arr])
            vals = np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)
        else:
            cog_path = LAYERS_DIR / layer["filename"]
            if not cog_path.exists():
                print(f"[warn] {cog_path.name} not found; skipping {layer_id}")
                return layer_id, np.full(len(lats), np.nan)
            scale = layer.get("scale_factor") or 1.0
            offset = layer.get("add_offset") or 0.0
            vals = _sample_cog_batch(cog_path, layer_id, lats[arr], lons[arr], scale, offset)
            vtype = layer.get("value_type", "")
            if vtype in ("nominal", "ordinal"):
                valid = _valid_class_ids(layer_id)
                if valid is not None:
                    finite = np.isfinite(vals)
                    int_vals = np.rint(np.where(finite, vals, 0)).astype(np.int64)
                    vals[finite & ~np.isin(int_vals, np.array(sorted(valid), dtype=np.int64))] = np.nan

        full = np.full(len(lats), np.nan, dtype=np.float64)
        full[arr] = vals
        return layer_id, full

    # Sentinel key used so the combined terrain job is distinguishable in the futures map.
    terrain_sentinel = "__terrain_combined__"

    def _sample_terrain_combined() -> tuple[str, list[tuple[str, np.ndarray]]]:
        """One combined pass over elevation.tif for all terrain layers simultaneously."""
        ids = sorted(_terrain_ids)
        idx_sets = [set(layer_rows[lid].tolist()) for lid in ids]
        common_set = idx_sets[0].intersection(*idx_sets[1:])
        if not common_set:
            return terrain_sentinel, []
        common_arr = np.array(sorted(common_set), dtype=np.int64)
        combo = sample_elevation_terrain_batch(
            lats[common_arr], lons[common_arr],
            want_elevation=elev_layer_id in _terrain_ids,
            want_slope="slope" in _terrain_ids,
            want_aspect="aspect" in _terrain_ids,
        )
        out: list[tuple[str, np.ndarray]] = []
        for lid, raw in combo.items():
            full = np.full(len(lats), np.nan, dtype=np.float64)
            vals = np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)
            if lid == elev_layer_id:
                meta = layer_meta.get(lid)
                if meta and meta.get("filename"):
                    scale = meta.get("scale_factor") or 1.0
                    offset = meta.get("add_offset") or 0.0
                    vals = vals * scale + offset
            full[common_arr] = vals
            out.append((lid, full))
        # Straggler rows (not in the common intersection) handled individually below
        for lid in ids:
            remaining = np.setdiff1d(layer_rows[lid], common_arr)
            if remaining.size > 0:
                _, arr_result = _sample_layer(lid)
                out.append((lid + "_straggler", arr_result))  # merged by caller
        return terrain_sentinel, out

    # Build the work queue: one job per non-terrain layer, one combined job for terrain.
    # Sort so in-memory layers run before ds.sample() layers (elevation, landcover).
    # With workers=1 this means the numpy-loaded layers (fast once in RAM) all
    # complete before the slow tile-by-tile passes start. Largest in-memory layers
    # (SoilGrids) go first within the in-memory group.
    def _layer_unc_mb(lid: str) -> float:
        meta = layer_meta.get(lid)
        if not meta or not meta.get("filename"):
            return 0.0
        p = LAYERS_DIR / meta["filename"]
        if not p.exists():
            return 0.0
        try:
            with rasterio.open(p) as ds:
                itemsize = np.dtype(ds.dtypes[0]).itemsize
                return ds.width * ds.height * itemsize / 1e6
        except Exception:
            return 0.0

    candidate_ids = [lid for lid in layer_rows if lid not in _terrain_ids]
    unc_mb_map = {lid: _layer_unc_mb(lid) for lid in candidate_ids}
    non_terrain_ids = sorted(
        candidate_ids,
        key=lambda lid: (0 if unc_mb_map[lid] <= _MEMORY_MB_THRESHOLD else 1, -unc_mb_map[lid]),
    )
    total = len(non_terrain_ids) + (len(_terrain_ids) if _terrain_ids else 0)
    layer_results: dict[str, np.ndarray] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=_LAYER_WORKERS) as executor:
        futures: dict = {}
        if len(_terrain_ids) > 1:
            futures[executor.submit(_sample_terrain_combined)] = terrain_sentinel
        else:
            # Single terrain layer — no combined pass benefit, just submit normally.
            for lid in _terrain_ids:
                futures[executor.submit(_sample_layer, lid)] = lid
        for lid in non_terrain_ids:
            futures[executor.submit(_sample_layer, lid)] = lid

        for future in as_completed(futures):
            result = future.result()
            if result[0] == terrain_sentinel:
                _, terrain_pairs = result
                for lid, arr_result in terrain_pairs:
                    base = lid.removesuffix("_straggler")
                    if base not in layer_results:
                        layer_results[base] = arr_result
                    else:
                        # Merge straggler: fill in any NaN slots from the combined pass
                        mask = np.isnan(layer_results[base])
                        layer_results[base][mask] = arr_result[mask]
                completed += len(_terrain_ids)
                for lid in sorted(_terrain_ids):
                    print(f"[process] layer {completed}/{total}: {lid} (combined terrain pass)")
            else:
                layer_id, full_values = result
                layer_results[layer_id] = full_values
                completed += 1
                print(f"[process] layer {completed}/{total}: {layer_id}  rss={_rss_mb():.0f}MB")

    rss_after_sample = _rss_mb()
    print(f"[process] sampling done — rss={rss_after_sample:.0f}MB")

    # Precompute a boolean mask per layer (size = worklist rows) marking which
    # rows this batch actually sampled that layer for.
    n_worklist = len(lats)
    layer_mask: dict[str, np.ndarray] = {}
    for lid, arr in layer_rows.items():
        m = np.zeros(n_worklist, dtype=bool)
        m[arr] = True
        layer_mask[lid] = m

    # Build one staging column per sampled layer, null everywhere this batch
    # didn't sample that row (so the later COALESCE join leaves those alone —
    # NaN is a real "sampled, no coverage" value and must stay distinguishable
    # from "not touched this batch").
    row_has_update = np.zeros(n_worklist, dtype=bool)
    update_arrays: dict[str, pa.Array] = {}
    for layer_id, full_values in layer_results.items():
        mask = layer_mask.get(layer_id)
        if mask is None or not mask.any():
            continue
        row_has_update |= mask
        update_arrays[layer_id] = pa.array(full_values, type=pa.float64(), mask=~mask)

    if not row_has_update.any() or not update_arrays:
        gc.collect()
        pa.default_memory_pool().release_unused()
        return None

    idx = pa.array(np.where(row_has_update)[0])
    staging = pa.table({
        "catalogNumber": pa.array(catalogs, type=pa.string()).take(idx),
        **{lid: arr.take(idx) for lid, arr in update_arrays.items()},
    })

    gc.collect()
    pa.default_memory_pool().release_unused()
    print(f"[process] batch done  rows_updated={staging.num_rows}  rss={_rss_mb():.0f}MB")
    return staging


def _sample_cog(
    path: Path,
    layer_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    scale: float,
    offset: float,
) -> list:
    """Compatibility shim around _sample_cog_batch; returns list[float | None]."""
    arr = _sample_cog_batch(path, layer_id, lats, lons, scale, offset)
    return [None if np.isnan(v) else float(v) for v in arr]


def _write_staging_batch(batch_idx: int, table: pa.Table | None) -> None:
    if table is None or table.num_rows == 0:
        return
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, STAGING_DIR / f"batch_{batch_idx:05d}.parquet")


def _finalize_enrichment(layer_ids: list[str]) -> None:
    """Join every staged batch's updates into occurrences.parquet in one pass,
    and drop any GIS/temporal columns no longer in the layer catalog.

    Replaces what used to be a read-modify-atomic-rewrite of every taxon's
    file, once per worklist batch it appeared in, with a single rewrite of
    the whole consolidated file.
    """
    staged = list(STAGING_DIR.glob("*.parquet")) if STAGING_DIR.exists() else []
    existing = _existing_columns()
    stale = _stale_gis_columns(layer_ids, existing)
    if not staged and not stale:
        return

    con = _duckdb_connect()
    try:
        base_cols = [
            r[0] for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{OCCURRENCES_FILE.as_posix()}') LIMIT 0"
            ).fetchall()
        ]

        update_cols: list[str] = []
        if staged:
            staging_glob = (STAGING_DIR / "*.parquet").as_posix()
            update_cols = [
                r[0] for r in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{staging_glob}', union_by_name=True) LIMIT 0"
                ).fetchall()
                if r[0] != "catalogNumber"
            ]
            # A catalogNumber should only appear in one batch's staging output
            # (each worklist row is scanned once per run), but dedupe defensively.
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW updates AS
                SELECT * FROM read_parquet('{staging_glob}', union_by_name=True)
                QUALIFY row_number() OVER (PARTITION BY "catalogNumber") = 1
            """)

        overlapping = [c for c in update_cols if c in base_cols]
        new_cols = [c for c in update_cols if c not in base_cols]
        exclude_cols = sorted(set(overlapping) | set(stale))

        select_parts = ["o.*"]
        if exclude_cols:
            select_parts[0] += " EXCLUDE (" + ", ".join(f'"{c}"' for c in exclude_cols) + ")"
        if overlapping:
            select_parts.append(", ".join(f'COALESCE(u."{c}", o."{c}") AS "{c}"' for c in overlapping))
        if new_cols:
            select_parts.append(", ".join(f'u."{c}"' for c in new_cols))
        select_clause = ", ".join(select_parts)

        from_clause = f"read_parquet('{OCCURRENCES_FILE.as_posix()}') o"
        if staged:
            from_clause += ' LEFT JOIN updates u USING ("catalogNumber")'

        tmp_dest = OCCURRENCES_FILE.with_suffix(".parquet.tmp")
        con.execute(f"""
            COPY (
                SELECT {select_clause} FROM {from_clause} ORDER BY taxon_key
            ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """)
    finally:
        con.close()
    tmp_dest.replace(OCCURRENCES_FILE)
    if stale:
        print(f"[finalize] dropped stale columns: {stale}")


def main() -> None:
    layers = _load_layers()
    if VARS_TO_ENRICH is not None:
        layers = [layer for layer in layers if layer["id"] in VARS_TO_ENRICH]
    layer_ids = [layer["id"] for layer in layers]

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    batch_count = 0
    for batch in _iter_worklist_batches(layer_ids, CONFIG.plantae_key, row_limit=ROW_LIMIT):
        if batch.num_rows == 0:
            continue
        batch_count += 1
        print(f"[worklist] processing batch {batch_count}")
        staging = _process_batch(batch, layers)
        _write_staging_batch(batch_count, staging)

    print("[finalize] merging staged updates into occurrences.parquet...")
    # Run in a fresh subprocess rather than in-process: the sampling loop
    # above was confirmed (via rebuild.log + kernel OOM logs) to still be
    # holding ~35GB RSS by the time it finishes — rasterio/GDAL buffers and
    # numpy sample arrays that either aren't fully released or, more likely,
    # freed by Python but not returned to the OS by glibc's allocator. That
    # baseline stacks on top of whatever _finalize_enrichment's own DuckDB
    # query needs, so tuning DuckDB's memory_limit down didn't help — the
    # process was already most of the way to the ceiling before the query
    # even started. spawn (not fork) gives the child a genuinely fresh
    # interpreter and heap, independent of whatever the parent accumulated.
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_finalize_enrichment, args=(layer_ids,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"_finalize_enrichment subprocess failed (exit code {proc.exitcode})")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("Completed GIS enrichment.")


if __name__ == "__main__":  # pragma: no cover
    main()
