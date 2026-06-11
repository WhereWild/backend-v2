"""
Enrich per-taxon occurrence parquets with GIS layer values.

Two sampling paths based on raster size:
- Small rasters (≤ MEMORY_MB_THRESHOLD): loaded fully into RAM on first use,
  then sampled with vectorized numpy indexing — no GDAL overhead per point.
- Large rasters (elevation, landcover, soilgrids, etc.): sampled with
  rasterio ds.sample() on hilbert-sorted coords so GDAL's block cache is
  effective. GDAL_CACHEMAX is set to 4 GB at startup.

Layers are processed in parallel threads.
"""

from __future__ import annotations

import functools
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rasterio
import rasterio.transform

from config.config import ZERO_NODATA_LAYERS, load_config
from util.gis import (
    DERIVED_FROM_ELEVATION,
    sample_aspect_batch,
    sample_elevation_terrain_batch,
    sample_slope_batch,
)
from util.taxa import TaxonRecord, load_catalog

CONFIG = load_config("global")

TREE_ROOT = Path("data/taxonomy/tree")
LAYERS_DIR = Path("data/gis/layers")
CATALOG_PATH = Path("config/gis/catalog.json")
OCCURRENCE_FILE = "occurrence.parquet"
ROW_LIMIT = 2_500_000

_LAYER_WORKERS = int(os.environ.get("ENRICH_LAYER_WORKERS", "1"))
# Rasters whose uncompressed footprint fits under this limit are loaded fully
# into RAM and sampled with vectorized numpy indexing. The array is held only
# for the duration of the sampling call and freed immediately after — no
# persistent cache, so at most _LAYER_WORKERS rasters live in memory at once.
# Default 24 GB covers SoilGrids (23 GB each). With _LAYER_WORKERS=1 this means
# peak RAM for rasters is one SoilGrids array at a time — safe on 64 GB hosts.
# Raising workers above 1 requires lowering this threshold proportionally.
_MEMORY_MB_THRESHOLD = int(os.environ.get("ENRICH_MEMORY_MB_THRESHOLD", "24000"))

# GDAL block cache for the ds.sample() large-raster path (elevation, landcover).
# In-memory rasters bypass GDAL entirely so this doesn't affect them.
# Default: total RAM − 16 GB OS floor − in-memory raster budget − 4 GB working data,
# capped at 48 GB. Overridable via GDAL_CACHEMAX in the environment (MB).
def _default_gdal_cachemax_mb() -> int:
    try:
        total_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (AttributeError, ValueError):
        total_mb = 16 * 1024  # fallback: assume 16 GB, cache gets 512 MB minimum
    floor_mb = 16 * 1024
    raster_budget_mb = _LAYER_WORKERS * _MEMORY_MB_THRESHOLD
    working_mb = 4 * 1024
    safe_mb = max(512, total_mb - floor_mb - raster_budget_mb - working_mb)
    return min(safe_mb, 48 * 1024)

os.environ.setdefault("GDAL_CACHEMAX", str(_default_gdal_cachemax_mb()))

_raw_vars = os.environ.get("VARS_TO_ENRICH", "")
VARS_TO_ENRICH: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

# Columns written by populate_tree — everything else is a GIS layer.
_BASE_COLS = frozenset([
    "decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx",
    "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
])
_REQUIRED_COLS = ("decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx")


def _load_layers() -> list[dict]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return [
        layer
        for category in cat["categories"]
        for layer in category["layers"]
        # Include raster layers (have a filename) and derived-from-elevation layers
        if layer.get("filename") or layer.get("id") in DERIVED_FROM_ELEVATION
    ]


def _atomic_write(path: Path, table: pa.Table) -> None:
    from util.storage import atomic_write_parquet
    atomic_write_parquet(path, table, row_group_size=50_000)


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


def _drop_stale_gis_columns(df, layer_ids: list[str], data_path: Path) -> None:
    """Remove GIS columns that are no longer in the layer catalog."""
    allowed = _BASE_COLS | set(layer_ids)
    temporal_ids = _temporal_layer_ids()
    stale = [
        col for col in df.columns
        if col not in allowed
        and not any(col.startswith(tid + "_") for tid in temporal_ids)
    ]
    if not stale:
        return
    df.drop(columns=stale, inplace=True)
    _atomic_write(data_path, pa.Table.from_pandas(df, preserve_index=False))


def _missing_rows_for_taxon(taxon: TaxonRecord, layer_ids: list[str]) -> pa.Table | None:
    """Return a worklist chunk for rows missing GIS values, or None if nothing to do.

    Only rows with at least one null layer value are included; per-row
    missingLayers lists only the layers that are actually null for that row.
    Rows already fully enriched (carry_forward or a prior run) are skipped.
    """
    data_path = TREE_ROOT / taxon["path"] / OCCURRENCE_FILE
    if not data_path.exists():
        return None
    table = pq.read_table(data_path)
    if table.num_rows == 0:
        return None
    df = table.to_pandas()
    if any(col not in df.columns for col in _REQUIRED_COLS):
        return None
    _drop_stale_gis_columns(df, layer_ids, data_path)

    # Use pc.is_null on the Arrow table (not pandas isna) so that no-coverage
    # sentinels (NaN for continuous, -1 for nominal) are not re-queued.
    # pc.is_null returns False for both NaN and -1 since they are real values,
    # so a single check handles all cases.
    null_cols = [
        np.asarray(pc.is_null(table.column(lid))) if lid in table.schema.names
        else np.ones(table.num_rows, dtype=bool)
        for lid in layer_ids
    ]
    if not null_cols:
        return None
    null_matrix = np.column_stack(null_cols)  # shape: (n_rows, n_layers)
    has_missing = null_matrix.any(axis=1)
    if not has_missing.any():
        return None  # all rows already fully enriched

    df_f = df[has_missing].reset_index(drop=True)
    null_f = null_matrix[has_missing]
    layer_arr = np.array(layer_ids)
    missing_layers = [layer_arr[row].tolist() for row in null_f]

    return pa.table({
        "catalogNumber":    pa.array(df_f["catalogNumber"].astype(str).tolist(), type=pa.string()),
        "hilbertIdx":       pa.array(df_f["hilbertIdx"].to_numpy(),               type=pa.int32()),
        "decimalLatitude":  pa.array(df_f["decimalLatitude"].to_numpy(),          type=pa.float64()),
        "decimalLongitude": pa.array(df_f["decimalLongitude"].to_numpy(),         type=pa.float64()),
        "missingLayers":    pa.array(missing_layers,                              type=pa.list_(pa.large_string())),
        "taxonKey":         pa.array([taxon["taxon_key"]] * len(df_f),            type=pa.string()),
        "dataPath":         pa.array([str(data_path)] * len(df_f),                type=pa.string()),
    })


def _iter_leaf_taxa(root_key: str | int) -> Iterable[TaxonRecord]:
    """Yield all leaf-rank taxa that are descendants of root_key (inclusive)."""
    root = load_catalog().get(str(root_key))
    if root is None:
        return
    prefix = root["path"]
    for taxon in load_catalog().values():
        if taxon["rank"] not in CONFIG.leaf_rank_set:
            continue
        if taxon["path"].startswith(prefix):
            yield taxon


def _iter_worklist_batches(
    layer_ids: list[str],
    root_key: str | int,
    *,
    row_limit: int,
) -> Iterable[pa.Table]:
    """Yield worklist batches sorted by hilbertIdx, capped at row_limit rows."""
    chunks: list[pa.Table] = []
    total_rows = 0
    batch_rows = 0
    for idx, taxon in enumerate(_iter_leaf_taxa(root_key), 1):
        chunk = _missing_rows_for_taxon(taxon, layer_ids)
        if chunk is None or chunk.num_rows == 0:
            continue
        chunks.append(chunk)
        total_rows += chunk.num_rows
        batch_rows += chunk.num_rows
        if idx % 1000 == 0:
            print(f"[worklist] scanned {idx} taxa, captured {total_rows} rows")
        if batch_rows >= row_limit:
            print(f"[worklist] concatenating {len(chunks)} chunks ({batch_rows} rows)")
            worklist = pa.concat_tables(chunks).combine_chunks().sort_by([("hilbertIdx", "ascending")])
            print(f"[worklist] batch rows pending GIS lookup: {worklist.num_rows}")
            yield worklist
            chunks = []
            batch_rows = 0
    if not chunks:
        return
    print(f"[worklist] concatenating {len(chunks)} chunks ({batch_rows} rows)")
    worklist = pa.concat_tables(chunks).combine_chunks().sort_by([("hilbertIdx", "ascending")])
    print(f"[worklist] batch rows pending GIS lookup: {worklist.num_rows}")
    yield worklist


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


def _process_batch(worklist: pa.Table, layers: list[dict]) -> None:
    """Sample all layers for every row in the worklist and flush results."""
    df = worklist.to_pandas()
    if df.empty:
        return
    df.sort_values("hilbertIdx", inplace=True)
    df.reset_index(drop=True, inplace=True)

    lats = df["decimalLatitude"].to_numpy(dtype=float)
    lons = df["decimalLongitude"].to_numpy(dtype=float)
    catalogs = df["catalogNumber"].astype(str).to_numpy()
    taxon_keys = df["taxonKey"].to_numpy()
    data_paths_arr = df["dataPath"].to_numpy()

    taxon_paths: dict[str, str] = {}
    for tk, dp in zip(taxon_keys, data_paths_arr):
        taxon_paths.setdefault(tk, dp)

    # Map taxon_key → worklist row indices
    unique_taxa, inverse = np.unique(taxon_keys, return_inverse=True)
    taxon_to_rows: dict[str, np.ndarray] = {
        tk: np.where(inverse == i)[0] for i, tk in enumerate(unique_taxa)
    }

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
        else:
            cog_path = LAYERS_DIR / layer["filename"]
            if not cog_path.exists():
                print(f"[warn] {cog_path.name} not found; skipping {layer_id}")
                return layer_id, np.full(len(lats), np.nan)
            scale = layer["scale_factor"] if layer["scale_factor"] is not None else 1.0
            offset = layer["add_offset"] if layer["add_offset"] is not None else 0.0
            vals = _sample_cog_batch(cog_path, layer_id, lats[arr], lons[arr], scale, offset)

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
                    scale = meta["scale_factor"] if meta["scale_factor"] is not None else 1.0
                    offset = meta["add_offset"] if meta["add_offset"] is not None else 0.0
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
                print(f"[process] layer {completed}/{total}: {layer_id}")

    # Flush per taxon — read parquet once, assign all layers at once
    for taxon_key, row_indices in taxon_to_rows.items():
        data_path = taxon_paths.get(taxon_key)
        if not data_path:
            continue
        data_file = Path(data_path)
        if not data_file.exists():
            continue

        # Include ALL sampled rows (even NaN = no-coverage) so we can stamp the
        # no-coverage sentinel and avoid re-processing them on future rebuilds.
        taxon_updates: dict[str, np.ndarray] = {}
        for layer_id, full_values in layer_results.items():
            t_vals = full_values[row_indices]
            if t_vals.size > 0:
                taxon_updates[layer_id] = t_vals

        if not taxon_updates:
            continue

        table = pq.read_table(data_file)
        df_taxon = table.to_pandas()
        if df_taxon.empty or "catalogNumber" not in df_taxon.columns:
            continue

        catalog_arr = df_taxon["catalogNumber"].astype(str).to_numpy()
        catalog_index = {v: i for i, v in enumerate(catalog_arr)}

        for layer_id, t_vals in taxon_updates.items():
            if layer_id not in df_taxon.columns:
                df_taxon[layer_id] = np.nan
            col = df_taxon[layer_id].to_numpy(dtype=np.float64, copy=True)
            df_indices = np.array([catalog_index.get(c, -1) for c in catalogs[row_indices]])
            valid = df_indices >= 0
            col[df_indices[valid]] = t_vals[valid]
            df_taxon[layer_id] = col

        # Stamp NaN (not null) for no-coverage rows so they aren't re-queued on
        # future rebuilds. pa.Table.from_pandas converts NaN→null for float cols,
        # so we restore NaN in Arrow after the conversion.
        # NaN is ignored by all stats computation (same as for continuous layers).
        arrow_table = pa.Table.from_pandas(df_taxon, preserve_index=False)
        new_columns = {}
        for layer_id in taxon_updates:
            if layer_id not in arrow_table.schema.names:
                continue
            col = arrow_table.column(layer_id)
            if pa.types.is_floating(col.type):
                new_columns[layer_id] = pc.if_else(pc.is_null(col), float("nan"), col)
        if new_columns:
            for col_name, new_col in new_columns.items():
                idx = arrow_table.schema.get_field_index(col_name)
                arrow_table = arrow_table.set_column(idx, col_name, new_col)

        _atomic_write(data_file, arrow_table)


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


def _flush_taxon_updates(
    taxon_key: str,
    data_path: str,
    pending: dict,
) -> None:
    """Write pending (catalogNumber, value) pairs for taxon_key to its parquet file."""
    if taxon_key not in pending:
        return
    colmap = pending.pop(taxon_key)
    parquet_path = Path(data_path)
    if not parquet_path.exists():
        return
    table = pq.read_table(parquet_path)
    if table.num_rows == 0:
        return
    df = table.to_pandas()
    if "catalogNumber" not in df.columns:
        return
    catalog_index = {v: i for i, v in enumerate(df["catalogNumber"].astype(str))}
    for col, pairs in colmap.items():
        if col not in df.columns:
            df[col] = np.nan
        col_arr = df[col].to_numpy(dtype=np.float64, copy=True)
        for cat_num, value in pairs:
            idx = catalog_index.get(str(cat_num))
            if idx is not None:
                col_arr[idx] = float(value)
        df[col] = col_arr
    _atomic_write(parquet_path, pa.Table.from_pandas(df, preserve_index=False))


def main() -> None:
    layers = _load_layers()
    if VARS_TO_ENRICH is not None:
        layers = [layer for layer in layers if layer["id"] in VARS_TO_ENRICH]
    layer_ids = [layer["id"] for layer in layers]
    batch_count = 0
    for batch in _iter_worklist_batches(layer_ids, CONFIG.plantae_key, row_limit=ROW_LIMIT):
        if batch.num_rows == 0:
            continue
        batch_count += 1
        print(f"[worklist] processing batch {batch_count}")
        _process_batch(batch, layers)
    print("Completed GIS enrichment.")


if __name__ == "__main__":  # pragma: no cover
    main()
