"""
Enrich per-taxon occurrence parquets with GIS layer values.

COG files are sampled in Hilbert-sorted observation order for spatial cache
locality — one file open per layer per batch, all points sampled in a single
pass rather than per-tile open/close cycles.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

from config.config import load_config
from util.taxa import TaxonRecord, load_catalog

CONFIG = load_config("global")

TREE_ROOT = Path("data/taxonomy/tree")
LAYERS_DIR = Path("data/gis/layers")
CATALOG_PATH = Path("config/gis/catalog.json")
OCCURRENCE_FILE = "occurrence.parquet"
ROW_LIMIT = 10_000_000

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
    # Skip temporal/non-raster layers (they have no filename and are handled by enrich_temporal)
    return [
        layer
        for category in cat["categories"]
        for layer in category["layers"]
        if layer.get("filename")
    ]


def _atomic_write(path: Path, table: pa.Table) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _drop_stale_gis_columns(df, layer_ids: list[str], data_path: Path) -> None:
    """Remove GIS columns that are no longer in the layer catalog."""
    allowed = _BASE_COLS | set(layer_ids)
    stale = [col for col in df.columns if col not in allowed]
    if not stale:
        return
    df.drop(columns=stale, inplace=True)
    _atomic_write(data_path, pa.Table.from_pandas(df, preserve_index=False))


def _missing_rows_for_taxon(taxon: TaxonRecord, layer_ids: list[str]) -> pa.Table | None:
    """Return a worklist chunk for rows missing GIS values, or None if nothing to do."""
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
    return pa.table({
        "catalogNumber": pa.array(df["catalogNumber"].astype(str).tolist(), type=pa.string()),
        "hilbertIdx":    pa.array(df["hilbertIdx"].to_numpy(),              type=pa.int32()),
        "decimalLatitude":  pa.array(df["decimalLatitude"].to_numpy(),      type=pa.float64()),
        "decimalLongitude": pa.array(df["decimalLongitude"].to_numpy(),     type=pa.float64()),
        "missingLayers": pa.array([layer_ids] * len(df),                    type=pa.list_(pa.string())),
        "taxonKey":  pa.array([taxon["taxon_key"]] * len(df),               type=pa.string()),
        "dataPath":  pa.array([str(data_path)] * len(df),                   type=pa.string()),
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


def _sample_cog(
    path: Path,
    layer_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    scale: float,
    offset: float,
) -> list[float | None]:
    """Sample a COG at the given coordinates. Returns physical values (scale+offset applied)."""
    if lats.size == 0:
        return []
    coords = list(zip(lons.tolist(), lats.tolist()))
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        results: list[float | None] = []
        for point in ds.sample(coords):
            v = float(point[0])
            if nodata is not None and v == nodata:
                # SWE nodata means no snow cover, not missing data.
                results.append(0.0 if layer_id == "swe" else None)
            else:
                results.append(v * scale + offset)
    return results


def _flush_taxon_updates(
    taxon_key: str,
    data_path: str,
    pending: dict[str, dict[str, list[tuple[str, float]]]],
) -> None:
    """Write accumulated GIS values for one taxon back to its parquet file."""
    updates = pending.pop(taxon_key, None)
    if not updates:
        return
    data_file = Path(data_path)
    if not data_file.exists():
        return
    table = pq.read_table(data_file)
    df = table.to_pandas()
    if df.empty or "catalogNumber" not in df.columns:
        return
    catalog_index = {v: i for i, v in enumerate(df["catalogNumber"].astype(str))}
    for layer_id, entries in updates.items():
        if layer_id not in df.columns:
            df[layer_id] = np.nan
        for catalog, value in entries:
            idx = catalog_index.get(str(catalog))
            if idx is None:
                continue
            df.at[idx, layer_id] = float(value)
    _atomic_write(data_file, pa.Table.from_pandas(df, preserve_index=False))


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

    layer_rows: dict[str, list[int]] = defaultdict(list)
    for row_idx, missing in enumerate(df["missingLayers"].tolist()):
        if missing is None or len(missing) == 0:
            continue
        for lid in missing:
            layer_rows[lid].append(row_idx)

    layer_meta = {layer["id"]: layer for layer in layers}
    pending: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    total = len(layer_rows)
    for n, (layer_id, row_indices) in enumerate(layer_rows.items(), 1):
        layer = layer_meta.get(layer_id)
        if layer is None:
            print(f"[process] unknown layer {layer_id!r}, skipping")
            continue
        cog_path = LAYERS_DIR / layer["filename"]
        if not cog_path.exists():
            print(f"[process] {layer['filename']} not found, skipping")
            continue
        scale = layer["scale_factor"] if layer["scale_factor"] is not None else 1.0
        offset = layer["add_offset"] if layer["add_offset"] is not None else 0.0
        arr = np.array(row_indices, dtype=int)
        values = _sample_cog(cog_path, layer_id, lats[arr], lons[arr], scale, offset)
        for row_idx, value in zip(arr, values):
            if value is None:
                continue
            pending[taxon_keys[row_idx]][layer_id].append((catalogs[row_idx], value))
        print(f"[process] layer {n}/{total}: {layer_id}")

    for taxon_key in list(pending.keys()):
        _flush_taxon_updates(taxon_key, taxon_paths[taxon_key], pending)


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
