"""
Builds and queries occurrence_index.parquet for taxon nodes.

Each column in the file is named after a GIS layer_id and holds a struct array of
    {catalogNumber, originId, lat, lon, value}
sorted ascending by value. All columns are null-padded to the same row count.

Schema metadata keys (parquet):
    b"origin_map"       JSON [{id, taxon_key}, ...]
    b"column_lengths"   JSON {layer_id: int}  (true length before null padding)
    b"catalog_column"   b"catalogNumber"
    b"category_offsets" JSON {layer_id: {class_str: {value, start, count}}, ...}
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from util.storage import ParquetStorageProxy, atomic_write_parquet
from util.taxa import TaxonRecord, get_children

_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent.parent,
)

TREE_ROOT = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "tree"
OCCURRENCE_INDEX_FILE = "occurrence_index.parquet"

_STRUCT_FIELDS_FLOAT = [
    pa.field("catalogNumber", pa.large_string()),
    pa.field("originId", pa.int32()),
    pa.field("lat", pa.float64()),
    pa.field("lon", pa.float64()),
    pa.field("value", pa.float64()),
]
_STRUCT_FIELDS_INT = [
    pa.field("catalogNumber", pa.large_string()),
    pa.field("originId", pa.int32()),
    pa.field("lat", pa.float64()),
    pa.field("lon", pa.float64()),
    pa.field("value", pa.int64()),
]


def _is_categorical(layer: dict) -> bool:
    return layer.get("value_type") == "nominal"


def _build_category_offsets(sorted_vals: pa.Array) -> dict[str, dict]:
    """Build {class_str: {value, start, count}} from a sorted value array."""
    offsets: dict[str, dict] = {}
    py_values = sorted_vals.to_pylist()
    current = None
    start = 0
    for i, v in enumerate(py_values):
        if current is None:
            current, start = v, i
            continue
        if v != current:
            offsets[str(current)] = {"value": current, "start": start, "count": i - start}
            current, start = v, i
    if current is not None:
        offsets[str(current)] = {"value": current, "start": start, "count": len(py_values) - start}
    return offsets


def _build_struct_col_np(
    val_np: np.ndarray,
    cat_arr: pa.Array,
    orig_arr: pa.Array,
    lat_arr: pa.Array,
    lon_arr: pa.Array,
    categorical: bool,
) -> tuple[pa.StructArray, pa.Array] | None:
    """Build a sorted struct column: numpy argsort on values, pc.take on pre-built PyArrow arrays."""
    valid = np.isfinite(val_np) if not categorical else ~np.isnan(val_np)
    if not valid.any():
        return None

    v = val_np[valid]
    local_sort = np.argsort(v, kind="stable")

    # Map local (within-valid) sort indices to global row positions, then take via PyArrow.
    global_idx = pa.array(np.where(valid)[0][local_sort], type=pa.int64())

    fields = _STRUCT_FIELDS_INT if categorical else _STRUCT_FIELDS_FLOAT
    val_type = pa.int64() if categorical else pa.float64()
    sorted_vals = pa.array(v[local_sort].astype(np.int64 if categorical else np.float64), type=val_type)

    struct_arr = pa.StructArray.from_arrays(
        [
            pc.take(cat_arr, global_idx),
            pc.take(orig_arr, global_idx),
            pc.take(lat_arr, global_idx),
            pc.take(lon_arr, global_idx),
            sorted_vals,
        ],
        fields=fields,
    )
    return struct_arr, sorted_vals


def _remap_struct_origins(col: pa.Array, remap: dict[int, int]) -> pa.Array:
    """Return a new struct array with originId values remapped."""
    catalogs = pc.struct_field(col, "catalogNumber")
    old_origins = pc.struct_field(col, "originId").to_pylist()
    lats = pc.struct_field(col, "lat")
    lons = pc.struct_field(col, "lon")
    values = pc.struct_field(col, "value")
    new_origins = pa.array([remap.get(o, 0) for o in old_origins], type=pa.int32())
    categorical = pa.types.is_integer(values.type)
    fields = _STRUCT_FIELDS_INT if categorical else _STRUCT_FIELDS_FLOAT
    return pa.StructArray.from_arrays(
        [catalogs, new_origins, lats, lons, values], fields=fields
    )


def _finalize_and_write(
    taxon_dir: Path,
    index_columns: dict[str, pa.Array],
    column_lengths: dict[str, int],
    category_offsets: dict[str, dict],
    origin_map: list[dict],
) -> None:
    if not index_columns:
        return
    max_len = max(len(a) for a in index_columns.values())
    padded: dict[str, pa.Array] = {}
    for layer_id, arr in index_columns.items():
        if len(arr) < max_len:
            arr = pa.concat_arrays([arr, pa.nulls(max_len - len(arr), type=arr.type)])
        padded[layer_id] = arr

    table = pa.table(padded)
    meta: dict[bytes, bytes] = {
        b"origin_map": json.dumps(origin_map).encode(),
        b"column_lengths": json.dumps(column_lengths).encode(),
        b"catalog_column": b"catalogNumber",
        b"category_offsets": json.dumps(category_offsets).encode(),
    }
    table = table.replace_schema_metadata(meta)

    atomic_write_parquet(taxon_dir / OCCURRENCE_INDEX_FILE, table, row_group_size=max_len)


# ---------------------------------------------------------------------------
# Public build entry points
# ---------------------------------------------------------------------------

def build_leaf_index(
    taxon_dir: Path,
    df: pd.DataFrame,
    layer_meta: dict[str, dict],
    taxon_key: str,
) -> None:
    """Build occurrence_index.parquet from a pre-filtered occurrence DataFrame."""
    valid = (
        df["catalogNumber"].notna()
        & df["decimalLatitude"].notna()
        & df["decimalLongitude"].notna()
    )
    df = df[valid]
    if df.empty:
        return

    n = len(df)
    # Pre-build shared PyArrow arrays once — string/float arrays are faster via pc.take than
    # repeated numpy object → PyArrow conversions per layer.
    cat_arr = pa.array(df["catalogNumber"].to_numpy(dtype=object), type=pa.large_string())
    lat_arr = pa.array(df["decimalLatitude"].to_numpy(dtype=np.float64), type=pa.float64())
    lon_arr = pa.array(df["decimalLongitude"].to_numpy(dtype=np.float64), type=pa.float64())
    orig_arr = pa.array(np.zeros(n, dtype=np.int32), type=pa.int32())

    origin_map = [{"id": 0, "taxon_key": taxon_key}]
    index_columns: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    category_offsets: dict[str, dict] = {}

    for layer_id, layer in layer_meta.items():
        if layer_id not in df.columns:
            continue
        categorical = _is_categorical(layer)
        col = df[layer_id]
        val_np = (col.to_numpy(dtype=np.float64, na_value=np.nan)
                  if col.dtype == np.float64
                  else pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64))
        result = _build_struct_col_np(val_np, cat_arr, orig_arr, lat_arr, lon_arr, categorical)
        if result is None:
            continue
        struct_arr, sorted_vals = result
        index_columns[layer_id] = struct_arr
        column_lengths[layer_id] = len(struct_arr)
        if categorical:
            off = _build_category_offsets(sorted_vals)
            if off:
                category_offsets[layer_id] = off

    _finalize_and_write(taxon_dir, index_columns, column_lengths, category_offsets, origin_map)


def build_nonleaf_index(taxon: TaxonRecord, taxon_dir: Path) -> None:
    """Build occurrence_index.parquet for a non-leaf taxon by merging children's indexes.

    Reads each direct child's occurrence_index.parquet, expands their origin_maps into
    a flat global origin_map for this node, remaps originIds accordingly, then merges
    and re-sorts every layer column by value.

    Must be called bottom-up so children's indexes are already built.
    """
    origin_map: list[dict] = []
    origin_counter = 0
    pending: dict[str, list[pa.Array]] = defaultdict(list)

    for child in get_children(taxon["taxon_key"]):
        child_path = TREE_ROOT / child["path"] / OCCURRENCE_INDEX_FILE
        if not child_path.exists():
            continue

        child_schema = pq.read_schema(child_path)
        child_meta = child_schema.metadata or {}
        child_origin_map: list[dict] = json.loads(child_meta.get(b"origin_map", b"[]"))
        child_col_lengths: dict[str, int] = json.loads(child_meta.get(b"column_lengths", b"{}"))

        # Skip children with old flat-schema indexes (no struct columns)
        if child_schema.names and not pa.types.is_struct(child_schema.field(child_schema.names[0]).type):
            continue

        remap: dict[int, int] = {}
        for entry in child_origin_map:
            new_id = origin_counter
            origin_map.append({"id": new_id, "taxon_key": entry["taxon_key"]})
            remap[int(entry["id"])] = new_id
            origin_counter += 1

        child_table = pq.read_table(child_path)
        for layer_id in child_table.schema.names:
            col = child_table.column(layer_id).combine_chunks()
            true_len = child_col_lengths.get(layer_id)
            col = col.slice(0, true_len) if true_len is not None else col.filter(pc.invert(pc.is_null(col)))
            if len(col) == 0:
                continue
            needs_remap = any(new != old for old, new in remap.items())
            if needs_remap:
                col = _remap_struct_origins(col, remap)
            pending[layer_id].append(col)

    if not pending:
        return

    index_columns: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    category_offsets: dict[str, dict] = {}

    for layer_id, arrays in pending.items():
        merged = pa.concat_arrays(arrays)
        values_arr = pc.struct_field(merged, "value")
        sort_idx = pc.sort_indices(values_arr)
        merged = pc.take(merged, sort_idx)
        sorted_vals = pc.take(values_arr, sort_idx)

        index_columns[layer_id] = merged
        column_lengths[layer_id] = len(merged)

        if pa.types.is_integer(values_arr.type):
            off = _build_category_offsets(sorted_vals)
            if off:
                category_offsets[layer_id] = off

    _finalize_and_write(taxon_dir, index_columns, column_lengths, category_offsets, origin_map)


# ---------------------------------------------------------------------------
# Public query entry points
# ---------------------------------------------------------------------------

def read_slice(
    index_path: Path,
    layer_id: str,
    *,
    value_min: float | None = None,
    value_max: float | None = None,
    circular_wrap: bool = False,
    class_value: float | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return observations in a value range (or class) using binary search.

    Numeric: O(log n + m) where m is the number of matching records.
    Categorical: O(1) offset lookup + O(m) extraction.
    """
    schema = _storage.read_schema(index_path)
    if layer_id not in schema.names:
        return []
    if not pa.types.is_struct(schema.field(layer_id).type):
        return []  # old flat-schema index

    meta = schema.metadata or {}
    origin_lookup: dict[int, str] = {
        int(e["id"]): e["taxon_key"]
        for e in json.loads(meta.get(b"origin_map", b"[]"))
    }
    col_lengths: dict[str, int] = json.loads(meta.get(b"column_lengths", b"{}"))
    true_len = col_lengths.get(layer_id)

    table = _storage.read_table(index_path, columns=[layer_id])
    col = table.column(layer_id).combine_chunks()
    col = col.slice(0, true_len) if true_len is not None else col.filter(pc.invert(pc.is_null(col)))

    if len(col) == 0:
        return []

    values = pc.struct_field(col, "value")

    if class_value is not None:
        cat_offsets: dict[str, dict] = json.loads(meta.get(b"category_offsets", b"{}"))
        layer_off = cat_offsets.get(layer_id, {})
        key = str(int(class_value)) if float(class_value).is_integer() else str(class_value)
        entry = layer_off.get(key)
        if entry is None:
            return []
        col_slice = col.slice(entry["start"], entry["count"])
    elif circular_wrap:
        # value_min > value_max: wraps around (e.g. aspect 350→10)
        vals_np = values.to_numpy(zero_copy_only=False)
        lo1 = int(np.searchsorted(vals_np, value_min, side="left"))
        hi2 = int(np.searchsorted(vals_np, value_max, side="right"))
        col_slice = pa.concat_arrays([col.slice(lo1), col.slice(0, hi2)])
    else:
        vals_np = values.to_numpy(zero_copy_only=False)
        lo = int(np.searchsorted(vals_np, value_min, side="left"))
        hi = int(np.searchsorted(vals_np, value_max, side="right"))
        col_slice = col.slice(lo, hi - lo)

    if limit is not None:
        col_slice = col_slice.slice(0, limit)

    if len(col_slice) == 0:
        return []

    catalogs = pc.struct_field(col_slice, "catalogNumber").to_pylist()
    origins = pc.struct_field(col_slice, "originId").to_pylist()
    lats = pc.struct_field(col_slice, "lat").to_pylist()
    lons = pc.struct_field(col_slice, "lon").to_pylist()
    vals = pc.struct_field(col_slice, "value").to_pylist()

    return [
        {
            "catalogNumber": cat,
            "taxon_key": origin_lookup.get(orig),
            "latitude": lat,
            "longitude": lon,
            "value": float(val) if val is not None else None,
        }
        for cat, orig, lat, lon, val in zip(catalogs, origins, lats, lons, vals)
        if lat is not None and lon is not None
    ]


def lookup_value(index_path: Path, layer_id: str, catalog_number: str) -> float | None:
    """Return the stored GIS value for a specific observation. O(n) linear scan."""
    schema = _storage.read_schema(index_path)
    if layer_id not in schema.names:
        return None
    if not pa.types.is_struct(schema.field(layer_id).type):
        return None

    meta = schema.metadata or {}
    col_lengths: dict[str, int] = json.loads(meta.get(b"column_lengths", b"{}"))
    true_len = col_lengths.get(layer_id)

    table = _storage.read_table(index_path, columns=[layer_id])
    col = table.column(layer_id).combine_chunks()
    if true_len is not None:
        col = col.slice(0, true_len)

    catalogs = pc.struct_field(col, "catalogNumber").to_pylist()
    vals = pc.struct_field(col, "value").to_pylist()
    for cat, val in zip(catalogs, vals):
        if cat == catalog_number and val is not None:
            return float(val)
    return None


def read_all_values(index_path: Path, layer_id: str) -> list[tuple[str, float]]:
    """Return all (catalogNumber, raw_value) pairs for a layer. No range filtering."""
    try:
        schema = _storage.read_schema(index_path)
    except Exception:
        return []
    if layer_id not in schema.names:
        return []
    if not pa.types.is_struct(schema.field(layer_id).type):
        return []

    meta = schema.metadata or {}
    col_lengths: dict[str, int] = json.loads(meta.get(b"column_lengths", b"{}"))
    true_len = col_lengths.get(layer_id)

    table = _storage.read_table(index_path, columns=[layer_id])
    col = table.column(layer_id).combine_chunks()
    col = col.slice(0, true_len) if true_len is not None else col

    catalogs = pc.struct_field(col, "catalogNumber").to_pylist()
    vals = pc.struct_field(col, "value").to_pylist()
    return [
        (str(cat), float(val))
        for cat, val in zip(catalogs, vals)
        if cat is not None and val is not None
    ]
