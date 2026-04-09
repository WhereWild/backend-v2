'''
Summary stats and index query helpers for GIS variables.
'''

from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any, Sequence, Optional, Iterable
from collections import Counter, defaultdict
import sys
import json
import math
import re
import random
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from fastdigest import TDigest as _FastTDigest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from util import gis_lookup, taxa_navigation
from util.config import load_config
from util.storage import ParquetStorageProxy

# ---- Path bootstrap ----
CONFIG = load_config("global")
PARQUET = ParquetStorageProxy(CONFIG.data_root, CONFIG.project_root)

significant_category_threshold = 0.05

density_graph_filename = "density_graph.parquet"

density_max_samples = 20000


excluded_numeric_columns = frozenset(
            {
                "decimalLatitude",
                "decimalLongitude",
                "eventTimestamp",
                "coordinateUncertaintyInMeters",
            }
        )

sys.path.append(str(CONFIG.project_root))

# ---- Configuration ----

# ---- Internal helpers ----
def _layer_value_type(layer_id: str) -> str | None:
    """Returns the configured value type for a layer id.
    
    Args:
        layer_id: Layer id to inspect.
    
    Returns:
        The value_type string (e.g., "numeric" or "categorical"), if present.
    """
    metadata = gis_lookup.load_layer_metadata().get(str(layer_id))
    if not metadata:
        return None
    value_type = metadata.get("value_type")
    if isinstance(value_type, str):
        return value_type.strip().lower()
    return None


def _is_circular_variable(layer_id: str) -> bool:
    return _layer_value_type(layer_id) == "circular"


def _empty_numeric_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "min": None,
        "1st percentile": None,
        "10th percentile": None,
        "25th percentile": None,
        "median": None,
        "75th percentile": None,
        "90th percentile": None,
        "99th percentile": None,
        "max": None,
        "mean": None,
        "std": None,
        "interquartile range": None,
        "10-90 range": None,
        "1-99 range": None,
        "range": None,
    }


def _wrap_degrees(value: float) -> float:
    wrapped = float(value) % 360.0
    if math.isclose(wrapped, 360.0):
        return 0.0
    if math.isclose(wrapped, -0.0):
        return 0.0
    return wrapped


def _storage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "count",
        "min",
        "10th percentile",
        "25th percentile",
        "median",
        "75th percentile",
        "90th percentile",
        "max",
        "mean",
        "std",
        "10-90 range",
        "range",
    )
    return {key: summary.get(key) for key in keys}


def summarize_layer_values(layer_id: str, values: Sequence[float]) -> dict[str, Any]:
    return summarize_values(values, circular=_is_circular_variable(layer_id))


def summarize_layer_values_for_storage(layer_id: str, values: Sequence[float]) -> dict[str, Any]:
    return _storage_summary(summarize_layer_values(layer_id, values))


def _legend_for_layer(layer_id: str) -> Dict[int, str]:
    """Loads a categorical legend mapping for a layer id.
    
    Args:
        layer_id: Layer id whose legend should be loaded.
    
    Returns:
        A mapping of category ids to display labels.
    """
    legend = gis_lookup.load_layer_legend(layer_id)
    mapping: Dict[int, str] = {}
    for key, entry in legend.items():
        if not isinstance(entry, dict):
            continue
        class_id = entry.get("id")
        name = entry.get("name")
        if class_id is None or not name:
            continue
        try:
            mapping[int(class_id)] = str(name)
        except (TypeError, ValueError):
            continue
    return mapping


def _slugify_metric(name: str | None, fallback: str) -> str:
    """Normalizes a metric name for use as a key.
    
    Args:
        name: Metric name to normalize.
        fallback: Fallback value when name is empty.
    
    Returns:
        A lowercased, slugified metric name.
    """
    if not name:
        return fallback
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or fallback


def _prepare_index_column(index_parquet: Path, layer_id: str):
    """Loads an indexed layer column and its lookup metadata.
    
    Args:
        index_parquet: Path to the occurrence index parquet.
        layer_id: Layer id to load from the index.
    
    Returns:
        A dict with the struct column plus origin and category offset metadata.
    """
    if not PARQUET.exists(index_parquet):
        raise FileNotFoundError(index_parquet)

    table = PARQUET.read_table(index_parquet, columns=[layer_id])
    if table.num_columns == 0:
        return None

    column = table.column(layer_id).combine_chunks()
    schema_meta = table.schema.metadata or {}
    origin_meta = schema_meta.get(b"origin_map")
    if origin_meta is None:
        raise ValueError("index parquet missing origin_map metadata")
    catalog_col_name = (
        schema_meta.get(b"catalog_column", b"catalogNumber").decode("utf-8")
    )
    origin_entries = json.loads(origin_meta.decode("utf-8"))
    origin_lookup = {entry["id"]: entry for entry in origin_entries}
    category_offsets_meta = schema_meta.get(b"category_offsets")
    layer_offsets: dict[str, dict[str, int | float]] = {}
    if category_offsets_meta:
        parsed_offsets = json.loads(category_offsets_meta.decode("utf-8"))
        layer_offsets = parsed_offsets.get(layer_id, {})
    return {
        "column": column,
        "catalog_column": catalog_col_name,
        "origin_lookup": origin_lookup,
        "index_dir": index_parquet.parent,
        "category_offsets": layer_offsets,
    }


def _make_dataset_loader(
    origin_lookup: dict[int, dict[str, Any]],
    index_dir: Path,
    *,
    catalog_column: str,
    layer_id: str,
    data_filename: str,
    lat_col: str,
    lon_col: str,
):
    """Creates a loader that resolves index origin ids to filtered datasets.
    
    Args:
        origin_lookup: Mapping of origin id to origin metadata from index metadata.
        index_dir: Directory containing the occurrence index parquet.
        catalog_column: Catalog number column name to load.
        layer_id: Variable id column to load.
        data_filename: Occurrence parquet filename to load per origin.
        lat_col: Latitude column name to load.
        lon_col: Longitude column name to load.
    
    Returns:
        A callable that takes an origin id and returns a cached dataset dict
        (table, index map, layer values, latitudes, longitudes).
    """
    datasets: dict[int, dict[str, Any]] = {}

    def _load_dataset(origin_id: int) -> dict[str, Any] | None:
        """Loads and caches the occurrence table for a given origin id."""
        if origin_id in datasets:
            return datasets[origin_id]
        info = origin_lookup.get(origin_id)
        if info is None:
            return None
        rel_path = info["relative_path"]
        data_path = (index_dir / rel_path / data_filename).resolve()
        if not PARQUET.exists(data_path):
            return None
        table_cols = [
            catalog_column,
            lat_col,
            lon_col,
            layer_id,
            "obscured",
            "coordinateUncertaintyInMeters",
        ]
        parquet_table = PARQUET.read_table(data_path, columns=table_cols).combine_chunks()

        mask = pc.equal(parquet_table["obscured"], "No")
        coord_col = parquet_table["coordinateUncertaintyInMeters"]
        coord_mask = pc.less_equal(coord_col, 500)
        mask = pc.and_(mask, coord_mask)
        filtered_table = parquet_table.filter(mask).combine_chunks()

        catalog_values = filtered_table[catalog_column].to_pylist()
        index_map = {value: idx for idx, value in enumerate(catalog_values)}
        datasets[origin_id] = {
            "table": filtered_table,
            "index": index_map,
            "layer_values": filtered_table.column(layer_id),
            "latitudes": filtered_table.column(lat_col),
            "longitudes": filtered_table.column(lon_col),
        }
        return datasets[origin_id]

    return _load_dataset


def get_sorted_layer_records(
    index_parquet: Path,
    layer_id: str,
    start: int = 0,
    stop: int | None = None,
    data_filename: str = CONFIG.occurrence_parquet_filename,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
) -> list[tuple[str, float | None, float | None, float | None]]:
    """Loads value-sorted records for a layer from the occurrence index.
    
    Args:
        index_parquet: Path to the occurrence index parquet.
        layer_id: Layer id to read.
        start: Start offset within the sorted index column.
        stop: End offset within the sorted index column.
        data_filename: Occurrence parquet filename to resolve origin ids.
        lat_col: Latitude column name to include.
        lon_col: Longitude column name to include.
    
    Returns:
        A list of (catalog, latitude, longitude, value) records in value order.
    """
    prepared = _prepare_index_column(index_parquet, layer_id)
    if prepared is None:
        return []
    return _slice_records(
        prepared,
        layer_id,
        start=start,
        stop=stop,
        data_filename=data_filename,
        lat_col=lat_col,
        lon_col=lon_col,
    )


def _slice_records(
    prepared: dict[str, Any],
    layer_id: str,
    *,
    start: int,
    stop: int | None,
    data_filename: str,
    lat_col: str,
    lon_col: str,
) -> list[tuple[str, float | None, float | None, float | None]]:
    """Slices a prepared index column and resolves records to lat/lon/value.
    
    Args:
        prepared: Output from _prepare_index_column with column + origin metadata.
        layer_id: Layer id being sliced.
        start: Start index within the sorted column.
        stop: End index within the sorted column.
        data_filename: Occurrence parquet filename to resolve origin ids.
        lat_col: Latitude column name to include.
        lon_col: Longitude column name to include.
    
    Returns:
        A list of (catalog, latitude, longitude, value) rows for the slice.
    """
    column = prepared["column"]
    total = len(column)
    if total == 0 or start >= total:
        return []
    if stop is None or stop > total:
        stop = total
    if stop <= start:
        return []

    length = stop - start
    slice_arr = column.slice(start, length)

    catalogs = slice_arr.field("catalogNumber").to_pylist()
    origins = slice_arr.field("originId").to_pylist()

    load_dataset = _make_dataset_loader(
        prepared["origin_lookup"],
        prepared["index_dir"],
        catalog_column=prepared["catalog_column"],
        layer_id=layer_id,
        data_filename=data_filename,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    results: list[tuple[str, float | None, float | None, float | None]] = []
    field_names = {field.name for field in slice_arr.type}
    values = (
        slice_arr.field("value").to_pylist() if "value" in field_names else [None] * len(catalogs)
    )

    for catalog, origin_id, stored_value in zip(catalogs, origins, values):
        dataset = load_dataset(origin_id)
        if not dataset:
            continue
        idx = dataset["index"].get(catalog)
        if idx is None:
            continue
        lat = dataset["latitudes"][idx].as_py()
        lon = dataset["longitudes"][idx].as_py()
        value = dataset["layer_values"][idx].as_py()
        if value is None:
            value = stored_value
        results.append((catalog, lat, lon, value))

    return results


def get_sorted_layer_records_in_value_range(
    index_parquet: Path,
    layer_id: str,
    value_min: float | None,
    value_max: float | None,
    *,
    limit: int | None = None,
    data_filename: str = CONFIG.occurrence_parquet_filename,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
) -> list[tuple[str, float | None, float | None, float | None]]:
    """Returns sorted records within a numeric value range.
    
    Args:
        index_parquet: Path to the occurrence index parquet.
        layer_id: Layer id to filter.
        value_min: Minimum value to include.
        value_max: Maximum value to include.
        limit: Optional maximum number of records to return.
        data_filename: Occurrence parquet filename to resolve origin ids.
        lat_col: Latitude column name to include.
        lon_col: Longitude column name to include.
    
    Returns:
        A list of (catalog, latitude, longitude, value) rows in value order.
    """
    prepared = _prepare_index_column(index_parquet, layer_id)
    if prepared is None:
        return []
    column = prepared["column"]
    total = len(column)
    if total == 0:
        return []

    catalogs_arr = column.field("catalogNumber")
    origins_arr = column.field("originId")
    column_field_names = {field.name for field in column.type}
    values_arr = column.field("value") if "value" in column_field_names else None

    load_dataset = _make_dataset_loader(
        prepared["origin_lookup"],
        prepared["index_dir"],
        catalog_column=prepared["catalog_column"],
        layer_id=layer_id,
        data_filename=data_filename,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    def _value_at(position: int) -> float | None:
        """Gets the numeric value at a sorted index position."""
        if values_arr is not None:
            scalar = values_arr[position]
            try:
                return float(scalar.as_py())
            except (TypeError, ValueError):
                return None
        catalog_scalar = catalogs_arr[position]
        origin_scalar = origins_arr[position]
        catalog = catalog_scalar.as_py()
        origin_id = origin_scalar.as_py()
        dataset = load_dataset(origin_id)
        if not dataset:
            return None
        idx = dataset["index"].get(catalog)
        if idx is None:
            return None
        value = dataset["layer_values"][idx].as_py()
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _lower_bound(target: float) -> int:
        """Finds the first index with value >= target (binary search)."""
        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi) // 2
            value = _value_at(mid)
            if value is None:
                return _linear_lower_bound(target)
            if value < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _upper_bound(target: float) -> int:
        """Finds the first index with value > target (binary search)."""
        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi) // 2
            value = _value_at(mid)
            if value is None:
                return _linear_upper_bound(target)
            if value <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _linear_lower_bound(target: float) -> int:
        """Fallback linear scan for the lower bound when values are missing."""
        for idx in range(total):
            value = _value_at(idx)
            if value is not None and value >= target:
                return idx
        return total

    def _linear_upper_bound(target: float) -> int:
        """Fallback linear scan for the upper bound when values are missing."""
        for idx in range(total):
            value = _value_at(idx)
            if value is None:
                continue
            if value > target:
                return idx
        return total

    start_idx = 0
    end_idx = total
    if value_min is not None:
        start_idx = _lower_bound(value_min)
    if value_max is not None:
        end_idx = _upper_bound(value_max)

    if start_idx >= end_idx:
        return []

    if limit is not None and limit > 0:
        end_idx = min(end_idx, start_idx + limit)

    slice_len = end_idx - start_idx
    catalogs = catalogs_arr.slice(start_idx, slice_len).to_pylist()
    origins = origins_arr.slice(start_idx, slice_len).to_pylist()

    results: list[tuple[str, float | None, float | None, float | None]] = []
    for catalog, origin_id in zip(catalogs, origins):
        dataset = load_dataset(origin_id)
        if not dataset:
            continue
        idx = dataset["index"].get(catalog)
        if idx is None:
            continue
        lat = dataset["latitudes"][idx].as_py()
        lon = dataset["longitudes"][idx].as_py()
        raw_value = dataset["layer_values"][idx].as_py()
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None:
            if value_min is not None and numeric_value < value_min:
                continue
            if value_max is not None and numeric_value > value_max:
                break
        results.append((catalog, lat, lon, raw_value))

    return results


def get_layer_records_for_class(
    index_parquet: Path,
    layer_id: str,
    class_value: int | float | str,
    data_filename: str = CONFIG.occurrence_parquet_filename,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
) -> list[tuple[str, float | None, float | None, float | None]]:
    """Returns records matching a categorical class value.
    
    Args:
        index_parquet: Path to the occurrence index parquet.
        layer_id: Layer id to filter.
        class_value: Categorical class value to match.
        data_filename: Occurrence parquet filename to resolve origin ids.
        lat_col: Latitude column name to include.
        lon_col: Longitude column name to include.
    
    Returns:
        A list of (catalog, latitude, longitude, value) rows for the class.
    """
    class_value = resolve_categorical_class_value(layer_id, class_value)
    prepared = _prepare_index_column(index_parquet, layer_id)
    if prepared is None:
        return []
    offsets = prepared.get("category_offsets") or {}
    entry = offsets.get(str(class_value))
    if entry is None and isinstance(class_value, (int, float)):
        numeric = float(class_value)
        entry = offsets.get(str(numeric))
        if entry is None and math.isfinite(numeric) and numeric.is_integer():
            entry = offsets.get(str(int(numeric)))
    if not entry:
        return []
    start = int(entry.get("start", 0))
    count = int(entry.get("count", 0))
    if count <= 0:
        return []
    stop = start + count
    return _slice_records(
        prepared,
        layer_id,
        start=start,
        stop=stop,
        data_filename=data_filename,
        lat_col=lat_col,
        lon_col=lon_col,
    )

# ---- Public API: schema & metadata ----
def get_schema(parquet_path: Path):
    """Loads the parquet schema for a file.
    
    Args:
        parquet_path: Parquet file to inspect.
    
    Returns:
        The pyarrow schema for the parquet file.
    """
    return PARQUET.read_schema(parquet_path)


def get_num_rows(parquet_path: Path) -> int:
    """Returns the number of rows in a parquet file.
    
    Args:
        parquet_path: Parquet file to inspect.
    
    Returns:
        The number of rows in the parquet file.
    """
    meta = PARQUET.read_metadata(parquet_path)
    return meta.num_rows


def get_column_names(parquet_path: Path) -> List[str]:
    """Returns the column names in a parquet file.
    
    Args:
        parquet_path: Parquet file to inspect.
    
    Returns:
        A list of column names.
    """
    schema = get_schema(parquet_path)
    return schema.names


def get_column_types(parquet_path: Path) -> Dict[str, str]:
    """Returns column names mapped to type strings for a parquet file.
    
    Args:
        parquet_path: Parquet file to inspect.
    
    Returns:
        A mapping of column name to its type string.
    """
    schema = get_schema(parquet_path)
    return {field.name: str(field.type) for field in schema}

def code_to_name(variable_code: str) -> str:
    """Maps a variable code to its display name via the GIS catalog.
    
    Args:
        variable_code: Variable id to resolve.
    
    Returns:
        The display name if found, otherwise None.
    """
    with PARQUET.open_input_file(CONFIG.gis_catalog_path) as handle:
        d = json.loads(handle.read())
        for category in d["categories"]:
            for layer in category["layers"]:
                if layer["id"] == variable_code:
                    return layer["display_name"]
    return None

# ---- Public API: aggregations ----
def column_null_counts(parquet_path: Path) -> Dict[str, int]:
    """Counts null values per column in a parquet file.
    
    Args:
        parquet_path: Parquet file to inspect.
    
    Returns:
        A mapping of column name to null count.
    """
    table = PARQUET.read_table(parquet_path)
    return {
        col: pc.sum(pc.is_null(table[col])).as_py()
        for col in table.column_names
    }


ObservationSample = Dict[str, Any]


def categorical_value_key(raw_value: Any) -> tuple[str, Any]:
    """Normalizes a categorical value into a comparable key.
    
    Args:
        raw_value: Raw class value from the data.
    
    Returns:
        A tuple of (string key, normalized value) for comparison/labels.
    """
    if isinstance(raw_value, (int, float)) and math.isfinite(raw_value):
        numeric = float(raw_value)
        normalized = int(numeric) if numeric.is_integer() else numeric
        return str(normalized), normalized
    try:
        numeric = float(raw_value)
        if math.isfinite(numeric):
            normalized = int(numeric) if numeric.is_integer() else numeric
            return str(normalized), normalized
    except (TypeError, ValueError):
        pass
    text = str(raw_value)
    return text, text


def _format_category_label(metric: str) -> str:
    """Formats a category metric into a display label.
    
    Args:
        metric: Raw metric key (possibly slugified).
    
    Returns:
        A title-cased label string.
    """
    cleaned = metric.replace("::", " ")
    cleaned = re.sub(r"[_\s]+", " ", cleaned).strip()
    return cleaned.title() if cleaned else metric


def _legend_key(value: object) -> str:
    """Normalize legend lookup keys so numeric codes match string ids."""
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def load_categorical_distribution(
    data_dir: Path,
    variable_id: str,
) -> Optional[dict[str, Any]]:
    """Loads categorical distribution stats for a variable from disk.
    
    Args:
        data_dir: Taxon directory containing categorical stats parquet.
        variable_id: Variable id to load.
    
    Returns:
        A dict with distribution, dominant classes, and totals, or None.
    """
    stats_path = data_dir / "categorical_stats.parquet"
    if not PARQUET.exists(stats_path):
        return None
    try:
        table = PARQUET.read_table(
            stats_path,
            columns=["variable", "metric", "value"],
        ).combine_chunks()
    except Exception:
        return None
    try:
        mask = pc.equal(table["variable"], variable_id)
        filtered = table.filter(mask).combine_chunks()
    except Exception:
        return None
    if filtered.num_rows == 0:
        return None
    metrics = filtered.column("metric").to_pylist()
    values = filtered.column("value").to_pylist()
    legend_lookup = gis_lookup.load_layer_legend(variable_id)
    totals: dict[str, float] = {}
    distribution: list[dict[str, Any]] = []
    for metric, raw_value in zip(metrics, values):
        key = str(metric)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None
        lowered = key.lower()
        if lowered == "total_samples" and numeric_value is not None:
            totals["total_samples"] = numeric_value
            continue
        if lowered == "unique_classes" and numeric_value is not None:
            totals["unique_classes"] = numeric_value
            continue
        if lowered == "significant_unique_classes" and numeric_value is not None:
            totals["significant_unique_classes"] = numeric_value
            continue
        if numeric_value is None:
            continue
        slug = re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()
        legend_entry = legend_lookup.get(slug)
        class_id = legend_entry.get("id") if legend_entry else None
        if class_id is None:
            if lowered.startswith("class_"):
                try:
                    class_id = int(lowered.split("_", 1)[1])
                except (ValueError, IndexError):
                    class_id = None
        # If metric keys were stored as fallback class_<id>, resolve legend by id now.
        if legend_entry is None and class_id is not None:
            legend_entry = legend_lookup.get(str(class_id))
        class_name = legend_entry.get("name") if legend_entry else _format_category_label(key)
        description = legend_entry.get("description") if legend_entry else None
        distribution.append(
            {
                "value": class_id if class_id is not None else key,
                "class_name": class_name,
                "description": description,
                "fraction": numeric_value,
                "slug": slug,
            }
        )
    if not distribution:
        return None
    total_samples = totals.get("total_samples")
    for entry in distribution:
        fraction = entry.get("fraction") or 0.0
        if total_samples is not None:
            entry["count"] = int(round(total_samples * fraction))
        else:
            entry["count"] = fraction
    distribution.sort(key=lambda row: row.get("fraction", 0), reverse=True)
    dominant = distribution[: min(5, len(distribution))]
    return {
        "distribution": distribution,
        "dominant": dominant,
        "totals": totals,
    }


def resolve_categorical_class_value(layer_id: str, class_value: Any) -> Any:
    """Resolves a categorical class value from a slug/label to its stored id.
    
    Args:
        layer_id: GIS layer id.
        class_value: Raw class value from the API.
    
    Returns:
        The resolved class id when possible, otherwise the input value.
    """
    try:
        parsed_value = float(class_value)
        if math.isfinite(parsed_value):
            return int(parsed_value) if parsed_value.is_integer() else parsed_value
    except (TypeError, ValueError):
        pass
    if isinstance(class_value, str):
        text = class_value.strip()
        lowered = text.lower()
        if lowered.startswith("class_"):
            tail = text.split("_", 1)[1]
            try:
                parsed_value = float(tail)
                if math.isfinite(parsed_value):
                    return int(parsed_value) if parsed_value.is_integer() else parsed_value
            except (TypeError, ValueError):
                return tail
        legend_lookup = gis_lookup.load_layer_legend(layer_id)
        entry = legend_lookup.get(text)
        if entry is None:
            slug = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
            entry = legend_lookup.get(slug)
        if entry and isinstance(entry, dict):
            resolved = entry.get("id")
            if resolved is not None:
                return resolved
    return class_value


def build_categorical_stats_for_location(
    taxon_id: int,
    variable_id: str,
    location_gid: str,
    *,
    sample_limit: int,
) -> Optional[dict[str, Any]]:
    """Builds a categorical distribution for a taxon filtered to a location.
    
    Args:
        taxon_id: Taxon id to sample.
        variable_id: Categorical layer id to analyze.
        location_gid: Location GID to filter observations.
        sample_limit: Max samples to keep per class.
    
    Returns:
        A dict with distribution, dominant classes, totals, and samples, or None.
    """
    counts: dict[str, int] = {}
    sample_map: dict[str, list[str]] = {}
    value_lookup: dict[str, Any] = {}
    for table in taxa_navigation.iter_filtered_occurrence_tables(
        taxon_id,
        extra_columns=[variable_id],
        location_gid=location_gid,
    ):
        value_col = table[variable_id]
        mask = pc.invert(pc.is_null(value_col))
        filtered = table.filter(mask).combine_chunks()
        if filtered.num_rows == 0:
            continue
        catalogs = filtered["catalogNumber"].to_pylist()
        values = filtered[variable_id].to_pylist()
        for catalog, value in zip(catalogs, values):
            key, normalized_value = categorical_value_key(value)
            counts[key] = counts.get(key, 0) + 1
            value_lookup.setdefault(key, normalized_value)
            bucket = sample_map.setdefault(key, [])
            if len(bucket) < sample_limit:
                bucket.append(str(catalog))
    if not counts:
        return None
    total = sum(counts.values())
    legend_lookup = gis_lookup.load_layer_legend(variable_id)
    distribution: list[dict[str, Any]] = []
    for key, count in counts.items():
        normalized_value = value_lookup.get(key, key)
        normalized_key = _legend_key(normalized_value)
        entry = legend_lookup.get(normalized_key) or legend_lookup.get(
            re.sub(r"[^a-z0-9]+", " ", normalized_key.lower()).strip()
        )
        value_field = entry["id"] if entry and "id" in entry else normalized_value
        class_name = entry["name"] if entry else str(normalized_value)
        description = entry.get("description") if entry else None
        distribution.append(
            {
                "value": value_field,
                "class_name": class_name,
                "description": description,
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )
    distribution.sort(key=lambda row: row.get("fraction", 0), reverse=True)
    dominant = distribution[: min(5, len(distribution))]
    significant = [
        entry for entry in distribution if entry.get("fraction", 0) >= significant_category_threshold
    ]
    totals = {
        "total_samples": total,
        "unique_classes": len(distribution),
        "significant_unique_classes": len(significant),
    }
    samples = [
        {"value": value_lookup.get(key, key), "observationIds": ids}
        for key, ids in sample_map.items()
        if ids
    ]
    return {
        "distribution": distribution,
        "dominant": dominant,
        "totals": totals,
        "samples": samples,
    }


def build_categorical_samples(
    data_dir: Path,
    variable_id: str,
    categories: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Builds observation id samples for categorical classes.
    
    Args:
        data_dir: Taxon directory containing occurrence index parquet.
        variable_id: Categorical layer id to sample.
        categories: Category entries with class values.
    
    Returns:
        A list of sample dicts with class value and observationIds.
    """
    index_path = data_dir / "occurrence_index.parquet"
    if not PARQUET.exists(index_path):
        return []
    samples: list[dict[str, Any]] = []
    for entry in categories:
        value = entry.get("value")
        if not isinstance(value, (int, float, str)):
            continue
        rows = []
        try:
            rows = get_layer_records_for_class(index_path, variable_id, value)
        except Exception:
            rows = []
        if not rows:
            continue
        catalogs = [str(record[0]) for record in rows]
        samples.append(
            {
                "value": value,
                "observationIds": catalogs,
            }
        )
    return samples


def summarize_values(values: Sequence[float], *, circular: bool = False) -> dict[str, Any]:
    """Summarizes numeric values using the same metrics as saved summary stats.

    Args:
        values: Numeric values to summarize.
        circular: Whether values should be treated as wrapped degrees on [0, 360).

    Returns:
        A summary dict with count, percentiles, mean, std, and range metrics.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return _empty_numeric_summary()
    if circular:
        normalized = np.mod(array, 360.0)
        radians = np.deg2rad(normalized)
        sin_mean = float(np.sin(radians).mean())
        cos_mean = float(np.cos(radians).mean())
        mean_angle = _wrap_degrees(np.degrees(np.arctan2(sin_mean, cos_mean)))
        shifted = ((normalized - mean_angle + 180.0) % 360.0) - 180.0
        q1, q10, q25, q50, q75, q90, q99 = np.percentile(shifted, [1, 10, 25, 50, 75, 90, 99])
        min_shift = float(shifted.min())
        max_shift = float(shifted.max())
        resultant = float(math.hypot(sin_mean, cos_mean))
        std_val = None
        if resultant > 0.0 and math.isfinite(resultant):
            std_val = float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))
        return {
            "count": count,
            "min": _wrap_degrees(mean_angle + min_shift),
            "1st percentile": _wrap_degrees(mean_angle + float(q1)),
            "10th percentile": _wrap_degrees(mean_angle + float(q10)),
            "25th percentile": _wrap_degrees(mean_angle + float(q25)),
            "median": _wrap_degrees(mean_angle + float(q50)),
            "75th percentile": _wrap_degrees(mean_angle + float(q75)),
            "90th percentile": _wrap_degrees(mean_angle + float(q90)),
            "99th percentile": _wrap_degrees(mean_angle + float(q99)),
            "max": _wrap_degrees(mean_angle + max_shift),
            "mean": mean_angle,
            "std": std_val,
            "interquartile range": float(q75 - q25),
            "10-90 range": float(q90 - q10),
            "1-99 range": float(q99 - q1),
            "range": float(max_shift - min_shift),
        }
    q1, q10, q25, q50, q75, q90, q99 = np.percentile(array, [1, 10, 25, 50, 75, 90, 99])
    min_val = float(array.min())
    max_val = float(array.max())
    mean_val = float(array.mean())
    std_val = float(array.std())
    return {
        "count": count,
        "min": min_val,
        "1st percentile": float(q1),
        "10th percentile": float(q10),
        "25th percentile": float(q25),
        "median": float(q50),
        "75th percentile": float(q75),
        "90th percentile": float(q90),
        "99th percentile": float(q99),
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "interquartile range": float(q75 - q25),
        "10-90 range": float(q90 - q10),
        "1-99 range": float(q99 - q1),
        "range": float(max_val - min_val),
    }


def gather_numeric_records(
    taxon_id: int,
    data_dir: Path,
    variable_id: str,
    *,
    location_gid: Optional[str] = None,
) -> List[ObservationSample]:
    """Collects numeric observation samples for a taxon and variable.
    
    Args:
        taxon_id: Taxon id to sample.
        data_dir: Taxon directory containing occurrence data/index.
        variable_id: Numeric variable id to extract.
        location_gid: Optional location GID to filter observations.
    
    Returns:
        A list of numeric observation samples.
    """
    if location_gid:
        return gather_numeric_records_from_tables(taxon_id, variable_id, location_gid)
    index_path = data_dir / "occurrence_index.parquet"
    if PARQUET.exists(index_path):
        try:
            rows = get_sorted_layer_records(index_path, variable_id)
        except Exception:
            rows = []
        samples: list[ObservationSample] = []
        for catalog, lat, lon, value in rows:
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            samples.append(
                {
                    "catalog_id": str(catalog),
                    "value": numeric,
                    "latitude": lat if isinstance(lat, (int, float)) else None,
                    "longitude": lon if isinstance(lon, (int, float)) else None,
                }
            )
        if samples:
            return samples

    for candidate in (
        CONFIG.occurrence_parquet_filename,
        taxa_navigation.combined_parquet_filename,
    ):
        path = data_dir / candidate
        if not PARQUET.exists(path):
            continue
        try:
            samples = read_numeric_from_parquet(path, variable_id)
        except (KeyError, pa.lib.ArrowInvalid):
            continue
        if samples:
            return samples
    return gather_numeric_records_from_tables(taxon_id, variable_id, None)


def read_numeric_from_parquet(
    parquet_path: Path,
    variable_id: str,
) -> List[ObservationSample]:
    """Reads numeric samples for a variable from a single occurrence parquet.
    
    Args:
        parquet_path: Path to the parquet file to read.
        variable_id: Name of the numeric column to extract.
    
    Returns:
        A list of numeric observation samples with catalog id and coordinates.
    """
    table = PARQUET.read_table(
        parquet_path,
        columns=[
            "catalogNumber",
            "decimalLatitude",
            "decimalLongitude",
            "obscured",
            "coordinateUncertaintyInMeters",
            variable_id,
        ],
    ).combine_chunks()
    mask = taxa_navigation.base_observation_mask(table)
    value_col = table[variable_id]
    mask = pc.and_(mask, pc.invert(pc.is_null(value_col)))
    if pa.types.is_floating(value_col.type):
        mask = pc.and_(mask, pc.invert(pc.is_nan(value_col)))
    filtered = table.filter(mask).combine_chunks()
    if filtered.num_rows == 0:
        return []
    catalogs = filtered["catalogNumber"].to_pylist()
    latitudes = filtered["decimalLatitude"].to_pylist()
    longitudes = filtered["decimalLongitude"].to_pylist()
    values = filtered[variable_id].to_pylist()
    samples: list[ObservationSample] = []
    for catalog, lat, lon, value in zip(catalogs, latitudes, longitudes, values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        samples.append(
            {
                "catalog_id": str(catalog),
                "value": numeric,
                "latitude": float(lat) if isinstance(lat, (int, float)) else None,
                "longitude": float(lon) if isinstance(lon, (int, float)) else None,
            }
        )
    return samples


def gather_numeric_records_from_tables(
    taxon_id: int,
    variable_id: str,
    location_gid: Optional[str],
) -> List[ObservationSample]:
    """Collects numeric samples from filtered occurrence tables.
    
    Args:
        taxon_id: Taxon id to sample.
        variable_id: Numeric variable id to extract.
        location_gid: Optional location GID to filter observations.
    
    Returns:
        A list of numeric observation samples.
    """
    samples: list[ObservationSample] = []
    for table in taxa_navigation.iter_filtered_occurrence_tables(
        taxon_id,
        extra_columns=[variable_id],
        location_gid=location_gid,
    ):
        value_col = table[variable_id]
        mask = pc.invert(pc.is_null(value_col))
        if pa.types.is_floating(value_col.type):
            mask = pc.and_(mask, pc.invert(pc.is_nan(value_col)))
        filtered = table.filter(mask).combine_chunks()
        if filtered.num_rows == 0:
            continue
        catalogs = filtered["catalogNumber"].to_pylist()
        latitudes = filtered["decimalLatitude"].to_pylist()
        longitudes = filtered["decimalLongitude"].to_pylist()
        values = filtered[variable_id].to_pylist()
        for catalog, lat, lon, value in zip(catalogs, latitudes, longitudes, values):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            samples.append(
                {
                    "catalog_id": str(catalog),
                    "value": numeric,
                    "latitude": float(lat) if isinstance(lat, (int, float)) else None,
                    "longitude": float(lon) if isinstance(lon, (int, float)) else None,
                }
            )
    return samples


def categorical_class_samples_for_location(
    taxon_id: int,
    variable_id: str,
    class_value: Any,
    *,
    location_gid: str,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    """Returns categorical class samples within a location.
    
    Args:
        taxon_id: Taxon id to sample.
        variable_id: Categorical layer id to filter.
        class_value: Class value to match.
        location_gid: Location GID to filter observations.
        limit: Optional maximum number of samples to return.
    
    Returns:
        A list of observation dicts with catalogNumber/lat/lon/value.
    """
    resolved_value = resolve_categorical_class_value(variable_id, class_value)
    target_key, _ = categorical_value_key(resolved_value)
    observations: list[dict[str, Any]] = []
    for table in taxa_navigation.iter_filtered_occurrence_tables(
        taxon_id,
        extra_columns=[variable_id],
        location_gid=location_gid,
    ):
        value_col = table[variable_id]
        mask = pc.invert(pc.is_null(value_col))
        filtered = table.filter(mask).combine_chunks()
        if filtered.num_rows == 0:
            continue
        catalogs = filtered["catalogNumber"].to_pylist()
        latitudes = filtered["decimalLatitude"].to_pylist()
        longitudes = filtered["decimalLongitude"].to_pylist()
        values = filtered[variable_id].to_pylist()
        for catalog, lat, lon, value in zip(catalogs, latitudes, longitudes, values):
            value_key, normalized_value = categorical_value_key(value)
            if value_key != target_key:
                continue
            observations.append(
                {
                    "catalogNumber": str(catalog),
                    "latitude": float(lat) if isinstance(lat, (int, float)) else None,
                    "longitude": float(lon) if isinstance(lon, (int, float)) else None,
                    "value": normalized_value,
                }
            )
            if limit is not None and len(observations) >= limit:
                break
        if limit is not None and len(observations) >= limit:
            break
    return observations


def numeric_range_samples_for_location(
    taxon_id: int,
    variable_id: str,
    min_value: float,
    max_value: float,
    *,
    location_gid: str,
    limit: Optional[int],
) -> list[tuple[str, float | None, float | None, float | None]]:
    """Returns numeric samples within a value range for a location.
    
    Args:
        taxon_id: Taxon id to sample.
        variable_id: Numeric layer id to filter.
        min_value: Minimum value to include.
        max_value: Maximum value to include.
        location_gid: Location GID to filter observations.
        limit: Optional maximum number of samples to return.
    
    Returns:
        A list of (catalog, latitude, longitude, value) rows.
    """
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for table in taxa_navigation.iter_filtered_occurrence_tables(
        taxon_id,
        extra_columns=[variable_id],
        location_gid=location_gid,
    ):
        value_col = table[variable_id]
        mask = pc.invert(pc.is_null(value_col))
        mask = pc.and_(mask, pc.greater_equal(value_col, min_value))
        mask = pc.and_(mask, pc.less_equal(value_col, max_value))
        filtered = table.filter(mask).combine_chunks()
        if filtered.num_rows == 0:
            continue
        catalogs = filtered["catalogNumber"].to_pylist()
        latitudes = filtered["decimalLatitude"].to_pylist()
        longitudes = filtered["decimalLongitude"].to_pylist()
        values = filtered[variable_id].to_pylist()
        for catalog, lat, lon, value in zip(catalogs, latitudes, longitudes, values):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                (
                    str(catalog),
                    float(lat) if isinstance(lat, (int, float)) else None,
                    float(lon) if isinstance(lon, (int, float)) else None,
                    numeric,
                )
            )
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    return rows

_STATS_NODE_PATH: ContextVar[Path | None] = ContextVar(
    "STATS_NODE_PATH", default=None
)


@contextmanager
def stats_context(node_path: Path):
    """Context manager for summary stats generation.
    
    Args:
        node_path: Taxon directory to set as the current stats root.
    
    Returns:
        A context manager that sets the stats node path.
    """
    token = _STATS_NODE_PATH.set(Path(node_path))
    try:
        yield
    finally:
        _STATS_NODE_PATH.reset(token)


def _iter_descendant_tables(parquet_path: Path) -> Iterable[pa.Table]:
    """Yields occurrence tables for a taxon and its descendants.
    
    Args:
        parquet_path: Path to the occurrence parquet to locate under descendants.
    
    Yields:
        Pyarrow tables from descendant directories.
    """
    taxon_dir = parquet_path.parent
    filename = parquet_path.name
    taxon_key = taxa_navigation.taxon_key_from_path(taxon_dir)
    taxon = taxa_navigation.get_taxon_by_id(taxon_key)
    if taxon is None:
        return
    for record in taxa_navigation.iter_descendants(taxon, include_self=True):
        data_file = Path(record["path"]) / filename
        if PARQUET.exists(data_file):
            yield PARQUET.read_table(data_file)


def _digest_quantile(digest: Any, q: float) -> float | None:
    if hasattr(digest, "quantile"):
        return float(digest.quantile(q))
    if hasattr(digest, "percentile"):
        return float(digest.percentile(q * 100))
    return None


def _init_streaming_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "min_value": None,
        "max_value": None,
        "digest": _FastTDigest(),
    }


def _update_streaming_stats(stats: Dict[str, Any], values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy()
    if numeric.size == 0:
        return
    numeric = numeric[~np.isnan(numeric)]
    if numeric.size == 0:
        return

    count = int(numeric.size)
    mean_new = float(np.mean(numeric))
    m2_new = float(np.var(numeric, ddof=0) * count)

    if stats["count"] == 0:
        stats["mean"] = mean_new
        stats["m2"] = m2_new
    else:
        delta = mean_new - stats["mean"]
        total = stats["count"] + count
        stats["mean"] += delta * count / total
        stats["m2"] += m2_new + delta * delta * stats["count"] * count / total
    stats["count"] += count

    min_new = float(np.min(numeric))
    max_new = float(np.max(numeric))
    stats["min_value"] = (
        min_new if stats["min_value"] is None else min(stats["min_value"], min_new)
    )
    stats["max_value"] = (
        max_new if stats["max_value"] is None else max(stats["max_value"], max_new)
    )

    if hasattr(stats["digest"], "batch_update"):
        stats["digest"].batch_update(numeric.tolist())
    else:
        for value in numeric:
            stats["digest"].update(float(value))


def numeric_column_stats(*, streaming: bool = True) -> Dict[str, Dict[str, float]]:
    """Computes numeric and categorical stats for the current stats context.
    
    Args:
        streaming: Whether to use streaming approximations for percentiles.
    
    Returns:
        A mapping of numeric variable id to summary statistics.
    """
    node_path = _STATS_NODE_PATH.get()
    if node_path is None:
        raise RuntimeError(
            "numeric_column_stats() must be called inside stats_context()."
        )
    parquet_path = Path(node_path) / CONFIG.occurrence_parquet_filename
    existing_stats = _load_summary_stats(str(parquet_path.parent)) or {}
    existing_categorical = _load_categorical_stats(str(parquet_path.parent)) or {}
    existing_numeric_vars = set(existing_stats.keys())
    existing_categorical_vars = set(existing_categorical.keys())
    if streaming:
        return _numeric_column_stats_streaming(
            parquet_path,
            existing_numeric=existing_numeric_vars,
            existing_categorical=existing_categorical_vars,
        )
    return _numeric_column_stats_exact(
        parquet_path,
        existing_numeric=existing_numeric_vars,
        existing_categorical=existing_categorical_vars,
    )


def _numeric_column_stats_exact(
    parquet_path: Path,
    *,
    existing_numeric: set[str] | None = None,
    existing_categorical: set[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    tables = list(_iter_descendant_tables(parquet_path))
    if not tables:
        return {}

    if len(tables) == 1:
        df = tables[0].to_pandas()
    else:
        dfs = [tbl.to_pandas() for tbl in tables]
        df = pd.concat(dfs, ignore_index=True)  # don't keep the original index

    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[df["coordinateUncertaintyInMeters"] <= 500]

    categorical_cols = [
        col for col in df.columns if _layer_value_type(col) == "categorical"
    ]
    if existing_categorical:
        categorical_cols = [col for col in categorical_cols if col not in existing_categorical]
    categorical_entries = _collect_categorical_stats(df, categorical_cols)
    _write_categorical_stats(
        parquet_path.parent,
        categorical_entries,
        merge_existing=True,
    )

    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in excluded_numeric_columns and _layer_value_type(col) != "categorical"
    ]
    if existing_numeric:
        numeric_cols = [col for col in numeric_cols if col not in existing_numeric]
    if not numeric_cols:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        stats[col] = summarize_layer_values_for_storage(col, df[col].tolist())

    _write_summary_stats(parquet_path.parent, stats, merge_existing=True)
    return stats


def _numeric_column_stats_streaming(
    parquet_path: Path,
    *,
    existing_numeric: set[str] | None = None,
    existing_categorical: set[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    categorical_counts: Dict[str, Counter] = defaultdict(Counter)
    categorical_totals: Dict[str, int] = defaultdict(int)
    numeric_stats: Dict[str, Dict[str, Any]] = {}
    circular_values: Dict[str, list[float]] = defaultdict(list)

    tables_seen = False
    for table in _iter_descendant_tables(parquet_path):
        tables_seen = True
        df = table.to_pandas()

        if "obscured" in df.columns:
            df = df[df["obscured"] == "No"]
        if "coordinateUncertaintyInMeters" in df.columns:
            df = df[df["coordinateUncertaintyInMeters"] <= 500]

        categorical_cols = [
            col for col in df.columns if _layer_value_type(col) == "categorical"
        ]
        if existing_categorical:
            categorical_cols = [
                col for col in categorical_cols if col not in existing_categorical
            ]
        for column in categorical_cols:
            series = df[column].dropna()
            if series.empty:
                continue
            categorical_totals[column] += int(series.count())
            counts = series.value_counts(dropna=True)
            for raw_value, count in counts.items():
                categorical_counts[column][raw_value] += int(count)

        numeric_cols = [
            col
            for col in df.select_dtypes(include=["number"]).columns
            if col not in excluded_numeric_columns and _layer_value_type(col) != "categorical"
        ]
        if existing_numeric:
            numeric_cols = [col for col in numeric_cols if col not in existing_numeric]
        for column in numeric_cols:
            if _is_circular_variable(column):
                values = pd.to_numeric(df[column], errors="coerce").to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    circular_values[column].extend(values.astype(float).tolist())
                continue
            stats_entry = numeric_stats.get(column)
            if stats_entry is None:
                stats_entry = _init_streaming_stats()
                numeric_stats[column] = stats_entry
            _update_streaming_stats(stats_entry, df[column])

    if not tables_seen:
        return {}

    categorical_entries = _collect_categorical_stats_from_counts(
        categorical_counts, categorical_totals
    )
    _write_categorical_stats(
        parquet_path.parent,
        categorical_entries,
        merge_existing=True,
    )

    if not numeric_stats:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for column, values in numeric_stats.items():
        if values["count"] == 0:
            continue
        q10 = _digest_quantile(values["digest"], 0.10) or 0.0
        q25 = _digest_quantile(values["digest"], 0.25) or 0.0
        q50 = _digest_quantile(values["digest"], 0.50) or 0.0
        q75 = _digest_quantile(values["digest"], 0.75) or 0.0
        q90 = _digest_quantile(values["digest"], 0.90) or 0.0
        std = math.sqrt(values["m2"] / values["count"]) if values["count"] else 0.0
        min_value = values["min_value"] if values["min_value"] is not None else 0.0
        max_value = values["max_value"] if values["max_value"] is not None else 0.0
        stats[column] = {
            "count": int(values["count"]),
            "min": float(min_value),
            "10th percentile": float(q10),
            "25th percentile": float(q25),
            "median": float(q50),
            "75th percentile": float(q75),
            "90th percentile": float(q90),
            "max": float(max_value),
            "mean": float(values["mean"]),
            "std": float(std),
            "10-90 range": float(q90 - q10),
            "range": float(max_value - min_value),
        }

    for column, values in circular_values.items():
        if not values:
            continue
        stats[column] = summarize_layer_values_for_storage(column, values)

    _write_summary_stats(parquet_path.parent, stats, merge_existing=True)
    return stats


def _collect_categorical_stats(df: pd.DataFrame, categorical_cols: Sequence[str]) -> List[Dict[str, Any]]:
    """Builds categorical distribution entries for selected columns.
    
    Args:
        df: Dataframe containing occurrence data.
        categorical_cols: Column names to treat as categorical.
    
    Returns:
        A list of tall categorical stats entries (variable/metric/value).
    """
    entries: List[Dict[str, Any]] = []
    for column in categorical_cols:
        if column not in df.columns:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        try:
            total = int(series.count())
        except (TypeError, ValueError):
            total = len(series)
        counts = series.value_counts(dropna=True)
        entries.append({"variable": column, "metric": "total_samples", "value": total})
        entries.append({"variable": column, "metric": "unique_classes", "value": int(len(counts))})
        significant = 0
        if total > 0:
            for count in counts:
                if (count / total) >= significant_category_threshold:
                    significant += 1
        entries.append(
            {
                "variable": column,
                "metric": "significant_unique_classes",
                "value": int(significant),
            }
        )
        for raw_class_value, count in counts.items():
            try:
                class_id = int(raw_class_value)
            except (TypeError, ValueError):
                class_id = raw_class_value
            metric_name = f"class_{class_id}"
            percentage = float(count) / float(total) if total else 0.0
            entries.append(
                {
                    "variable": column,
                    "metric": metric_name,
                    "value": percentage,
                }
            )
    return entries


def _collect_categorical_stats_from_counts(
    counts_by_column: Dict[str, Counter],
    totals_by_column: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Builds categorical distribution entries from aggregated counts."""
    entries: List[Dict[str, Any]] = []
    for column, counts in counts_by_column.items():
        total = int(totals_by_column.get(column, 0))
        if total <= 0:
            continue
        entries.append({"variable": column, "metric": "total_samples", "value": total})
        entries.append({"variable": column, "metric": "unique_classes", "value": int(len(counts))})
        significant = 0
        for count in counts.values():
            if (count / total) >= significant_category_threshold:
                significant += 1
        entries.append(
            {
                "variable": column,
                "metric": "significant_unique_classes",
                "value": int(significant),
            }
        )
        for raw_class_value, count in counts.items():
            try:
                class_id = int(raw_class_value)
            except (TypeError, ValueError):
                class_id = raw_class_value
            metric_name = f"class_{class_id}"
            percentage = float(count) / float(total) if total else 0.0
            entries.append(
                {
                    "variable": column,
                    "metric": metric_name,
                    "value": percentage,
                }
            )
    return entries

def _write_summary_stats(
    directory: Path,
    stats: Dict[str, Dict[str, Any]],
    *,
    merge_existing: bool = False,
) -> None:
    """Writes numeric summary stats to summary_stats.parquet.
    
    Args:
        directory: Output directory to write the stats parquet.
        stats: Mapping of variable id to metric dicts.
    """
    stats = {variable: values for variable, values in stats.items() if values}
    if not stats:
        return
    frame = pd.DataFrame.from_dict(stats, orient="index") # passing index means we use the keys of the provided dict as the rows (each variable makes sense as a row, metrics are the columns)
    serialized = frame.reset_index().rename(columns={"index": "variable"}) # simply reset and rename the index to the variable column
    stats_path = directory / "summary_stats.parquet"
    try:
        if merge_existing and stats_path.exists():
            existing = pd.read_parquet(stats_path)
            if not existing.empty:
                combined = pd.concat([existing, serialized], ignore_index=True)
                combined = combined.drop_duplicates(subset=["variable"], keep="last")
                serialized = combined
        serialized.to_parquet(stats_path, index=False)
        _load_summary_stats.cache_clear()
    except Exception:
        pass


def _density_point_count(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 32:
        return max(8, count)
    if count <= 200:
        return 64
    return 128


def _build_density_curve(values: Sequence[float], point_count: int) -> Optional[dict[str, Any]]:
    if not values or point_count <= 0:
        return None
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    count = len(array)
    min_val = float(array.min())
    max_val = float(array.max())
    if math.isclose(min_val, max_val):
        span = 1.0 if math.isclose(min_val, 0.0) else abs(min_val) * 0.1 or 1.0
        min_val -= span
        max_val += span
    std = float(array.std()) or 1.0
    bandwidth = 1.06 * std * (count ** (-0.2))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = (max_val - min_val) / 20 or 1.0
    xs = np.linspace(min_val, max_val, point_count)
    diffs = (xs[:, None] - array[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2)
    factor = 1.0 / (count * bandwidth * math.sqrt(2 * math.pi))
    densities = kernel.sum(axis=1) * factor
    return {
        "points": [float(value) for value in xs.tolist()],
        "density": [float(value) for value in densities.tolist()],
        "min": min_val,
        "max": max_val,
        "bandwidth": bandwidth,
    }


def write_density_graph(directory: Path) -> None:
    """Precompute density curves for numeric columns and write density_graph.parquet."""
    parquet_path = Path(directory) / CONFIG.occurrence_parquet_filename
    rows: list[dict[str, Any]] = []
    counts: Dict[str, int] = {}
    samples: Dict[str, list[float]] = {}

    for table in _iter_descendant_tables(parquet_path):
        df = table.to_pandas()

        if "obscured" in df.columns:
            df = df[df["obscured"] == "No"]
        if "coordinateUncertaintyInMeters" in df.columns:
            df = df[df["coordinateUncertaintyInMeters"] <= 500]

        numeric_cols = [
            col
            for col in df.select_dtypes(include=["number"]).columns
            if col not in excluded_numeric_columns and _layer_value_type(col) != "categorical"
        ]
        for column in numeric_cols:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                continue
            values = series.astype(float).tolist()
            if not values:
                continue
            sample_bucket = samples.setdefault(column, [])
            seen = counts.get(column, 0)
            for value in values:
                seen += 1
                if len(sample_bucket) < density_max_samples:
                    sample_bucket.append(float(value))
                else:
                    j = random.randrange(seen)
                    if j < density_max_samples:
                        sample_bucket[j] = float(value)
            counts[column] = seen

    for column, sample_values in samples.items():
        count = counts.get(column, 0)
        if count <= 0 or not sample_values:
            continue
        point_count = _density_point_count(count)
        curve = _build_density_curve(sample_values, point_count)
        if not curve:
            continue
        rows.append(
            {
                "variable": column,
                "count": int(count),
                "sampleCount": int(len(sample_values)),
                "pointCount": int(point_count),
                "points": curve["points"],
                "density": curve["density"],
                "min": curve["min"],
                "max": curve["max"],
                "bandwidth": curve["bandwidth"],
            }
        )

    out_path = Path(directory) / density_graph_filename
    if not rows:
        out_path.unlink(missing_ok=True)
        return
    table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _write_categorical_stats(
    directory: Path,
    entries: List[Dict[str, Any]],
    *,
    merge_existing: bool = False,
) -> None:
    """Writes categorical stats to categorical_stats.parquet.
    
    Args:
        directory: Output directory to write the stats parquet.
        entries: Tall categorical stats entries (variable/metric/value).
    """
    stats_path = directory / "categorical_stats.parquet"
    entries = [entry for entry in entries if entry]
    if not entries:
        if merge_existing:
            return
        stats_path.unlink(missing_ok=True)
        return
    frame = pd.DataFrame(entries)
    try:
        if merge_existing and stats_path.exists():
            existing = pd.read_parquet(stats_path)
            if not existing.empty:
                combined = pd.concat([existing, frame], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["variable", "metric"],
                    keep="last",
                )
                frame = combined
        frame.to_parquet(stats_path, index=False)
        _load_categorical_stats.cache_clear()
    except Exception:
        pass

@lru_cache(maxsize=4096)
def _load_summary_stats(path_str: str) -> Dict[str, Any] | None:
    """Loads cached numeric summary stats for a taxon directory.
    
    Args:
        path_str: Taxon directory path as a string.
    
    Returns:
        A mapping of variable id to metric dicts, or None if missing.
    """
    stats_path = Path(path_str) / "summary_stats.parquet"
    if not PARQUET.exists(stats_path):
        return None
    try:
        frame = PARQUET.read_table(stats_path).to_pandas()
        stats = _dataframe_to_stats(frame)
        return stats
    except (OSError, ValueError):
        return None


def load_density_graph(path_str: str, variable_id: str) -> Dict[str, Any] | None:
    """Loads a precomputed density graph for a variable if available."""
    density_path = Path(path_str) / density_graph_filename
    local_exists = density_path.exists()
    if not PARQUET.exists(density_path) and not local_exists:
        try:
            density_path.stat()
            stat_msg = "stat-ok"
        except Exception as exc:
            stat_msg = f"stat-fail:{exc}"
        print(f"[density] missing file {density_path} ({stat_msg})")
        return None
    try:
        columns = [
            "variable",
            "points",
            "density",
            "min",
            "max",
            "bandwidth",
            "count",
            "sampleCount",
            "pointCount",
        ]
        if PARQUET.is_remote and local_exists:
            table = pq.read_table(density_path, columns=columns)
        else:
            table = PARQUET.read_table(density_path, columns=columns)
    except (OSError, ValueError) as exc:
        print(f"[density] read failed {density_path}: {exc}")
        return None
    if not table.num_rows:
        print(f"[density] empty file {density_path}")
        return None
    try:
        variable_col = pc.cast(table["variable"], pa.string())
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        variable_col = table["variable"]
    mask = pc.equal(variable_col, variable_id)
    filtered = table.filter(mask).combine_chunks()
    if not filtered.num_rows:
        preview = table.column("variable").to_pylist()[:10]
        print(
            f"[density] missing var={variable_id} in {density_path} "
            f"(first={preview})"
        )
        return None
    row = filtered.to_pylist()[0]
    return {
        "points": row.get("points") or [],
        "density": row.get("density") or [],
        "min": row.get("min"),
        "max": row.get("max"),
        "bandwidth": row.get("bandwidth"),
        "count": row.get("count"),
        "sampleCount": row.get("sampleCount"),
        "pointCount": row.get("pointCount"),
    }


def load_numeric_summary(path_str: str, variable_id: str) -> Dict[str, Any] | None:
    stats = _load_summary_stats(path_str)
    if not stats:
        stats_path = Path(path_str) / "summary_stats.parquet"
        if PARQUET.exists(stats_path):
            _load_summary_stats.cache_clear()
            stats = _load_summary_stats(path_str)
        if not stats:
            return None
    return stats.get(variable_id)


@lru_cache(maxsize=4096)
def _load_categorical_stats(path_str: str) -> Dict[str, Dict[str, Any]]:
    """Loads cached categorical stats for a taxon directory.
    
    Args:
        path_str: Taxon directory path as a string.
    
    Returns:
        A mapping of variable id to metric dicts.
    """
    stats_path = Path(path_str) / "categorical_stats.parquet"
    if not PARQUET.exists(stats_path):
        return {}
    try:
        frame = PARQUET.read_table(stats_path).to_pandas()
    except (OSError, ValueError):
        return {}
    return _tall_dataframe_to_stats(frame)


def _dataframe_to_stats(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Converts a wide stats dataframe to a nested dict.
    
    Args:
        frame: Dataframe with variables as rows and metrics as columns.
    
    Returns:
        A mapping of variable id to metric/value dicts.
    """
    if frame.empty:
        return {}
    working = frame.copy()
    if "variable" in working.columns:
        working = working.set_index("variable")
    result: Dict[str, Dict[str, Any]] = {}
    for variable, row in working.iterrows():
        entries = {}
        for key, value in row.items():
            if pd.isna(value):
                continue
            if hasattr(value, "item"):
                try:
                    entries[key] = value.item()
                    continue
                except ValueError:
                    pass
            entries[key] = value
        result[str(variable)] = entries
    return result


def _tall_dataframe_to_stats(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Converts a tall stats dataframe to a nested dict.
    
    Args:
        frame: Dataframe with columns variable/metric/value.
    
    Returns:
        A mapping of variable id to metric/value dicts.
    """
    if frame.empty:
        return {}
    working = frame.copy()
    required = {"variable", "metric", "value"}
    if not required.issubset(set(working.columns)):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in working.itertuples(index=False):
        variable = getattr(row, "variable", None)
        metric = getattr(row, "metric", None)
        value = getattr(row, "value", None)
        if variable is None or metric is None:
            continue
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        bucket = result.setdefault(str(variable), {})
        bucket[str(metric)] = value
    return result
