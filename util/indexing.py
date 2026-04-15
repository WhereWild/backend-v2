"""
Index creation and ranking/query helpers for occurrence and ranking parquets.
"""

from __future__ import annotations

from pathlib import Path
import bisect
from functools import lru_cache
import logging
from typing import Any, Sequence, Optional, List, Tuple, Dict, NamedTuple, cast
import json
import math
import os
import re
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from util import gis_lookup, taxa_navigation
from util.config import load_config
from util.request_cancellation import CancelCheck
from util.summary_stats import _load_summary_stats, _load_categorical_stats
from util.storage import ParquetStorageProxy

CONFIG = load_config("global")
PARQUET = ParquetStorageProxy(CONFIG.data_root, CONFIG.project_root)
LOGGER = logging.getLogger("uvicorn.error")
PC: Any = pc

descendant_rank_order = (
    "KINGDOM",
    "PHYLUM",
    "CLASS",
    "ORDER",
    "FAMILY",
    "GENUS",
    "SPECIES",
    "SUBSPECIES",
)


def _validated_descendant_rank(value: str | None, *, required: bool = False) -> str:
    canonical = taxa_navigation.canonical_rank(value)
    if not canonical:
        if required:
            raise ValueError("descendant_rank is required")
        return ""
    if not taxa_navigation.is_valid_descendant_rank(canonical):
        raise ValueError(f"Unknown descendant_rank: {value}")
    return canonical


def _rank_index_storage_rank(descendant_rank: str | None) -> str:
    canonical_rank = _validated_descendant_rank(descendant_rank, required=True)
    if canonical_rank in CONFIG.subspecies_equivalents:
        return "SUBSPECIES"
    return canonical_rank


relative_rank_metrics = (
    "count",
    "min",
    "10th percentile",
    "25th percentile",
    "median",
    "75th percentile",
    "90th percentile",
    "mean",
    "std",
    "10-90 range",
    "range",
    "1st percentile",
    "99th percentile",
    "interquartile range",
    "1-99 range",
    "max",
)

relative_rank_global_dirname = "_relative_ranks_positions"


def global_relative_positions_dir() -> Path:
    return CONFIG.taxonomy_root / relative_rank_global_dirname


@lru_cache(maxsize=1)
def _temporal_registry_config() -> tuple[frozenset[str], tuple[str, ...]]:
    """Return expanded temporal variable ids and temporal base ids.

    Expanded ids are built from registry layer + windows definitions and include
    snapshot ids directly.
    """
    registry = gis_lookup.load_temporal_registry() or {}
    default_windows = registry.get("windows", []) or []
    expanded_ids: set[str] = set()
    base_ids: list[str] = []
    for layer in registry.get("layers", []) or []:
        base_id = str(layer.get("id") or "").strip()
        if not base_id:
            continue
        base_ids.append(base_id)
        agg = str(layer.get("agg") or "avg").strip().lower()
        if agg == "snapshot":
            expanded_ids.add(base_id)
            continue
        windows = layer.get("windows") or default_windows
        for hours in windows:
            try:
                hour_value = int(hours)
            except (TypeError, ValueError):
                continue
            if hour_value <= 0:
                continue
            expanded_ids.add(f"{base_id}_{agg}_{hour_value}h")
    return frozenset(expanded_ids), tuple(sorted(set(base_ids)))


def _extract_variable_from_metric_column(column_key: str) -> str:
    if "::" not in column_key:
        return column_key
    variable, _metric = column_key.split("::", 1)
    return variable


def _is_temporal_variable_id(variable_id: str) -> bool:
    variable = str(variable_id or "").strip()
    if not variable:
        return False
    expanded_ids, base_ids = _temporal_registry_config()
    if variable in expanded_ids:
        return True
    # Fallback for expanded temporal names not listed in registry windows.
    if re.search(r"_\d+h$", variable):
        for base in base_ids:
            if variable.startswith(f"{base}_"):
                return True
    return False


def _is_temporal_metric_column(column_key: str) -> bool:
    return _is_temporal_variable_id(_extract_variable_from_metric_column(column_key))


def _load_global_relative_rows(
    taxon_key: str,
    variable_id: str,
    metric_names: Optional[Sequence[str]] = None,
) -> pa.Table | None:
    if PARQUET.is_remote:
        return None
    base = global_relative_positions_dir()
    if not base.exists():
        return None
    try:
        dataset = pds.dataset(str(base), format="parquet")
    except (OSError, ValueError):
        return None
    try:
        filter_expr = (pds.field("taxonKey") == str(taxon_key)) & (pds.field("variable") == str(variable_id))
        requested_metrics = [str(name).strip() for name in (metric_names or ()) if str(name).strip()]
        if requested_metrics:
            filter_expr = filter_expr & pds.field("metric").isin(requested_metrics)
        table = dataset.to_table(
            columns=[
                "variable",
                "metric",
                "position",
                "count",
                "sampleCount",
                "contextTaxonId",
                "contextLabel",
            ],
            filter=filter_expr,
        )
    except (OSError, ValueError):
        return None
    if not table.num_rows:
        return None
    return table


def _harmonize_numeric_arrays(arrays: list[pa.Array]) -> list[pa.Array]:
    if not arrays:
        return arrays
    types = {arr.type for arr in arrays}
    if len(types) <= 1:
        return arrays
    if all(pa.types.is_integer(t) or pa.types.is_floating(t) for t in types):
        target = pa.float64() if any(pa.types.is_floating(t) for t in types) else pa.int64()
        return [PC.cast(arr, target) if arr.type != target else arr for arr in arrays]
    if all(pa.types.is_string(t) or pa.types.is_large_string(t) for t in types):
        target = pa.string()
        return [PC.cast(arr, target) if arr.type != target else arr for arr in arrays]
    return arrays


def index_targets_for_columns(
    available_columns: set[str],
    *,
    layer_catalog: Optional[dict[str, dict[str, Any]]] = None,
) -> list[tuple[str, str]]:
    """Resolve index target columns from catalog metadata and available columns."""
    catalog = layer_catalog or gis_lookup.load_layer_metadata()
    available = set(available_columns)
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for layer_id, layer in catalog.items():
        if not layer_id:
            continue
        value_type = str(layer.get("value_type") or "").lower()
        agg = str(layer.get("agg") or "").strip().lower()
        candidates: list[str] = [layer_id]
        if agg and agg != "snapshot":
            prefix = f"{layer_id}_{agg}_"
            temporal_candidates = sorted(
                column for column in available if column.startswith(prefix) and column.endswith("h")
            )
            if temporal_candidates:
                candidates = temporal_candidates
        for candidate in candidates:
            if candidate not in available or candidate in seen:
                continue
            seen.add(candidate)
            targets.append((candidate, value_type))
    return targets


def build_index_parquet(node_path: Path) -> None:
    """Builds/overwrites an occurrence_index.parquet for a given node in the tree.
        The index is a sorted list of tuples `(catalogNumber, originId, value)` for each GIS variable.
        This means a sorted list of observations for a species by any GIS variable can be done in O(n) time
        where n is the number of observations. `originId` is the parquet the row came from; if 0 the row came from
        the taxon in question, but if numbered, it came from one of its children subspecies (in-order) so it can be
        implicitly reconstructed if needed.
        TODO: lat/lon are not stored implicitly here for size reasons but do make queries that need them O(nlogn)
        in practice. We need to profile the API eventually and see if it makes sense to move to storing them in the index as well
        to reduce API time, but this increases storage size by probably a non-trivial amount overall (storing two more floats for each row).

    Args:
        node_path: The path to the node in question..

    Returns:
        Nothing, but writes the index parquet as described above.
    """
    node_path = Path(node_path)
    catalog_number_col = "catalogNumber"
    data_parquet = Path(node_path) / CONFIG.occurrence_parquet_filename
    index_parquet = Path(node_path) / "occurrence_index.parquet"
    layer_catalog = gis_lookup.load_layer_metadata()
    indexed_layer_types: dict[str, str] = {}
    category_offsets: dict[str, dict[str, dict[str, int | float]]] = {}
    data_parquet_exists = data_parquet.exists()
    parent_dir = data_parquet.parent

    if data_parquet_exists:
        table = PARQUET.read_table(data_parquet)
        if catalog_number_col not in table.schema.names:
            raise ValueError(f"{catalog_number_col} not found in {data_parquet}")
    else:
        table = pa.table({catalog_number_col: pa.array([], type=pa.string())})

    # Ensure parent is sorted by catalogNumber
    if len(table) > 1:
        cat_col = table[catalog_number_col]
        is_sorted = PC.all(
            PC.less_equal(
                cat_col.slice(0, len(cat_col) - 1),
                cat_col.slice(1),
            )
        ).as_py()

        if not is_sorted and data_parquet_exists:
            sort_indices = PC.sort_indices(cat_col)
            table = table.take(sort_indices)

            # atomic rewrite of occurrence parquet
            data_parquet = data_parquet.resolve()
            with tempfile.NamedTemporaryFile(
                dir=data_parquet.parent,
                suffix=".parquet",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)

            try:
                pq.write_table(table, tmp_path)
                os.replace(tmp_path, data_parquet)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

    # Collect datasets for the index. Each dataset is a parquet. We might need multiple if the node is a species node (has its own parquet) but also has subspecies children (also have their own parquet)
    parent_catalog_numbers = table[catalog_number_col].combine_chunks()
    if not (pa.types.is_string(parent_catalog_numbers.type) or pa.types.is_large_string(parent_catalog_numbers.type)):
        parent_catalog_numbers = PC.cast(parent_catalog_numbers, pa.string())
    datasets = [
        {
            "origin_id": 0,
            "table": table,
            "catalog_numbers": parent_catalog_numbers,
            "relative_path": ".",
            "path": parent_dir,
            "taxon_key": taxa_navigation.taxon_key_from_path(parent_dir),
        }
    ]

    # We also build a map for this
    origin_map = [
        {
            "id": 0,
            "relative_path": ".",
            "taxon_key": taxa_navigation.taxon_key_from_path(parent_dir),
        }
    ]

    # We iterate over children, adding their datasets
    parent_taxon_key = taxa_navigation.taxon_key_from_path(parent_dir)
    origin_id_counter = 1
    for child_taxon in taxa_navigation.get_children(parent_taxon_key):
        child_dir = Path(child_taxon["path"])
        child_parquet = child_dir / data_parquet.name
        if not child_parquet.exists():
            continue

        child_table = PARQUET.read_table(child_parquet)
        if catalog_number_col not in child_table.schema.names:
            continue

        child_catalog_numbers = child_table[catalog_number_col].combine_chunks()
        if not (pa.types.is_string(child_catalog_numbers.type) or pa.types.is_large_string(child_catalog_numbers.type)):
            child_catalog_numbers = PC.cast(child_catalog_numbers, pa.string())
        datasets.append(
            {
                "origin_id": origin_id_counter,
                "table": child_table,
                "catalog_numbers": child_catalog_numbers,
                "relative_path": child_dir.name,
                "path": child_dir,
                "taxon_key": child_taxon["taxon_key"],
            }
        )
        origin_map.append(
            {
                "id": origin_id_counter,
                "relative_path": child_dir.name,
                "taxon_key": child_taxon["taxon_key"],
            }
        )
        origin_id_counter += 1

    available_columns: set[str] = set()
    for dataset in datasets:
        available_columns.update(dataset["table"].schema.names)

    index_targets = index_targets_for_columns(
        available_columns,
        layer_catalog=layer_catalog,
    )
    indexed_layer_types = {layer_id: value_type for layer_id, value_type in index_targets}

    # ---- Build index columns ----
    existing_table: pa.Table | None = None
    existing_columns: set[str] = set()
    existing_column_lengths: dict[str, int] = {}
    existing_category_offsets: dict[str, dict[str, dict[str, int | float]]] = {}
    existing_origin_map: list[dict[str, Any]] | None = None
    existing_catalog_column: str | None = None

    if index_parquet.exists():
        try:
            existing_schema = pq.read_schema(index_parquet)
            existing_columns = set(existing_schema.names)
            metadata = dict(existing_schema.metadata or {})
            raw_lengths = metadata.get(b"column_lengths")
            if raw_lengths:
                existing_column_lengths = json.loads(raw_lengths.decode("utf-8"))
            raw_offsets = metadata.get(b"category_offsets")
            if raw_offsets:
                existing_category_offsets = json.loads(raw_offsets.decode("utf-8"))
            raw_origin_map = metadata.get(b"origin_map")
            if raw_origin_map:
                existing_origin_map = json.loads(raw_origin_map.decode("utf-8"))
            raw_catalog = metadata.get(b"catalog_column")
            if raw_catalog:
                existing_catalog_column = raw_catalog.decode("utf-8")
        except Exception:
            existing_columns = set()
            existing_column_lengths = {}
            existing_category_offsets = {}
            existing_origin_map = None
            existing_catalog_column = None

    if existing_origin_map:
        datasets = []
        for entry in existing_origin_map:
            rel_path = entry.get("relative_path") or "."
            origin_id = int(entry.get("id", 0))
            data_dir = parent_dir if rel_path == "." else parent_dir / rel_path
            data_file = data_dir / data_parquet.name
            if not data_file.exists():
                continue
            try:
                entry_table = pq.read_table(data_file)
            except Exception:
                continue
            if catalog_number_col not in entry_table.schema.names:
                continue
            entry_catalog_numbers = entry_table[catalog_number_col].combine_chunks()
            if not (
                pa.types.is_string(entry_catalog_numbers.type) or pa.types.is_large_string(entry_catalog_numbers.type)
            ):
                entry_catalog_numbers = PC.cast(entry_catalog_numbers, pa.string())
            datasets.append(
                {
                    "origin_id": origin_id,
                    "table": entry_table,
                    "catalog_numbers": entry_catalog_numbers,
                    "relative_path": rel_path,
                    "path": data_dir,
                    "taxon_key": entry.get("taxon_key"),
                }
            )
        origin_map = existing_origin_map

    pending_targets = [
        (layer_id, value_type)
        for layer_id, value_type in index_targets
        if layer_id and layer_id not in existing_columns
    ]

    if not pending_targets and existing_columns:
        print(f"skip indexing {str(parent_dir).split('/')[-1]} (already built)")
        return

    index_columns: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    max_len = 0

    for layer_id, _value_type in pending_targets:
        combined_values = []
        combined_catalogs = []
        combined_origins = []

        for dataset in datasets:
            table = dataset["table"]
            if layer_id not in table.schema.names:
                continue

            layer_col = table[layer_id]

            # after grabbing the column with the layer we care about, mask out nulls so we don't index them
            mask = PC.invert(PC.is_null(layer_col))
            if pa.types.is_floating(layer_col.type):
                mask = PC.and_(mask, PC.invert(PC.is_nan(layer_col)))
            try:
                obscured_col = table["obscured"]
                mask = PC.and_(mask, PC.equal(obscured_col, "No"))
            except KeyError:
                pass
            try:
                coord_col = table["coordinateUncertaintyInMeters"]
                mask = PC.and_(mask, PC.less_equal(coord_col, 500))
            except KeyError:
                pass

            # filter the catalogs the same way so we have indexes of equal length as the original cols
            filtered_values = PC.filter(layer_col, mask).combine_chunks()
            filtered_catalogs = PC.filter(
                dataset["catalog_numbers"],
                mask,
            ).combine_chunks()

            if len(filtered_values) == 0:
                continue

            # Normalize types across datasets to avoid concat type mismatches
            target_type = pa.int64() if indexed_layer_types.get(layer_id) == "categorical" else pa.float64()
            try:
                filtered_values = PC.cast(filtered_values, target_type)
            except pa.ArrowInvalid:
                filtered_values = PC.cast(filtered_values, pa.float64())

            # append the filtered values and catalogs and origins. origin allows the index user to correctly identify which file it came from
            combined_values.append(filtered_values)
            combined_catalogs.append(filtered_catalogs)
            combined_origins.append(
                pa.array(
                    [dataset["origin_id"]] * len(filtered_values),
                    type=pa.int32(),
                )
            )

        if not combined_values:
            continue

        combined_values = _harmonize_numeric_arrays(combined_values)
        combined_catalogs = _harmonize_numeric_arrays(combined_catalogs)

        values = pa.concat_arrays(combined_values)
        catalogs = pa.concat_arrays(combined_catalogs)
        origins = pa.concat_arrays(combined_origins)

        # sort the indices by their values, and take the respective catalogs and origins after this sorting
        sort_indices = PC.sort_indices(values)
        sorted_catalogs = PC.take(catalogs, sort_indices)
        sorted_origins = PC.take(origins, sort_indices)
        sorted_values = PC.take(values, sort_indices)

        # store the sorted catalogs and origins
        struct_array = pa.StructArray.from_arrays(
            [sorted_catalogs, sorted_origins, sorted_values],
            fields=[
                pa.field("catalogNumber", sorted_catalogs.type),
                pa.field("originId", sorted_origins.type),
                pa.field("value", sorted_values.type),
            ],
        )

        # get relevant data
        index_columns[layer_id] = struct_array
        column_lengths[layer_id] = len(struct_array)
        max_len = max(max_len, len(struct_array))

        if indexed_layer_types.get(layer_id) == "categorical":
            offsets: dict[str, dict[str, int | float]] = {}
            py_values = sorted_values.to_pylist()
            current_value = None
            start_idx = 0
            for idx, value in enumerate(py_values):
                if current_value is None:
                    current_value = value
                    start_idx = idx
                    continue
                if value != current_value:
                    offsets[str(current_value)] = cast(
                        dict[str, int | float],
                        {
                            "value": current_value if current_value is not None else None,
                            "start": start_idx,
                            "count": idx - start_idx,
                        },
                    )
                    current_value = value
                    start_idx = idx
            if current_value is not None:
                offsets[str(current_value)] = cast(
                    dict[str, int | float],
                    {
                        "value": current_value if current_value is not None else None,
                        "start": start_idx,
                        "count": len(py_values) - start_idx,
                    },
                )
            if offsets:
                category_offsets[layer_id] = offsets

    if not index_columns and existing_columns:
        print(f"skip indexing {str(parent_dir).split('/')[-1]} (no new layers)")
        return

    if existing_columns:
        existing_table = pq.read_table(index_parquet)
        assert existing_table is not None
        existing_arrays: dict[str, pa.Array] = {}
        for name in existing_table.schema.names:
            existing_arrays[name] = existing_table[name].combine_chunks()
            if name not in existing_column_lengths:
                nulls = int(PC.sum(PC.is_null(existing_arrays[name])).as_py())
                existing_column_lengths[name] = len(existing_arrays[name]) - nulls

        existing_max_len = existing_table.num_rows
        max_len = max(max_len, existing_max_len)

        for key, arr in list(existing_arrays.items()):
            if len(arr) < max_len:
                pad = pa.nulls(max_len - len(arr), type=arr.type)
                existing_arrays[key] = pa.concat_arrays([arr, pad])

        for key, arr in list(index_columns.items()):
            if len(arr) < max_len:
                pad = pa.nulls(max_len - len(arr), type=arr.type)
                index_columns[key] = pa.concat_arrays([arr, pad])

        merged_arrays = {**existing_arrays, **index_columns}
        index_table = pa.table(merged_arrays)
        metadata = dict(index_table.schema.metadata or {})
        merged_column_lengths = {**existing_column_lengths, **column_lengths}
        merged_category_offsets = {**existing_category_offsets, **category_offsets}
        metadata[b"origin_map"] = json.dumps(existing_origin_map or origin_map).encode("utf-8")
        metadata[b"column_lengths"] = json.dumps(merged_column_lengths).encode("utf-8")
        metadata[b"catalog_column"] = (existing_catalog_column or catalog_number_col).encode("utf-8")
        metadata[b"category_offsets"] = json.dumps(merged_category_offsets).encode("utf-8")
        index_table = index_table.replace_schema_metadata(metadata)
    else:
        # pad rows with nulls at the end
        for key, arr in list(index_columns.items()):
            if len(arr) < max_len:
                pad = pa.nulls(max_len - len(arr), type=arr.type)
                index_columns[key] = pa.concat_arrays([arr, pad])

        index_table = pa.table(index_columns)

        # enrich the metadata with computed stuffs
        metadata = dict(index_table.schema.metadata or {})
        metadata[b"origin_map"] = json.dumps(origin_map).encode("utf-8")
        metadata[b"column_lengths"] = json.dumps(column_lengths).encode("utf-8")
        metadata[b"catalog_column"] = catalog_number_col.encode("utf-8")
        metadata[b"category_offsets"] = json.dumps(category_offsets).encode("utf-8")
        index_table = index_table.replace_schema_metadata(metadata)

    # ---- Atomic write occurrence_index.parquet ----
    index_parquet = index_parquet.resolve()
    with tempfile.NamedTemporaryFile(
        dir=index_parquet.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pq.write_table(index_table, tmp_path)
        os.replace(tmp_path, index_parquet)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def build_descendant_catalog_parquet(
    ancestor_taxon_id: str,
    descendant_rank: str,
    aggregate_ranks: Sequence[str] | None = None,
) -> None:
    """Builds a descendant catalog parquet for an ancestor/rank, simply containing a list
        of all taxon ids and their sample count that match the rank that are children of the ancestor.

    Args:
        ancestor_taxon_id: Taxon id whose descendants should be cataloged.
        descendant_rank: Rank to include in the catalog (e.g., SPECIES).
        aggregate_ranks: Optional additional ranks to merge into the catalog.

    Returns:
        None. Writes a `{rank}.parquet` file under the ancestor directory.
    """
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    canonical_desc_rank = taxa_navigation.canonical_rank(descendant_rank)
    if not canonical_desc_rank:
        raise ValueError("descendant_rank is required")

    ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or "ROOT"
    output_parquet = Path(ancestor["path"]) / f"{canonical_desc_rank.lower()}.parquet"

    if canonical_desc_rank == "SUBSPECIES" and ancestor_rank != "SPECIES":
        output_parquet.unlink(missing_ok=True)
        return

    target_ranks = (
        [taxa_navigation.canonical_rank(rank) for rank in aggregate_ranks] if aggregate_ranks else [canonical_desc_rank]
    )
    target_ranks = [rank for rank in target_ranks if rank]

    descendants: list[Any] = []
    seen: set[str] = set()
    for rank in target_ranks:
        for taxon in taxa_navigation.iter_descendants_by_rank(ancestor, rank):
            if taxon["taxon_key"] in seen:
                continue
            seen.add(taxon["taxon_key"])
            descendants.append(taxon)

    _write_descendant_catalog(output_parquet, descendants)


def _sorted_unique_descendants(descendants: Sequence[Any]) -> list[Any]:
    by_key: dict[str, Any] = {}
    for taxon in descendants:
        key = str(taxon.get("taxon_key") or "")
        if not key:
            continue
        by_key.setdefault(key, taxon)
    ordered = list(by_key.values())
    ordered.sort(
        key=lambda taxon: (
            0 if taxa_navigation.taxon_id_as_int(taxon["taxon_key"]) is not None else 1,
            taxa_navigation.taxon_id_as_int(taxon["taxon_key"]) or taxon["taxon_key"],
        )
    )
    return ordered


def _write_descendant_catalog(output_parquet: Path, descendants: Sequence[Any]) -> None:
    ordered = _sorted_unique_descendants(descendants)
    if not ordered:
        output_parquet.unlink(missing_ok=True)
        return
    taxon_keys: list[str] = []
    sample_counts: list[int] = []
    for taxon in ordered:
        taxon_key = str(taxon["taxon_key"])
        taxon_keys.append(taxon_key)
        sample_counts.append(int(_infer_sample_count(taxon)))
    table = pa.table(
        {
            "taxon_key": pa.array(taxon_keys, type=pa.string()),
            "sample_count": pa.array(sample_counts, type=pa.int64()),
        }
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_parquet.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, output_parquet)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _infer_sample_count(taxon) -> int:
    """Infers a sample count for a taxon from summary stats or parquet metadata.

    Args:
        taxon: Taxon record to inspect for sample counts.

    Returns:
        Estimated sample count for the taxon.
    """
    taxon_key = str(taxon.get("taxon_key") or "")
    return _infer_sample_count_cached(taxon_key, str(taxon["path"]))


@lru_cache(maxsize=131072)
def _infer_sample_count_cached(taxon_key: str, taxon_path: str) -> int:
    stats = _load_summary_stats(taxon_path)
    if stats:
        for metrics in stats.values():
            count = metrics.get("count")
            if count is None:
                continue
            try:
                return int(count)
            except (TypeError, ValueError):
                continue
    taxon = taxa_navigation.get_taxon_by_id(taxon_key)
    if taxon is None:
        return 0
    direct = taxa_navigation.count_taxon_rows(taxon)
    if direct is not None:
        return int(direct)
    return 0


def reset_rank_build_caches() -> None:
    """Clear in-process caches used during descendant/rank materialization."""
    _infer_sample_count_cached.cache_clear()
    _cached_metric_rows_for_taxon.cache_clear()


def _descendant_rank_targets(ancestor_rank: str) -> list[str]:
    """Returns descendant ranks that sit below an ancestor rank.

    Args:
        ancestor_rank: Canonical rank string for the ancestor taxon.

    Returns:
        Ordered list of descendant ranks to consider.
    """
    rank_for_order = ancestor_rank
    if ancestor_rank in CONFIG.subspecies_equivalents:
        rank_for_order = "SUBSPECIES"
    try:
        start_idx = descendant_rank_order.index(rank_for_order)
    except ValueError:
        start_idx = -1
    return list(descendant_rank_order[start_idx + 1 :])


def build_descendant_catalogs_for_ancestor(
    ancestor_taxon_id: str,
    *,
    verbose: bool = True,
) -> None:
    """Builds descendant catalog parquets for all ranks below an ancestor.

    Args:
        ancestor_taxon_id: Taxon id whose descendant rank catalogs should be written.

    Returns:
        None. Writes `{rank}.parquet` files under the ancestor directory.
    """

    def _log(message: str) -> None:
        if verbose:
            print(message)

    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    start = time.perf_counter()
    _log(
        f"[desc-catalog] start ancestor={ancestor.get('scientific_name') or ancestor['taxon_key']} "
        f"({ancestor['taxon_key']})"
    )
    ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or "ROOT"
    targets = _descendant_rank_targets(ancestor_rank)
    descendants = taxa_navigation.iter_descendants(ancestor, include_self=False)
    by_rank: dict[str, list[Any]] = {}
    descendant_count = 0
    for taxon in descendants:
        descendant_count += 1
        rank = taxa_navigation.canonical_rank(taxon.get("rank"))
        if not rank:
            continue
        by_rank.setdefault(rank, []).append(taxon)
    _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} descendants={descendant_count} targets={','.join(targets)}")
    normalized_subspecies = tuple(taxa_navigation.canonical_rank(rank) for rank in CONFIG.subspecies_equivalents)
    normalized_subspecies = tuple(rank for rank in normalized_subspecies if rank)
    canonical_species = taxa_navigation.canonical_rank(CONFIG.species_rank) or "SPECIES"
    species_group = {CONFIG.species_rank, *CONFIG.subspecies_equivalents}
    species_group = {
        taxa_navigation.canonical_rank(rank) for rank in species_group if taxa_navigation.canonical_rank(rank)
    }
    for rank in targets:
        output_parquet = Path(ancestor["path"]) / f"{rank.lower()}.parquet"
        if rank == "SUBSPECIES":
            if ancestor_rank != "SPECIES":
                output_parquet.unlink(missing_ok=True)
                _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} rank={rank} skip (non-species ancestor)")
                continue
            if PARQUET.exists(output_parquet):
                _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} rank={rank} skip existing {output_parquet}")
                continue
            descendants_for_rank: list[Any] = []
            for alt in normalized_subspecies:
                descendants_for_rank.extend(by_rank.get(alt, []))
        elif rank == "SPECIES" and ancestor_rank not in species_group:
            if PARQUET.exists(output_parquet):
                _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} rank={rank} skip existing {output_parquet}")
                continue
            descendants_for_rank = list(by_rank.get(canonical_species, []))
            for alt in normalized_subspecies:
                descendants_for_rank.extend(by_rank.get(alt, []))
        else:
            if PARQUET.exists(output_parquet):
                _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} rank={rank} skip existing {output_parquet}")
                continue
            descendants_for_rank = list(by_rank.get(rank, []))
        _log(f"[desc-catalog] ancestor={ancestor['taxon_key']} rank={rank} rows={len(descendants_for_rank)}")
        _write_descendant_catalog(output_parquet, descendants_for_rank)
    elapsed = time.perf_counter() - start
    _log(f"[desc-catalog] done ancestor={ancestor['taxon_key']} elapsed={elapsed:.2f}s")


def _collect_metric_entries_for_taxon(
    taxon,
    fallback_samples: int,
    *,
    exclude_columns: Optional[set[str]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collects ranking entries from summary stats for a single taxon.

    Args:
        taxon: Taxon record to read summary stats from.
        fallback_samples: Sample count to use when stats omit a count field.

    Returns:
        A mapping of "variable::metric" to entry lists shaped like
        {"taxon_key": ..., "value": <metric>, "sample_count": <count>}.
    """
    rows = _cached_metric_rows_for_taxon(
        str(taxon["taxon_key"]),
        str(taxon["path"]),
    )
    if not rows:
        return {}
    entries: dict[str, list[dict[str, Any]]] = {}
    fallback_value = _normalize_fallback_samples(fallback_samples)
    for column_key, numeric_value, sample_count in rows:
        if _is_temporal_metric_column(column_key):
            continue
        if exclude_columns and column_key in exclude_columns:
            continue
        resolved_samples = sample_count if sample_count is not None else fallback_value
        if resolved_samples <= 0:
            continue
        bucket = entries.setdefault(column_key, [])
        bucket.append(
            {
                "taxon_key": taxon["taxon_key"],
                "value": numeric_value,
                "sample_count": resolved_samples,
            }
        )
    return entries


def _normalize_fallback_samples(value: Any) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return 0
    return resolved if resolved > 0 else 0


def _normalize_sample_count(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


@lru_cache(maxsize=131072)
def _cached_metric_rows_for_taxon(
    taxon_key: str,
    taxon_path: str,
) -> tuple[tuple[str, float, int | None], ...]:
    stats = _load_summary_stats(taxon_path) or {}
    categorical_stats = _load_categorical_stats(taxon_path) or {}
    combined: dict[str, dict[str, Any]] = {}
    for source in (stats, categorical_stats):
        if not source:
            continue
        for variable, metrics in source.items():
            if not metrics:
                continue
            bucket = combined.setdefault(str(variable), {})
            bucket.update(metrics)
    if not combined:
        return ()
    rows: list[tuple[str, float, int | None]] = []
    for variable, metrics in combined.items():
        sample_count = _normalize_sample_count(metrics.get("count"))
        for metric_name, raw_value in metrics.items():
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            rows.append(
                (
                    f"{variable}::{metric_name}",
                    numeric_value,
                    sample_count,
                )
            )
    return tuple(rows)


def _build_rank_index_arrays(
    column_entries: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, pa.Array], dict[str, int], set[str], int]:
    if not column_entries:
        return {}, {}, set(), 0
    struct_fields = [
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int32()),
    ]
    max_len = 0
    column_lengths: dict[str, int] = {}
    arrays: dict[str, pa.Array] = {}
    metric_names: set[str] = set()

    for column_name, entries in column_entries.items():
        if not entries:
            continue
        sorted_entries = sorted(
            entries,
            key=lambda entry: (entry["value"], entry["taxon_key"]),
        )
        if "::" in column_name:
            _, metric_name = column_name.split("::", 1)
        else:
            metric_name = column_name
        metric_names.add(metric_name)
        column_lengths[column_name] = len(sorted_entries)
        taxon_keys = pa.array([str(entry["taxon_key"]) for entry in sorted_entries], type=pa.string())
        values = pa.array([float(entry["value"]) for entry in sorted_entries], type=pa.float64())
        samples = pa.array(
            [int(entry["sample_count"]) for entry in sorted_entries],
            type=pa.int32(),
        )
        struct_array = pa.StructArray.from_arrays(
            [taxon_keys, values, samples],
            fields=struct_fields,
        )
        arrays[column_name] = struct_array
        max_len = max(max_len, len(struct_array))

    if not arrays:
        return {}, {}, set(), 0

    for column_name, arr in list(arrays.items()):
        if len(arr) < max_len:
            pad = pa.nulls(max_len - len(arr), type=arr.type)
            arrays[column_name] = pa.concat_arrays([arr, pad])

    return arrays, column_lengths, metric_names, max_len


def _write_rank_index(
    index_path: Path,
    column_entries: dict[str, list[dict[str, Any]]],
    *,
    merge_existing: bool = False,
) -> None:
    """Writes a rank index parquet from metric entry lists.

    Args:
        index_path: Output path for the rank index parquet.
        column_entries: Mapping of "variable::metric" to entry dicts with taxon_key, value, and sample_count.

    Example:
        Output file name: "<ancestor_path>/<rank>_index.parquet" (e.g. "genus_index.parquet").
        Output columns are struct arrays keyed by "variable::metric", for example.
    """
    arrays, column_lengths, metric_names, max_len = _build_rank_index_arrays(column_entries)
    if not arrays:
        if not merge_existing:
            index_path.unlink(missing_ok=True)
        return

    table: pa.Table
    metadata: dict[bytes, bytes]
    if merge_existing and index_path.exists():
        try:
            existing_schema = PARQUET.read_schema(index_path)
            existing_table = PARQUET.read_table(index_path)
        except (OSError, ValueError):
            existing_schema = None
            existing_table = None
        if existing_schema is not None and existing_table is not None:
            existing_arrays: dict[str, pa.Array] = {}
            for name in existing_schema.names:
                existing_arrays[name] = existing_table[name].combine_chunks()
            existing_lengths = _load_column_lengths(index_path)
            existing_metric_names: set[str] = set()
            for name in existing_schema.names:
                if "::" not in name:
                    continue
                _variable, metric_name = name.split("::", 1)
                existing_metric_names.add(metric_name)
            max_len = max(max_len, existing_table.num_rows)
            for key, arr in list(existing_arrays.items()):
                if len(arr) < max_len:
                    pad = pa.nulls(max_len - len(arr), type=arr.type)
                    existing_arrays[key] = pa.concat_arrays([arr, pad])
            for key, arr in list(arrays.items()):
                if len(arr) < max_len:
                    pad = pa.nulls(max_len - len(arr), type=arr.type)
                    arrays[key] = pa.concat_arrays([arr, pad])
            merged_arrays = {**existing_arrays, **arrays}
            table = pa.table(merged_arrays)
            metadata = dict(existing_schema.metadata or {})
            merged_lengths = {**existing_lengths, **column_lengths}
            merged_metrics = sorted(existing_metric_names | metric_names)
            metadata[b"column_lengths"] = json.dumps(merged_lengths).encode("utf-8")
            metadata[b"metrics"] = json.dumps(merged_metrics).encode("utf-8")
            table = table.replace_schema_metadata(metadata)
        else:
            table = pa.table(arrays)
            metadata = dict(table.schema.metadata or {})
            metadata[b"column_lengths"] = json.dumps(column_lengths).encode("utf-8")
            metadata[b"metrics"] = json.dumps(sorted(metric_names)).encode("utf-8")
            table = table.replace_schema_metadata(metadata)
    else:
        table = pa.table(arrays)
        metadata = dict(table.schema.metadata or {})
        metadata[b"column_lengths"] = json.dumps(column_lengths).encode("utf-8")
        metadata[b"metrics"] = json.dumps(sorted(metric_names)).encode("utf-8")
        table = table.replace_schema_metadata(metadata)

    index_path = index_path.resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=index_path.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, index_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return None


def _build_rank_index_parquet(
    ancestor,
    canonical_rank: str,
    *,
    verbose: bool = True,
) -> None:
    """Builds a rank index parquet for a given ancestor and descendant rank.

    Args:
        ancestor: Ancestor taxon record whose descendant catalog will be indexed.
        canonical_rank: Canonical descendant rank to build (e.g., SPECIES).

    Returns:
        None. Writes a `{rank}_index.parquet` under the ancestor directory.
    """

    def _log(message: str) -> None:
        if verbose:
            print(message)

    ancestor_path = Path(ancestor["path"])
    catalog_path = ancestor_path / f"{canonical_rank.lower()}.parquet"
    index_path = ancestor_path / f"{canonical_rank.lower()}_index.parquet"
    start = time.perf_counter()
    _log(f"[rank-index] start ancestor={ancestor.get('taxon_key')} rank={canonical_rank} catalog={catalog_path}")
    if not PARQUET.exists(catalog_path):
        _log(f"[rank-index] missing catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return None
    try:
        frame = PARQUET.read_table(
            catalog_path,
            columns=["taxon_key", "sample_count"],
        ).to_pandas()
    except (OSError, ValueError):
        _log(f"[rank-index] failed reading catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return None
    if frame.empty:
        _log(f"[rank-index] empty catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return None
    _log(f"[rank-index] catalog rows ancestor={ancestor['taxon_key']} rank={canonical_rank} rows={len(frame)}")

    existing_columns: set[str] = set()
    if index_path.exists():
        try:
            schema = PARQUET.read_schema(index_path)
            existing_columns = set(schema.names)
        except Exception:
            existing_columns = set()
    existing_metric_columns = [name for name in existing_columns if "::" in name]
    existing_temporal_metric_columns = [name for name in existing_metric_columns if _is_temporal_metric_column(name)]
    existing_non_temporal_metric_columns = [
        name for name in existing_metric_columns if not _is_temporal_metric_column(name)
    ]
    _log(
        f"[rank-index] existing columns ancestor={ancestor['taxon_key']} "
        f"rank={canonical_rank} count={len(existing_metric_columns)} "
        f"temporal={len(existing_temporal_metric_columns)}"
    )
    incremental_mode = bool(existing_non_temporal_metric_columns) and not bool(existing_temporal_metric_columns)
    if existing_temporal_metric_columns:
        _log(
            f"[rank-index] rebuild {ancestor_path} {canonical_rank} "
            f"(removing {len(existing_temporal_metric_columns)} temporal metric columns)"
        )
    elif incremental_mode:
        _log(
            f"[rank-index] completeness-check {ancestor_path} {canonical_rank} "
            f"(existing_non_temporal_cols={len(existing_non_temporal_metric_columns)})"
        )

    column_entries: dict[str, list[dict[str, Any]]] = {}
    existing_column_set = set(existing_non_temporal_metric_columns) if incremental_mode else set()
    taxa_seen = 0
    taxa_with_entries = 0
    for record in frame.itertuples(index=False):
        taxa_seen += 1
        taxon_key = getattr(record, "taxon_key", None)
        _log(
            f"[rank-index] taxon ancestor={ancestor['taxon_key']} rank={canonical_rank} "
            f"{taxa_seen}/{len(frame)} taxon_key={taxon_key}"
        )
        if taxon_key is None:
            _log(
                f"[rank-index] taxon ancestor={ancestor['taxon_key']} rank={canonical_rank} "
                f"{taxa_seen}/{len(frame)} skip (missing taxon_key)"
            )
            continue
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
        if taxon is None:
            _log(
                f"[rank-index] taxon ancestor={ancestor['taxon_key']} rank={canonical_rank} "
                f"{taxa_seen}/{len(frame)} skip (taxon lookup failed)"
            )
            continue
        fallback_samples = _normalize_fallback_samples(getattr(record, "sample_count", None))
        metric_entries = _collect_metric_entries_for_taxon(
            taxon,
            fallback_samples,
            exclude_columns=existing_column_set if incremental_mode else None,
        )
        if not metric_entries:
            _log(
                f"[rank-index] taxon ancestor={ancestor['taxon_key']} rank={canonical_rank} "
                f"{taxa_seen}/{len(frame)} no new metrics"
            )
            continue
        taxa_with_entries += 1
        metric_count = sum(len(v) for v in metric_entries.values())
        for column_name, entries in metric_entries.items():
            bucket = column_entries.setdefault(column_name, [])
            bucket.extend(entries)
        _log(
            f"[rank-index] taxon ancestor={ancestor['taxon_key']} rank={canonical_rank} "
            f"{taxa_seen}/{len(frame)} added_cols={len(metric_entries)} "
            f"added_entries={metric_count} total_new_cols={len(column_entries)}"
        )
    total_entries = sum(len(v) for v in column_entries.values())
    _log(
        f"[rank-index] collected ancestor={ancestor['taxon_key']} rank={canonical_rank} "
        f"taxa={taxa_seen} with_entries={taxa_with_entries} "
        f"new_cols={len(column_entries)} entries={total_entries}"
    )
    if column_entries:
        sample_columns = ", ".join(sorted(column_entries.keys())[:8])
        _log(f"[rank-index] new column sample ancestor={ancestor['taxon_key']} rank={canonical_rank}: {sample_columns}")

    if not column_entries:
        if incremental_mode:
            _log(f"[rank-index] up-to-date ancestor={ancestor['taxon_key']} rank={canonical_rank} (no missing columns)")
            return None
        _log(f"[rank-index] no stats entries for {ancestor_path} {canonical_rank}")
        index_path.unlink(missing_ok=True)
        return None

    _write_rank_index(
        index_path,
        column_entries,
        merge_existing=incremental_mode,
    )
    _log(
        f"[rank-index] wrote new index ancestor={ancestor['taxon_key']} "
        f"rank={canonical_rank} cols={len(column_entries)} "
        f"mode={'incremental' if incremental_mode else 'full'} "
        f"elapsed={time.perf_counter() - start:.2f}s"
    )
    return None


def build_rank_indexes_for_ancestor(
    ancestor_taxon_id: str,
    *,
    verbose: bool = True,
) -> None:
    """Builds rank index parquets for all descendant ranks of an ancestor.

    Args:
        ancestor_taxon_id: Taxon id whose descendant rank indexes should be written.

    Returns:
        None. Writes `{rank}_index.parquet` files under the ancestor directory.
    """
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or "ROOT"
    targets = _descendant_rank_targets(ancestor_rank)
    ancestor_path = Path(ancestor["path"])
    for rank in targets:
        catalog_path = ancestor_path / f"{rank.lower()}.parquet"
        index_path = ancestor_path / f"{rank.lower()}_index.parquet"
        if not PARQUET.exists(catalog_path):
            index_path.unlink(missing_ok=True)
            continue
        _build_rank_index_parquet(ancestor, rank, verbose=verbose)
    return None


def build_density_curve(
    values: Sequence[float],
    *,
    point_count: int,
    circular: bool = False,
) -> Optional[dict[str, Any]]:
    """Builds a kernel density estimate curve for numeric values.

    Args:
        values: Numeric values to estimate a density curve for.
        point_count: Number of points to sample along the curve.
        circular: If True, treats the variable as circular over [0, 360) using a
            wrapped Gaussian kernel so the density is continuous across 0°/360°.
            Intended for directional variables like aspect in degrees.

    Returns:
        A dict with sampled points, density values, and min/max/bandwidth metadata.
    """
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    count = len(array)

    if circular:
        std = float(array.std()) or 1.0
        bandwidth = 1.06 * std * (count ** (-0.2))
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 18.0  # ~360/20 fallback
        # Pad with ±360 ghost copies so the kernel sees across the 0/360 seam.
        padded = np.concatenate([np.subtract(array, 360.0), array, np.add(array, 360.0)])
        xs = np.linspace(0, 360, point_count)
        diffs = (xs[:, None] - padded[None, :]) / bandwidth
        kernel = np.exp(-0.5 * diffs**2)
        factor = 1.0 / (count * bandwidth * math.sqrt(2 * math.pi))
        densities = kernel.sum(axis=1) * factor
        return {
            "points": [float(v) for v in xs.tolist()],
            "density": [float(v) for v in densities.tolist()],
            "min": 0.0,
            "max": 360.0,
            "bandwidth": bandwidth,
        }

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
    kernel = np.exp(-0.5 * diffs**2)
    factor = 1.0 / (count * bandwidth * math.sqrt(2 * math.pi))
    densities = kernel.sum(axis=1) * factor
    return {
        "points": [float(value) for value in xs.tolist()],
        "density": [float(value) for value in densities.tolist()],
        "min": min_val,
        "max": max_val,
        "bandwidth": bandwidth,
    }


def _ancestor_contexts(taxon_path: Path) -> List[Any]:
    """Collects ancestor taxon records from a taxonomy path.

    Args:
        taxon_path: Filesystem path for the taxon directory.

    Returns:
        A list of ancestor TaxonRecords from closest parent up to the taxonomy root.
    """
    contexts: list[Any] = []
    taxonomy_root = CONFIG.taxonomy_root.resolve()
    current = taxon_path.resolve()
    while True:
        parent = current.parent
        if parent == current:
            break
        try:
            parent.relative_to(taxonomy_root)
        except ValueError:
            break
        if parent == taxonomy_root:
            break
        name = parent.name
        taxon_key = name.rsplit("_", 1)[-1].strip() if "_" in name else ""
        if not taxon_key:
            current = parent
            continue
        ancestor = taxa_navigation.get_taxon_by_id(taxon_key)
        if ancestor is not None:
            resolved = dict(ancestor)
            resolved["path"] = parent
            contexts.append(resolved)
        else:
            scientific_guess = name.rsplit("_", 1)[0].replace("_", " ").strip()
            contexts.append(
                {
                    "taxon_key": taxon_key,
                    "path": parent,
                    "scientific_name": scientific_guess or name,
                    "common_name": None,
                    "rank": "",
                }
            )
        current = parent
    return contexts


def _is_categorical_class_metric(metric_name: str) -> bool:
    return str(metric_name or "").strip().lower().startswith("class_")


def _all_requested_metrics_are_class_metrics(
    requested_metrics: set[str],
) -> bool:
    if not requested_metrics:
        return False
    return all(_is_categorical_class_metric(metric_name) for metric_name in requested_metrics)


def _context_column_rank(storage_rank: str, ancestor_rank: str) -> str:
    if storage_rank == "SUBSPECIES" and ancestor_rank != "SPECIES":
        return "SPECIES"
    return storage_rank


@lru_cache(maxsize=65536)
def _descendant_catalog_sample_counts(
    ancestor_taxon_id: str,
    rank: str,
) -> dict[int, int]:
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        return {}
    catalog_path = Path(ancestor["path"]) / f"{str(rank).lower()}.parquet"
    if not PARQUET.exists(catalog_path):
        return {}
    try:
        table = PARQUET.read_table(catalog_path, columns=["taxon_key", "sample_count"]).combine_chunks()
    except Exception:
        return {}
    if not table.num_rows:
        return {}
    taxon_keys = table.column("taxon_key").to_pylist()
    sample_counts = table.column("sample_count").to_pylist()
    mapping: dict[int, int] = {}
    for taxon_key, sample_count in zip(taxon_keys, sample_counts):
        taxon_id = taxa_navigation.taxon_id_as_int(str(taxon_key))
        if taxon_id is None:
            continue
        try:
            numeric_count = int(sample_count)
        except (TypeError, ValueError):
            numeric_count = 0
        if numeric_count <= 0:
            continue
        mapping[int(taxon_id)] = numeric_count
    return mapping


def _eligible_context_taxon_ids(
    *,
    ancestor_taxon_id: str,
    target_rank: str,
    storage_rank: str,
    include_species_like: bool,
    allowed_taxa: Optional[frozenset[int]],
    min_samples: int = 0,
    location_counts: Optional[dict[int, int]] = None,
) -> set[int]:
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        return set()
    ancestor_rank = taxa_navigation.canonical_rank(ancestor.get("rank")) or ""
    column_rank = _context_column_rank(storage_rank, ancestor_rank)
    base_counts = _descendant_catalog_sample_counts(str(ancestor_taxon_id), column_rank)
    if not base_counts:
        return set()
    target_rank_canonical = taxa_navigation.canonical_rank(target_rank) or target_rank
    # Fast-path: for species+species-like comparisons, descendant species catalogs
    # already represent the universe we compare against.
    if (
        allowed_taxa is None
        and min_samples == 0
        and location_counts is None
        and target_rank_canonical == "SPECIES"
        and include_species_like
    ):
        return set(base_counts.keys())
    species_like_ranks = {CONFIG.species_rank, *CONFIG.subspecies_equivalents}
    species_like_ranks = {
        taxa_navigation.canonical_rank(rank) for rank in species_like_ranks if taxa_navigation.canonical_rank(rank)
    }
    min_samples = max(0, int(min_samples or 0))
    eligible: set[int] = set()
    for taxon_id, base_sample_count in base_counts.items():
        if allowed_taxa is not None and taxon_id not in allowed_taxa:
            continue
        resolved_sample_count = (
            int(location_counts.get(taxon_id, 0)) if location_counts is not None else int(base_sample_count)
        )
        if min_samples and resolved_sample_count < min_samples:
            continue
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
        if taxon is None:
            continue
        taxon_rank = taxa_navigation.canonical_rank(taxon.get("rank"))
        if target_rank_canonical == "SPECIES":
            if include_species_like:
                if taxon_rank not in species_like_ranks:
                    continue
            elif taxon_rank != "SPECIES":
                continue
        elif taxon_rank != target_rank_canonical:
            continue
        eligible.add(taxon_id)
    return eligible


@lru_cache(maxsize=65536)
def _eligible_context_taxon_count_cached(
    ancestor_taxon_id: str,
    target_rank: str,
    storage_rank: str,
    include_species_like: bool,
) -> int:
    return len(
        _eligible_context_taxon_ids(
            ancestor_taxon_id=ancestor_taxon_id,
            target_rank=target_rank,
            storage_rank=storage_rank,
            include_species_like=include_species_like,
            allowed_taxa=None,
            min_samples=0,
            location_counts=None,
        )
    )


def load_relative_ranks(
    taxon_dir: Path,
    variable_id: str,
    metric_names: Optional[Sequence[str]] = None,
    location_gid: Optional[str] = None,
) -> List[dict[str, Any]]:
    """Loads relative rank positions for a taxon across ancestor contexts.

    Args:
        taxon_dir: Filesystem path of the taxon directory to rank.
        variable_id: Environmental variable id to rank on.
        metric_names: Optional metric names to include. If omitted, defaults to
            canonical relative-rank metrics.
        location_gid: Optional location GID to filter ranks to taxa that occur there.

    Returns:
        A list of relative-rank entries for each ancestor context and metric.
    """
    name = taxon_dir.name
    if "_" not in name:
        return []
    taxon_key = name.split("_")[-1]
    if not taxon_key:
        return []
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
    if taxon is None:
        return []
    requested_metrics = {str(name).strip().lower() for name in (metric_names or ()) if str(name).strip()}
    target_taxon_id = taxa_navigation.taxon_id_as_int(str(taxon_key))
    target_rank = taxa_navigation.canonical_rank(taxon["rank"]) or "SPECIES"
    storage_rank = "SUBSPECIES" if target_rank in CONFIG.subspecies_equivalents else target_rank
    contexts = _ancestor_contexts(Path(taxon["path"]))
    if not contexts:
        return []
    allowed_taxa: Optional[frozenset[int]] = None
    normalized_location = location_gid.strip() if location_gid else None
    if normalized_location:
        _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
        allowed_taxa = _location_taxa_with_optional_ancestor_rollup(scope, target)
        if not allowed_taxa:
            return []
        if target_taxon_id is not None and target_taxon_id not in allowed_taxa:
            return []
    results: list[dict[str, Any]] = []
    if allowed_taxa is None:
        skip_global_for_class_metrics = _all_requested_metrics_are_class_metrics(requested_metrics)
        table = None
        if not skip_global_for_class_metrics:
            table = _load_global_relative_rows(
                str(taxon_key),
                variable_id,
                metric_names=tuple(requested_metrics) if requested_metrics else None,
            )
        if table is None:
            positions_path = taxon_dir / "relative_ranks_positions.parquet"
            local_exists = positions_path.exists()
            if not PARQUET.exists(positions_path) and not local_exists:
                return []
            filters: list[tuple[str, str, Any]] = [("variable", "=", variable_id)]
            if requested_metrics:
                filters.append(("metric", "in", sorted(requested_metrics)))
            columns = [
                "variable",
                "metric",
                "position",
                "count",
                "sampleCount",
                "contextTaxonId",
                "contextLabel",
            ]
            try:
                table = PARQUET.read_table(
                    positions_path,
                    columns=columns,
                    filters=filters,
                )
            except TypeError:
                try:
                    if PARQUET.is_remote and local_exists:
                        table = pq.read_table(
                            positions_path,
                            columns=columns,
                            filters=filters,
                        )
                    else:
                        table = PARQUET.read_table(
                            positions_path,
                            columns=columns,
                            filters=filters,
                        )
                except (OSError, ValueError):
                    return []
            except (OSError, ValueError):
                return []
        if not table.num_rows:
            return []
        filtered = table.combine_chunks()
        if not filtered.num_rows:
            return []
        for entry in filtered.to_pylist():
            if str(entry.get("variable") or "") != str(variable_id):
                continue
            metric_name = str(entry.get("metric") or "")
            normalized_metric = metric_name.strip().lower()
            if requested_metrics:
                if normalized_metric not in requested_metrics:
                    continue
            elif metric_name not in relative_rank_metrics:
                continue
            count = entry.get("count")
            position = entry.get("position")
            if count is None or count <= 0 or position is None:
                continue
            try:
                count = int(count)
                position = int(position)
            except (TypeError, ValueError):
                continue
            if _is_categorical_class_metric(normalized_metric):
                ancestor_taxon_id = str(entry.get("contextTaxonId") or "").strip()
                if ancestor_taxon_id:
                    adjusted_total = _eligible_context_taxon_count_cached(
                        ancestor_taxon_id,
                        target_rank,
                        storage_rank,
                        (target_rank == "SPECIES"),
                    )
                    if adjusted_total > count:
                        # Missing class_* rows imply zero share and sort before positive shares.
                        position += adjusted_total - count
                        count = adjusted_total
            percentile = position / max(count - 1, 1) if count > 1 else 0.0
            ancestor_label = (entry.get("contextLabel") or "").replace("_", " ") or None
            ancestor_taxon_id = entry.get("contextTaxonId")
            results.append(
                {
                    "metric": metric_name,
                    "context": ancestor_label,
                    "label": ancestor_label,
                    "rank": target_rank,
                    "ancestorTaxonId": ancestor_taxon_id,
                    "count": int(count),
                    "position": int(position) + 1,
                    "percentile": percentile,
                    "sampleCount": (int(entry.get("sampleCount")) if entry.get("sampleCount") is not None else None),
                }
            )
        return results
    for ancestor in contexts:
        ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or ""
        column_rank = storage_rank if not (storage_rank == "SUBSPECIES" and ancestor_rank != "SPECIES") else "SPECIES"
        lookup_key = taxon_key
        index_path = Path(ancestor["path"]) / f"{column_rank.lower()}_index.parquet"
        if not PARQUET.exists(index_path):
            continue
        column_lengths = _load_column_lengths(index_path)
        lookup_taxon = taxa_navigation.get_taxon_by_id(str(lookup_key))
        if lookup_taxon is None:
            continue
        stats = _load_summary_stats(str(lookup_taxon["path"])) or {}
        categorical_stats = _load_categorical_stats(str(lookup_taxon["path"])) or {}
        metrics = dict(stats.get(variable_id, {}))
        metrics.update(categorical_stats.get(variable_id, {}))
        metric_iterable: Sequence[str]
        if requested_metrics:
            metric_iterable = sorted(requested_metrics)
        else:
            metric_iterable = relative_rank_metrics
        for metric_name in metric_iterable:
            try:
                column_name = _resolve_column_name(index_path, variable_id, metric_name)
            except ValueError:
                continue
            column_length = column_lengths.get(column_name)
            if not column_length:
                continue
            raw_value = metrics.get(metric_name)
            if raw_value is None:
                continue
            try:
                target_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(target_value):
                continue
            column = _load_struct_column(index_path, column_name, column_length)
            taxon_keys = column.field("taxonKey").to_pylist()
            values = column.field("value").to_pylist()
            samples = column.field("sampleCount").to_pylist()
            left = bisect.bisect_left(values, target_value)
            right = bisect.bisect_right(values, target_value)
            if left == right:
                continue
            block = taxon_keys[left:right]
            try:
                block_idx = block.index(str(lookup_key))
            except ValueError:
                continue
            zero_based = left + block_idx
            sample_count = samples[zero_based]
            filtered_total = 0
            filtered_position = None
            for idx, entry_taxon_key in enumerate(taxon_keys):
                taxon_id_value = taxa_navigation.taxon_id_as_int(str(entry_taxon_key))
                if taxon_id_value is None or taxon_id_value not in allowed_taxa:
                    continue
                if idx == zero_based:
                    filtered_position = filtered_total
                filtered_total += 1
            if filtered_position is None or filtered_total == 0:
                continue
            count = filtered_total
            position = filtered_position
            normalized_metric = str(metric_name).strip().lower()
            if _is_categorical_class_metric(normalized_metric):
                ancestor_taxon_id = str(ancestor.get("taxon_key") or "").strip()
                eligible_ids = _eligible_context_taxon_ids(
                    ancestor_taxon_id=ancestor_taxon_id,
                    target_rank=target_rank,
                    storage_rank=storage_rank,
                    include_species_like=(target_rank == "SPECIES"),
                    allowed_taxa=allowed_taxa,
                    min_samples=0,
                    location_counts=None,
                )
                adjusted_total = len(eligible_ids)
                if adjusted_total > count:
                    position += adjusted_total - count
                    count = adjusted_total
            percentile = position / max(count - 1, 1) if count > 1 else 0.0
            ancestor_label = (
                (ancestor["scientific_name"] or "").replace("_", " ").strip()
                or (ancestor["common_name"] or "").replace("_", " ").strip()
                or ancestor["taxon_key"]
            )
            entry = {
                "metric": metric_name,
                "context": ancestor_label,
                "label": ancestor_label,
                "rank": target_rank,
                "ancestorTaxonId": ancestor["taxon_key"],
                "count": count,
                "position": position + 1,
                "percentile": percentile,
                "sampleCount": int(sample_count) if sample_count is not None else None,
            }
            results.append(entry)
    return results


def _load_column_lengths(parquet_path: Path) -> dict[str, int]:
    """Loads per-column lengths stored in parquet metadata.

    Args:
        parquet_path: Parquet file containing padded rank index columns.

    Returns:
        A mapping of column name to its true length (before padding).
    """
    try:
        schema = PARQUET.read_schema(parquet_path)
    except (OSError, ValueError):
        return {}
    metadata = schema.metadata or {}
    raw = metadata.get(b"column_lengths")
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _resolve_column_name(parquet_path: Path, variable: str, metric: str) -> str:
    """Resolves a variable/metric column name in a parquet schema.

    Args:
        parquet_path: Parquet file to inspect.
        variable: Variable id portion of the column name.
        metric: Metric name portion of the column name.

    Returns:
        The actual column name as stored in the parquet schema.
    """
    target = f"{variable}::{metric}".strip().lower()
    try:
        schema = PARQUET.read_schema(parquet_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to read schema for {parquet_path}: {exc}") from exc
    for name in schema.names:
        if name.lower() == target:
            return name
    raise ValueError(f"Column {variable}::{metric} not found in {parquet_path}")


def _load_struct_column(parquet_path: Path, column_name: str, column_length: int) -> pa.StructArray:
    """Loads a column of per-row records and trims padding.

    Args:
        parquet_path: Parquet file containing the column.
        column_name: Name of the column to load (each row is a record like
            {taxonKey, value, sampleCount}).
        column_length: True number of rows before padding nulls were added.

    Returns:
        The column sliced to its real length, excluding padded null rows.
    """
    table = PARQUET.read_table(parquet_path, columns=[column_name])
    column = table.column(column_name).combine_chunks()
    if column_length < len(column):
        column = column.slice(0, column_length)
    return column


def list_rank_metric_options(
    ancestor_taxon_id: str,
    descendant_rank: str,
) -> List[Dict[str, Any]]:
    """List valid variable/metric ranking options for an ancestor/rank scope."""
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    storage_rank = _rank_index_storage_rank(descendant_rank)
    ancestor_path = Path(ancestor["path"])
    index_path = ancestor_path / f"{storage_rank.lower()}_index.parquet"
    if not PARQUET.exists(index_path):
        return []
    try:
        schema = PARQUET.read_schema(index_path)
    except (OSError, ValueError):
        return []

    column_lengths = _load_column_lengths(index_path)
    options: list[dict[str, Any]] = []
    for column_name in schema.names:
        if "::" not in column_name:
            continue
        count = int(column_lengths.get(column_name, 0) or 0)
        if count <= 0:
            continue
        variable, metric = column_name.split("::", 1)
        options.append(
            {
                "variable": variable,
                "metric": metric,
                "column": column_name,
                "count": count,
            }
        )
    options.sort(key=lambda entry: (entry["variable"], entry["metric"]))
    return options


def child_relative_rankings(
    ancestor_taxon_id: str,
    descendant_rank: str,
    layer: str,
    metric: str,
    limit: int = 50,
    order: str | None = "asc",
    min_samples: int | None = None,
    include_species_like: bool = False,
    return_distribution: bool = True,
    location_gid: Optional[str] = None,
    candidate_taxon_ids: Optional[Sequence[int | str]] = None,
    name_query: Optional[str] = None,
    cancel_check: CancelCheck | None = None,
) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    """Returns ranked descendant taxa by a variable/metric.

    Args:
        ancestor_taxon_id: Taxon id whose descendants will be ranked.
        descendant_rank: Rank of descendants to include (e.g., SPECIES).
        layer: Environmental variable id to rank by.
        metric: Metric name to rank by (e.g., mean, min).
        limit: Maximum number of ranked entries to return.
        order: Sort order ("asc" or "desc").
        min_samples: Minimum sample count required to be included.
        include_species_like: When rank=SPECIES, include subspecies equivalents.
        return_distribution: Whether to return raw metric values for density curves.
        location_gid: Optional location GID to filter descendants by occurrence membership.
        candidate_taxon_ids: Optional taxon ids to intersect with the descendant scope.
        name_query: Optional text query applied within the scoped leaderboard.

    Returns:
        A tuple of (ranked entries, optional metric value list for distributions).
    """
    start = time.perf_counter()
    candidate_ids: Optional[frozenset[int]] = None
    candidate_count = 0
    location_applied = False
    try:
        ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
        if ancestor is None:
            raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")

        canonical_rank = _validated_descendant_rank(descendant_rank, required=True)
        storage_rank = _rank_index_storage_rank(canonical_rank)

        order_normalized = (order or "asc").strip().lower()
        if order_normalized not in {"asc", "desc"}:
            raise ValueError("order must be either 'asc' or 'desc'")

        ancestor_path = Path(ancestor["path"])
        index_path = ancestor_path / f"{storage_rank.lower()}_index.parquet"
        if not PARQUET.exists(index_path):
            return [], None

        try:
            column_name = _resolve_column_name(index_path, layer, metric)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        column_lengths = _load_column_lengths(index_path)
        column_length = column_lengths.get(column_name)
        if not column_length:
            return [], None

        column = _load_struct_column(index_path, column_name, column_length)

        allowed_taxa: Optional[frozenset[int]] = None
        location_counts: Optional[dict[int, int]] = None
        if candidate_taxon_ids is not None:
            normalized_candidate_ids: set[int] = set()
            for raw_id in candidate_taxon_ids:
                normalized_id = taxa_navigation.taxon_id_as_int(str(raw_id))
                if normalized_id is not None:
                    normalized_candidate_ids.add(normalized_id)
            candidate_ids = frozenset(normalized_candidate_ids)
            candidate_count = len(candidate_ids)
            if not candidate_ids:
                return [], None
        location_filter = _resolve_location_filter(
            location_gid,
            include_species_rollup=(canonical_rank == "SPECIES"),
        )
        location_applied = location_filter.applied
        allowed_taxa = location_filter.allowed_taxa
        location_counts = location_filter.location_counts
        location_scope_target = location_filter.scope_target
        if location_filter.applied and not allowed_taxa:
            return [], None

        taxon_values = column.field("taxonKey").to_pylist()
        metric_values = column.field("value").to_pylist()
        sample_counts = column.field("sampleCount").to_pylist()

        min_samples = max(0, int(min_samples or 0))
        limit = max(1, int(limit or 1))

        query_matched_total = 0
        eligible: list[tuple[int, Any, float, int, Optional[int], Optional[float]]] = []
        for idx, (taxon_key, value, sample_count) in enumerate(zip(taxon_values, metric_values, sample_counts)):
            if cancel_check is not None:
                cancel_check()
            taxon_id_value = taxa_navigation.taxon_id_as_int(str(taxon_key))
            if candidate_ids is not None:
                if taxon_id_value is None or taxon_id_value not in candidate_ids:
                    continue
            if allowed_taxa is not None:
                if taxon_id_value is None or taxon_id_value not in allowed_taxa:
                    continue

            taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
            if taxon is None:
                continue
            if not _taxon_is_within_scope(
                taxon,
                ancestor_taxon_id=None,
                descendant_rank=canonical_rank,
                include_species_like=include_species_like,
            ):
                continue
            query_match_score: Optional[float] = None
            if name_query:
                query_match_score = taxa_navigation.taxon_name_match_score(taxon, name_query)
                if query_match_score is None:
                    continue
                query_matched_total += 1

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            resolved_sample_count: Optional[int]
            if location_filter.applied:
                resolved_sample_count = _location_sample_count_for_taxon(
                    taxon_id_value,
                    location_scope_target=location_scope_target,
                    location_counts=location_counts,
                )
            else:
                resolved_sample_count = int(sample_count) if sample_count is not None else None
            if min_samples and (resolved_sample_count is None or resolved_sample_count < min_samples):
                continue
            effective_sample_count = int(resolved_sample_count) if resolved_sample_count is not None else 0
            eligible.append((idx, taxon, numeric_value, effective_sample_count, taxon_id_value, query_match_score))

        filtered_total = len(eligible)
        if filtered_total == 0:
            return [], None

        metric_is_class = _is_categorical_class_metric(metric)
        class_zero_prefix = 0
        if metric_is_class:
            eligible_ids = _eligible_context_taxon_ids(
                ancestor_taxon_id=str(ancestor_taxon_id),
                target_rank=canonical_rank,
                storage_rank=storage_rank,
                include_species_like=include_species_like,
                allowed_taxa=allowed_taxa,
                min_samples=min_samples,
                location_counts=location_counts,
            )
            if candidate_ids is not None:
                eligible_ids = {taxon_id for taxon_id in eligible_ids if taxon_id in candidate_ids}
            adjusted_total = len(eligible_ids)
            if adjusted_total > filtered_total:
                class_zero_prefix = adjusted_total - filtered_total
                filtered_total = adjusted_total

        distribution_values = [entry[2] for entry in eligible] if return_distribution else None

        if order_normalized == "desc":
            eligible = list(reversed(eligible))

        denominator = max(filtered_total - 1, 1)
        results: list[dict[str, Any]] = []
        for output_idx, entry in enumerate(eligible[:limit]):
            if cancel_check is not None:
                cancel_check()
            _original_index, taxon, numeric_value, sample_count, taxon_id, query_match_score = entry
            taxon_id = taxon_id if taxon_id is not None else taxa_navigation.taxon_id_as_int(taxon["taxon_key"])
            canonical_taxon_rank = taxa_navigation.canonical_rank(taxon["rank"])
            if order_normalized == "desc":
                rank_index = filtered_total - 1 - output_idx
            else:
                rank_index = output_idx + class_zero_prefix
            percentile = rank_index / denominator if filtered_total > 1 else 0.0
            common_names = taxa_navigation.extract_common_names_for_language(
                taxon,
                language=CONFIG.common_name_language,
            )
            common_name = common_names[0] if common_names else None
            media_record = taxa_navigation.resolve_taxon_media(taxon["taxon_key"])
            preferred_image = taxa_navigation.preferred_image_payload(taxon)
            record = {
                "taxonId": taxon_id if taxon_id is not None else taxon["taxon_key"],
                "taxon_id": taxon_id if taxon_id is not None else taxon["taxon_key"],
                "taxon_key": taxon["taxon_key"],
                "scientificName": taxon["scientific_name"],
                "scientific_name": taxon["scientific_name"],
                "commonName": common_name,
                "common_name": common_name,
                "common_names": common_names,
                "rank": canonical_taxon_rank,
                "value": numeric_value,
                "sort_value": numeric_value,
                "sampleCount": sample_count,
                "sample_count": sample_count,
                "count": filtered_total,
                "position": rank_index + 1,
                "percentile": percentile,
                "metric": metric,
                "sort_metric": metric,
                "variable": layer,
                "sort_variable": layer,
            }
            if query_match_score is not None:
                record["match_score"] = query_match_score
                record["matched_count"] = query_matched_total
            if preferred_image:
                record.update(preferred_image)
            elif media_record:
                record["image_url"] = media_record.get("url")
                record["image_license"] = media_record.get("license")
                record["image_creator"] = media_record.get("creator")
                record["image_rights_holder"] = media_record.get("rightsHolder")
                record["image_references"] = media_record.get("references")
            results.append(record)

        return results, distribution_values
    finally:
        LOGGER.info(
            "[child-relative-rankings] elapsed=%.3fs ancestor=%r rank=%r variable=%r metric=%r order=%r "
            "candidate_count=%s location=%r location_applied=%s min_samples=%s include_species_like=%s",
            time.perf_counter() - start,
            ancestor_taxon_id,
            descendant_rank,
            layer,
            metric,
            order,
            candidate_count,
            location_gid,
            location_applied,
            min_samples,
            include_species_like,
        )


def _taxon_is_within_scope(
    taxon: Any,
    ancestor_taxon_id: Optional[str],
    descendant_rank: Optional[str],
    include_species_like: bool,
) -> bool:
    ancestor_key = str(ancestor_taxon_id or "").strip()
    target_rank = taxa_navigation.canonical_rank(descendant_rank)
    taxon_rank = taxa_navigation.canonical_rank(taxon.get("rank"))

    if target_rank:
        if target_rank == "SPECIES" and include_species_like:
            if taxon_rank not in ({"SPECIES"} | set(CONFIG.subspecies_equivalents)):
                return False
        elif taxon_rank != target_rank:
            return False

    if not ancestor_key:
        return True
    current = taxa_navigation.get_parent_taxon(cast(Any, taxon))
    while current is not None:
        if str(current.get("taxon_key") or "") == ancestor_key:
            return True
        current = taxa_navigation.get_parent_taxon(cast(Any, current))
    return False


def _taxon_metric_record(
    taxon: dict[str, Any],
    variable_id: str,
    metric_name: str,
) -> tuple[Optional[float], Optional[int]]:
    taxon_path = str(taxon.get("path") or "")
    if not taxon_path:
        return None, None
    stats = _load_summary_stats(taxon_path) or {}
    categorical_stats = _load_categorical_stats(taxon_path) or {}
    metrics = dict(stats.get(variable_id, {}))
    metrics.update(categorical_stats.get(variable_id, {}))
    raw_value = metrics.get(metric_name)
    if raw_value is None:
        return None, None
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(numeric_value):
        return None, None
    sample_count = _normalize_sample_count(metrics.get("count"))
    return numeric_value, sample_count


class LocationFilterResult(NamedTuple):
    """Resolved location filter state for ranking queries.

    Attributes:
        applied: Whether a location filter was requested and resolved.
        allowed_taxa: Taxon ids allowed by the location filter, or None when no
            location filter applies.
        location_counts: Optional per-taxon sample counts scoped to the
            requested location.
        scope_target: Resolved location scope and target used for per-taxon
            count fallback when the aggregated lookup is unavailable.
    """

    applied: bool
    allowed_taxa: Optional[frozenset[int]]
    location_counts: Optional[dict[int, int]]
    scope_target: Optional[tuple[str, str]]


def _location_taxa_with_optional_ancestor_rollup(scope: str, target: str) -> frozenset[int]:
    try:
        return gis_lookup.location_taxa_for(scope, target, include_ancestor_rollup=True)
    except TypeError as exc:
        if "include_ancestor_rollup" not in str(exc):
            raise
        return gis_lookup.location_taxa_for(scope, target)


def _location_counts_with_optional_ancestor_rollup(
    scope: str,
    target: str,
    *,
    include_species_rollup: bool,
) -> Optional[dict[int, int]]:
    try:
        return gis_lookup.location_taxon_counts(
            scope,
            target,
            include_species_rollup=include_species_rollup,
            include_ancestor_rollup=True,
        )
    except TypeError as exc:
        if "include_ancestor_rollup" not in str(exc):
            raise
        return gis_lookup.location_taxon_counts(
            scope,
            target,
            include_species_rollup=include_species_rollup,
        )


def _location_sample_count_for_taxon(
    taxon_id: int | None,
    *,
    location_scope_target: Optional[tuple[str, str]],
    location_counts: Optional[dict[int, int]],
) -> int | None:
    if taxon_id is None:
        return None

    if location_counts is not None:
        count = location_counts.get(taxon_id)
        if count is not None:
            return int(count)

    if location_scope_target is not None:
        scope, target = location_scope_target
        count = gis_lookup.location_counts_for_taxon(int(taxon_id)).get((scope, target))
        return _normalize_sample_count(count)

    return None


def _matched_taxon_sample_count(
    match: dict[str, Any],
    *,
    location_scope_target: Optional[tuple[str, str]],
    location_counts: Optional[dict[int, int]],
) -> int | None:
    taxon = match.get("taxon")
    taxon_id = match.get("taxon_id")

    location_sample_count = _location_sample_count_for_taxon(
        taxon_id,
        location_scope_target=location_scope_target,
        location_counts=location_counts,
    )
    if location_sample_count is not None:
        return location_sample_count
    if location_scope_target is not None:
        return None

    if not taxon:
        return None
    return _infer_sample_count(taxon)


def _taxon_matches_direct_location_membership(
    taxon: Any,
    taxon_id: int | None,
    direct_allowed_taxa: frozenset[int],
    *,
    cancel_check: CancelCheck | None = None,
) -> bool:
    if taxon_id is not None and taxon_id in direct_allowed_taxa:
        return True
    try:
        descendants = taxa_navigation.iter_descendants(cast(Any, taxon), include_self=False)
    except FileNotFoundError:
        return False
    for descendant in descendants:
        if cancel_check is not None:
            cancel_check()
        descendant_id = taxa_navigation.taxon_id_as_int(descendant.get("taxon_key"))
        if descendant_id is not None and descendant_id in direct_allowed_taxa:
            return True
    return False


def _filter_matched_taxa(
    matched_taxa: list[dict[str, Any]],
    *,
    min_samples: int,
    include_species_like: bool,
    within_taxon_id: Optional[str],
    descendant_rank: Optional[str],
    location_gid: Optional[str],
    use_direct_location_membership_fast_path: bool = False,
    cancel_check: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    normalized_location = (location_gid or "").strip()
    needs_sample_count = min_samples > 0
    location_scope_target: Optional[tuple[str, str]] = None
    allowed_taxa: Optional[frozenset[int]] = None
    location_counts: Optional[dict[int, int]] = None
    direct_location_taxa: Optional[frozenset[int]] = None
    if normalized_location and not needs_sample_count and use_direct_location_membership_fast_path:
        _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
        direct_location_taxa = gis_lookup.location_taxa_for(scope, target)
        if not direct_location_taxa:
            return []
    else:
        location_filter = _resolve_location_filter(
            location_gid,
            include_species_rollup=(
                descendant_rank is None or taxa_navigation.canonical_rank(descendant_rank) == "SPECIES"
            ),
        )
        allowed_taxa = location_filter.allowed_taxa
        location_counts = location_filter.location_counts
        if location_filter.applied and not allowed_taxa:
            return []
        if normalized_location:
            location_scope_target = location_filter.scope_target

    filtered: list[dict[str, Any]] = []
    for match in matched_taxa:
        if cancel_check is not None:
            cancel_check()
        taxon = match.get("taxon")
        if not taxon:
            continue
        if not _taxon_is_within_scope(
            taxon,
            ancestor_taxon_id=within_taxon_id,
            descendant_rank=descendant_rank,
            include_species_like=include_species_like,
        ):
            continue
        taxon_id = match.get("taxon_id")
        if direct_location_taxa is not None:
            if not _taxon_matches_direct_location_membership(
                taxon,
                taxon_id,
                direct_location_taxa,
                cancel_check=cancel_check,
            ):
                continue
        elif allowed_taxa is not None and taxon_id not in allowed_taxa:
            continue
        sample_count: int | None = None
        if needs_sample_count:
            sample_count = _matched_taxon_sample_count(
                match,
                location_scope_target=location_scope_target,
                location_counts=location_counts,
            )
        if min_samples > 0:
            if sample_count is None or sample_count < min_samples:
                continue
        if sample_count is not None:
            enriched_match = dict(match)
            enriched_match["sample_count"] = sample_count
            filtered.append(enriched_match)
        else:
            filtered.append(match)
    return filtered


def _resolve_location_filter(
    location_gid: Optional[str],
    *,
    include_species_rollup: bool,
) -> LocationFilterResult:
    """Resolve location-scoped taxa membership and optional sample counts.

    Args:
        location_gid: Location identifier used to scope taxa membership. When
            empty, no location filtering is applied.
        include_species_rollup: Whether location counts should roll subspecies-
            like occurrences up to species totals.

    Returns:
        A named result describing whether filtering was applied, which taxon ids
        are allowed for the location, and any per-taxon location sample counts.
    """
    normalized_location = (location_gid or "").strip()
    if not normalized_location:
        return LocationFilterResult(False, None, None, None)

    _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
    scope_target = (scope, target)
    allowed_taxa = _location_taxa_with_optional_ancestor_rollup(scope, target)
    if not allowed_taxa:
        return LocationFilterResult(True, frozenset(), None, scope_target)

    try:
        location_counts = _location_counts_with_optional_ancestor_rollup(
            scope,
            target,
            include_species_rollup=include_species_rollup,
        )
    except Exception:
        location_counts = None

    if location_counts:
        allowed_taxa = frozenset(set(allowed_taxa) | set(location_counts))
        location_counts = {taxon_id: count for taxon_id, count in location_counts.items() if taxon_id in allowed_taxa}
        if not location_counts:
            location_counts = None

    return LocationFilterResult(True, allowed_taxa, location_counts, scope_target)


def _query_taxa_response(
    *,
    total: int,
    results: list[dict[str, Any]],
    matched_total: int,
    eligible_total: int,
    empty_reason: str | None,
) -> dict[str, Any]:
    """Build the internal taxa query payload shape.

    Args:
        total: Total number of matches or ranked rows before pagination.
        results: Normalized result rows for the current page.

    Returns:
        The normalized query payload consumed by the API layer.
    """
    return {
        "total": total,
        "matched_total": matched_total,
        "eligible_total": eligible_total,
        "empty_reason": empty_reason,
        "results": results,
    }


def _query_empty_reason(
    *,
    has_query: bool,
    has_sort: bool,
    matched_total: int,
    eligible_total: int,
    total: int,
) -> str | None:
    if total > 0:
        return None
    if not has_query and not has_sort:
        return "no_query"
    if has_query and matched_total == 0:
        return "no_text_matches"
    if has_sort:
        return "ranking_ineligible"
    if has_query and eligible_total == 0:
        return "filtered_out"
    return "filtered_out"


def _query_result_from_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "taxon": match.get("taxon"),
        "taxon_id": match.get("taxon_id"),
        "match_score": match.get("match_score"),
        "sample_count": match.get("sample_count"),
        "sort_value": None,
        "sort_variable": None,
        "sort_metric": None,
        "position": None,
        "percentile": None,
    }


def _normalize_ranked_query_row(row: dict[str, Any]) -> dict[str, Any]:
    taxon_id = row.get("taxon_id")
    if taxon_id is None:
        taxon_id = row.get("taxonId")
    normalized_taxon_id = taxa_navigation.taxon_id_as_int(str(taxon_id)) if taxon_id is not None else None
    return {
        "taxon_id": normalized_taxon_id if normalized_taxon_id is not None else taxon_id,
        "sample_count": row.get("sample_count") if row.get("sample_count") is not None else row.get("sampleCount"),
        "sort_value": row.get("sort_value") if row.get("sort_value") is not None else row.get("value"),
        "sort_variable": row.get("sort_variable") if row.get("sort_variable") is not None else row.get("variable"),
        "sort_metric": row.get("sort_metric") if row.get("sort_metric") is not None else row.get("metric"),
        "position": row.get("position"),
        "percentile": row.get("percentile"),
    }


def _query_result_from_ranked_row(
    row: dict[str, Any],
    matched_rows_by_taxon: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    normalized_row = _normalize_ranked_query_row(row)
    normalized_taxon_id = normalized_row.get("taxon_id")
    match_row = matched_rows_by_taxon.get(normalized_taxon_id or -1, {})
    return {
        "taxon": match_row.get("taxon") or row.get("taxon"),
        "taxon_id": normalized_taxon_id,
        "match_score": match_row.get("match_score") if match_row else row.get("match_score"),
        "sample_count": normalized_row.get("sample_count"),
        "sort_value": normalized_row.get("sort_value"),
        "sort_variable": normalized_row.get("sort_variable"),
        "sort_metric": normalized_row.get("sort_metric"),
        "position": normalized_row.get("position"),
        "percentile": normalized_row.get("percentile"),
    }


def _rank_candidate_taxa(
    matched_taxa: list[dict[str, Any]],
    variable_id: str,
    metric_name: str,
    *,
    order: str,
    min_samples: int,
    include_species_like: bool,
    within_taxon_id: Optional[str],
    descendant_rank: Optional[str],
    location_gid: Optional[str],
    cancel_check: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    location_filter = _resolve_location_filter(
        location_gid,
        include_species_rollup=(
            descendant_rank is None or taxa_navigation.canonical_rank(descendant_rank) == "SPECIES"
        ),
    )
    allowed_taxa = location_filter.allowed_taxa
    location_counts = location_filter.location_counts
    location_scope_target = location_filter.scope_target
    if location_filter.applied and not allowed_taxa:
        return []

    eligible: list[tuple[dict[str, Any], float, int]] = []
    for match in matched_taxa:
        if cancel_check is not None:
            cancel_check()
        taxon = match.get("taxon")
        if not taxon:
            continue
        if not _taxon_is_within_scope(
            taxon,
            ancestor_taxon_id=within_taxon_id,
            descendant_rank=descendant_rank,
            include_species_like=include_species_like,
        ):
            continue
        taxon_id = match.get("taxon_id")
        if allowed_taxa is not None and taxon_id not in allowed_taxa:
            continue
        numeric_value, sample_count = _taxon_metric_record(taxon, variable_id, metric_name)
        if numeric_value is None:
            continue
        if location_filter.applied:
            resolved_sample_count = _location_sample_count_for_taxon(
                taxon_id,
                location_scope_target=location_scope_target,
                location_counts=location_counts,
            )
        else:
            resolved_sample_count = sample_count
        if min_samples and (resolved_sample_count is None or resolved_sample_count < min_samples):
            continue
        effective_sample_count = int(resolved_sample_count) if resolved_sample_count is not None else 0
        eligible.append((match, numeric_value, effective_sample_count))

    reverse = order == "desc"
    if reverse:
        eligible.sort(key=lambda item: (-item[1], -float(item[0].get("match_score", 0.0))))
    else:
        eligible.sort(key=lambda item: (item[1], -float(item[0].get("match_score", 0.0))))
    filtered_total = len(eligible)
    if filtered_total == 0:
        return []

    denominator = max(filtered_total - 1, 1)
    results: list[dict[str, Any]] = []
    for position, (match, numeric_value, sample_count) in enumerate(eligible, start=1):
        if cancel_check is not None:
            cancel_check()
        taxon = match["taxon"]
        taxon_id = match.get("taxon_id")
        common_names = taxa_navigation.extract_common_names_for_language(
            taxon,
            language=CONFIG.common_name_language,
        )
        common_name = common_names[0] if common_names else None
        if reverse:
            rank_index = filtered_total - position
        else:
            rank_index = position - 1
        percentile = rank_index / denominator if filtered_total > 1 else 0.0
        record = {
            "taxon_id": taxon_id if taxon_id is not None else taxon.get("taxon_key"),
            "taxon_key": taxon.get("taxon_key"),
            "scientific_name": taxon.get("scientific_name"),
            "common_name": common_name,
            "common_names": common_names,
            "rank": taxa_navigation.canonical_rank(taxon.get("rank")),
            "sort_value": numeric_value,
            "sample_count": sample_count,
            "count": filtered_total,
            "position": rank_index + 1,
            "percentile": percentile,
            "sort_metric": metric_name,
            "sort_variable": variable_id,
        }
        preferred_image = taxa_navigation.preferred_image_payload(taxon)
        media_record = (
            None if preferred_image else taxa_navigation.resolve_taxon_media(str(taxon.get("taxon_key") or ""))
        )
        if preferred_image:
            record.update(preferred_image)
        elif media_record:
            record["image_url"] = media_record.get("url")
            record["image_license"] = media_record.get("license")
            record["image_creator"] = media_record.get("creator")
            record["image_rights_holder"] = media_record.get("rightsHolder")
            record["image_references"] = media_record.get("references")
        results.append(record)
    return results


def _count_scoped_query_matches(
    ancestor_taxon_id: str,
    descendant_rank: str,
    name_query: str,
    *,
    location_gid: str | None = None,
    include_species_like: bool,
    cancel_check: CancelCheck | None = None,
) -> int:
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        return 0

    canonical_rank = _validated_descendant_rank(descendant_rank, required=True)
    allowed_taxa: Optional[frozenset[int]] = None
    normalized_location = (location_gid or "").strip()
    if normalized_location:
        _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
        allowed_taxa = _location_taxa_with_optional_ancestor_rollup(scope, target)
        if not allowed_taxa:
            return 0
    matched_total = 0
    for taxon in taxa_navigation.iter_descendants(ancestor, include_self=False):
        if cancel_check is not None:
            cancel_check()
        if not _taxon_is_within_scope(
            taxon,
            ancestor_taxon_id=str(ancestor_taxon_id),
            descendant_rank=canonical_rank,
            include_species_like=include_species_like,
        ):
            continue
        if allowed_taxa is not None:
            taxon_id = taxa_navigation.taxon_id_as_int(str(taxon.get("taxon_key")))
            if taxon_id is None or taxon_id not in allowed_taxa:
                continue
        if taxa_navigation.taxon_name_match_score(taxon, name_query) is None:
            continue
        matched_total += 1
    return matched_total


def _search_scoped_taxa_by_name(
    name_query: str,
    *,
    within_taxon_id: str,
    descendant_rank: str | None,
    include_species_like: bool,
    cancel_check: CancelCheck | None = None,
) -> list[dict[str, Any]]:
    ancestor = taxa_navigation.get_taxon_by_id(str(within_taxon_id))
    if ancestor is None:
        return []

    matches: list[dict[str, Any]] = []
    for taxon in taxa_navigation.iter_descendants(ancestor, include_self=False):
        if cancel_check is not None:
            cancel_check()
        if not _taxon_is_within_scope(
            taxon,
            ancestor_taxon_id=str(within_taxon_id),
            descendant_rank=descendant_rank,
            include_species_like=include_species_like,
        ):
            continue
        match_score = taxa_navigation.taxon_name_match_score(taxon, name_query)
        if match_score is None:
            continue
        taxon_id = taxa_navigation.taxon_id_as_int(taxon.get("taxon_key"))
        if taxon_id is None:
            continue
        matches.append(
            {
                "taxon_id": taxon_id,
                "taxon": taxon,
                "match_score": float(match_score),
            }
        )

    matches.sort(
        key=lambda row: (
            -float(row.get("match_score", 0.0)),
            int(row.get("taxon_id") or 0),
            str((row.get("taxon") or {}).get("scientific_name") or ""),
        )
    )
    return matches


def query_taxa(
    q: Optional[str] = None,
    within_taxon_id: Optional[str] = None,
    descendant_rank: Optional[str] = None,
    sort_variable: Optional[str] = None,
    sort_metric: Optional[str] = None,
    sort_order: str | None = "asc",
    limit: int = 12,
    offset: int = 0,
    min_samples: int | None = None,
    include_species_like: bool = False,
    location_gid: Optional[str] = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Query taxa for page results using text matching and optional ranking."""
    normalized_query = (q or "").strip()
    normalized_within_taxon_id = (within_taxon_id or "").strip() or None
    normalized_rank = (descendant_rank or "").strip() or None
    normalized_sort_variable = (sort_variable or "").strip() or None
    normalized_sort_metric = (sort_metric or "").strip() or None
    normalized_location = (location_gid or "").strip() or None
    normalized_order = (sort_order or "asc").strip().lower() or "asc"
    limit = max(1, int(limit or 1))
    offset = max(0, int(offset or 0))

    if normalized_order not in {"asc", "desc"}:
        raise ValueError("order must be either 'asc' or 'desc'")

    has_sort = bool(normalized_sort_variable or normalized_sort_metric)
    if bool(normalized_sort_variable) != bool(normalized_sort_metric):
        raise ValueError("sort_variable and sort_metric must be provided together")
    if normalized_rank is not None:
        normalized_rank = _validated_descendant_rank(normalized_rank)
    if not normalized_query and not has_sort:
        return _query_taxa_response(
            total=0,
            matched_total=0,
            eligible_total=0,
            empty_reason="no_query",
            results=[],
        )
    if has_sort and not normalized_query and (not normalized_within_taxon_id or not normalized_rank):
        raise ValueError("within_taxon and descendant_rank are required for ranked taxon queries")

    matched_taxa: list[dict[str, Any]] = []
    candidate_taxon_ids: Optional[list[int]] = None
    scoped_ranked_text_query = bool(has_sort and normalized_query and normalized_within_taxon_id and normalized_rank)
    if normalized_query and not scoped_ranked_text_query:
        if normalized_within_taxon_id:
            matched_taxa = _search_scoped_taxa_by_name(
                normalized_query,
                within_taxon_id=normalized_within_taxon_id,
                descendant_rank=normalized_rank,
                include_species_like=include_species_like,
                cancel_check=cancel_check,
            )
            candidate_taxon_ids = [row["taxon_id"] for row in matched_taxa if row.get("taxon_id") is not None]
        else:
            # Plain text queries intentionally bound the search window so common
            # names do not force a full fuzzy scan of the global name index.
            # We over-fetch relative to the requested page to leave room for
            # downstream scope/location filtering while keeping latency bounded.
            search_limit = max((offset + limit) * 25, 250)
            search_kwargs: dict[str, Any] = {"limit": search_limit}
            if cancel_check is not None:
                search_kwargs["cancel_check"] = cancel_check
            search_rows = taxa_navigation.search_taxa_by_name(normalized_query, **search_kwargs)
            candidate_taxon_ids = []
            for taxon, score in search_rows:
                if cancel_check is not None:
                    cancel_check()
                taxon_id = taxa_navigation.taxon_id_as_int(taxon.get("taxon_key"))
                if taxon_id is None:
                    continue
                matched_taxa.append(
                    {
                        "taxon_id": taxon_id,
                        "taxon": taxon,
                        "match_score": float(score),
                    }
                )
                candidate_taxon_ids.append(taxon_id)

    matched_total = len(matched_taxa)
    matched_rows_by_taxon = {row["taxon_id"]: row for row in matched_taxa if row.get("taxon_id") is not None}

    if scoped_ranked_text_query:
        ranked_rows, _distribution = child_relative_rankings(
            cast(str, normalized_within_taxon_id),
            cast(str, normalized_rank),
            normalized_sort_variable or "",
            normalized_sort_metric or "",
            limit=limit + offset,
            order=normalized_order,
            min_samples=min_samples,
            include_species_like=include_species_like,
            return_distribution=False,
            location_gid=normalized_location,
            name_query=normalized_query,
            cancel_check=cancel_check,
        )
        ranked_total = ranked_rows[0]["count"] if ranked_rows else 0
        matched_total = (
            int(ranked_rows[0].get("matched_count", ranked_total))
            if ranked_rows
            else _count_scoped_query_matches(
                cast(str, normalized_within_taxon_id),
                cast(str, normalized_rank),
                normalized_query,
                location_gid=normalized_location,
                include_species_like=include_species_like,
                cancel_check=cancel_check,
            )
        )
        return _query_taxa_response(
            total=ranked_total,
            matched_total=matched_total,
            eligible_total=ranked_total,
            empty_reason=_query_empty_reason(
                has_query=True,
                has_sort=True,
                matched_total=matched_total,
                eligible_total=ranked_total,
                total=ranked_total,
            ),
            results=[
                _query_result_from_ranked_row(row, matched_rows_by_taxon)
                for row in ranked_rows[offset : offset + limit]
            ],
        )

    filtered_matches = _filter_matched_taxa(
        matched_taxa,
        min_samples=max(0, int(min_samples or 0)),
        include_species_like=include_species_like,
        within_taxon_id=normalized_within_taxon_id,
        descendant_rank=normalized_rank,
        location_gid=normalized_location,
        use_direct_location_membership_fast_path=not has_sort,
        cancel_check=cancel_check,
    )
    eligible_total = len(filtered_matches)

    if not has_sort:
        return _query_taxa_response(
            total=len(filtered_matches),
            matched_total=matched_total,
            eligible_total=eligible_total,
            empty_reason=_query_empty_reason(
                has_query=bool(normalized_query),
                has_sort=False,
                matched_total=matched_total,
                eligible_total=eligible_total,
                total=len(filtered_matches),
            ),
            results=[_query_result_from_match(row) for row in filtered_matches[offset : offset + limit]],
        )
    if normalized_query and (not normalized_within_taxon_id or not normalized_rank):
        ranked_rows = _rank_candidate_taxa(
            filtered_matches,
            normalized_sort_variable or "",
            normalized_sort_metric or "",
            order=normalized_order,
            min_samples=max(0, int(min_samples or 0)),
            include_species_like=include_species_like,
            within_taxon_id=normalized_within_taxon_id,
            descendant_rank=normalized_rank,
            location_gid=normalized_location,
            cancel_check=cancel_check,
        )
        ranked_total = len(ranked_rows)
        return _query_taxa_response(
            total=ranked_total,
            matched_total=matched_total,
            eligible_total=ranked_total,
            empty_reason=_query_empty_reason(
                has_query=True,
                has_sort=True,
                matched_total=matched_total,
                eligible_total=ranked_total,
                total=ranked_total,
            ),
            results=[
                _query_result_from_ranked_row(row, matched_rows_by_taxon)
                for row in ranked_rows[offset : offset + limit]
            ],
        )

    ranked_rows, _distribution = child_relative_rankings(
        normalized_within_taxon_id or "",
        normalized_rank or "",
        normalized_sort_variable or "",
        normalized_sort_metric or "",
        limit=limit + offset,
        order=normalized_order,
        min_samples=min_samples,
        include_species_like=include_species_like,
        return_distribution=False,
        location_gid=normalized_location,
        candidate_taxon_ids=candidate_taxon_ids,
        cancel_check=cancel_check,
    )
    ranked_total = ranked_rows[0]["count"] if ranked_rows else 0

    return _query_taxa_response(
        total=ranked_total,
        matched_total=matched_total,
        eligible_total=ranked_total,
        empty_reason=_query_empty_reason(
            has_query=bool(normalized_query),
            has_sort=True,
            matched_total=matched_total,
            eligible_total=ranked_total,
            total=ranked_total,
        ),
        results=[
            _query_result_from_ranked_row(row, matched_rows_by_taxon) for row in ranked_rows[offset : offset + limit]
        ],
    )
