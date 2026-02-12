'''
Index creation and ranking/query helpers for occurrence and ranking parquets.
'''

from __future__ import annotations

from pathlib import Path
import bisect
from typing import Any, Sequence, Optional, List, Tuple, Dict
import json
import math
import os
import tempfile

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from util import gis_lookup, taxa_navigation
from util.config import load_config
from util.summary_stats import _load_summary_stats, _load_categorical_stats

CONFIG = load_config("global")

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

relative_rank_metrics = (
        "min",
        "mean",
        "max",
        "std",
        "1-99 range",
    )


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
    allow_missing_parent = True
    layer_catalog = gis_lookup.load_layer_metadata()
    categorical_layers = {
        layer_id: (layer.get("value_type") or "").lower() == "categorical"
        for layer_id, layer in layer_catalog.items()
    }
    category_offsets: dict[str, dict[str, dict[str, int | float]]] = {}
    data_parquet_exists = data_parquet.exists()
    if not data_parquet_exists and not allow_missing_parent:
        raise FileNotFoundError(data_parquet)

    parent_dir = data_parquet.parent

    if data_parquet_exists:
        table = pq.read_table(data_parquet)
        if catalog_number_col not in table.schema.names:
            raise ValueError(f"{catalog_number_col} not found in {data_parquet}")
    else:
        table = pa.table({catalog_number_col: pa.array([], type=pa.string())})

    # Ensure parent is sorted by catalogNumber
    if len(table) > 1:
        cat_col = table[catalog_number_col]
        is_sorted = pc.all(
            pc.less_equal(
                cat_col.slice(0, len(cat_col) - 1),
                cat_col.slice(1),
            )
        ).as_py()

        if not is_sorted and data_parquet_exists:
            sort_indices = pc.sort_indices(cat_col)
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
    datasets = [
        {
            "origin_id": 0,
            "table": table,
            "catalog_numbers": table[catalog_number_col].combine_chunks(),
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

        child_table = pq.read_table(child_parquet)
        if catalog_number_col not in child_table.schema.names:
            continue

        datasets.append(
            {
                "origin_id": origin_id_counter,
                "table": child_table,
                "catalog_numbers": child_table[catalog_number_col].combine_chunks(),
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

    # ---- Build index columns ----
    index_columns: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    max_len = 0

    for layer_id, layer in layer_catalog.items():
        if not layer_id:
            continue

        combined_values = []
        combined_catalogs = []
        combined_origins = []

        for dataset in datasets:
            table = dataset["table"]
            if layer_id not in table.schema.names:
                continue

            layer_col = table[layer_id]

            # after grabbing the column with the layer we care about, mask out nulls so we don't index them
            mask = pc.invert(pc.is_null(layer_col))
            if pa.types.is_floating(layer_col.type):
                mask = pc.and_(mask, pc.invert(pc.is_nan(layer_col)))
            try:
                obscured_col = table["obscured"]
                mask = pc.and_(mask, pc.equal(obscured_col, "No"))
            except KeyError:
                pass
            try:
                coord_col = table["coordinateUncertaintyInMeters"]
                mask = pc.and_(mask, pc.less_equal(coord_col, 500))
            except KeyError:
                pass

            # filter the catalogs the same way so we have indexes of equal length as the original cols
            filtered_values = pc.filter(layer_col, mask).combine_chunks()
            filtered_catalogs = pc.filter(
                dataset["catalog_numbers"],
                mask,
            ).combine_chunks()

            if len(filtered_values) == 0:
                continue

            # Normalize types across datasets to avoid concat type mismatches
            target_type = pa.int64() if categorical_layers.get(layer_id) else pa.float64()
            try:
                filtered_values = pc.cast(filtered_values, target_type)
            except pa.ArrowInvalid:
                filtered_values = pc.cast(filtered_values, pa.float64())

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

        values = pa.concat_arrays(combined_values)
        catalogs = pa.concat_arrays(combined_catalogs)
        origins = pa.concat_arrays(combined_origins)

        # sort the indices by their values, and take the respective catalogs and origins after this sorting
        sort_indices = pc.sort_indices(values)
        sorted_catalogs = pc.take(catalogs, sort_indices)
        sorted_origins = pc.take(origins, sort_indices)
        sorted_values = pc.take(values, sort_indices)

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

        if categorical_layers.get(layer_id):
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
                    offsets[str(current_value)] = {
                        "value": current_value if current_value is not None else None,
                        "start": start_idx,
                        "count": idx - start_idx,
                    }
                    current_value = value
                    start_idx = idx
            if current_value is not None:
                offsets[str(current_value)] = {
                    "value": current_value if current_value is not None else None,
                    "start": start_idx,
                    "count": len(py_values) - start_idx,
                }
            if offsets:
                category_offsets[layer_id] = offsets

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
        [taxa_navigation.canonical_rank(rank) for rank in aggregate_ranks]
        if aggregate_ranks
        else [canonical_desc_rank]
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

    if not descendants:
        output_parquet.unlink(missing_ok=True)
        return

    descendants.sort(
        key=lambda taxon: (
            0 if taxa_navigation.taxon_id_as_int(taxon["taxon_key"]) is not None else 1,
            taxa_navigation.taxon_id_as_int(taxon["taxon_key"]) or taxon["taxon_key"],
        )
    )

    records: list[dict[str, Any]] = []
    for taxon in descendants:
        sample_count = _infer_sample_count(taxon)
        records.append(
            {
                "taxon_key": taxon["taxon_key"],
                "sample_count": int(sample_count),
            }
        )

    frame = pd.DataFrame(records, columns=["taxon_key", "sample_count"])
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_parquet.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        frame.to_parquet(tmp_path, index=False)
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
    stats = _load_summary_stats(str(taxon["path"]))
    if stats:
        for metrics in stats.values():
            count = metrics.get("count")
            if count is None:
                continue
            try:
                return int(count)
            except (TypeError, ValueError):
                continue
    direct = taxa_navigation.count_taxon_rows(taxon)
    if direct is not None:
        return int(direct)
    return 0


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


def build_descendant_catalogs_for_ancestor(ancestor_taxon_id: str) -> None:
    """Builds descendant catalog parquets for all ranks below an ancestor.
    
    Args:
        ancestor_taxon_id: Taxon id whose descendant rank catalogs should be written.
    
    Returns:
        None. Writes `{rank}.parquet` files under the ancestor directory.
    """
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or "ROOT"
    targets = _descendant_rank_targets(ancestor_rank)
    for rank in targets:
        aggregate: list[str] | None = None
        if rank == "SUBSPECIES":
            if ancestor_rank != "SPECIES":
                continue
            aggregate = list(CONFIG.subspecies_equivalents)
        elif (
            rank == "SPECIES"
            and ancestor_rank
            not in {CONFIG.species_rank, *CONFIG.subspecies_equivalents}
        ):
            aggregate = [CONFIG.species_rank, *CONFIG.subspecies_equivalents]
        build_descendant_catalog_parquet(
            ancestor_taxon_id,
            rank,
            aggregate_ranks=aggregate,
        )


def _collect_metric_entries_for_taxon(
    taxon,
    fallback_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    """Collects ranking entries from summary stats for a single taxon.
    
    Args:
        taxon: Taxon record to read summary stats from.
        fallback_samples: Sample count to use when stats omit a count field.
    
    Returns:
        A mapping of "variable::metric" to entry lists shaped like
        {"taxon_key": ..., "value": <metric>, "sample_count": <count>}.
    """
    stats = _load_summary_stats(str(taxon["path"])) or {}
    categorical_stats = _load_categorical_stats(str(taxon["path"])) or {}
    combined: dict[str, dict[str, Any]] = {}
    for source in (stats, categorical_stats):
        if not source:
            continue
        for variable, metrics in source.items():
            if not metrics:
                continue
            bucket = combined.setdefault(variable, {})
            bucket.update(metrics)
    if not combined:
        return {}
    entries: dict[str, list[dict[str, Any]]] = {}
    for variable, metrics in combined.items():
        if not metrics:
            continue
        sample_count = metrics.get("count")
        if sample_count is None:
            sample_count = metrics.get("total_samples")
        if sample_count is None:
            sample_count = fallback_samples
        sample_count = int(sample_count)
        if sample_count <= 0:
            continue
        for metric_name, raw_value in metrics.items():
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric_value):
                continue
            column_key = f"{variable}::{metric_name}"
            bucket = entries.setdefault(column_key, [])
            bucket.append(
                {
                    "taxon_key": taxon["taxon_key"],
                    "value": numeric_value,
                    "sample_count": sample_count,
                }
            )
    return entries


def _write_rank_index(
    index_path: Path,
    column_entries: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Writes a rank index parquet from metric entry lists.
    
    Args:
        index_path: Output path for the rank index parquet.
        column_entries: Mapping of "variable::metric" to entry dicts with taxon_key, value, and sample_count.
    
    Returns:
        A positions payload mapping "variable::metric" to taxon_key/position/sample_count entries.
    
    Example:
        Output file name: "<ancestor_path>/<rank>_index.parquet" (e.g. "genus_index.parquet").
        Output columns are struct arrays keyed by "variable::metric", for example.
    """
    if not column_entries:
        index_path.unlink(missing_ok=True)
        return {}

    struct_fields = [
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int32()),
    ]
    max_len = 0
    column_lengths: dict[str, int] = {}
    arrays: dict[str, pa.Array] = {}
    metric_names: set[str] = set()
    position_payload: dict[str, list[dict[str, Any]]] = {}

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
        column_positions: list[dict[str, Any]] = []
        for idx, entry in enumerate(sorted_entries):
            column_positions.append(
                {
                    "taxon_key": str(entry["taxon_key"]),
                    "position": idx,
                    "sample_count": entry["sample_count"],
                }
            )
        position_payload[column_name] = column_positions

    if not arrays:
        index_path.unlink(missing_ok=True)
        return {}

    for column_name, arr in list(arrays.items()):
        if len(arr) < max_len:
            pad = pa.nulls(max_len - len(arr), type=arr.type)
            arrays[column_name] = pa.concat_arrays([arr, pad])

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

    return position_payload


def _build_rank_index_parquet(ancestor, canonical_rank: str) -> None:
    """Builds a rank index parquet for a given ancestor and descendant rank.
    
    Args:
        ancestor: Ancestor taxon record whose descendant catalog will be indexed.
        canonical_rank: Canonical descendant rank to build (e.g., SPECIES).
    
    Returns:
        None. Writes a `{rank}_index.parquet` under the ancestor directory.
    """
    ancestor_path = Path(ancestor["path"])
    catalog_path = ancestor_path / f"{canonical_rank.lower()}.parquet"
    index_path = ancestor_path / f"{canonical_rank.lower()}_index.parquet"
    if not catalog_path.exists():
        print(f"[rank-index] missing catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return
    try:
        frame = pd.read_parquet(catalog_path, columns=["taxon_key", "sample_count"])
    except (OSError, ValueError):
        print(f"[rank-index] failed reading catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return
    if frame.empty:
        print(f"[rank-index] empty catalog {catalog_path}")
        index_path.unlink(missing_ok=True)
        return

    column_entries: dict[str, list[dict[str, Any]]] = {}
    for record in frame.itertuples(index=False):
        taxon_key = getattr(record, "taxon_key", None)
        if taxon_key is None:
            continue
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
        if taxon is None:
            continue
        fallback_samples = int(getattr(record, "sample_count", None))
        metric_entries = _collect_metric_entries_for_taxon(taxon, fallback_samples)
        if not metric_entries:
            continue
        for column_name, entries in metric_entries.items():
            bucket = column_entries.setdefault(column_name, [])
            bucket.extend(entries)

    if not column_entries:
        print(f"[rank-index] no stats entries for {ancestor_path} {canonical_rank}")
        index_path.unlink(missing_ok=True)
        return

    _write_rank_index(index_path, column_entries)


def build_rank_indexes_for_ancestor(ancestor_taxon_id: str) -> None:
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
    for rank in targets:
        _build_rank_index_parquet(ancestor, rank)

def build_density_curve(
    values: Sequence[float],
    *,
    point_count: int,
) -> Optional[dict[str, Any]]:
    """Builds a kernel density estimate curve for numeric values.
    
    Args:
        values: Numeric values to estimate a density curve for.
        point_count: Number of points to sample along the curve.
    
    Returns:
        A dict with sampled points, density values, and min/max/bandwidth metadata.
    """
    if not values:
        return None
    array = np.asarray(values, dtype=float)
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
        if "_" not in name:
            break
        taxon_key = name.split("_")[-1]
        if not taxon_key:
            break
        ancestor = taxa_navigation.get_taxon_by_id(taxon_key)
        if ancestor:
            contexts.append(ancestor)
        current = parent
    return contexts


def load_relative_ranks(
    taxon_dir: Path,
    variable_id: str,
    location_gid: Optional[str] = None,
) -> List[dict[str, Any]]:
    """Loads relative rank positions for a taxon across ancestor contexts.
    
    Args:
        taxon_dir: Filesystem path of the taxon directory to rank.
        variable_id: Environmental variable id to rank on.
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
    target_taxon_id = taxa_navigation.taxon_id_as_int(str(taxon_key))
    target_rank = taxa_navigation.canonical_rank(taxon["rank"]) or "SPECIES"
    storage_rank = (
        "SUBSPECIES"
        if target_rank in CONFIG.subspecies_equivalents
        else target_rank
    )
    contexts = _ancestor_contexts(Path(taxon["path"]))
    if not contexts:
        return []
    allowed_taxa: Optional[frozenset[int]] = None
    normalized_location = location_gid.strip() if location_gid else None
    if normalized_location:
        _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
        membership = gis_lookup.location_taxa_membership()
        allowed_taxa = membership.get((scope, target))
        if not allowed_taxa:
            return []
        if target_taxon_id is not None and target_taxon_id not in allowed_taxa:
            return []
    results: list[dict[str, Any]] = []
    for ancestor in contexts:
        ancestor_rank = taxa_navigation.canonical_rank(ancestor["rank"]) or ""
        column_rank = (
            storage_rank
            if not (storage_rank == "SUBSPECIES" and ancestor_rank != "SPECIES")
            else "SPECIES"
        )
        lookup_key = taxon_key
        index_path = Path(ancestor["path"]) / f"{column_rank.lower()}_index.parquet"
        if not index_path.exists():
            continue
        column_lengths = _load_column_lengths(index_path)
        if not column_lengths:
            continue
        for metric_name in relative_rank_metrics:
            try:
                column_name = _resolve_column_name(
                    index_path, variable_id, metric_name
                )
            except ValueError:
                continue
            column_length = column_lengths.get(column_name)
            if not column_length:
                continue
            column = _load_struct_column(index_path, column_name, column_length)
            taxon_keys = column.field("taxonKey").to_pylist()
            values = column.field("value").to_pylist()
            samples = column.field("sampleCount").to_pylist()
            lookup_taxon = taxa_navigation.get_taxon_by_id(str(lookup_key))
            if lookup_taxon is None:
                continue
            stats = _load_summary_stats(str(lookup_taxon["path"])) or {}
            categorical_stats = _load_categorical_stats(str(lookup_taxon["path"])) or {}
            metrics = dict(stats.get(variable_id, {}))
            metrics.update(categorical_stats.get(variable_id, {}))
            raw_value = metrics.get(metric_name)
            if raw_value is None:
                continue
            try:
                target_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(target_value):
                continue
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
            count = column_length
            position = zero_based
            percentile = (
                position / max(count - 1, 1) if count > 1 else 0.0
            )
            if allowed_taxa is not None:
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
                percentile = (
                    position / max(count - 1, 1) if count > 1 else 0.0
                )
            ancestor_label = (
                ancestor["scientific_name"]
                or ancestor["common_name"]
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
        schema = pq.read_schema(parquet_path)
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
        schema = pq.read_schema(parquet_path)
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
    table = pq.read_table(parquet_path, columns=[column_name])
    column = table.column(column_name).combine_chunks()
    if column_length < len(column):
        column = column.slice(0, column_length)
    return column


def list_rank_metric_options(
    ancestor_taxon_id: str,
    descendant_rank: str,
) -> List[Dict[str, Any]]:
    """Lists available variable/metric columns for a rank index parquet.
    
    Args:
        ancestor_taxon_id: Taxon id whose descendant rank index should be inspected.
        descendant_rank: Rank to inspect (e.g., SPECIES).
    
    Returns:
        A list of entries describing each available variable/metric column.
    """
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")
    canonical_rank = taxa_navigation.canonical_rank(descendant_rank)
    if not canonical_rank:
        raise ValueError("descendant_rank is required")
    ancestor_path = Path(ancestor["path"])
    index_path = ancestor_path / f"{canonical_rank.lower()}_index.parquet"
    if not index_path.exists():
        return []
    try:
        schema = pq.read_schema(index_path)
    except (OSError, ValueError):
        return []
    column_lengths = _load_column_lengths(index_path)
    options: list[Dict[str, Any]] = []
    for column_name in schema.names:
        if "::" not in column_name:
            continue
        variable, metric = column_name.split("::", 1)
        options.append(
            {
                "variable": variable,
                "metric": metric,
                "column": column_name,
                "count": column_lengths.get(column_name, 0),
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
    
    Returns:
        A tuple of (ranked entries, optional metric value list for distributions).
    """
    ancestor = taxa_navigation.get_taxon_by_id(str(ancestor_taxon_id))
    if ancestor is None:
        raise ValueError(f"Unknown ancestor taxon id {ancestor_taxon_id}")

    canonical_rank = taxa_navigation.canonical_rank(descendant_rank)
    if not canonical_rank:
        raise ValueError("descendant_rank is required")

    order_normalized = (order or "asc").strip().lower()
    if order_normalized not in {"asc", "desc"}:
        raise ValueError("order must be either 'asc' or 'desc'")

    ancestor_path = Path(ancestor["path"])
    index_path = ancestor_path / f"{canonical_rank.lower()}_index.parquet"
    if not index_path.exists():
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
    if column_length == 0:
        return [], None

    allowed_taxa: Optional[frozenset[int]] = None
    normalized_location = location_gid.strip() if location_gid else None
    if normalized_location:
        _column, scope, target = gis_lookup.location_lookup_for_gid(normalized_location)
        membership = gis_lookup.location_taxa_membership()
        allowed_taxa = membership.get((scope, target))
        if not allowed_taxa:
            return [], None

    taxon_values = column.field("taxonKey").to_pylist()
    metric_values = column.field("value").to_pylist()
    sample_counts = column.field("sampleCount").to_pylist()

    media_index = taxa_navigation.load_taxon_media()

    min_samples = max(0, int(min_samples or 0))

    eligible: list[tuple[int, Any, float, int, Optional[int]]] = []
    for idx, (taxon_key, value, sample_count) in enumerate(
        zip(taxon_values, metric_values, sample_counts)
    ):
        if sample_count is None or (min_samples and sample_count < min_samples):
            continue
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_key))
        if taxon is None:
            continue
        taxon_id_value = taxa_navigation.taxon_id_as_int(taxon["taxon_key"])
        if allowed_taxa is not None:
            if taxon_id_value is None or taxon_id_value not in allowed_taxa:
                continue
        taxon_rank = taxa_navigation.canonical_rank(taxon["rank"])
        if (
            canonical_rank == "SPECIES"
            and not include_species_like
            and taxon_rank != "SPECIES"
        ):
            continue
        eligible.append((idx, taxon, float(value), int(sample_count), taxon_id_value))

    filtered_total = len(eligible)
    if filtered_total == 0:
        return [], None

    distribution_values = [entry[2] for entry in eligible] if return_distribution else None

    if order_normalized == "desc":
        eligible = list(reversed(eligible))

    denominator = max(filtered_total - 1, 1)
    results: list[dict[str, Any]] = []
    limit = max(1, int(limit or 1))
    for output_idx, entry in enumerate(eligible[:limit]):
        _original_index, taxon, numeric_value, sample_count, taxon_id = entry
        taxon_id = taxon_id if taxon_id is not None else taxa_navigation.taxon_id_as_int(taxon["taxon_key"])
        canonical_taxon_rank = taxa_navigation.canonical_rank(taxon["rank"])
        if order_normalized == "desc":
            rank_index = filtered_total - 1 - output_idx
        else:
            rank_index = output_idx
        percentile = rank_index / denominator if filtered_total > 1 else 0.0
        common_names = taxa_navigation.extract_common_names_for_language(
            taxon,
            language=CONFIG.common_name_language,
        )
        common_name = common_names[0] if common_names else None
        media_record = media_index.get(taxon["taxon_key"])
        record = {
            "taxonId": taxon_id if taxon_id is not None else taxon["taxon_key"],
            "taxon_id": taxon_id if taxon_id is not None else taxon["taxon_key"],
            "taxon_key": taxon["taxon_key"],
            "scientificName": taxon["scientific_name"],
            "commonName": common_name,
            "common_names": common_names,
            "rank": canonical_taxon_rank,
            "value": numeric_value,
            "sampleCount": sample_count,
            "count": filtered_total,
            "position": output_idx + 1,
            "percentile": percentile,
            "metric": metric,
            "variable": layer,
        }
        if media_record:
            record["image_url"] = media_record.get("url")
            record["image_license"] = media_record.get("license")
            record["image_creator"] = media_record.get("creator")
            record["image_rights_holder"] = media_record.get("rightsHolder")
            record["image_references"] = media_record.get("references")
        results.append(record)

    return results, distribution_values
