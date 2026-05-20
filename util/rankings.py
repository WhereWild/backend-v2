"""
Relative ranking artifacts for the taxonomy tree.

For each ancestor taxon (top-down, level by level), builds:
  {rank}.parquet                    — catalog of all descendants of that rank
  {rank}_index.parquet              — descendants sorted by each variable::metric
  relative_ranks_positions.parquet  — per-taxon position in ancestor rank indexes

Runs after the stats pass (which produces numerical_stats.parquet and
nominal_stats.parquet). Called from scripts/process_tree.py.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from config.config import METRICS_BY_TYPE, ValueType, load_config
from util.stats import NOMINAL_STATS_FILE, NUMERICAL_STATS_FILE
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

TREE_ROOT = Path("data/taxonomy/tree")
POSITION_FILE = "relative_ranks_positions.parquet"

# Canonical taxonomy rank order used to determine descendant catalog targets.
_RANK_ORDER: tuple[str, ...] = (
    "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES", "SUBSPECIES",
)

_CATALOG_SCHEMA = pa.schema([
    pa.field("taxon_key", pa.string()),
    pa.field("path", pa.string()),
    pa.field("scientific_name", pa.string()),
    pa.field("common_name", pa.string()),
    pa.field("rank", pa.string()),
    pa.field("sample_count", pa.int64()),
])

_POSITION_SCHEMA = pa.schema([
    pa.field("variable", pa.string()),
    pa.field("metric", pa.string()),
    pa.field("position", pa.int64()),
    pa.field("count", pa.int64()),
    pa.field("sampleCount", pa.int64()),
    pa.field("contextTaxonId", pa.string()),
    pa.field("contextLabel", pa.string()),
])

_STRUCT_FIELDS = [
    pa.field("taxonKey", pa.string()),
    pa.field("value", pa.float64()),
    pa.field("sampleCount", pa.int64()),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _resolve_context_label(taxon: TaxonRecord) -> str:
    sci = (taxon.get("scientific_name") or "").replace("_", " ").strip()
    if sci:
        return sci
    common = (taxon.get("common_name") or "").replace("_", " ").strip()
    if common:
        return common
    return str(taxon["taxon_key"])


def _descendant_rank_targets(ancestor_rank: str) -> list[str]:
    """Return canonical ranks below ancestor_rank in taxonomy order."""
    try:
        idx = _RANK_ORDER.index(ancestor_rank)
    except ValueError:
        return []
    return list(_RANK_ORDER[idx + 1:])


def _metrics_for_vtype(layer: dict, vtype: ValueType) -> tuple[str, ...]:
    """Return rankable metric names for a value type.

    Raises NotImplementedError for CIRCULAR (not yet supported).
    Returns () for types with no ranking metrics (ORDINAL, AGGREGATE).
    """
    match vtype:
        case ValueType.RATIO | ValueType.INTERVAL:
            return METRICS_BY_TYPE[vtype]
        case ValueType.NOMINAL:
            return METRICS_BY_TYPE[ValueType.NOMINAL]
        case ValueType.CIRCULAR:
            raise NotImplementedError(
                f"Relative ranking not implemented for circular layers"
                f" (layer: {layer.get('id')!r})"
            )
        case _:
            return ()


def _infer_sample_count(taxon_dir: Path) -> int:
    """Return observation count from stats files or occurrence index."""
    stats_path = taxon_dir / NUMERICAL_STATS_FILE
    if stats_path.exists():
        try:
            tbl = pq.read_table(stats_path, columns=["count"])
            for val in tbl.column("count").to_pylist():
                if val is not None:
                    try:
                        n = int(val)
                        if n > 0:
                            return n
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    nom_path = taxon_dir / NOMINAL_STATS_FILE
    if nom_path.exists():
        try:
            df = pq.read_table(nom_path).to_pandas()
            rows = df[df["metric"] == "total_samples"]
            if not rows.empty:
                return int(float(rows.iloc[0]["value"]))
        except Exception:
            pass
    idx_path = taxon_dir / "occurrence_index.parquet"
    if idx_path.exists():
        try:
            return pq.read_metadata(idx_path).num_rows
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# Descendant catalogs
# ---------------------------------------------------------------------------

def _write_descendant_catalog(out_path: Path, taxa: list[TaxonRecord]) -> None:
    if not taxa:
        out_path.unlink(missing_ok=True)
        return
    rows = []
    for t in taxa:
        rows.append({
            "taxon_key": str(t["taxon_key"]),
            "path": t["path"],
            "scientific_name": t.get("scientific_name") or "",
            "common_name": t.get("common_name") or "",
            "rank": t.get("rank") or "",
            "sample_count": _infer_sample_count(TREE_ROOT / t["path"]),
        })
    _atomic_write(out_path, pa.Table.from_pylist(rows, schema=_CATALOG_SCHEMA))


def build_descendant_catalogs(ancestor: TaxonRecord) -> None:
    """Write {rank}.parquet catalog files under the ancestor's tree directory."""
    ancestor_rank = ancestor.get("rank") or ""
    targets = _descendant_rank_targets(ancestor_rank)
    if not targets:
        return

    equiv = frozenset(CONFIG.subspecies_equivalents)
    species_rank = CONFIG.species_rank

    by_rank: dict[str, list[TaxonRecord]] = {}
    for desc in iter_descendants(ancestor, include_self=False):
        rank = desc.get("rank") or ""
        canonical = "SUBSPECIES" if rank in equiv else rank
        by_rank.setdefault(canonical, []).append(desc)

    ancestor_dir = TREE_ROOT / ancestor["path"]
    for rank in targets:
        out_path = ancestor_dir / f"{rank.lower()}.parquet"
        if rank == "SUBSPECIES":
            if ancestor_rank != species_rank:
                out_path.unlink(missing_ok=True)
                continue
            _write_descendant_catalog(out_path, by_rank.get("SUBSPECIES", []))
        elif rank == "SPECIES" and ancestor_rank not in (species_rank, *CONFIG.subspecies_equivalents):
            # Non-species ancestors: SPECIES catalog includes leaf-rank taxa of all species-group ranks
            combined = list(by_rank.get("SPECIES", []))
            combined += by_rank.get("SUBSPECIES", [])
            _write_descendant_catalog(out_path, combined)
        else:
            _write_descendant_catalog(out_path, by_rank.get(rank, []))


# ---------------------------------------------------------------------------
# Rank index
# ---------------------------------------------------------------------------

def _collect_entries_from_numerical_stats(
    taxon_key: str,
    taxon_dir: Path,
    sample_count: int,
    layers: list[dict],
) -> dict[str, dict[str, Any]]:
    """Read numerical_stats.parquet → {variable::metric: entry dict}."""
    stats_path = taxon_dir / NUMERICAL_STATS_FILE
    if not stats_path.exists():
        return {}
    try:
        df = pq.read_table(stats_path).to_pandas()
    except Exception:
        return {}

    layer_by_id = {lay["id"]: lay for lay in layers}
    entries: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        variable = str(row.get("variable") or "")
        if not variable or variable not in layer_by_id:
            continue
        layer = layer_by_id[variable]
        try:
            vtype = ValueType(layer.get("value_type", ""))
        except ValueError:
            continue
        try:
            metrics = _metrics_for_vtype(layer, vtype)
        except NotImplementedError:
            continue
        for metric in metrics:
            val = row.get(metric)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):
                continue
            entries[f"{variable}::{metric}"] = {
                "taxon_key": taxon_key,
                "value": fval,
                "sample_count": sample_count,
            }
    return entries


def _collect_entries_from_nominal_stats(
    taxon_key: str,
    taxon_dir: Path,
    sample_count: int,
    layers: list[dict],
) -> dict[str, dict[str, Any]]:
    """Read nominal_stats.parquet → {variable::metric: entry dict}."""
    stats_path = taxon_dir / NOMINAL_STATS_FILE
    if not stats_path.exists():
        return {}
    try:
        df = pq.read_table(stats_path).to_pandas()
    except Exception:
        return {}

    nominal_ids = {lay["id"] for lay in layers if lay.get("value_type") == ValueType.NOMINAL}
    nominal_metrics = set(METRICS_BY_TYPE[ValueType.NOMINAL])
    entries: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        variable = str(row.get("variable") or "")
        metric = str(row.get("metric") or "")
        if variable not in nominal_ids or metric not in nominal_metrics:
            continue
        val = row.get("value")
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fval):
            continue
        entries[f"{variable}::{metric}"] = {
            "taxon_key": taxon_key,
            "value": fval,
            "sample_count": sample_count,
        }
    return entries


def _collect_all_entries(
    taxon_key: str,
    taxon_dir: Path,
    sample_count: int,
    layers: list[dict],
) -> dict[str, dict[str, Any]]:
    entries = _collect_entries_from_numerical_stats(taxon_key, taxon_dir, sample_count, layers)
    entries.update(_collect_entries_from_nominal_stats(taxon_key, taxon_dir, sample_count, layers))
    return entries


def _build_rank_index(
    catalog_path: Path,
    index_path: Path,
    layers: list[dict],
) -> None:
    """Read catalog, collect per-taxon metrics, write sorted struct array index."""
    try:
        catalog = pq.read_table(
            catalog_path, columns=["taxon_key", "path", "sample_count"]
        ).to_pandas()
    except Exception:
        index_path.unlink(missing_ok=True)
        return
    if catalog.empty:
        index_path.unlink(missing_ok=True)
        return

    column_entries: dict[str, list[dict[str, Any]]] = {}
    for _, row in catalog.iterrows():
        taxon_key = str(row["taxon_key"])
        taxon_path = str(row.get("path") or "")
        sample_count = int(row.get("sample_count") or 0)
        if not taxon_path:
            continue
        for col_key, entry in _collect_all_entries(
            taxon_key, TREE_ROOT / taxon_path, sample_count, layers
        ).items():
            column_entries.setdefault(col_key, []).append(entry)

    if not column_entries:
        index_path.unlink(missing_ok=True)
        return

    struct_type = pa.struct(_STRUCT_FIELDS)
    max_len = 0
    arrays: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}

    for col_name, entries in column_entries.items():
        sorted_entries = sorted(entries, key=lambda e: (e["value"], e["taxon_key"]))
        column_lengths[col_name] = len(sorted_entries)
        max_len = max(max_len, len(sorted_entries))
        arr = pa.StructArray.from_arrays(
            [
                pa.array([e["taxon_key"] for e in sorted_entries], type=pa.string()),
                pa.array([e["value"] for e in sorted_entries], type=pa.float64()),
                pa.array([e["sample_count"] for e in sorted_entries], type=pa.int64()),
            ],
            fields=_STRUCT_FIELDS,
        )
        arrays[col_name] = arr

    for col_name, arr in arrays.items():
        if len(arr) < max_len:
            arrays[col_name] = pa.concat_arrays(
                [arr, pa.nulls(max_len - len(arr), type=struct_type)]
            )

    table = pa.table(arrays)
    metadata = {b"column_lengths": json.dumps(column_lengths).encode("utf-8")}
    _atomic_write(index_path, table.replace_schema_metadata(metadata))


def build_rank_indexes(ancestor: TaxonRecord, layers: list[dict]) -> None:
    """Build {rank}_index.parquet files under the ancestor's directory."""
    ancestor_rank = ancestor.get("rank") or ""
    targets = _descendant_rank_targets(ancestor_rank)
    if not targets:
        return

    ancestor_dir = TREE_ROOT / ancestor["path"]
    for rank in targets:
        catalog_path = ancestor_dir / f"{rank.lower()}.parquet"
        index_path = ancestor_dir / f"{rank.lower()}_index.parquet"
        if not catalog_path.exists():
            index_path.unlink(missing_ok=True)
            continue
        _build_rank_index(catalog_path, index_path, layers)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def _load_column_lengths(index_path: Path) -> dict[str, int]:
    try:
        schema = pq.read_schema(index_path)
        raw = (schema.metadata or {}).get(b"column_lengths")
        if not raw:
            return {}
        return {k: int(v) for k, v in json.loads(raw.decode("utf-8")).items() if int(v) > 0}
    except Exception:
        return {}


def _load_existing_positions(positions_path: Path) -> list[dict[str, Any]]:
    if not positions_path.exists():
        return []
    try:
        rows = pq.read_table(positions_path).to_pylist()
        return [r for r in rows if r.get("variable") and r.get("metric")]
    except Exception:
        return []


def _distribute_positions(ancestor: TaxonRecord, index_path: Path) -> None:
    """Read one rank index and upsert position rows into each descendant's file."""
    if not index_path.exists():
        return
    try:
        schema = pq.read_schema(index_path)
    except Exception:
        return

    metric_columns = [n for n in schema.names if "::" in n]
    if not metric_columns:
        return

    column_lengths = _load_column_lengths(index_path)
    context_taxon_id = str(ancestor["taxon_key"])
    context_label = _resolve_context_label(ancestor)

    try:
        table = pq.read_table(index_path, columns=metric_columns)
    except Exception:
        return

    rows_by_taxon: dict[str, list[dict[str, Any]]] = {}
    for col_name in metric_columns:
        variable, metric = col_name.split("::", 1)
        column = table.column(col_name).combine_chunks()
        col_len = min(column_lengths.get(col_name, len(column)), len(column))
        if col_len <= 0:
            continue
        for position in range(col_len):
            entry = column[position].as_py()
            if entry is None:
                continue
            taxon_key = entry.get("taxonKey")
            if taxon_key is None:
                continue
            sample_count = entry.get("sampleCount")
            rows_by_taxon.setdefault(str(taxon_key), []).append({
                "variable": variable,
                "metric": metric,
                "position": position,
                "count": col_len,
                "sampleCount": int(sample_count) if sample_count is not None else None,
                "contextTaxonId": context_taxon_id,
                "contextLabel": context_label,
            })

    for taxon_key, new_rows in rows_by_taxon.items():
        taxon = get_taxon_by_id(taxon_key)
        if taxon is None:
            continue
        positions_path = TREE_ROOT / taxon["path"] / POSITION_FILE
        existing = _load_existing_positions(positions_path)
        existing_keys = {(r["variable"], r["metric"], r["contextTaxonId"]) for r in existing}
        unique_new = [
            r for r in new_rows
            if (r["variable"], r["metric"], r["contextTaxonId"]) not in existing_keys
        ]
        if not unique_new:
            continue
        merged = existing + unique_new
        tbl = pa.Table.from_arrays(
            [
                pa.array([r["variable"] for r in merged], type=pa.string()),
                pa.array([r["metric"] for r in merged], type=pa.string()),
                pa.array([r["position"] for r in merged], type=pa.int64()),
                pa.array([r["count"] for r in merged], type=pa.int64()),
                pa.array([r["sampleCount"] for r in merged], type=pa.int64()),
                pa.array([r["contextTaxonId"] for r in merged], type=pa.string()),
                pa.array([r["contextLabel"] for r in merged], type=pa.string()),
            ],
            schema=_POSITION_SCHEMA,
        )
        _atomic_write(positions_path, tbl)


def distribute_all_positions(ancestor: TaxonRecord) -> None:
    """Distribute position rows from all rank indexes under ancestor's directory."""
    ancestor_rank = ancestor.get("rank") or ""
    targets = _descendant_rank_targets(ancestor_rank)
    ancestor_dir = TREE_ROOT / ancestor["path"]
    for rank in targets:
        index_path = ancestor_dir / f"{rank.lower()}_index.parquet"
        _distribute_positions(ancestor, index_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_relative_ranks(ancestor: TaxonRecord, layers: list[dict]) -> None:
    """Build descendant catalogs, rank indexes, and distribute positions for one ancestor.

    Designed to be called per-ancestor in a top-down (shallowest-first) BFS pass
    after the bottom-up stats pass is complete.
    """
    build_descendant_catalogs(ancestor)
    build_rank_indexes(ancestor, layers)
    distribute_all_positions(ancestor)
