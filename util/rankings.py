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

import csv
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from config.config import METRICS_BY_TYPE, ValueType, load_config
from util.stats import CIRCULAR_STATS_FILE, NOMINAL_STATS_FILE, NUMERICAL_STATS_FILE
from util.storage import atomic_write_parquet
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants, search_taxa_by_name

CONFIG = load_config("global")

TREE_ROOT = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "tree"
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

def _safe_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

def _atomic_write(path: Path, table: pa.Table) -> None:
    atomic_write_parquet(path, table, row_group_size=256)


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


# Circular metrics that are angular bearings — included in the sort index but
# excluded from relative_ranks_positions.parquet (no percentile/position display).
_ANGULAR_METRICS: frozenset[str] = frozenset({"circular_mean", "mode"})


def _metrics_for_vtype(layer: dict, vtype: ValueType) -> tuple[str, ...]:
    """Return rankable metric names for a value type.

    Returns () for types with no ranking metrics (ORDINAL, AGGREGATE).
    """
    match vtype:
        case ValueType.RATIO | ValueType.INTERVAL:
            return METRICS_BY_TYPE[vtype]
        case ValueType.NOMINAL:
            return METRICS_BY_TYPE[ValueType.NOMINAL]
        case ValueType.CIRCULAR:
            return METRICS_BY_TYPE[ValueType.CIRCULAR]
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

    for record in df.to_dict("records"):
        variable = str(record.get("variable") or "")
        if not variable or variable not in layer_by_id:
            continue
        layer = layer_by_id[variable]
        try:
            vtype = ValueType(layer.get("value_type", ""))
        except ValueError:
            continue
        if vtype not in (ValueType.RATIO, ValueType.INTERVAL):
            continue
        for metric in _metrics_for_vtype(layer, vtype):
            val = record.get(metric)
            if val is None:
                continue
            if not _safe_finite(val):
                continue
            entries[f"{variable}::{metric}"] = {
                "taxon_key": taxon_key,
                "value": float(val),
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

    for record in df.to_dict("records"):
        variable = str(record.get("variable") or "")
        metric = str(record.get("metric") or "")
        if variable not in nominal_ids:
            continue
        if metric not in nominal_metrics and not metric.startswith("class_"):
            continue
        val = record.get("value")
        if not _safe_finite(val):
            continue
        entries[f"{variable}::{metric}"] = {
            "taxon_key": taxon_key,
            "value": float(val),
            "sample_count": sample_count,
        }
    return entries


def _collect_entries_from_circular_stats(
    taxon_key: str,
    taxon_dir: Path,
    sample_count: int,
    layers: list[dict],
) -> dict[str, dict[str, Any]]:
    """Read circular_stats.parquet → {variable::metric: entry dict}."""
    stats_path = taxon_dir / CIRCULAR_STATS_FILE
    if not stats_path.exists():
        return {}
    try:
        df = pq.read_table(stats_path).to_pandas()
    except Exception:
        return {}

    layer_by_id = {lay["id"]: lay for lay in layers}
    entries: dict[str, dict[str, Any]] = {}

    for record in df.to_dict("records"):
        variable = str(record.get("variable") or "")
        if not variable or variable not in layer_by_id:
            continue
        layer = layer_by_id[variable]
        try:
            vtype = ValueType(layer.get("value_type", ""))
        except ValueError:
            continue
        if vtype != ValueType.CIRCULAR:
            continue
        for metric in METRICS_BY_TYPE[ValueType.CIRCULAR]:
            val = record.get(metric)
            if val is None:
                continue
            if not _safe_finite(val):
                continue
            entries[f"{variable}::{metric}"] = {
                "taxon_key": taxon_key,
                "value": float(val),
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
    entries.update(_collect_entries_from_circular_stats(taxon_key, taxon_dir, sample_count, layers))
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
    for row in catalog.itertuples(index=False):
        taxon_key = str(row.taxon_key)
        taxon_path = str(row.path or "")
        sample_count = int(row.sample_count or 0)
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
        if metric in _ANGULAR_METRICS:
            continue  # bearings have no meaningful percentile position
        is_class_metric = metric.startswith("class_")
        column = table.column(col_name).combine_chunks()
        col_len = min(column_lengths.get(col_name, len(column)), len(column))
        if col_len <= 0:
            continue

        # Extract struct fields into flat arrays — much faster than per-entry as_py()
        col_slice = column[:col_len]
        taxon_keys_list = col_slice.field("taxonKey").to_pylist()
        values_list = col_slice.field("value").to_pylist()
        sample_counts_list = col_slice.field("sampleCount").to_pylist()

        prev_value: float | None = None
        min_rank_pos = 0
        for position in range(col_len):
            taxon_key = taxon_keys_list[position]
            if taxon_key is None:
                continue
            value = values_list[position]
            # Track min-rank position: all tied values share the first position
            # in their group rather than getting arbitrary alphabetical positions.
            if value != prev_value:
                min_rank_pos = position
                prev_value = value
            # Zero-class entries exist in the index for search but are meaningless
            # on the species page (the species was never observed in that class).
            if is_class_metric and value == 0.0:
                continue
            sample_count = sample_counts_list[position]
            rows_by_taxon.setdefault(taxon_key, []).append({
                "variable": variable,
                "metric": metric,
                "position": min_rank_pos,
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
        # Keep rows from other ancestor contexts; replace rows from this context.
        kept = [r for r in existing if r["contextTaxonId"] != context_taxon_id]
        merged = kept + new_rows
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


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

_LOCATIONS_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "gis" / "locations"
_LOC_TAXA_PATH = _LOCATIONS_DIR / "location_taxa.parquet"
_HIERARCHY_CSV = _LOCATIONS_DIR / "hierarchy.csv"


@lru_cache(maxsize=1)
def _load_gid_levels() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open(_HIERARCHY_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = (row.get("gid") or "").strip()
                try:
                    level = int(row.get("level", ""))
                except (ValueError, TypeError):
                    continue
                if gid:
                    result[gid] = level
    except Exception:
        pass
    return result


def _gid_to_scope(gid: str) -> str:
    level = _load_gid_levels().get(gid)
    if level is not None:
        return CONFIG.location_scope_by_level.get(level, "gbif_region")
    return "gbif_region"


@lru_cache(maxsize=256)
def _location_taxon_keys(gid: str) -> tuple[frozenset[str], dict[str, int]]:
    """Return (taxon_key set, per-taxon observation counts) for a GID."""
    scope = _gid_to_scope(gid)
    try:
        tbl = pq.read_table(
            _LOC_TAXA_PATH,
            filters=[("scope", "=", scope), ("gid", "=", gid)],
        )
        keys = frozenset(str(k) for k in tbl.column("taxon_key").to_pylist())
        counts = {
            str(k): int(c)
            for k, c in zip(
                tbl.column("taxon_key").to_pylist(),
                tbl.column("count").to_pylist(),
            )
        }
        return keys, counts
    except Exception:
        return frozenset(), {}


def _read_index_entries(index_path: Path, col_name: str, col_len: int) -> list[dict]:
    """Read one struct column from a rank_index.parquet, returning up to col_len entries."""
    try:
        tbl = pq.read_table(index_path, columns=[col_name])
        column = tbl.column(col_name).combine_chunks()
        result = []
        for i in range(min(col_len, len(column))):
            entry = column[i].as_py()
            if entry is not None:
                result.append(entry)
        return result
    except Exception:
        return []


def _taxon_metric_value(taxon_dir: Path, variable_id: str, metric_id: str) -> float | None:
    """Read one variable::metric value from a taxon's stats files."""
    num_path = taxon_dir / NUMERICAL_STATS_FILE
    if num_path.exists():
        try:
            df = pq.read_table(num_path).to_pandas()
            rows = df[df["variable"] == variable_id]
            if not rows.empty and metric_id in rows.columns:
                val = rows.iloc[0][metric_id]
                if val is not None:
                    fval = float(val)
                    if math.isfinite(fval):
                        return fval
        except Exception:
            pass
    nom_path = taxon_dir / NOMINAL_STATS_FILE
    if nom_path.exists():
        try:
            df = pq.read_table(nom_path).to_pandas()
            rows = df[(df["variable"] == variable_id) & (df["metric"] == metric_id)]
            if not rows.empty:
                val = rows.iloc[0]["value"]
                if val is not None:
                    fval = float(val)
                    if math.isfinite(fval):
                        return fval
        except Exception:
            pass
    circ_path = taxon_dir / CIRCULAR_STATS_FILE
    if circ_path.exists():
        try:
            df = pq.read_table(circ_path).to_pandas()
            rows = df[df["variable"] == variable_id]
            if not rows.empty and metric_id in rows.columns:
                val = rows.iloc[0][metric_id]
                if val is not None:
                    fval = float(val)
                    if math.isfinite(fval):
                        return fval
        except Exception:
            pass
    return None


def _accepted_ranks(descendant_rank: str, include_species_like: bool) -> frozenset[str] | None:
    """Return accepted taxon rank set for filtering, or None if no rank filter needed."""
    if descendant_rank == CONFIG.species_rank:
        if include_species_like:
            return frozenset({CONFIG.species_rank} | set(CONFIG.subspecies_equivalents))
        return frozenset({CONFIG.species_rank})
    return None


def _empty_result(empty_reason: str, eligible_total: int = 0) -> dict:
    return {
        "total": 0,
        "matched_total": 0,
        "eligible_total": eligible_total,
        "empty_reason": empty_reason,
        "results": [],
    }


def _query_ranked_scoped(
    *,
    q: str | None,
    within_taxon: TaxonRecord,
    descendant_rank: str,
    sort_variable: str,
    sort_metric: str,
    sort_order: str,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    loc_keys: frozenset[str] | None,
    loc_counts: dict[str, int],
    reference_value: float | None = None,
    min_rbar: float | None = None,
) -> dict:
    rank_lower = "subspecies" if descendant_rank in CONFIG.subspecies_equivalents else descendant_rank.lower()
    ancestor_dir = TREE_ROOT / within_taxon["path"]
    index_path = ancestor_dir / f"{rank_lower}_index.parquet"

    if not index_path.exists():
        return _empty_result("no_index")

    col_name = f"{sort_variable}::{sort_metric}"
    col_len = _load_column_lengths(index_path).get(col_name)
    if not col_len:
        return _empty_result("no_column")

    entries = _read_index_entries(index_path, col_name, col_len)
    if not entries:
        return _empty_result("no_column")

    # Build reverse map: taxon_key → (raw_position, value, sample_count)
    index_map: dict[str, tuple[int, float, int]] = {}
    for pos, entry in enumerate(entries):
        tk = str(entry.get("taxonKey") or "")
        if tk:
            index_map[tk] = (pos, float(entry.get("value") or 0.0), int(entry.get("sampleCount") or 0))

    # Mode 3: restrict to text-matched taxon keys
    candidate_keys: frozenset[str] | None = None
    match_scores: dict[str, float] = {}
    if q:
        text_matches = search_taxa_by_name(q, limit=max(limit * 10, 200))
        candidate_keys = frozenset(str(t["taxon_key"]) for t, _, _ in text_matches if str(t["taxon_key"]) in index_map)
        match_scores = {str(t["taxon_key"]): score for t, score, _ in text_matches}

    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like)

    is_circular_bearing = sort_metric in _ANGULAR_METRICS and reference_value is not None

    # For circular sorts, optionally load rbar values for min_rbar filtering
    rbar_map: dict[str, float] = {}
    if is_circular_bearing and min_rbar is not None:
        rbar_col = f"{sort_variable}::rbar"
        rbar_col_len = _load_column_lengths(index_path).get(rbar_col)
        if rbar_col_len:
            for entry in _read_index_entries(index_path, rbar_col, rbar_col_len):
                tk = str(entry.get("taxonKey") or "")
                if tk:
                    rbar_map[tk] = float(entry.get("value") or 0.0)

    # Filter
    filtered: list[tuple[int, str, float, int]] = []  # (raw_pos, taxon_key, value, sample_count)
    for tk, (pos, val, sc) in index_map.items():
        if candidate_keys is not None and tk not in candidate_keys:
            continue
        if loc_keys is not None and tk not in loc_keys:
            continue
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        if rbar_map and rbar_map.get(tk, 0.0) < min_rbar:
            continue
        if accepted_ranks is not None:
            taxon = get_taxon_by_id(tk)
            if taxon is None or taxon.get("rank") not in accepted_ranks:
                continue
        filtered.append((pos, tk, val, sc))

    if is_circular_bearing:
        ref = float(reference_value)  # type: ignore[arg-type]
        if sort_order == "desc":
            filtered.sort(key=lambda e: ((ref - e[2]) % 360.0, e[1]))
        else:
            filtered.sort(key=lambda e: ((e[2] - ref) % 360.0, e[1]))
    else:
        reverse = (sort_order == "desc")
        filtered.sort(key=lambda e: (e[2], e[1]), reverse=reverse)

    total = len(filtered)
    page = filtered[offset:offset + limit]

    results = []
    for local_rank, (raw_pos, tk, val, sc) in enumerate(page, start=offset + 1):
        taxon = get_taxon_by_id(tk)
        if taxon is None:
            continue
        percentile = (raw_pos / col_len * 100) if col_len > 0 else None
        results.append({
            "taxon": taxon,
            "match_score": match_scores.get(tk),
            "sample_count": loc_counts.get(tk) or sc or None,
            "sort_value": val,
            "location_count": loc_counts.get(tk) or None,
            "position": local_rank,
            "percentile": percentile,
        })

    return {
        "total": total,
        "matched_total": total,
        "eligible_total": col_len,
        "empty_reason": None if results else "no_results",
        "results": results,
    }


def _query_ranked_text(
    *,
    q: str,
    sort_variable: str,
    sort_metric: str,
    sort_order: str,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    loc_keys: frozenset[str] | None,
    loc_counts: dict[str, int],
    reference_value: float | None = None,
    min_rbar: float | None = None,
) -> dict:
    candidates = search_taxa_by_name(q, limit=max((limit + offset) * 5, 200))
    if not candidates:
        return _empty_result("no_text_matches")

    is_circular_bearing = sort_metric in _ANGULAR_METRICS and reference_value is not None

    enriched: list[tuple[TaxonRecord, float, float, int]] = []  # taxon, score, sort_val, sc
    for taxon, score, _ in candidates:
        tk = str(taxon["taxon_key"])
        if loc_keys is not None and tk not in loc_keys:
            continue
        taxon_dir = TREE_ROOT / taxon["path"]
        val = _taxon_metric_value(taxon_dir, sort_variable, sort_metric)
        if val is None:
            continue
        sc = _infer_sample_count(taxon_dir)
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        if is_circular_bearing and min_rbar is not None:
            rbar = _taxon_metric_value(taxon_dir, sort_variable, "rbar")
            if rbar is None or rbar < min_rbar:
                continue
        enriched.append((taxon, score, val, sc))

    if is_circular_bearing:
        ref = float(reference_value)  # type: ignore[arg-type]
        if sort_order == "desc":
            enriched.sort(key=lambda e: ((ref - e[2]) % 360.0, str(e[0]["taxon_key"])))
        else:
            enriched.sort(key=lambda e: ((e[2] - ref) % 360.0, str(e[0]["taxon_key"])))
    else:
        reverse = (sort_order == "desc")
        enriched.sort(key=lambda e: (e[2], str(e[0]["taxon_key"])), reverse=reverse)

    total = len(enriched)
    page = enriched[offset:offset + limit]

    results = []
    for taxon, score, val, sc in page:
        tk = str(taxon["taxon_key"])
        results.append({
            "taxon": taxon,
            "match_score": score,
            "sample_count": loc_counts.get(tk) or sc or None,
            "sort_value": val,
            "location_count": loc_counts.get(tk) or None,
            "position": None,
            "percentile": None,
        })

    return {
        "total": total,
        "matched_total": len(candidates),
        "eligible_total": total,
        "empty_reason": None if results else "no_results",
        "results": results,
    }


def _query_text(
    *,
    q: str,
    within_taxon: TaxonRecord | None,
    descendant_rank: str | None,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    loc_keys: frozenset[str] | None,
    loc_counts: dict[str, int],
) -> dict:
    candidates = search_taxa_by_name(q, limit=max((limit + offset) * 5, 200))
    if not candidates:
        return _empty_result("no_text_matches")

    scope_keys: frozenset[str] | None = None
    if within_taxon is not None and descendant_rank is not None:
        scope_keys = _load_scope_keys(within_taxon, descendant_rank, include_species_like)

    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like) if descendant_rank else None

    filtered: list[tuple[TaxonRecord, float, int]] = []
    for taxon, score, _ in candidates:
        tk = str(taxon["taxon_key"])
        if scope_keys is not None and tk not in scope_keys:
            continue
        if loc_keys is not None and tk not in loc_keys:
            continue
        if accepted_ranks is not None and taxon.get("rank") not in accepted_ranks:
            continue
        sc = _infer_sample_count(TREE_ROOT / taxon["path"])
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        filtered.append((taxon, score, sc))

    total = len(filtered)
    page = filtered[offset:offset + limit]

    results = []
    for taxon, score, sc in page:
        tk = str(taxon["taxon_key"])
        results.append({
            "taxon": taxon,
            "match_score": score,
            "sample_count": loc_counts.get(tk) or sc or None,
            "sort_value": None,
            "location_count": loc_counts.get(tk) or None,
            "position": None,
            "percentile": None,
        })

    return {
        "total": total,
        "matched_total": len(candidates),
        "eligible_total": total,
        "empty_reason": None if results else ("no_text_matches" if not candidates else "no_results"),
        "results": results,
    }


def _load_scope_keys(
    within_taxon: TaxonRecord,
    descendant_rank: str,
    include_species_like: bool,
) -> frozenset[str]:
    """Return taxon_key set for all descendants of within_taxon at descendant_rank."""
    rank_lower = "subspecies" if descendant_rank in CONFIG.subspecies_equivalents else descendant_rank.lower()
    catalog_path = TREE_ROOT / within_taxon["path"] / f"{rank_lower}.parquet"
    if catalog_path.exists():
        try:
            tbl = pq.read_table(catalog_path, columns=["taxon_key"])
            return frozenset(str(k) for k in tbl.column("taxon_key").to_pylist())
        except Exception:
            pass
    # Fall back to live DFS if catalog is missing
    accepted_ranks_set: set[str] = {descendant_rank}
    if descendant_rank == CONFIG.species_rank and include_species_like:
        accepted_ranks_set |= set(CONFIG.subspecies_equivalents)
    return frozenset(
        str(t["taxon_key"])
        for t in iter_descendants(within_taxon, include_self=False)
        if (t.get("rank") or "") in accepted_ranks_set
    )


def _query_catalog(
    *,
    within_taxon: TaxonRecord,
    descendant_rank: str,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    loc_keys: frozenset[str] | None,
    loc_counts: dict[str, int],
) -> dict:
    rank_lower = "subspecies" if descendant_rank in CONFIG.subspecies_equivalents else descendant_rank.lower()
    catalog_path = TREE_ROOT / within_taxon["path"] / f"{rank_lower}.parquet"
    if not catalog_path.exists():
        return _empty_result("no_catalog")

    try:
        rows = pq.read_table(catalog_path).to_pylist()
    except Exception:
        return _empty_result("no_catalog")

    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like)

    filtered: list[tuple[TaxonRecord, int]] = []
    for row in rows:
        tk = str(row.get("taxon_key") or "")
        sc = int(row.get("sample_count") or 0)
        if loc_keys is not None and tk not in loc_keys:
            continue
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        taxon = get_taxon_by_id(tk)
        if taxon is None:
            continue
        if accepted_ranks is not None and taxon.get("rank") not in accepted_ranks:
            continue
        filtered.append((taxon, sc))

    total = len(filtered)
    page = filtered[offset:offset + limit]

    results = []
    for taxon, sc in page:
        tk = str(taxon["taxon_key"])
        results.append({
            "taxon": taxon,
            "match_score": None,
            "sample_count": loc_counts.get(tk) or sc or None,
            "sort_value": None,
            "location_count": loc_counts.get(tk) or None,
            "position": None,
            "percentile": None,
        })

    return {
        "total": total,
        "matched_total": total,
        "eligible_total": len(rows),
        "empty_reason": None if results else "no_results",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Public query entry point
# ---------------------------------------------------------------------------

def query_taxa(
    q: str | None,
    within_taxon: TaxonRecord | None,
    descendant_rank: str | None,
    sort_variable: str | None,
    sort_metric: str | None,
    sort_order: str,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    location_gid: str | None,
    reference_value: float | None = None,
    min_rbar: float | None = None,
) -> dict:
    """Search and rank taxa.

    Returns a dict with keys: total, matched_total, eligible_total, empty_reason, results.
    Each result has: taxon, match_score, sample_count, sort_value, location_count, position, percentile.

    ``reference_value`` and ``min_rbar`` are used when sorting by a circular bearing metric
    (circular_mean or mode): results are ordered by forward clockwise distance from
    reference_value, and taxa with rbar below min_rbar are excluded.
    """
    has_q = bool(q)
    has_scope = within_taxon is not None and bool(descendant_rank)
    has_sort = bool(sort_variable) and bool(sort_metric)

    loc_keys: frozenset[str] | None = None
    loc_counts: dict[str, int] = {}
    if location_gid:
        loc_keys, loc_counts = _location_taxon_keys(location_gid)

    if has_scope and has_sort:
        return _query_ranked_scoped(
            q=q, within_taxon=within_taxon, descendant_rank=descendant_rank,
            sort_variable=sort_variable, sort_metric=sort_metric,
            sort_order=sort_order, limit=limit, offset=offset,
            min_samples=min_samples, include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
            reference_value=reference_value, min_rbar=min_rbar,
        )
    if has_q and has_sort:
        return _query_ranked_text(
            q=q, sort_variable=sort_variable, sort_metric=sort_metric,
            sort_order=sort_order, limit=limit, offset=offset,
            min_samples=min_samples, include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
            reference_value=reference_value, min_rbar=min_rbar,
        )
    if has_q:
        return _query_text(
            q=q, within_taxon=within_taxon, descendant_rank=descendant_rank,
            limit=limit, offset=offset, min_samples=min_samples,
            include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
        )
    if has_scope:
        return _query_catalog(
            within_taxon=within_taxon, descendant_rank=descendant_rank,
            limit=limit, offset=offset, min_samples=min_samples,
            include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
        )
    return _empty_result("no_query")
