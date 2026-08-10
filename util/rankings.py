# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
import gc
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.config import METRICS_BY_TYPE, ValueType, load_config
from util.stats import (
    CIRCULAR_STATS_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
)
from util.storage import ParquetStorageProxy
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants, search_taxa_by_name

_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent.parent,
)

# Module-level cache populated by preload_stats_cache().
# Format: taxon_key → (sample_count: int, values: np.ndarray float32)
# values is indexed by _metric_vocab; NaN means metric not present for this taxon.
# ~1.5GB total vs ~8GB if Python float dicts were used.
_stats_cache: dict[str, tuple[int, np.ndarray]] | None = None
_metric_vocab: list[str] = []       # sorted list of all metric keys
_metric_to_idx: dict[str, int] = {} # reverse lookup
_rankings_mask: np.ndarray | None = None  # shape (len(_metric_vocab),), dtype bool; None = include all

def _build_rankings_mask() -> None:
    global _rankings_mask
    _rankings_mask = None  # include all metrics (temporal vars included)

CONFIG = load_config("global")

TREE_ROOT = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "tree"
_CACHE_FILE = TREE_ROOT.parent / "stats_cache.pkl.gz"
POSITION_FILE = "relative_ranks_positions.parquet"
RANKINGS_FILE = "relative_rankings.parquet"
# Same underlying rows as POSITION_FILE (one taxon's position within one ancestor
# context), just re-sorted by (contextTaxonId, rank, variable, metric, position)
# instead of (taxon_key, variable) — that sort order lets a scoped ranking browse
# query (e.g. "species under Cactaceae, sorted by bio1 mean, page 3") prune
# straight to the matching row group(s) and read rows already in position order,
# instead of reading/sorting the whole context's data at request time.

# Canonical taxonomy rank order used to determine descendant catalog targets.
_RANK_ORDER: tuple[str, ...] = (
    "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES", "SUBSPECIES",
)

MIN_RANKING_SAMPLES = 10  # taxa with fewer observations for a variable are excluded from rankings


_POSITION_SCHEMA = pa.schema([
    pa.field("taxon_key", pa.large_string()),
    pa.field("variable", pa.large_string()),
    pa.field("metric", pa.large_string()),
    pa.field("value", pa.float64()),
    pa.field("position", pa.int32()),
    pa.field("count", pa.int32()),
    pa.field("sampleCount", pa.int32()),
    pa.field("contextTaxonId", pa.large_string()),
    pa.field("rank", pa.large_string()),
    pa.field("contextLabel", pa.large_string()),
])

_STRUCT_FIELDS = [
    pa.field("taxonKey", pa.large_string()),
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
# excluded from relative_ranks_positions.parquet for circular variables only.
_CIRCULAR_ANGULAR_METRICS: frozenset[str] = frozenset({"circular_mean", "mode"})

# Metrics excluded from ranking for nominal/ordinal variables.
# Ordinal percentiles are ordinal-scale class IDs — numeric ordering is meaningful
# within a variable but not comparable across taxa.  Mode is a class ID too.
_NOMINAL_SKIP_RANK_METRICS: frozenset[str] = frozenset({"mode"})
_ORDINAL_SKIP_RANK_METRICS: frozenset[str] = frozenset({
    "mode",
    "10th_percentile", "25th_percentile", "median", "75th_percentile", "90th_percentile",
})


def _metrics_for_vtype(layer: dict, vtype: ValueType) -> tuple[str, ...]:
    """Return rankable metric names for a value type.

    Returns () for types with no ranking metrics (AGGREGATE, etc.).
    """
    match vtype:
        case ValueType.RATIO | ValueType.INTERVAL:
            return METRICS_BY_TYPE[vtype]
        case ValueType.NOMINAL:
            return METRICS_BY_TYPE[ValueType.NOMINAL]
        case ValueType.ORDINAL:
            return METRICS_BY_TYPE[ValueType.ORDINAL]
        case ValueType.CIRCULAR:
            return METRICS_BY_TYPE[ValueType.CIRCULAR]
        case _:
            return ()


def _wide_seen_and_counts(
    path: Path, kind: str, ids: set[str],
) -> tuple[set[str], pd.Series | None]:
    """Cheap structural pass over a wide-format stats file (numerical or
    circular): which taxa have >=1 row for a variable in `ids`, and —
    numerical only — each taxon's sample count from the `count` column.

    Reads only taxon_key/variable/(count) via Arrow column selection, never
    the metric-value columns — those are the bulk of numerical_stats.parquet
    (~1.8GB on disk for the real catalog) and aren't needed for this pass.
    """
    if not path.exists():
        return set(), None
    tbl = pq.read_table(path)
    col_names = set(tbl.schema.names)
    if "taxon_key" not in col_names or "variable" not in col_names:
        return set(), None
    select_cols = ["taxon_key", "variable"]
    has_count = kind == "numerical" and "count" in col_names
    if has_count:
        select_cols.append("count")
    df = tbl.select(select_cols).to_pandas()
    df = df[df["variable"].isin(ids)]
    if df.empty:
        return set(), None
    seen = set(df["taxon_key"].unique())
    sample_counts = None
    if has_count:
        cnt = df.loc[df["count"].fillna(0) > 0, ["taxon_key", "count"]]
        if not cnt.empty:
            sample_counts = cnt.groupby("taxon_key")["count"].first().astype("int64")
    return seen, sample_counts


def _wide_fill(
    values: np.ndarray, taxon_pos: pd.Series, metric_pos: pd.Series,
    path: Path, kind: str, ids: set[str], metrics_by_var: dict[str, tuple[str, ...]] | None,
    circ_metrics: tuple[str, ...] = (),
) -> None:
    """Fill `values` (shape n_taxa x n_metrics) from a wide-format stats
    file, one metric COLUMN at a time via vectorized numpy/pandas ops.

    Deliberately does *not* reshape the file into a taxon*metric "long"
    table (melt): that reshape multiplies row count by metric count, and on
    the real numerical_stats.parquet (~1.8GB, many metric columns) that
    intermediate ended up *larger* than the old per-row Python-dict
    accumulation it was meant to replace — confirmed by an actual rebuild
    OOMing faster than before. Filling column-by-column keeps every
    intermediate proportional to the file's own row count, never multiplied
    by metric count, and only pulls in the metric columns this call needs
    (via Arrow column selection) rather than the whole table.
    """
    if not path.exists():
        return
    tbl = pq.read_table(path)
    col_names = set(tbl.schema.names)
    if "taxon_key" not in col_names:
        return

    if kind == "numerical":
        needed_metrics: set[str] = set()
        for var, metrics in (metrics_by_var or {}).items():
            if var in ids:
                needed_metrics.update(metrics)
        metric_cols = [m for m in needed_metrics if m in col_names]
    else:
        metric_cols = [m for m in circ_metrics if m in col_names]
    if not metric_cols:
        return

    df = tbl.select(["taxon_key", "variable", *metric_cols]).to_pandas()
    df = df[df["variable"].isin(ids)]
    if df.empty:
        return
    variable = df["variable"].astype(str).to_numpy()
    row_pos_all = taxon_pos.reindex(df["taxon_key"]).to_numpy()

    for m in metric_cols:
        col_vals = pd.to_numeric(df[m], errors="coerce").to_numpy()
        finite = np.isfinite(col_vals)
        if not finite.any():
            continue
        if kind == "numerical":
            # Metrics can in principle differ by variable (e.g. RATIO vs
            # INTERVAL) — only assign where this metric actually applies to
            # that row's variable.
            vars_using_m = np.array(sorted(
                v for v, metrics in (metrics_by_var or {}).items() if m in metrics
            ))
            applies = np.isin(variable[finite], vars_using_m)
            sel = np.where(finite)[0][applies]
        else:
            sel = np.where(finite)[0]
        if sel.size == 0:
            continue
        metric_keys = np.char.add(np.char.add(variable[sel].astype(str), "::"), m)
        c = metric_pos.reindex(metric_keys).to_numpy()
        valid = ~np.isnan(c)
        if not valid.any():
            continue
        r = row_pos_all[sel][valid]
        cc = c[valid].astype(np.int64)
        values[r, cc] = col_vals[sel][valid]


def _tall_seen_counts_and_vocab(
    path: Path, ids: set[str], metrics: set[str],
) -> tuple[set[str], pd.Series | None, set[str]]:
    """Structural + sample-count pass over a tall-format stats file
    (nominal/ordinal). Unlike the wide-format files, these are already
    one-value-per-row and much smaller in practice (tens of MB vs
    numerical's ~1.8GB), so reading variable+metric+value together here is
    fine — no separate cheap/heavy split needed, and no melt is involved
    since the file is already in long form."""
    if not path.exists():
        return set(), None, set()
    tbl = pq.read_table(path)
    if "taxon_key" not in tbl.schema.names:
        return set(), None, set()
    df = tbl.to_pandas()
    variable = df["variable"].astype(str)
    metric = df["metric"].astype(str)
    mask = variable.isin(ids) & (metric.isin(metrics) | metric.str.startswith("class_"))
    df = df.loc[mask]
    if df.empty:
        return set(), None, set()
    variable = variable.loc[mask]
    metric = metric.loc[mask]
    seen = set(df["taxon_key"].unique())
    vocab = set((variable + "::" + metric).unique())

    # First non-zero total_samples row per taxon, in file row order — same
    # "only set once" semantics as the old entry["__sample_count__"] == 0 guard.
    sample_counts = None
    ts_mask = (metric == "total_samples").to_numpy()
    if ts_mask.any():
        ts_val = pd.to_numeric(df.loc[ts_mask, "value"], errors="coerce").fillna(0)
        nonzero = ts_val > 0
        if nonzero.any():
            sample_counts = (
                pd.Series(ts_val[nonzero].to_numpy(), index=df.loc[ts_mask, "taxon_key"].to_numpy()[nonzero.to_numpy()])
                .groupby(level=0).first().astype("int64")
            )
    return seen, sample_counts, vocab


def _tall_fill(values: np.ndarray, taxon_pos: pd.Series, metric_pos: pd.Series,
               path: Path, ids: set[str], metrics: set[str]) -> None:
    """Fill `values` from a tall-format stats file. No melt needed (already
    long-form); see _tall_seen_counts_and_vocab for why a single full read
    is fine for these smaller files."""
    if not path.exists():
        return
    tbl = pq.read_table(path)
    if "taxon_key" not in tbl.schema.names:
        return
    df = tbl.to_pandas()
    variable = df["variable"].astype(str)
    metric = df["metric"].astype(str)
    mask = variable.isin(ids) & (metric.isin(metrics) | metric.str.startswith("class_"))
    df = df.loc[mask]
    if df.empty:
        return
    variable = variable.loc[mask]
    metric = metric.loc[mask]
    value = pd.to_numeric(df["value"], errors="coerce")
    finite = np.isfinite(value)
    if not finite.any():
        return
    metric_key = (variable[finite] + "::" + metric[finite]).to_numpy()
    r = taxon_pos.reindex(df.loc[finite, "taxon_key"]).to_numpy()
    c = metric_pos.reindex(metric_key).to_numpy()
    valid = ~np.isnan(c)
    if not valid.any():
        return
    values[r[valid], c[valid].astype(np.int64)] = value[finite].to_numpy(dtype=np.float32)[valid]


def preload_stats_cache(layers: list[dict]) -> None:
    """Read the (already-consolidated) global stats files once and populate
    the module-level cache.

    Call this before the rankings pass so every lookup is an O(1) dict access
    instead of a disk read. Since scripts/process_tree.py::run_stats() now
    streams stats straight into one sorted global file per type instead of
    thousands of per-taxon files, this is four bulk column reads instead of
    a thread pool scanning one file per taxon.
    """
    import time as _time

    global _stats_cache
    _stats_cache = {}
    layer_by_id = {lay["id"]: lay for lay in layers}
    nominal_ids = {lay["id"] for lay in layers if lay.get("value_type") == ValueType.NOMINAL}
    nominal_metrics = set(METRICS_BY_TYPE[ValueType.NOMINAL]) - _NOMINAL_SKIP_RANK_METRICS
    ordinal_ids = {lay["id"] for lay in layers if lay.get("value_type") == ValueType.ORDINAL}
    ordinal_metrics = set(METRICS_BY_TYPE[ValueType.ORDINAL]) - _ORDINAL_SKIP_RANK_METRICS

    ratio_interval_ids: set[str] = set()
    circular_ids: set[str] = set()
    layer_metrics: dict[str, tuple[str, ...]] = {}
    for lid, lay in layer_by_id.items():
        try:
            vtype = ValueType(lay.get("value_type", ""))
        except ValueError:
            continue
        if vtype in (ValueType.RATIO, ValueType.INTERVAL):
            ratio_interval_ids.add(lid)
            layer_metrics[lid] = _metrics_for_vtype(lay, vtype)
        elif vtype == ValueType.CIRCULAR:
            circular_ids.add(lid)
    circ_metrics = METRICS_BY_TYPE[ValueType.CIRCULAR]

    import gzip as _gzip
    import pickle as _pickle

    # Fast path: load from disk cache if it exists
    if _CACHE_FILE.exists():
        print(f"[rankings] loading stats cache from disk ({_CACHE_FILE.name})...")
        t0 = _time.monotonic()
        try:
            with _gzip.open(_CACHE_FILE, "rb") as f:
                saved = _pickle.load(f)
            _metric_vocab[:] = saved["vocab"]
            _metric_to_idx.update({k: i for i, k in enumerate(_metric_vocab)})
            _stats_cache.update(saved["cache"])
            _build_rankings_mask()
            print(f"[rankings] cache loaded from disk: {len(_stats_cache):,} taxa  [{_time.monotonic()-t0:.1f}s]")
            return
        except Exception as e:
            print(f"[rankings] disk cache load failed ({e}), rebuilding...")
            _stats_cache.clear()
            _metric_vocab.clear()
            _metric_to_idx.clear()

    t0 = _time.monotonic()
    print("[rankings] preloading stats cache from global stats files...")
    numerical_path = GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE
    nominal_path = GLOBAL_STATS_DIR / NOMINAL_STATS_FILE
    ordinal_path = GLOBAL_STATS_DIR / ORDINAL_STATS_FILE
    circular_path = GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE

    # Pass 1 (cheap): seen-taxa + sample counts via lightweight column
    # selections — never touches the wide files' metric-value columns, the
    # bulk of their size. Numerical/circular vocab is config-driven (every
    # variable::metric the layer catalog says is possible for that value
    # type), not data-driven, so it costs nothing to compute; nominal/
    # ordinal vocab still needs the (cheap, small-file) "class_N" scan since
    # those labels are data-dependent.
    seen_num, sc_num = _wide_seen_and_counts(numerical_path, "numerical", ratio_interval_ids)
    seen_circ, _ = _wide_seen_and_counts(circular_path, "circular", circular_ids)
    seen_nom, sc_nom, vocab_nom = _tall_seen_counts_and_vocab(nominal_path, nominal_ids, nominal_metrics)
    seen_ord, sc_ord, vocab_ord = _tall_seen_counts_and_vocab(ordinal_path, ordinal_ids, ordinal_metrics)

    vocab_num = {f"{v}::{m}" for v, metrics in layer_metrics.items() for m in metrics}
    vocab_circ = {f"{v}::{m}" for v in circular_ids for m in circ_metrics}
    _metric_vocab[:] = sorted(vocab_num | vocab_circ | vocab_nom | vocab_ord)
    _metric_to_idx.update({k: i for i, k in enumerate(_metric_vocab)})
    _build_rankings_mask()
    n_metrics = len(_metric_vocab)

    sample_counts: pd.Series | None = None
    for sc in (sc_num, sc_nom, sc_ord):  # priority order: numerical, nominal, ordinal
        if sc is not None:
            sample_counts = sc if sample_counts is None else sample_counts.combine_first(sc)

    taxon_union = seen_num | seen_circ | seen_nom | seen_ord
    if sample_counts is not None:
        taxon_union |= set(sample_counts.index)
    row_idx = pd.Index(sorted(taxon_union))

    values = np.full((len(row_idx), n_metrics), np.nan, dtype=np.float32)
    if n_metrics and len(row_idx):
        taxon_pos = pd.Series(np.arange(len(row_idx)), index=row_idx)
        metric_pos = pd.Series(np.arange(n_metrics), index=_metric_vocab)
        # Pass 2: fill the matrix directly, one metric column at a time —
        # see _wide_fill's docstring for why this replaced an earlier
        # melt-based version that made memory use worse, not better.
        _wide_fill(values, taxon_pos, metric_pos, numerical_path, "numerical", ratio_interval_ids, layer_metrics)
        _tall_fill(values, taxon_pos, metric_pos, nominal_path, nominal_ids, nominal_metrics)
        _tall_fill(values, taxon_pos, metric_pos, ordinal_path, ordinal_ids, ordinal_metrics)
        _wide_fill(values, taxon_pos, metric_pos, circular_path, "circular", circular_ids, None,
                   circ_metrics=circ_metrics)

    sc_full = (
        sample_counts.reindex(row_idx).fillna(0).astype("int64")
        if sample_counts is not None else pd.Series(0, index=row_idx, dtype="int64")
    )
    sc_values = sc_full.to_numpy()
    # values[i] is a view into the shared `values` matrix, not a per-taxon
    # copy — one contiguous allocation for all taxa instead of ~213k small
    # numpy allocations, and no further mutation happens after this point.
    for i, taxon_key in enumerate(row_idx):
        _stats_cache[taxon_key] = (int(sc_values[i]), values[i])

    elapsed = _time.monotonic() - t0
    print(f"[rankings] stats cache ready: {len(_stats_cache):,} taxa  {n_metrics:,} metrics  [{elapsed:.1f}s]")

    # Phase 3: persist to disk for fast restart
    print("[rankings] saving cache to disk...")
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        with _gzip.open(tmp, "wb", compresslevel=1) as f:
            _pickle.dump({"vocab": list(_metric_vocab), "cache": dict(_stats_cache)}, f, protocol=5)
        tmp.replace(_CACHE_FILE)
        print(f"[rankings] cache saved ({_CACHE_FILE.stat().st_size / 1e9:.2f}GB)  [{_time.monotonic()-t0:.1f}s total]")
    except Exception as e:
        print(f"[rankings] cache save failed (non-fatal): {e}")


def _batch_sample_counts(taxon_keys: list[str]) -> dict[str, int]:
    """Approximate each taxon's observation count as the first available
    per-variable 'count' from the global numerical_stats.parquet, falling
    back to nominal_stats.parquet's total_samples row for taxa with no
    numerical data — same approximation this always used, just as one or two
    batched filtered reads against the consolidated files instead of one
    per-taxon-directory file (those stopped existing once stats got
    consolidated straight into the global files)."""
    if not taxon_keys:
        return {}
    result: dict[str, int] = {}
    try:
        tbl = _storage.read_table(
            GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE,
            columns=["taxon_key", "count"],
            filters=[("taxon_key", "in", taxon_keys)],
        )
        for tk, val in zip(tbl.column("taxon_key").to_pylist(), tbl.column("count").to_pylist()):
            tk = str(tk)
            if tk in result or val is None:
                continue
            n = int(val)
            if n > 0:
                result[tk] = n
    except Exception:
        pass
    missing = [tk for tk in taxon_keys if tk not in result]
    if missing:
        try:
            tbl = _storage.read_table(
                GLOBAL_STATS_DIR / NOMINAL_STATS_FILE,
                columns=["taxon_key", "metric", "value"],
                filters=[("taxon_key", "in", missing), ("metric", "=", "total_samples")],
            )
            for tk, val in zip(tbl.column("taxon_key").to_pylist(), tbl.column("value").to_pylist()):
                tk = str(tk)
                if tk not in result and val is not None:
                    result[tk] = int(float(val))
        except Exception:
            pass
    return result


def _batch_metric_values(
    taxon_keys: list[str], variable: str, metric: str, *, need_rbar: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Look up one variable::metric value per taxon_key, batched across the
    global stats files that might hold it (numerical, nominal, circular —
    same fallback order a single-taxon lookup always tried), instead of one
    per-taxon-directory read per candidate.

    class_{id} metrics: a taxon missing from nominal_stats for that class has
    zero presence, not missing data (util/stats.py only writes nonzero class
    rows) — only treat it as truly absent if it also has no total_samples row.

    Returns (values, rbar_values); rbar_values is empty unless need_rbar.
    """
    if not taxon_keys:
        return {}, {}
    values: dict[str, float] = {}
    rbars: dict[str, float] = {}

    try:
        tbl = _storage.read_table(
            GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE,
            columns=["taxon_key", "variable", metric],
            filters=[("taxon_key", "in", taxon_keys), ("variable", "=", variable)],
        )
        for tk, val in zip(tbl.column("taxon_key").to_pylist(), tbl.column(metric).to_pylist()):
            if val is not None and _safe_finite(val):
                values[str(tk)] = float(val)
    except Exception:
        pass

    has_total: set[str] = set()
    try:
        want_metrics = [metric, "total_samples"]
        tbl = _storage.read_table(
            GLOBAL_STATS_DIR / NOMINAL_STATS_FILE,
            columns=["taxon_key", "variable", "metric", "value"],
            filters=[("taxon_key", "in", taxon_keys), ("variable", "=", variable), ("metric", "in", want_metrics)],
        )
        for tk, m, val in zip(
            tbl.column("taxon_key").to_pylist(), tbl.column("metric").to_pylist(), tbl.column("value").to_pylist(),
        ):
            tk = str(tk)
            if m == "total_samples":
                has_total.add(tk)
            elif val is not None and _safe_finite(val):
                values[tk] = float(val)
    except Exception:
        pass
    for tk in has_total:
        values.setdefault(tk, 0.0)

    try:
        columns = ["taxon_key", "variable", metric] + (["rbar"] if need_rbar and metric != "rbar" else [])
        tbl = _storage.read_table(
            GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE,
            columns=columns,
            filters=[("taxon_key", "in", taxon_keys), ("variable", "=", variable)],
        )
        for tk, val in zip(tbl.column("taxon_key").to_pylist(), tbl.column(metric).to_pylist()):
            tk = str(tk)
            if tk not in values and val is not None and _safe_finite(val):
                values[tk] = float(val)
        if need_rbar and "rbar" in tbl.column_names:
            for tk, val in zip(tbl.column("taxon_key").to_pylist(), tbl.column("rbar").to_pylist()):
                if val is not None:
                    rbars[str(tk)] = float(val)
    except Exception:
        pass

    return values, rbars


# ---------------------------------------------------------------------------
# Rank index build
# ---------------------------------------------------------------------------

def _descendants_for_rank(ancestor: TaxonRecord, rank: str) -> list[TaxonRecord]:
    """Return descendant taxa to include in a rank index, respecting species/subspecies combining."""
    ancestor_rank = ancestor.get("rank") or ""
    equiv = frozenset(CONFIG.subspecies_equivalents)
    species_rank = CONFIG.species_rank

    if rank == "SUBSPECIES":
        if ancestor_rank != species_rank:
            return []
        target_ranks = equiv
    elif rank == "SPECIES" and ancestor_rank not in (species_rank, *equiv):
        target_ranks = {species_rank} | equiv
    else:
        target_ranks = {rank}

    return [
        t for t in iter_descendants(ancestor, include_self=False)
        if (t.get("rank") or "").upper() in target_ranks
    ]


# ---------------------------------------------------------------------------
# Rank index
# ---------------------------------------------------------------------------

class RankingsSink:
    """Buffers one tree-depth level's computed ranking-position rows into a
    single shared staged parquet chunk (one streaming ParquetWriter, opened on
    first write and closed when the level finishes) — mirrors
    util.stats.StatsSink, just for the single position-row shape produced
    here instead of stats' several kinds. Replaces writing one
    {rank}_positions.parquet file per ancestor directory: scripts/process_tree.py
    creates one sink per level and sorts all levels' chunks into the two final
    global rankings files once the whole rankings pass completes (see
    _finalize_rankings).
    """

    def __init__(self, staging_dir: Path, level_id: str):
        self._path = staging_dir / f"{level_id}.parquet"
        self._writer: pq.ParquetWriter | None = None
        self._lock = threading.Lock()

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        with self._lock:
            if self._writer is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = pq.ParquetWriter(self._path, table.schema)
            self._writer.write_table(table)

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None


def _write_rank_positions(
    ancestor: TaxonRecord,
    rank: str,
    layers: list[dict],
    sink: RankingsSink,
    circular_ids: frozenset[str] | None = None,
) -> None:
    """Collect per-taxon metrics for all descendants of rank and stream a
    per-(ancestor,rank) positions row set into sink — one row per (variable, metric, taxon),
    already sorted by value with position/percentile-denominator baked in.

    This is always called with _stats_cache populated (preload_stats_cache()
    runs once before the whole rankings pass in scripts/process_tree.py, the
    only caller) — there's deliberately no per-taxon-directory disk-read
    fallback: that used to exist for a cache-miss case, but per-taxon stats
    files haven't existed since stats got consolidated straight into the
    global files, so a fallback here would just silently read nothing.
    """
    _circular_ids: frozenset[str] = circular_ids if circular_ids is not None else frozenset()
    descendants = _descendants_for_rank(ancestor, rank)
    if not descendants or _stats_cache is None:
        return

    # Collect lightweight (taxon_key, cached_dict) pairs — just references into _stats_cache,
    # no data copying. Then process one column at a time so only one Python list of entries
    # is alive at a time; each is immediately converted to a compact Arrow array (C memory)
    # and the Python list is discarded. This keeps peak Python heap usage to ~one column's
    # worth of data regardless of how many descendants or metrics there are.
    desc_data: list[tuple[str, tuple]] = []
    for t in descendants:
        taxon_key = str(t["taxon_key"])
        entry = _stats_cache.get(taxon_key)
        if entry is not None:
            desc_data.append((taxon_key, entry))

    if not desc_data:
        return

    all_taxon_keys = [tk for tk, _ in desc_data]
    all_sample_counts = np.array([e[0] for _, e in desc_data], dtype=np.int64)

    # Single pass using numpy: for each taxon, find non-NaN metric indices and
    # append (position, value) to per-column lists. No Python float objects created
    # for cached values — they stay as float32 in the numpy array until appended.
    col_idx: dict[str, list[int]] = {}
    col_val: dict[str, list[float]] = {}
    vocab = _metric_vocab
    include = _rankings_mask  # None = include all metrics
    for i, (_, entry) in enumerate(desc_data):
        values_arr = entry[1]
        active = ~np.isnan(values_arr)
        if include is not None:
            active &= include
        for metric_idx in np.where(active)[0]:
            k = vocab[metric_idx]
            # Check per-variable sample count threshold using {variable}::count
            # (continuous/circular) or {variable}::total_samples (nominal).
            variable = k.split("::")[0]
            count_idx = _metric_to_idx.get(f"{variable}::count")
            if count_idx is None:
                count_idx = _metric_to_idx.get(f"{variable}::total_samples")
            if count_idx is not None:
                var_count = values_arr[count_idx]
                if np.isnan(var_count) or int(var_count) < MIN_RANKING_SAMPLES:
                    continue
            v = float(values_arr[metric_idx])
            if k in col_idx:
                col_idx[k].append(i)
                col_val[k].append(v)
            else:
                col_idx[k] = [i]
                col_val[k] = [v]

    context_taxon_id = str(ancestor["taxon_key"])
    context_label = _resolve_context_label(ancestor)

    # Positions accumulators — numpy chunks for ints (no Python int objects),
    # plain lists for strings (references to existing objects, no copies).
    pos_tks:  list[str] = []
    pos_vars: list[str] = []
    pos_mets: list[str] = []
    pos_val_chunks: list[np.ndarray] = []
    pos_pos_chunks: list[np.ndarray] = []
    pos_cnt_chunks: list[np.ndarray] = []
    pos_sc_chunks:  list[np.ndarray] = []

    # class_{id} columns only hold taxa with nonzero presence in that
    # class (see util/stats.py) — the true population for percentile
    # purposes is {variable}::total_samples, which still has an entry
    # for every taxon with any data for the variable. Precompute per
    # variable before the main loop pops entries out of col_idx below.
    total_samples_len: dict[str, int] = {}
    for key, vals in col_idx.items():
        var, metric = key.split("::", 1)
        if metric == "total_samples":
            total_samples_len[var] = len(vals)

    for col_key in sorted(col_idx):
        idx_list = col_idx.pop(col_key)
        val_list = col_val.pop(col_key)
        if not idx_list:
            continue
        val_np = np.array(val_list, dtype=np.float64)
        idx_np = np.array(idx_list, dtype=np.int32)
        del val_list, idx_list
        order = np.argsort(val_np, kind="stable")
        sorted_tks = [all_taxon_keys[i] for i in idx_np[order]]
        sorted_scs = all_sample_counts[idx_np[order]]
        sorted_vals = val_np[order]
        n = len(sorted_tks)
        if n == 0:
            continue

        # Vectorised min_rank_pos: tied values share the first position in their group.
        # (Meaningful as a value-ascending rank for every metric, including circular
        # bearings — those just get re-sorted live by clockwise distance from a
        # request-time reference angle, since that can't be precomputed once.)
        is_new = np.empty(n, dtype=bool)
        is_new[0] = True
        if n > 1:
            is_new[1:] = sorted_vals[1:] != sorted_vals[:-1]
        group_starts = np.where(is_new)[0]
        group_ids = np.cumsum(is_new) - 1
        min_rank_arr = group_starts[group_ids].astype(np.int32)

        variable, metric = col_key.split("::", 1)
        if metric.startswith("class_"):
            indices = np.where(sorted_vals != 0.0)[0]
            # Real entries only occupy the top of the true population
            # (every taxon missing from this class has an implicit
            # 0.0, which is always <= any real value here) — offset
            # position by however many implicit zeros exist, and use
            # the full population as the percentile denominator.
            full_n = total_samples_len.get(variable, n)
            offset = max(full_n - n, 0)
        else:
            indices = np.arange(n, dtype=np.int32)
            full_n = n
            offset = 0

        if len(indices):
            pos_tks.extend([sorted_tks[i] for i in indices])
            pos_vars.extend([variable] * len(indices))
            pos_mets.extend([metric] * len(indices))
            pos_val_chunks.append(sorted_vals[indices])
            pos_pos_chunks.append((min_rank_arr[indices] + offset).astype(np.int32))
            pos_cnt_chunks.append(np.full(len(indices), full_n, dtype=np.int32))
            pos_sc_chunks.append(sorted_scs[indices].astype(np.int32))

    # Stream this ancestor/rank's position rows into the level's shared sink.
    if pos_tks and pos_pos_chunks:
        all_val = np.concatenate(pos_val_chunks)
        all_pos = np.concatenate(pos_pos_chunks)
        all_cnt = np.concatenate(pos_cnt_chunks)
        all_sc  = np.concatenate(pos_sc_chunks)
        del pos_val_chunks, pos_pos_chunks, pos_cnt_chunks, pos_sc_chunks
        n_pos = len(pos_tks)
        pos_table = pa.table({
            "taxon_key":     pa.array(pos_tks,  type=pa.large_string()),
            "variable":      pa.array(pos_vars, type=pa.large_string()),
            "metric":        pa.array(pos_mets, type=pa.large_string()),
            "value":         pa.array(all_val, type=pa.float64()),
            "position":      pa.array(all_pos),
            "count":         pa.array(all_cnt),
            "sampleCount":   pa.array(all_sc),
            "contextTaxonId": pa.array([context_taxon_id] * n_pos, type=pa.large_string()),
            "rank":           pa.array([rank] * n_pos, type=pa.large_string()),
            "contextLabel":   pa.array([context_label]    * n_pos, type=pa.large_string()),
        }, schema=_POSITION_SCHEMA)
        del pos_tks, pos_vars, pos_mets, all_val, all_pos, all_cnt, all_sc
        sink.write(pos_table)

    gc.collect()
    pa.default_memory_pool().release_unused()


def build_rank_indexes(ancestor: TaxonRecord, layers: list[dict], sink: RankingsSink) -> None:
    """Stream this ancestor's descendant rank positions into sink."""
    ancestor_rank = ancestor.get("rank") or ""
    targets = _descendant_rank_targets(ancestor_rank)
    if not targets:
        return

    circular_ids: frozenset[str] = frozenset(
        lay["id"] for lay in layers
        if lay.get("value_type") == ValueType.CIRCULAR and lay.get("id")
    )
    for rank in targets:
        _write_rank_positions(ancestor, rank, layers, sink, circular_ids)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_relative_ranks(ancestor: TaxonRecord, layers: list[dict], sink: RankingsSink) -> None:
    """Compute descendant rank positions for one ancestor and stream them into
    sink. scripts/process_tree.py sorts each level's staged sink output into
    the two final global rankings files once the whole pass completes (see
    _finalize_rankings)."""
    build_rank_indexes(ancestor, layers, sink)


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
        tbl = _storage.read_table(
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


def _read_rank_positions(context_id: str, rank: str, variable: str, metric: str) -> list[dict]:
    """Read this (context, rank, variable, metric) group's rows from the global
    consolidated rankings file — one filtered read, pruned straight to the
    matching row group(s) since the file is physically sorted by
    (contextTaxonId, rank, variable, metric, position). Used for the sort
    metric itself, and equally for the {variable}::total_samples population
    (implicit-zero reconstruction) and {variable}::rbar lookups, since those
    are just rows with a different `metric` value in the same file."""
    path = GLOBAL_STATS_DIR / RANKINGS_FILE
    if not path.exists():
        return []
    try:
        tbl = _storage.read_table(
            path,
            columns=["taxon_key", "value", "position", "count", "sampleCount"],
            filters=[
                ("contextTaxonId", "=", context_id),
                ("rank", "=", rank),
                ("variable", "=", variable),
                ("metric", "=", metric),
            ],
        )
        return tbl.to_pylist()
    except Exception:
        return []


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


# ---------------------------------------------------------------------------
# Arbitrary stat filters — "at least 10 observations in ecoregion X",
# "avg temp < 25C", "60%+ shrubland cover", chained with scope/sort/text.
#
# Deliberately NOT raw SQL text from the client: every filter is a strict
# variable:metric:op:value tuple, validated against the known layer catalog,
# and applied via _storage.read_table's typed filters= (structured predicate
# pushdown, not a string-interpolated query) — there's no path for a client
# to inject arbitrary SQL.
# ---------------------------------------------------------------------------

class StatFilter(NamedTuple):
    variable: str
    metric: str
    op: str
    value: float
    as_count: bool = False  # nominal/ordinal class_{id} only — see _filter_tall


_FILTER_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


def parse_stat_filter(raw: str) -> StatFilter:
    """Parse one 'variable:metric:op:value[:count]' filter string.

    ':count' only applies to nominal/ordinal class_{id} metrics: their value
    is stored as a fraction of the taxon's total observations (see
    util/stats.py), so ':count' reconstructs and compares the raw
    observation count instead (fraction * total_samples).

    Raises ValueError on anything malformed — callers should turn that into
    a 4xx, not swallow it, since a silently-ignored filter would look like a
    query that matched nothing was actually just unfiltered.
    """
    parts = raw.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(f"malformed filter {raw!r}: expected variable:metric:op:value[:count]")
    variable, metric, op, value_str, *rest = parts
    if op not in _FILTER_OPS:
        raise ValueError(f"unknown filter operator {op!r} in {raw!r}")
    try:
        value = float(value_str)
    except ValueError as exc:
        raise ValueError(f"non-numeric filter value in {raw!r}") from exc
    as_count = bool(rest) and rest[0] == "count"
    if rest and not as_count:
        raise ValueError(f"unknown filter modifier {rest[0]!r} in {raw!r}")
    return StatFilter(variable=variable, metric=metric, op=op, value=value, as_count=as_count)


def _filter_value_type(variable: str, layers: list[dict]) -> ValueType | None:
    layer = next((lay for lay in layers if lay.get("id") == variable), None)
    if layer is None:
        return None
    try:
        return ValueType(layer.get("value_type", ""))
    except ValueError:
        return None


def _filter_wide(path: Path, keys: frozenset[str], f: StatFilter) -> frozenset[str]:
    """Numerical/circular metrics are stored wide (one column per metric) —
    a single filtered column read, scoped to the candidate keys, then a
    direct comparison."""
    if not keys or not path.exists():
        return frozenset()
    try:
        tbl = _storage.read_table(
            path, columns=["taxon_key", f.metric],
            filters=[("taxon_key", "in", list(keys)), ("variable", "=", f.variable)],
        )
    except Exception:
        return frozenset()
    op = _FILTER_OPS[f.op]
    return frozenset(
        str(tk) for tk, v in zip(tbl.column("taxon_key").to_pylist(), tbl.column(f.metric).to_pylist())
        if v is not None and op(float(v), f.value)
    )


def _filter_tall(path: Path, keys: frozenset[str], f: StatFilter) -> frozenset[str]:
    """Nominal/ordinal metrics are stored tall (taxon_key, metric, value
    rows). class_{id} rows only exist for taxa with nonzero presence in that
    class (see util/stats.py) — a taxon with a total_samples row but no row
    for this specific class has an implicit value of 0.0, not missing data,
    so total_samples is always fetched alongside to tell "zero" apart from
    "no data for this variable at all"."""
    if not keys or not path.exists():
        return frozenset()
    try:
        tbl = _storage.read_table(
            path, columns=["taxon_key", "metric", "value"],
            filters=[
                ("taxon_key", "in", list(keys)), ("variable", "=", f.variable),
                ("metric", "in", [f.metric, "total_samples"]),
            ],
        )
    except Exception:
        return frozenset()
    values: dict[str, float] = {}
    totals: dict[str, float] = {}
    for tk, m, v in zip(
        tbl.column("taxon_key").to_pylist(), tbl.column("metric").to_pylist(), tbl.column("value").to_pylist(),
    ):
        if v is None:
            continue
        tk = str(tk)
        if m == "total_samples":
            totals[tk] = float(v)
        elif m == f.metric:
            values[tk] = float(v)
    op = _FILTER_OPS[f.op]
    result = set()
    for tk in keys:
        if tk not in totals and tk not in values:
            continue  # no data for this variable at all
        raw_value = values.get(tk, 0.0)  # implicit zero — see docstring
        compare_value = round(raw_value * totals.get(tk, 0.0)) if f.as_count else raw_value
        if op(compare_value, f.value):
            result.add(tk)
    return frozenset(result)


def _apply_stat_filters(candidate_keys: frozenset[str], filters: list[StatFilter], layers: list[dict]) -> frozenset[str]:
    """Narrow candidate_keys by ANDing every filter — one batched read per
    filter (not one read per taxon), scoped to whatever candidates already
    survived scope/text-search narrowing, so this stays cheap regardless of
    how large the underlying tree is."""
    keys = candidate_keys
    for f in filters:
        if not keys:
            return keys
        vtype = _filter_value_type(f.variable, layers)
        if vtype in (ValueType.RATIO, ValueType.INTERVAL):
            keys = _filter_wide(GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE, keys, f)
        elif vtype == ValueType.CIRCULAR:
            keys = _filter_wide(GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE, keys, f)
        elif vtype == ValueType.NOMINAL:
            keys = _filter_tall(GLOBAL_STATS_DIR / NOMINAL_STATS_FILE, keys, f)
        elif vtype == ValueType.ORDINAL:
            keys = _filter_tall(GLOBAL_STATS_DIR / ORDINAL_STATS_FILE, keys, f)
        else:
            return frozenset()  # unknown variable — can't match anything
    return keys


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
    stat_filters: list[StatFilter] | None = None,
    layers: list[dict] | None = None,
) -> dict:
    rank_key = "SUBSPECIES" if descendant_rank in CONFIG.subspecies_equivalents else descendant_rank
    context_id = str(within_taxon["taxon_key"])

    entries = _read_rank_positions(context_id, rank_key, sort_variable, sort_metric)
    if not entries:
        return _empty_result("no_column")

    # Build reverse map: taxon_key → (raw_position, value, sample_count).
    # `count` is the same for every row in this (context, rank, variable,
    # metric) group — it's this sort's full eligible population, computed
    # once at build time (see util/rankings.py::_write_rank_positions).
    index_map: dict[str, tuple[int, float, int]] = {}
    full_population = 0
    for entry in entries:
        tk = str(entry.get("taxon_key") or "")
        if tk:
            index_map[tk] = (int(entry["position"]), float(entry.get("value") or 0.0), int(entry.get("sampleCount") or 0))
            full_population = int(entry["count"])

    # class_{id} metrics are no longer zero-expanded at write time (see
    # util/stats.py _nominal_cat_entries) — the rankings file, and thus this
    # group, only has rows for taxa with real (nonzero) presence in the
    # class; full_population above already accounts for the implicit-zero
    # taxa (see _write_rank_positions), but we still need their actual
    # taxon_keys here to synthesize zero-value entries for pagination.
    is_class_metric = sort_metric.startswith("class_")
    implicit_zero: dict[str, int] = {}  # taxon_key → sample_count, value implicitly 0.0
    if is_class_metric:
        for entry in _read_rank_positions(context_id, rank_key, sort_variable, "total_samples"):
            tk = str(entry.get("taxon_key") or "")
            if tk and tk not in index_map:
                implicit_zero[tk] = int(entry.get("sampleCount") or 0)

    eligible_keys = frozenset(index_map) | frozenset(implicit_zero)

    stat_filter_keys: frozenset[str] | None = None
    if stat_filters:
        stat_filter_keys = _apply_stat_filters(eligible_keys, stat_filters, layers or [])

    # Mode 3: restrict to text-matched taxon keys
    candidate_keys: frozenset[str] | None = None
    match_scores: dict[str, float] = {}
    match_names: dict[str, str] = {}
    if q:
        text_matches = search_taxa_by_name(q, limit=max(limit * 10, 200))
        candidate_keys = frozenset(str(t["taxon_key"]) for t, _, _ in text_matches if str(t["taxon_key"]) in eligible_keys)
        match_scores = {str(t["taxon_key"]): score for t, score, _ in text_matches}
        match_names = {str(t["taxon_key"]): name for t, _, name in text_matches}

    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like)

    is_circular_bearing = sort_metric in _CIRCULAR_ANGULAR_METRICS and reference_value is not None

    # For circular sorts, optionally load rbar values for min_rbar filtering
    rbar_map: dict[str, float] = {}
    if is_circular_bearing and min_rbar is not None:
        for entry in _read_rank_positions(context_id, rank_key, sort_variable, "rbar"):
            tk = str(entry.get("taxon_key") or "")
            if tk:
                rbar_map[tk] = float(entry.get("value") or 0.0)

    # Filter — real (nonzero) entries first, then implicit-zero entries.
    # raw_pos for real entries is offset by the implicit-zero count so it
    # still reflects true ascending rank within the full population (used
    # for percentile below); implicit-zero entries occupy the remaining
    # low end of that range, tiebroken by taxon_key for stable pagination
    # since they're all tied at 0.0 with no other meaningful order.
    implicit_count = len(implicit_zero)
    filtered: list[tuple[int, str, float, int]] = []  # (raw_pos, taxon_key, value, sample_count)
    for tk, (pos, val, sc) in index_map.items():
        if candidate_keys is not None and tk not in candidate_keys:
            continue
        if loc_keys is not None and tk not in loc_keys:
            continue
        if stat_filter_keys is not None and tk not in stat_filter_keys:
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
        filtered.append((implicit_count + pos, tk, val, sc))

    for local_idx, tk in enumerate(sorted(implicit_zero)):
        if candidate_keys is not None and tk not in candidate_keys:
            continue
        if loc_keys is not None and tk not in loc_keys:
            continue
        if stat_filter_keys is not None and tk not in stat_filter_keys:
            continue
        sc = implicit_zero[tk]
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        if accepted_ranks is not None:
            taxon = get_taxon_by_id(tk)
            if taxon is None or taxon.get("rank") not in accepted_ranks:
                continue
        filtered.append((local_idx, tk, 0.0, sc))

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
        percentile = round(raw_pos / full_population * 100, 3) if full_population > 0 else None
        results.append({
            "taxon": taxon,
            "match_score": match_scores.get(tk),
            "match_name": match_names.get(tk),
            "sample_count": loc_counts.get(tk) or sc or None,
            "sort_value": val,
            "location_count": loc_counts.get(tk) or None,
            "position": raw_pos + 1,
            "percentile": percentile,
        })

    return {
        "total": total,
        "matched_total": total,
        "eligible_total": full_population,
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
    stat_filters: list[StatFilter] | None = None,
    layers: list[dict] | None = None,
) -> dict:
    candidates = search_taxa_by_name(q, limit=max((limit + offset) * 5, 200))
    if not candidates:
        return _empty_result("no_text_matches")

    is_circular_bearing = sort_metric in _CIRCULAR_ANGULAR_METRICS and reference_value is not None
    need_rbar = is_circular_bearing and min_rbar is not None

    candidate_keys = [str(t["taxon_key"]) for t, _, _ in candidates]
    values, rbars = _batch_metric_values(candidate_keys, sort_variable, sort_metric, need_rbar=need_rbar)
    sample_counts = _batch_sample_counts(candidate_keys)
    stat_filter_keys: frozenset[str] | None = None
    if stat_filters:
        stat_filter_keys = _apply_stat_filters(frozenset(candidate_keys), stat_filters, layers or [])

    enriched: list[tuple[TaxonRecord, float, float, int, str]] = []  # taxon, score, sort_val, sc, match_name
    for taxon, score, match_name in candidates:
        tk = str(taxon["taxon_key"])
        if loc_keys is not None and tk not in loc_keys:
            continue
        if stat_filter_keys is not None and tk not in stat_filter_keys:
            continue
        val = values.get(tk)
        if val is None:
            continue
        sc = sample_counts.get(tk, 0)
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        if need_rbar:
            rbar = rbars.get(tk)
            if rbar is None or rbar < min_rbar:
                continue
        enriched.append((taxon, score, val, sc, match_name))

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
    for taxon, score, val, sc, match_name in page:
        tk = str(taxon["taxon_key"])
        results.append({
            "taxon": taxon,
            "match_score": score,
            "match_name": match_name,
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
    stat_filters: list[StatFilter] | None = None,
    layers: list[dict] | None = None,
) -> dict:
    candidates = search_taxa_by_name(q, limit=max((limit + offset) * 5, 200))
    if not candidates:
        return _empty_result("no_text_matches")

    scope_keys: frozenset[str] | None = None
    if within_taxon is not None and descendant_rank is not None:
        scope_keys = _load_scope_keys(within_taxon, descendant_rank, include_species_like)

    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like) if descendant_rank else None
    candidate_keys = [str(t["taxon_key"]) for t, _, _ in candidates]
    sample_counts = _batch_sample_counts(candidate_keys)
    stat_filter_keys: frozenset[str] | None = None
    if stat_filters:
        stat_filter_keys = _apply_stat_filters(frozenset(candidate_keys), stat_filters, layers or [])

    filtered: list[tuple[TaxonRecord, float, int, str]] = []
    for taxon, score, match_name in candidates:
        tk = str(taxon["taxon_key"])
        if scope_keys is not None and tk not in scope_keys:
            continue
        if loc_keys is not None and tk not in loc_keys:
            continue
        if stat_filter_keys is not None and tk not in stat_filter_keys:
            continue
        if accepted_ranks is not None and taxon.get("rank") not in accepted_ranks:
            continue
        sc = sample_counts.get(tk, 0)
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        filtered.append((taxon, score, sc, match_name))

    total = len(filtered)
    page = filtered[offset:offset + limit]

    results = []
    for taxon, score, sc, match_name in page:
        tk = str(taxon["taxon_key"])
        results.append({
            "taxon": taxon,
            "match_score": score,
            "match_name": match_name,
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
    """Return taxon_key set for all descendants of within_taxon at descendant_rank.

    This is pure scope/membership — no ranking data involved — so it's
    resolved straight from the catalog (same DFS _descendants_for_rank uses
    at build time) rather than depending on any ranking file existing.
    """
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
    stat_filters: list[StatFilter] | None = None,
    layers: list[dict] | None = None,
) -> dict:
    """Browse mode: list scope members, no sort. Scope membership (which
    taxa) and ranking/sample-count data (how many observations) are two
    separate concerns here — resolve scope from the catalog, then batch-fetch
    sample counts for just that scope in one filtered read."""
    scope_keys = _load_scope_keys(within_taxon, descendant_rank, include_species_like)
    if not scope_keys:
        return _empty_result("no_catalog")

    eligible_total = len(scope_keys)
    sample_counts = _batch_sample_counts(list(scope_keys))
    stat_filter_keys: frozenset[str] | None = None
    if stat_filters:
        stat_filter_keys = _apply_stat_filters(scope_keys, stat_filters, layers or [])

    filtered: list[tuple[TaxonRecord, int]] = []
    for tk in sorted(scope_keys):
        if loc_keys is not None and tk not in loc_keys:
            continue
        if stat_filter_keys is not None and tk not in stat_filter_keys:
            continue
        sc = sample_counts.get(tk, 0)
        effective_sc = loc_counts.get(tk, 0) if loc_counts else sc
        if effective_sc < min_samples:
            continue
        taxon = get_taxon_by_id(tk)
        if taxon is None:
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
        "eligible_total": eligible_total,
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
    stat_filters: list[StatFilter] | None = None,
    layers: list[dict] | None = None,
) -> dict:
    """Search and rank taxa.

    Returns a dict with keys: total, matched_total, eligible_total, empty_reason, results.
    Each result has: taxon, match_score, sample_count, sort_value, location_count, position, percentile.

    ``reference_value`` and ``min_rbar`` are used when sorting by a circular bearing metric
    (circular_mean or mode): results are ordered by forward clockwise distance from
    reference_value, and taxa with rbar below min_rbar are excluded.

    ``stat_filters`` chains arbitrary summary-stat predicates on top of
    scope/sort/text/location — e.g. "avg temp < 25C AND at least 10
    observations in ecoregion X" — ANDed together and applied to whichever
    candidate pool the active mode already resolved (scope population, or
    text-search matches). ``layers`` (the full layer catalog) is required
    whenever stat_filters is non-empty, to resolve each filter's value type.
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
            stat_filters=stat_filters, layers=layers,
        )
    if has_q and has_sort:
        return _query_ranked_text(
            q=q, sort_variable=sort_variable, sort_metric=sort_metric,
            sort_order=sort_order, limit=limit, offset=offset,
            min_samples=min_samples, include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
            reference_value=reference_value, min_rbar=min_rbar,
            stat_filters=stat_filters, layers=layers,
        )
    if has_q:
        return _query_text(
            q=q, within_taxon=within_taxon, descendant_rank=descendant_rank,
            limit=limit, offset=offset, min_samples=min_samples,
            include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
            stat_filters=stat_filters, layers=layers,
        )
    if has_scope:
        return _query_catalog(
            within_taxon=within_taxon, descendant_rank=descendant_rank,
            limit=limit, offset=offset, min_samples=min_samples,
            include_species_like=include_species_like,
            loc_keys=loc_keys, loc_counts=loc_counts,
            stat_filters=stat_filters, layers=layers,
        )
    return _empty_result("no_query")
