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

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.config import METRICS_BY_TYPE, ValueType, load_config
from util.stats import CIRCULAR_STATS_FILE, NOMINAL_STATS_FILE, NUMERICAL_STATS_FILE
from util.storage import ParquetStorageProxy, atomic_write_parquet
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

CONFIG = load_config("global")

TREE_ROOT = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "tree"
_CACHE_FILE = TREE_ROOT.parent / "stats_cache.pkl.gz"
POSITION_FILE = "relative_ranks_positions.parquet"
POSITION_CTX_GLOB = "positions_ctx_*.parquet"  # per-context files written during rankings pass

# Canonical taxonomy rank order used to determine descendant catalog targets.
_RANK_ORDER: tuple[str, ...] = (
    "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES", "SUBSPECIES",
)


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


def _preload_one_taxon(
    num_path: Path,
    ratio_interval_ids: set[str],
    circular_ids: set[str],
    nominal_ids: set[str],
    layer_metrics: dict[str, tuple[str, ...]],
    nominal_metrics: set[str],
    circ_metrics: tuple[str, ...],
) -> tuple[str, dict]:
    taxon_key = num_path.parent.name.rsplit("_", 1)[-1]
    entry: dict = {"__sample_count__": 0}
    taxon_dir = num_path.parent

    # numerical stats — wide format: one row per variable, metric columns
    try:
        tbl = pq.ParquetFile(num_path).read()
        col_names = set(tbl.schema.names)
        variables = tbl.column("variable").to_pylist()
        counts = tbl.column("count").to_pylist() if "count" in col_names else [None] * len(variables)
        needed_metrics: set[str] = set()
        for var in variables:
            if var in ratio_interval_ids:
                needed_metrics.update(layer_metrics.get(var, ()))
        metric_cols = {m: tbl.column(m).to_pylist() for m in needed_metrics if m in col_names}
        for i, variable in enumerate(variables):
            if not variable or variable not in ratio_interval_ids:
                continue
            cnt = counts[i]
            if cnt and entry["__sample_count__"] == 0:
                try:
                    entry["__sample_count__"] = int(cnt)
                except (TypeError, ValueError):
                    pass
            for metric in layer_metrics.get(variable, ()):
                col = metric_cols.get(metric)
                if col is None:
                    continue
                val = col[i]
                if val is not None and _safe_finite(val):
                    entry[f"{variable}::{metric}"] = float(val)
    except Exception:
        pass

    # nominal stats — long format: columns variable, metric, value
    nom_path = taxon_dir / NOMINAL_STATS_FILE
    if nom_path.exists():
        try:
            tbl = pq.ParquetFile(nom_path).read()
            nom_variables = tbl.column("variable").to_pylist()
            nom_metrics_col = tbl.column("metric").to_pylist()
            nom_values = tbl.column("value").to_pylist()
            for variable, metric, val in zip(nom_variables, nom_metrics_col, nom_values):
                variable = str(variable or "")
                metric = str(metric or "")
                if variable not in nominal_ids:
                    continue
                if metric not in nominal_metrics and not metric.startswith("class_"):
                    continue
                if entry["__sample_count__"] == 0 and metric == "total_samples":
                    try:
                        entry["__sample_count__"] = int(float(val or 0))
                    except (TypeError, ValueError):
                        pass
                if _safe_finite(val):
                    entry[f"{variable}::{metric}"] = float(val)
        except Exception:
            pass

    # circular stats — wide format: one row per variable, metric columns
    circ_path = taxon_dir / CIRCULAR_STATS_FILE
    if circ_path.exists():
        try:
            tbl = pq.ParquetFile(circ_path).read()
            col_names = set(tbl.schema.names)
            circ_variables = tbl.column("variable").to_pylist()
            circ_metric_cols = {m: tbl.column(m).to_pylist() for m in circ_metrics if m in col_names}
            for i, variable in enumerate(circ_variables):
                if not variable or variable not in circular_ids:
                    continue
                for metric in circ_metrics:
                    col = circ_metric_cols.get(metric)
                    if col is None:
                        continue
                    val = col[i]
                    if val is not None and _safe_finite(val):
                        entry[f"{variable}::{metric}"] = float(val)
        except Exception:
            pass

    return taxon_key, entry


def preload_stats_cache(layers: list[dict]) -> None:
    """Walk all per-node stats files once and populate the module-level cache.

    Call this before the rankings pass so every lookup is an O(1) dict access
    instead of a disk read. Uses pyarrow column access directly (no pandas) and
    a thread pool so pyarrow I/O can overlap across threads (GIL released during reads).
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
    from functools import partial as _partial

    global _stats_cache
    _stats_cache = {}
    layer_by_id = {lay["id"]: lay for lay in layers}
    nominal_ids = {lay["id"] for lay in layers if lay.get("value_type") == ValueType.NOMINAL}
    nominal_metrics = set(METRICS_BY_TYPE[ValueType.NOMINAL])

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

    worker = _partial(
        _preload_one_taxon,
        ratio_interval_ids=ratio_interval_ids,
        circular_ids=circular_ids,
        nominal_ids=nominal_ids,
        layer_metrics=layer_metrics,
        nominal_metrics=nominal_metrics,
        circ_metrics=circ_metrics,
    )

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
            print(f"[rankings] cache loaded from disk: {len(_stats_cache):,} taxa  [{_time.monotonic()-t0:.1f}s]")
            return
        except Exception as e:
            print(f"[rankings] disk cache load failed ({e}), rebuilding...")
            _stats_cache.clear()
            _metric_vocab.clear()
            _metric_to_idx.clear()

    all_paths = list(TREE_ROOT.rglob(NUMERICAL_STATS_FILE))
    total_paths = len(all_paths)
    t0 = _time.monotonic()
    done = 0
    LOG_EVERY = 10_000

    # Phase 1: read all stats files in parallel, collect raw dicts
    raw: dict[str, dict] = {}
    print(f"[rankings] preloading stats cache for {total_paths:,} taxa...")
    with _TPE(max_workers=4) as executor:
        futs = {executor.submit(worker, p): p for p in all_paths}
        for fut in _as_completed(futs):
            try:
                taxon_key, entry = fut.result()
                raw[taxon_key] = entry
            except Exception:
                pass
            done += 1
            if done % LOG_EVERY == 0 or done == total_paths:
                elapsed = _time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_paths - done) / rate if rate > 0 else 0
                m, s = divmod(int(eta), 60)
                eta_str = f"{m}m{s:02d}s" if m else f"{s}s"
                print(f"[rankings/preload] {done:,}/{total_paths:,}  {rate:.0f}/s  eta={eta_str}")
    print(f"[rankings] read complete: {len(raw):,} taxa  [{_time.monotonic()-t0:.1f}s]  converting to compact format...")

    # Phase 2: build global metric vocab and convert to numpy float32 arrays (~6x RAM reduction)
    all_keys: set[str] = set()
    for entry in raw.values():
        all_keys.update(k for k in entry if k != "__sample_count__")
    _metric_vocab[:] = sorted(all_keys)
    _metric_to_idx.update({k: i for i, k in enumerate(_metric_vocab)})
    n_metrics = len(_metric_vocab)

    for taxon_key, entry in raw.items():
        sc = int(entry.get("__sample_count__", 0))
        arr = np.full(n_metrics, np.nan, dtype=np.float32)
        for k, v in entry.items():
            if k != "__sample_count__":
                idx = _metric_to_idx.get(k)
                if idx is not None:
                    arr[idx] = np.float32(v)
        _stats_cache[taxon_key] = (sc, arr)
    del raw

    elapsed = _time.monotonic() - t0
    print(f"[rankings] stats cache ready: {len(_stats_cache):,} taxa  {n_metrics:,} metrics  [{elapsed:.1f}s]")

    # Phase 3: persist to disk for fast restart
    print(f"[rankings] saving cache to disk...")
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        with _gzip.open(tmp, "wb", compresslevel=1) as f:
            _pickle.dump({"vocab": list(_metric_vocab), "cache": dict(_stats_cache)}, f, protocol=5)
        tmp.replace(_CACHE_FILE)
        print(f"[rankings] cache saved ({_CACHE_FILE.stat().st_size / 1e9:.2f}GB)  [{_time.monotonic()-t0:.1f}s total]")
    except Exception as e:
        print(f"[rankings] cache save failed (non-fatal): {e}")


def _infer_sample_count(taxon_dir: Path) -> int:
    """Return observation count from stats files."""
    if _stats_cache is not None:
        taxon_key = taxon_dir.name.rsplit("_", 1)[-1]
        entry = _stats_cache.get(taxon_key)
        return entry[0] if entry is not None else 0
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
    return 0


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
    if _stats_cache is not None:
        entry = _stats_cache.get(taxon_key)
        if not entry:
            return {}
        _, values_arr = entry
        non_nan = np.where(~np.isnan(values_arr))[0]
        return {
            _metric_vocab[i]: {"taxon_key": taxon_key, "value": float(values_arr[i]), "sample_count": sample_count}
            for i in non_nan
        }
    entries = _collect_entries_from_numerical_stats(taxon_key, taxon_dir, sample_count, layers)
    entries.update(_collect_entries_from_nominal_stats(taxon_key, taxon_dir, sample_count, layers))
    entries.update(_collect_entries_from_circular_stats(taxon_key, taxon_dir, sample_count, layers))
    return entries


def _build_rank_index(
    ancestor: TaxonRecord,
    rank: str,
    index_path: Path,
    layers: list[dict],
) -> None:
    """Collect per-taxon metrics for all descendants of rank and write sorted struct array index."""
    import time as _time
    ancestor_name = ancestor.get("scientific_name") or ancestor.get("taxon_key", "?")
    ancestor_rank = ancestor.get("rank", "?")
    t0 = _time.monotonic()

    descendants = _descendants_for_rank(ancestor, rank)
    if not descendants:
        index_path.unlink(missing_ok=True)
        return

    print(f"  [rank_index] {ancestor_rank} {ancestor_name} → {rank}: {len(descendants):,} descendants", flush=True)

    # Collect lightweight (taxon_key, cached_dict) pairs — just references into _stats_cache,
    # no data copying. Then process one column at a time so only one Python list of entries
    # is alive at a time; each is immediately converted to a compact Arrow array (C memory)
    # and the Python list is discarded. This keeps peak Python heap usage to ~one column's
    # worth of data regardless of how many descendants or metrics there are.
    if _stats_cache is not None:
        desc_data: list[tuple[str, tuple]] = []
        for t in descendants:
            taxon_key = str(t["taxon_key"])
            entry = _stats_cache.get(taxon_key)
            if entry is not None:
                desc_data.append((taxon_key, entry))

        if not desc_data:
            index_path.unlink(missing_ok=True)
            return

        all_taxon_keys = [tk for tk, _ in desc_data]
        all_sample_counts = np.array([e[0] for _, e in desc_data], dtype=np.int64)

        # Single pass using numpy: for each taxon, find non-NaN metric indices and
        # append (position, value) to per-column lists. No Python float objects created
        # for cached values — they stay as float32 in the numpy array until appended.
        col_idx: dict[str, list[int]] = {}
        col_val: dict[str, list[float]] = {}
        vocab = _metric_vocab
        print(f"    collecting {len(desc_data):,} taxa...", flush=True)
        for i, (_, entry) in enumerate(desc_data):
            values_arr = entry[1]
            for metric_idx in np.where(~np.isnan(values_arr))[0]:
                k = vocab[metric_idx]
                v = float(values_arr[metric_idx])
                if k in col_idx:
                    col_idx[k].append(i)
                    col_val[k].append(v)
                else:
                    col_idx[k] = [i]
                    col_val[k] = [v]

    else:
        # Fallback: collect via disk reads (no cache loaded)
        col_idx_fb: dict[str, list[tuple[str, float, int]]] = {}
        for t in descendants:
            taxon_key = str(t["taxon_key"])
            taxon_path = t.get("path", "")
            if not taxon_path:
                continue
            taxon_dir = TREE_ROOT / taxon_path
            sample_count = _infer_sample_count(taxon_dir)
            for col_key, entry in _collect_all_entries(taxon_key, taxon_dir, sample_count, layers).items():
                col_idx_fb.setdefault(col_key, []).append(
                    (entry["taxon_key"], entry["value"], entry["sample_count"])
                )
        if not col_idx_fb:
            index_path.unlink(missing_ok=True)
            return

    struct_type = pa.struct(_STRUCT_FIELDS)
    arrays: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    max_len = 0

    n_cols = len(col_idx) if _stats_cache is not None else len(col_idx_fb)  # type: ignore[possibly-undefined]
    print(f"    sorting+building {n_cols:,} columns...", flush=True)

    if _stats_cache is not None:
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
            n = len(sorted_tks)
            column_lengths[col_key] = n
            max_len = max(max_len, n)
            arrays[col_key] = pa.StructArray.from_arrays(
                [
                    pa.array(sorted_tks, type=pa.string()),
                    pa.array(val_np[order], type=pa.float64()),
                    pa.array(sorted_scs, type=pa.int64()),
                ],
                fields=_STRUCT_FIELDS,
            )
    else:
        for col_key, entries in col_idx_fb.items():  # type: ignore[possibly-undefined]
            entries.sort(key=lambda e: (e[1], e[0]))
            n = len(entries)
            column_lengths[col_key] = n
            max_len = max(max_len, n)
            arrays[col_key] = pa.StructArray.from_arrays(
                [
                    pa.array([e[0] for e in entries], type=pa.string()),
                    pa.array([e[1] for e in entries], type=pa.float64()),
                    pa.array([e[2] for e in entries], type=pa.int64()),
                ],
                fields=_STRUCT_FIELDS,
            )

    if not arrays:
        index_path.unlink(missing_ok=True)
        return

    for col_name, arr in arrays.items():
        if len(arr) < max_len:
            arrays[col_name] = pa.concat_arrays(
                [arr, pa.nulls(max_len - len(arr), type=struct_type)]
            )

    table = pa.table(arrays)
    metadata = {b"column_lengths": json.dumps(column_lengths).encode("utf-8")}
    _atomic_write(index_path, table.replace_schema_metadata(metadata))
    print(f"    wrote {n_cols:,} cols  max_len={max_len:,}  [{_time.monotonic()-t0:.1f}s]", flush=True)


def build_rank_indexes(ancestor: TaxonRecord, layers: list[dict]) -> None:
    """Build {rank}_index.parquet files under the ancestor's directory."""
    ancestor_rank = ancestor.get("rank") or ""
    targets = _descendant_rank_targets(ancestor_rank)
    if not targets:
        return

    ancestor_dir = TREE_ROOT / ancestor["path"]
    for rank in targets:
        index_path = ancestor_dir / f"{rank.lower()}_index.parquet"
        _build_rank_index(ancestor, rank, index_path, layers)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def _load_column_lengths(index_path: Path) -> dict[str, int]:
    try:
        schema = _storage.read_schema(index_path)
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

    import gc as _gc
    import time as _dtime
    _dt0 = _dtime.monotonic()
    anc_name = ancestor.get("scientific_name") or ancestor.get("taxon_key", "?")
    rank_label = index_path.stem
    print(f"  [distribute] {anc_name} {rank_label}: reading {len(metric_columns)} cols...", flush=True)

    try:
        table = pq.read_table(index_path, columns=metric_columns)
    except Exception:
        return
    tbl_mb = sum(col.nbytes for col in table.columns) / 1e6
    print(f"  [distribute] index loaded: {len(table):,} rows  {tbl_mb:.0f}MB  accumulating...", flush=True)

    # Store (variable, metric, position, count, sampleCount) tuples.
    rows_by_taxon: dict[str, list[tuple]] = {}
    for col_name in metric_columns:
        variable, metric = col_name.split("::", 1)
        if metric in _ANGULAR_METRICS:
            continue
        is_class_metric = metric.startswith("class_")
        column = table.column(col_name).combine_chunks()
        col_len = min(column_lengths.get(col_name, len(column)), len(column))
        if col_len <= 0:
            continue

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
            if value != prev_value:
                min_rank_pos = position
                prev_value = value
            if is_class_metric and value == 0.0:
                continue
            sample_count = sample_counts_list[position]
            rows_by_taxon.setdefault(taxon_key, []).append((
                variable, metric, min_rank_pos, col_len,
                int(sample_count) if sample_count is not None else None,
            ))

    total_rows = sum(len(v) for v in rows_by_taxon.values())
    print(f"  [distribute] accumulated {total_rows:,} rows for {len(rows_by_taxon):,} taxa  [{_dtime.monotonic()-_dt0:.1f}s]  freeing Arrow table...", flush=True)

    del table
    _gc.collect()

    # Write one per-context file per taxon — no reads needed, just overwrite.
    # Consolidation merges all positions_ctx_*.parquet → relative_ranks_positions.parquet.
    ctx_filename = f"positions_ctx_{context_taxon_id}.parquet"
    all_taxon_keys = list(rows_by_taxon.keys())
    n_total = len(all_taxon_keys)
    n_written = 0
    print(f"  [distribute] writing {ctx_filename} for {n_total:,} taxa...", flush=True)
    for taxon_key in all_taxon_keys:
        new_tuples = rows_by_taxon.pop(taxon_key)
        n_written += 1
        if n_written % 25_000 == 0:
            print(f"  [distribute] wrote {n_written:,}/{n_total:,}  [{_dtime.monotonic()-_dt0:.1f}s]", flush=True)
        if not new_tuples:
            continue
        taxon = get_taxon_by_id(taxon_key)
        if taxon is None:
            continue
        nt_vars, nt_mets, nt_pos, nt_cnt, nt_sc = map(list, zip(*new_tuples))
        n_new = len(nt_vars)
        tbl = pa.Table.from_arrays(
            [
                pa.array(nt_vars,                      type=pa.string()),
                pa.array(nt_mets,                      type=pa.string()),
                pa.array(nt_pos,                       type=pa.int64()),
                pa.array(nt_cnt,                       type=pa.int64()),
                pa.array(nt_sc,                        type=pa.int64()),
                pa.array([context_taxon_id] * n_new,   type=pa.string()),
                pa.array([context_label]    * n_new,   type=pa.string()),
            ],
            schema=_POSITION_SCHEMA,
        )
        _atomic_write(TREE_ROOT / taxon["path"] / ctx_filename, tbl)

    print(f"  [distribute] done  [{_dtime.monotonic()-_dt0:.1f}s]", flush=True)


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


def _read_index_entries(index_path: Path, col_name: str, col_len: int) -> list[dict]:
    """Read one struct column from a rank_index.parquet, returning up to col_len entries."""
    try:
        tbl = _storage.read_table(index_path, columns=[col_name])
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
    for path, filter_fn in [
        (taxon_dir / NUMERICAL_STATS_FILE,
         lambda df: df[df["variable"] == variable_id]),
        (taxon_dir / NOMINAL_STATS_FILE,
         lambda df: df[(df["variable"] == variable_id) & (df["metric"] == metric_id)]),
        (taxon_dir / CIRCULAR_STATS_FILE,
         lambda df: df[df["variable"] == variable_id]),
    ]:
        try:
            df = _storage.read_table(path).to_pandas()
            rows = filter_fn(df)
            if rows.empty:
                continue
            col = "value" if "value" in rows.columns else metric_id
            if col not in rows.columns:
                continue
            val = rows.iloc[0][col]
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
    index_path = TREE_ROOT / within_taxon["path"] / f"{rank_lower}_index.parquet"
    try:
        schema = _storage.read_schema(index_path)
        col_lengths: dict[str, int] = json.loads(schema.metadata.get(b"column_lengths", b"{}"))
        if col_lengths:
            first_col = next(iter(col_lengths))
            col_len = col_lengths[first_col]
            tbl = _storage.read_table(index_path, columns=[first_col])
            col = tbl.column(first_col).combine_chunks().slice(0, col_len)
            keys = pc.struct_field(col, "taxonKey").to_pylist()
            return frozenset(str(k) for k in keys if k is not None)
    except Exception:
        pass
    # Fall back to live DFS if index is missing
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
    index_path = TREE_ROOT / within_taxon["path"] / f"{rank_lower}_index.parquet"

    try:
        schema = _storage.read_schema(index_path)
        col_lengths: dict[str, int] = json.loads(schema.metadata.get(b"column_lengths", b"{}"))
        if not col_lengths:
            return _empty_result("no_catalog")
        first_col = next(iter(col_lengths))
        col_len = col_lengths[first_col]
        tbl = _storage.read_table(index_path, columns=[first_col])
        col = tbl.column(first_col).combine_chunks().slice(0, col_len)
        taxon_keys = pc.struct_field(col, "taxonKey").to_pylist()
        sample_counts_list = pc.struct_field(col, "sampleCount").to_pylist()
    except Exception:
        return _empty_result("no_catalog")

    tk_sc = {str(tk): int(sc or 0) for tk, sc in zip(taxon_keys, sample_counts_list) if tk}
    eligible_total = len(tk_sc)
    accepted_ranks = _accepted_ranks(descendant_rank, include_species_like)

    filtered: list[tuple[TaxonRecord, int]] = []
    for tk, sc in tk_sc.items():
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
