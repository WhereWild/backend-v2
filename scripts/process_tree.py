# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Compute per-taxon summary statistics, density graphs, and relative rankings.

Runs after enrich_tree has populated occurrence.parquets with GIS values.

Pass 1 — Stats (bottom-up, deepest first):
  Leaves use exact pandas/numpy stats; non-leaves stream descendant occurrence
  parquets with T-Digest approximations. Writes numerical_stats.parquet,
  nominal_stats.parquet, density.parquet, and occurrence_index.parquet.

Pass 2 — Rankings (top-down, shallowest first):
  Builds descendant rank catalogs ({rank}.parquet), rank index parquets
  ({rank}_index.parquet), and distributes position rows to each taxon's
  relative_ranks_positions.parquet. Pipelined per-ancestor so positions are
  written as soon as each ancestor's index is complete.
"""

from __future__ import annotations

import os

# OpenBLAS/OpenMP/MKL default to sizing their internal thread pool to the
# core count and spinning it up per call — fine for a few large calls, pure
# thread-spawn/teardown overhead for the many tiny per-taxon KDE/matrix ops
# this pipeline does. STATS_WORKERS=1 makes this process deliberately
# single-process, so there's no real parallelism to gain here, only
# contention. Must be set before numpy/scipy get imported anywhere in the
# process (BLAS reads these once at load time) — confirmed via a real
# 1411-taxon subtree benchmark: 5m00s -> 2m40s wall time (27m42s -> 2m36s
# CPU time) with these pinned to 1.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from config.config import load_config
from util.rankings import (
    POSITION_FILE,
    RANKINGS_FILE,
    RankingsSink,
    compute_relative_ranks,
    preload_stats_cache,
)
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    DENSITY_GRID_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    TREE_ROOT,
    StatsSink,
    clear_stats_occurrence_cache,
    compute_taxon_stats,
    preload_stats_occurrence_cache,
)
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants
from util.tiles import load_layers

CONFIG = load_config("global")

STATS_WORKERS = 1
RANK_WORKERS = 1
LOG_INTERVAL = 50

def _load_layers() -> list[dict]:
    return load_layers()


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _level_pass(
    by_depth: dict[int, list[TaxonRecord]],
    levels: list[int],
    task_fn,
    *,
    max_workers: int,
    label: str,
    total: int,
    should_skip_level=None,
    on_level_start=None,
    on_level_end=None,
) -> tuple[int, int]:
    """Run task_fn(node) over all taxa level by level, returning (completed, failed).

    should_skip_level(depth) / on_level_start(depth) / on_level_end(depth) are
    optional hooks used by run_stats() to checkpoint at the level boundary —
    skip whole already-finished levels on resume, and open/close a per-level
    StatsSink around each level's taxa.
    """
    completed = 0
    failed = 0
    t0 = time.monotonic()
    window = 500
    recent: deque[float] = deque(maxlen=window)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for depth in levels:
            level_taxa = by_depth[depth]
            if should_skip_level and should_skip_level(depth):
                completed += len(level_taxa)
                print(f"[{label}] level {depth}: already done, skipping ({len(level_taxa)} taxa)  [{time.monotonic()-t0:.1f}s]")
                continue
            if on_level_start:
                on_level_start(depth)
            futures = {executor.submit(task_fn, node): node for node in level_taxa}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                    completed += 1
                    now = time.monotonic()
                    recent.append(now)
                    if completed % LOG_INTERVAL == 0 or completed == total:
                        elapsed = now - t0
                        if len(recent) >= 2:
                            rate = (len(recent) - 1) / (recent[-1] - recent[0])
                        else:
                            rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(
                            f"[{label}] {completed}/{total}"
                            f"  elapsed={_fmt_duration(elapsed)}"
                            f"  eta={_fmt_duration(eta)}"
                            f"  rate={rate:.1f}/s"
                            f"  ({node['rank']} {node['scientific_name']})"
                        )
                except Exception as exc:
                    failed += 1
                    elapsed = time.monotonic() - t0
                    print(
                        f"[{label}] FAIL [{elapsed:.0f}s]"
                        f"  {node['rank']} {node['scientific_name']}: {exc}"
                    )
            if on_level_end:
                on_level_end(depth)

    elapsed = time.monotonic() - t0
    print(f"[{label}] done — {completed} ok, {failed} failed, {_fmt_duration(elapsed)} total")
    return completed, failed


def _setup() -> tuple[list[dict], dict[str, dict], dict[int, list[TaxonRecord]], list[int], list[int], int]:
    layers = _load_layers()
    layer_meta = {layer["id"]: layer for layer in layers}
    root = get_taxon_by_id(CONFIG.plantae_key)
    if root is None:
        raise RuntimeError(f"[process_tree] root taxon {CONFIG.plantae_key} not found")
    all_taxa = list(iter_descendants(root, include_self=True))
    total = len(all_taxa)
    by_depth: dict[int, list[TaxonRecord]] = defaultdict(list)
    for t in all_taxa:
        by_depth[t["path"].count("/")].append(t)
    stats_levels = sorted(by_depth.keys(), reverse=True)
    rank_levels = sorted(by_depth.keys())
    return layers, layer_meta, by_depth, stats_levels, rank_levels, total


_STATS_FILES = [
    ("numerical_stats", NUMERICAL_STATS_FILE),
    ("nominal_stats",   NOMINAL_STATS_FILE),
    ("ordinal_stats",   ORDINAL_STATS_FILE),
    ("circular_stats",  CIRCULAR_STATS_FILE),
    ("density",         DENSITY_FILE),
    ("density_grid",    DENSITY_GRID_FILE),
    # positions handled separately — built inline during rank index pass, merged at consolidation
]

_CONSOLIDATION_ROW_GROUP_SIZE = 50_000

# _finalize_stats and the rankings/positions builder below both do an
# ORDER BY + COPY over a glob of staged chunk files with a bare
# duckdb.connect() — the same shape that OOM-killed populate_tree's
# _consolidate, carry_forward's join+sort, and enrich_tree's finalize join
# (all now fixed the same way). Staged stat/ranking chunks are much smaller
# than raw occurrences, but applying the same defensive config here is cheap
# insurance against the same failure mode as this data grows.
_DUCKDB_SPILL_DIR = Path("data/tmp/duckdb_spill")
_DUCKDB_MEMORY_LIMIT = "28GB"


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    _DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")
    return con


_STATS_STAGING_DIRNAME = ".stats_staging"
_RANKINGS_STAGING_DIRNAME = ".rankings_staging"


def _stats_staging_dir() -> Path:
    return GLOBAL_STATS_DIR / _STATS_STAGING_DIRNAME


def _rankings_staging_dir() -> Path:
    return GLOBAL_STATS_DIR / _RANKINGS_STAGING_DIRNAME


def _level_marker_path(staging_dir: Path, depth: int) -> Path:
    return staging_dir / ".done" / f"level_{depth:04d}"


def _finalize_rankings(staging_dir: Path) -> None:
    """Sort the staged per-level ranking-position chunks into the two final
    global rankings files:
      - POSITION_FILE  sorted by (taxon_key, variable) — per-taxon lookup
        (main.py's "your rank in each ancestor context" display).
      - RANKINGS_FILE  sorted by (contextTaxonId, rank, variable, metric,
        position) — scoped ranking browse/search, pruned straight to the
        matching row group(s) and already in position order, so pagination
        never needs a live sort even for a near-root context.
    Same DuckDB-out-of-core-ORDER-BY pattern as _finalize_stats, just two
    passes (one per output sort order) over the same staged chunks instead
    of one.
    """
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if not staging_dir.exists() or not any(staging_dir.glob("*.parquet")):
        print("[rankings] no position data staged, skipping")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    glob_pattern = (staging_dir / "*.parquet").as_posix()
    con = _duckdb_connect()
    try:
        for filename, order_by in (
            (POSITION_FILE, '"taxon_key", "variable"'),
            (RANKINGS_FILE, '"contextTaxonId", "rank", "variable", "metric", "position"'),
        ):
            dest = GLOBAL_STATS_DIR / filename
            tmp_dest = dest.with_suffix(".parquet.tmp")
            con.execute(f"""
                COPY (
                    SELECT * FROM read_parquet('{glob_pattern}', union_by_name=True)
                    ORDER BY {order_by}
                ) TO '{tmp_dest.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {_CONSOLIDATION_ROW_GROUP_SIZE})
            """)
            tmp_dest.replace(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"[rankings] {filename} -> {size_mb:.0f}MB  [{time.monotonic()-t0:.1f}s]")
    finally:
        con.close()
    shutil.rmtree(staging_dir, ignore_errors=True)


def _finalize_stats(staging_dir: Path) -> None:
    """Sort each stat type's per-level chunks by taxon_key into its final
    global file — one DuckDB pass per type, replacing what used to be a
    separate glob-thousands-of-per-taxon-files consolidation stage, since
    run_stats() now streams straight into these staged chunks instead of
    one file per taxon directory."""
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    kinds = [*_STATS_FILES, ("phenology_counts", "phenology_counts.parquet")]
    con = _duckdb_connect()
    try:
        for kind, filename in kinds:
            chunk_dir = staging_dir / kind
            if not chunk_dir.exists() or not any(chunk_dir.glob("*.parquet")):
                continue
            dest = GLOBAL_STATS_DIR / filename
            tmp_dest = dest.with_suffix(".parquet.tmp")
            con.execute(f"""
                COPY (
                    SELECT * FROM read_parquet('{(chunk_dir / "*.parquet").as_posix()}', union_by_name=True)
                    ORDER BY taxon_key
                ) TO '{tmp_dest.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {_CONSOLIDATION_ROW_GROUP_SIZE})
            """)
            tmp_dest.replace(dest)
            print(f"[stats] {kind} -> {filename}  [{time.monotonic()-t0:.1f}s]")
    finally:
        con.close()
    shutil.rmtree(staging_dir, ignore_errors=True)


def run_consolidation() -> None:
    """Merge per-node ranking positions into a global file, and clean up
    now-unneeded per-node intermediate state (.acc accumulators, rank
    catalogs). Summary stats no longer need consolidating here —
    run_stats() already streams them straight into their final global
    files (see _finalize_stats)."""
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Remove accumulator state and rank catalogs — no longer needed once
    # rankings have been computed from them.
    removed = 0
    patterns = [
        "species.parquet", "subspecies.parquet", "genus.parquet",
        "family.parquet", "order.parquet", "variety.parquet", "form.parquet",
        ".acc",
        POSITION_FILE,  # old per-taxon positions files (new approach never creates them)
        "*_index.parquet",  # old per-ancestor wide struct-array rank indexes (superseded by RANKINGS_FILE)
        "*_positions.parquet",  # old per-ancestor position ctx files (superseded by RankingsSink staging)
    ]
    for filename in patterns:
        for path in TREE_ROOT.rglob(filename):
            path.unlink()
            removed += 1
    for path in TREE_ROOT.rglob("tmp*.parquet"):
        path.unlink()
        removed += 1
    print(f"[consolidate] removed {removed} per-node files")

    cache_file = GLOBAL_STATS_DIR.parent / "stats_cache.pkl.gz"
    if cache_file.exists():
        cache_file.unlink()
        print(f"[consolidate] removed {cache_file.name}")

    print(f"[consolidate] done — {time.monotonic() - t0:.1f}s total")


def run_stats(resume: bool = False) -> None:
    """Compute stats bottom-up, streaming each tree-depth level straight into
    shared per-type staging chunks (see util.stats.StatsSink) instead of one
    file per taxon directory, then sort those chunks into the final global
    stats files. Resume checkpoints at the level boundary: a level is skipped
    if it already has a completion marker from a prior run, otherwise the
    whole level is (re)computed — cheap, since per-taxon accumulator merging
    is already fast and only leaf levels are large.
    """
    layers, layer_meta, by_depth, stats_levels, _, total = _setup()
    print(f"[process_tree] {total} taxa — stats:{STATS_WORKERS} workers" + (" — RESUME" if resume else ""))

    staging_dir = _stats_staging_dir()
    if not resume and staging_dir.exists():
        shutil.rmtree(staging_dir)
    marker_dir = staging_dir / ".done"
    marker_dir.mkdir(parents=True, exist_ok=True)

    sink_holder: dict[str, StatsSink] = {}

    def _should_skip(depth: int) -> bool:
        return resume and _level_marker_path(staging_dir, depth).exists()

    def _on_start(depth: int) -> None:
        sink_holder["sink"] = StatsSink(staging_dir, f"level_{depth:04d}")

    def _on_end(depth: int) -> None:
        sink_holder.pop("sink").close()
        _level_marker_path(staging_dir, depth).touch()

    def _task(node: TaxonRecord) -> None:
        compute_taxon_stats(node, layers=layers, layer_meta=layer_meta, sink=sink_holder["sink"])

    # Every taxon's stats computation reads its own occurrence rows via a
    # fresh DuckDB query — at full-tree scale (187k+ taxa) each one's fixed
    # per-query planning/binding overhead (confirmed via EXPLAIN ANALYZE:
    # ~0.3-1s per call regardless of how little data it actually reads)
    # dominates completely, measured at ~0.9 taxa/s versus ~15/s on the old
    # per-taxon-file architecture. One bulk preload, grouped by taxon_key,
    # turns that into dict lookups. Falls back to the unchanged per-query
    # path (returns False) when the file is too large to safely hold in
    # memory this way — see _STATS_CACHE_MAX_ROWS.
    cached = preload_stats_occurrence_cache(layer_meta)
    try:
        _level_pass(
            by_depth, stats_levels, _task, max_workers=STATS_WORKERS, label="stats", total=total,
            should_skip_level=_should_skip, on_level_start=_on_start, on_level_end=_on_end,
        )
    finally:
        if cached:
            clear_stats_occurrence_cache()

    print("[stats] finalizing global stats files...")
    _finalize_stats(staging_dir)


def run_rankings() -> None:
    """Compute relative rankings, streaming each tree-depth level's computed
    position rows straight into a shared per-level staging chunk (see
    util.rankings.RankingsSink) instead of one {rank}_positions.parquet file
    per ancestor directory, then sort those chunks into the two final global
    rankings files. Same pattern as run_stats()/StatsSink — nothing gets
    written under the tree directories at all.
    """
    layers, _, by_depth, _, rank_levels, total = _setup()
    print(f"[process_tree] {total} taxa — rankings:{RANK_WORKERS} workers")
    preload_stats_cache(layers)

    staging_dir = _rankings_staging_dir()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    sink_holder: dict[str, RankingsSink] = {}

    def _on_start(depth: int) -> None:
        sink_holder["sink"] = RankingsSink(staging_dir, f"level_{depth:04d}")

    def _on_end(depth: int) -> None:
        sink_holder.pop("sink").close()

    def _task(node: TaxonRecord) -> None:
        compute_relative_ranks(node, layers=layers, sink=sink_holder["sink"])

    _level_pass(
        by_depth, rank_levels, _task, max_workers=RANK_WORKERS, label="rankings", total=total,
        on_level_start=_on_start, on_level_end=_on_end,
    )

    print("[rankings] finalizing global rankings files...")
    _finalize_rankings(staging_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["stats", "rankings", "consolidate", "all"],
        default="all",
        help="Run stats, rankings, consolidate, or all (default: all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tree-depth levels whose stats are already finalized (for restarts).",
    )
    args, _ = parser.parse_known_args()

    try:
        _setup()
    except RuntimeError as exc:
        print(str(exc))
        return

    print(f"[process_tree] phase:{args.phase}" + (" — RESUME" if args.resume else ""))

    if args.phase in ("stats", "all"):
        run_stats(resume=args.resume)
        if args.phase == "all":
            print("[process_tree] stats complete — starting rankings pass")

    if args.phase in ("rankings", "all"):
        run_rankings()
        if args.phase == "all":
            print("[process_tree] rankings complete — starting consolidation pass")

    if args.phase in ("consolidate", "all"):
        run_consolidation()


if __name__ == "__main__":  # pragma: no cover
    main()
