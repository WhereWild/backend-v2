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

import argparse
import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from config.config import load_config
from util.rankings import POSITION_CTX_GLOB, POSITION_FILE, compute_relative_ranks, preload_stats_cache
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    TREE_ROOT,
    StatsSink,
    compute_taxon_stats,
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
    # positions handled separately — built inline during rank index pass, merged at consolidation
]

_CONSOLIDATION_ROW_GROUP_SIZE = 50_000
_POS_MEM_BUDGET = 1_000_000_000  # 1 GB Arrow in-memory per sort run


def _consolidate_positions(t0: float) -> None:
    """External sort-merge of positions ctx files into one global sorted parquet."""
    import shutil as _shutil

    pos_files = sorted(TREE_ROOT.rglob(POSITION_CTX_GLOB))
    if not pos_files:
        print("[consolidate] positions: no ctx files found, skipping")
        return

    print(f"[consolidate] positions: {len(pos_files):,} ctx files, sort-merging...")
    runs_dir = GLOBAL_STATS_DIR / ".pos_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1 — build sorted runs bounded by _POS_MEM_BUDGET
        run_paths: list[Path] = []
        frames: list[pa.Table] = []
        current_bytes = 0

        total_rows = 0

        def _flush_run() -> None:
            nonlocal total_rows
            if not frames:
                return
            tbl = pa.concat_tables(frames).sort_by(
                [("taxon_key", "ascending"), ("variable", "ascending")]
            )
            total_rows += len(tbl)
            p = runs_dir / f"run_{len(run_paths):05d}.parquet"
            pq.write_table(tbl, p, compression="snappy",
                           row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE)
            run_paths.append(p)
            frames.clear()
            print(f"[consolidate/positions] run {len(run_paths):03d} written  "
                  f"{len(tbl):,} rows  {p.stat().st_size/1e6:.0f}MB  "
                  f"[{time.monotonic()-t0:.1f}s]", flush=True)

        file_idx = 0
        for f in pos_files:
            file_idx += 1
            try:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=500_000):
                    tbl = pa.Table.from_batches([batch])
                    frames.append(tbl)
                    current_bytes += tbl.nbytes
                    if current_bytes >= _POS_MEM_BUDGET:
                        _flush_run()
                        current_bytes = 0
            except Exception as e:
                print(f"[consolidate/positions] skip {f.name}: {e}", flush=True)
                continue
            if file_idx % 500 == 0:
                print(f"[consolidate/positions] scanned {file_idx:,}/{len(pos_files):,} files  "
                      f"{len(run_paths)} runs so far  [{time.monotonic()-t0:.1f}s]", flush=True)
        _flush_run()

        if not run_paths:
            print("[consolidate] positions: no data written")
            return

        print(f"[consolidate/positions] phase 1 done: {len(run_paths)} runs  "
              f"{total_rows:,} rows  [{time.monotonic()-t0:.1f}s]", flush=True)

        # Phase 2 — iterative merge until one run remains
        group = 10
        pass_num = 0
        while len(run_paths) > 1:
            pass_num += 1
            next_runs: list[Path] = []
            for i in range(0, len(run_paths), group):
                chunk = run_paths[i : i + group]
                tbls = [pq.read_table(p) for p in chunk]
                merged = pa.concat_tables(tbls).sort_by(
                    [("taxon_key", "ascending"), ("variable", "ascending")]
                )
                del tbls
                out = runs_dir / f"merge_p{pass_num}_{len(next_runs):04d}.parquet"
                pq.write_table(merged, out, compression="snappy",
                               row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE)
                del merged
                for p in chunk:
                    p.unlink(missing_ok=True)
                next_runs.append(out)
                print(f"[consolidate/positions] pass {pass_num} merge {len(next_runs)}/{-(-len(run_paths)//group)}  "
                      f"[{time.monotonic()-t0:.1f}s]", flush=True)
            run_paths = next_runs

        run_paths[0].replace(GLOBAL_STATS_DIR / POSITION_FILE)
        for f in pos_files:
            f.unlink(missing_ok=True)

        size_mb = (GLOBAL_STATS_DIR / POSITION_FILE).stat().st_size / 1e6
        print(f"[consolidate] positions: done  {total_rows:,} rows  {size_mb:.0f}MB  "
              f"[{time.monotonic()-t0:.1f}s]")

    finally:
        _shutil.rmtree(runs_dir, ignore_errors=True)


_STATS_STAGING_DIRNAME = ".stats_staging"


def _stats_staging_dir() -> Path:
    return GLOBAL_STATS_DIR / _STATS_STAGING_DIRNAME


def _level_marker_path(staging_dir: Path, depth: int) -> Path:
    return staging_dir / ".done" / f"level_{depth:04d}"


def _finalize_stats(staging_dir: Path) -> None:
    """Sort each stat type's per-level chunks by taxon_key into its final
    global file — one DuckDB pass per type, replacing what used to be a
    separate glob-thousands-of-per-taxon-files consolidation stage, since
    run_stats() now streams straight into these staged chunks instead of
    one file per taxon directory."""
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    kinds = [*_STATS_FILES, ("phenology_counts", "phenology_counts.parquet")]
    con = duckdb.connect()
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

    _consolidate_positions(t0)

    # Remove accumulator state and rank catalogs — no longer needed once
    # rankings have been computed from them.
    removed = 0
    patterns = [
        "species.parquet", "subspecies.parquet", "genus.parquet",
        "family.parquet", "order.parquet", "variety.parquet", "form.parquet",
        ".acc",
        POSITION_FILE,  # old per-taxon positions files (new approach never creates them)
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

    _level_pass(
        by_depth, stats_levels, _task, max_workers=STATS_WORKERS, label="stats", total=total,
        should_skip_level=_should_skip, on_level_start=_on_start, on_level_end=_on_end,
    )

    print("[stats] finalizing global stats files...")
    _finalize_stats(staging_dir)


def run_rankings() -> None:
    layers, _, by_depth, _, rank_levels, total = _setup()
    print(f"[process_tree] {total} taxa — rankings:{RANK_WORKERS} workers")
    removed = 0
    for pattern in ["tmp*.parquet", POSITION_CTX_GLOB, POSITION_FILE]:
        for p in TREE_ROOT.rglob(pattern):
            p.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"[process_tree] cleaned up {removed} stale position/tmp files")
    preload_stats_cache(layers)
    task = partial(compute_relative_ranks, layers=layers)
    _level_pass(by_depth, rank_levels, task, max_workers=RANK_WORKERS, label="rankings", total=total)


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
