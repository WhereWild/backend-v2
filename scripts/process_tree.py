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
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
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
    PHENOLOGY_COUNTS_FILE,
    TREE_ROOT,
    compute_taxon_stats,
)
from util.storage import atomic_write_parquet
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
) -> tuple[int, int]:
    """Run task_fn(node) over all taxa level by level, returning (completed, failed)."""
    completed = 0
    failed = 0
    t0 = time.monotonic()
    window = 500
    recent: deque[float] = deque(maxlen=window)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for depth in levels:
            level_taxa = by_depth[depth]
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


def run_consolidation() -> None:
    """Merge per-node stats files into global files under data/taxonomy/global/."""
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    print(f"[consolidate] building global stats files  [{time.monotonic()-t0:.1f}s]")

    tmp_dir = GLOBAL_STATS_DIR / ".tmp_consolidate"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for label, filename in _STATS_FILES:
        dest = GLOBAL_STATS_DIR / filename
        if dest.exists():
            print(f"[consolidate] {label}: already exists, skipping  [{time.monotonic()-t0:.1f}s]")
            continue
        tmp_path = tmp_dir / filename
        if tmp_path.exists():
            print(f"[consolidate] {label}: resuming from tmp  [{time.monotonic()-t0:.1f}s]")
            tmp_path.replace(dest)
            print(f"[consolidate] {label}: moved to global  [{time.monotonic()-t0:.1f}s]")
            continue
        print(f"[consolidate] {label}: scanning tree...  [{time.monotonic()-t0:.1f}s]")
        paths = sorted(TREE_ROOT.rglob(filename), key=lambda p: p.parent.name.rsplit("_", 1)[-1])
        print(f"[consolidate] {label}: {len(paths)} files found, reading...  [{time.monotonic()-t0:.1f}s]")
        if not paths:
            print(f"[consolidate] {label}: no files found, skipping")
            continue

        # Stream-write in batches to avoid holding all frames in memory at once.
        writer: pq.ParquetWriter | None = None
        batch: list[pa.Table] = []
        total_rows = 0
        batch_size = 100_000
        try:
            for n, path in enumerate(paths, 1):
                taxon_key = path.parent.name.rsplit("_", 1)[-1]
                tbl = pq.read_table(path)
                tbl = tbl.append_column(
                    pa.field("taxon_key", pa.string()),
                    pa.array([taxon_key] * len(tbl), type=pa.string()),
                )
                batch.append(tbl)
                if len(batch) >= batch_size:
                    chunk = pa.concat_tables(batch, promote_options="default")
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, chunk.schema)
                    writer.write_table(chunk, row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE)
                    total_rows += len(chunk)
                    batch.clear()
                if n % 10_000 == 0:
                    print(f"[consolidate] {label}: {n}/{len(paths)}  [{time.monotonic()-t0:.1f}s]")

            if batch:
                chunk = pa.concat_tables(batch, promote_options="default")
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, chunk.schema)
                writer.write_table(chunk, row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE)
                total_rows += len(chunk)
                batch.clear()
        finally:
            if writer:
                writer.close()

        # Move into final location only after fully written.
        tmp_path.replace(dest)
        print(
            f"[consolidate] {label}: {len(paths)} taxa  {total_rows} rows"
            f"  → {filename}  [{time.monotonic() - t0:.1f}s]"
        )

    # Extract per-taxon phenology counts from numerical_stats metadata → global file
    pheno_rows: list[dict] = []
    for path in sorted(TREE_ROOT.rglob(NUMERICAL_STATS_FILE)):
        taxon_key = path.parent.name.rsplit("_", 1)[-1]
        try:
            meta = pq.read_schema(path).metadata or {}
            raw = meta.get(b"phenology_counts")
            if raw:
                import json as _json
                for pheno_val, count in _json.loads(raw).items():
                    pheno_rows.append({"taxon_key": taxon_key, "phenology_value": pheno_val, "count": count})
        except Exception:
            pass
    if pheno_rows:
        pheno_tbl = pa.Table.from_pylist(pheno_rows)
        sort_idx = pc.sort_indices(pheno_tbl, sort_keys=[("taxon_key", "ascending")])
        pheno_tbl = pheno_tbl.take(sort_idx)
        atomic_write_parquet(
            GLOBAL_STATS_DIR / "phenology_counts.parquet", pheno_tbl,
            row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE,
        )
        n_taxa = len(set(r["taxon_key"] for r in pheno_rows))
        print(f"[consolidate] phenology: {len(pheno_rows)} rows for {n_taxa} taxa  [{time.monotonic() - t0:.1f}s]")

    # Remove per-node stats files, accumulator state, rank catalogs, and any tmp parquets
    removed = 0
    patterns = [filename for _, filename in _STATS_FILES] + [
        PHENOLOGY_COUNTS_FILE,
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
    layers, layer_meta, by_depth, stats_levels, _, total = _setup()
    print(f"[process_tree] {total} taxa — stats:{STATS_WORKERS} workers" + (" — RESUME" if resume else ""))
    task = partial(compute_taxon_stats, layers=layers, layer_meta=layer_meta, resume=resume)
    _level_pass(by_depth, stats_levels, task, max_workers=STATS_WORKERS, label="stats", total=total)


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
        help="Skip taxa whose stats files are already written (for restarts).",
    )
    args, _ = parser.parse_known_args()

    try:
        layers, layer_meta, by_depth, stats_levels, rank_levels, total = _setup()
    except RuntimeError as exc:
        print(str(exc))
        return

    print(
        f"[process_tree] {total} taxa across {len(stats_levels)} levels"
        f" — stats:{STATS_WORKERS} workers  rankings:{RANK_WORKERS} workers"
        f" — phase:{args.phase}"
        + (" — RESUME" if args.resume else "")
    )

    if args.phase in ("stats", "all"):
        task = partial(compute_taxon_stats, layers=layers, layer_meta=layer_meta, resume=args.resume)
        _level_pass(
            by_depth, stats_levels, task,
            max_workers=STATS_WORKERS, label="stats", total=total,
        )
        if args.phase == "all":
            print("[process_tree] stats complete — starting rankings pass")

    if args.phase in ("rankings", "all"):
        removed = 0
        for pattern in ["tmp*.parquet", POSITION_CTX_GLOB, POSITION_FILE]:
            for p in TREE_ROOT.rglob(pattern):
                p.unlink(missing_ok=True)
                removed += 1
        if removed:
            print(f"[process_tree] cleaned up {removed} stale position/tmp files")
        preload_stats_cache(layers)
        task = partial(compute_relative_ranks, layers=layers)
        _level_pass(
            by_depth, rank_levels, task,
            max_workers=RANK_WORKERS, label="rankings", total=total,
        )
        if args.phase == "all":
            print("[process_tree] rankings complete — starting consolidation pass")

    if args.phase in ("consolidate", "all"):
        run_consolidation()


if __name__ == "__main__":  # pragma: no cover
    main()
