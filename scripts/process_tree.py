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

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.config import load_config
from util.rankings import POSITION_FILE, POSITION_CTX_GLOB, compute_relative_ranks, preload_stats_cache
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    PHENOLOGY_COUNTS_FILE,
    TREE_ROOT,
    compute_taxon_stats,
    stats_complete,
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
    _WINDOW = 500
    recent: deque[float] = deque(maxlen=_WINDOW)

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
    ("circular_stats",  CIRCULAR_STATS_FILE),
    ("density",         DENSITY_FILE),
    ("positions",       POSITION_FILE),
]

_CONSOLIDATION_ROW_GROUP_SIZE = 256


def run_consolidation() -> None:
    """Merge per-node stats files into global files under data/taxonomy/global/."""
    GLOBAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Merge per-context position files → one relative_ranks_positions.parquet per taxon
    print("[consolidate] merging per-context position files...")
    ctx_dirs: dict[Path, list[Path]] = {}
    for ctx_file in TREE_ROOT.rglob(POSITION_CTX_GLOB):
        ctx_dirs.setdefault(ctx_file.parent, []).append(ctx_file)
    n_merged = 0
    for taxon_dir, ctx_files in ctx_dirs.items():
        try:
            combined = pa.concat_tables([pq.read_table(f) for f in ctx_files])
            atomic_write_parquet(taxon_dir / POSITION_FILE, combined, row_group_size=256)
            for f in ctx_files:
                f.unlink(missing_ok=True)
            n_merged += 1
        except Exception as e:
            print(f"[consolidate] positions merge failed {taxon_dir.name}: {e}")
    print(f"[consolidate] merged positions for {n_merged:,} taxa  [{time.monotonic()-t0:.1f}s]")

    print("[consolidate] building global stats files")

    tmp_dir = GLOBAL_STATS_DIR / ".tmp_consolidate"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for label, filename in _STATS_FILES:
            frames: list[pa.Table] = []
            for path in sorted(TREE_ROOT.rglob(filename)):
                taxon_key = path.parent.name.rsplit("_", 1)[-1]
                tbl = pq.read_table(path)
                tbl = tbl.append_column(
                    pa.field("taxon_key", pa.string()),
                    pa.array([taxon_key] * len(tbl), type=pa.string()),
                )
                frames.append(tbl)

            if not frames:
                print(f"[consolidate] {label}: no files found, skipping")
                continue

            combined = pa.concat_tables(frames, promote_options="default")
            sort_idx = pc.sort_indices(combined, sort_keys=[("taxon_key", "ascending")])
            combined = combined.take(sort_idx)
            atomic_write_parquet(
                tmp_dir / filename, combined,
                row_group_size=_CONSOLIDATION_ROW_GROUP_SIZE,
            )
            print(
                f"[consolidate] {label}: {len(frames)} taxa"
                f"  {len(combined)} rows"
                f"  → {filename}"
                f"  [{time.monotonic() - t0:.1f}s]"
            )

        # All files written successfully — move into place atomically
        for _, filename in _STATS_FILES:
            src = tmp_dir / filename
            if src.exists():
                src.replace(GLOBAL_STATS_DIR / filename)
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)

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
    ]
    for filename in patterns:
        for path in TREE_ROOT.rglob(filename):
            path.unlink()
            removed += 1
    for path in TREE_ROOT.rglob("tmp*.parquet"):
        path.unlink()
        removed += 1
    for path in TREE_ROOT.rglob(POSITION_CTX_GLOB):
        path.unlink()
        removed += 1
    print(f"[consolidate] removed {removed} per-node files")
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
