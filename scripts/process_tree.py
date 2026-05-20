"""
Compute per-taxon summary statistics, density graphs, and relative rankings.

Runs after enrich_tree has populated occurrence.parquets with GIS values.

Pass 1 — Stats (bottom-up, deepest first):
  Leaves use exact pandas/numpy stats; non-leaves stream descendant occurrence
  parquets with T-Digest approximations. Writes numerical_stats.parquet,
  nominal_stats.parquet, numerical_density.parquet, and occurrence_index.parquet.

Pass 2 — Rankings (top-down, shallowest first):
  Builds descendant rank catalogs ({rank}.parquet), rank index parquets
  ({rank}_index.parquet), and distributes position rows to each taxon's
  relative_ranks_positions.parquet. Pipelined per-ancestor so positions are
  written as soon as each ancestor's index is complete.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.config import load_config
from util.rankings import compute_relative_ranks
from util.stats import compute_taxon_stats
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

CATALOG_PATH = Path("config/gis/catalog.json")
STATS_WORKERS = 4
RANK_WORKERS = 4
LOG_INTERVAL = 50


def _load_layers() -> list[dict]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return [layer for category in cat["categories"] for layer in category["layers"]]


def _run_node(node: TaxonRecord, layers: list[dict]) -> str:
    compute_taxon_stats(node, layers)
    return node["taxon_key"]


def _run_rankings(node: TaxonRecord, layers: list[dict]) -> str:
    compute_relative_ranks(node, layers)
    return node["taxon_key"]


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _level_pass(
    by_depth: dict[int, list[TaxonRecord]],
    levels: list[int],
    task_fn,
    layers: list[dict],
    *,
    max_workers: int,
    label: str,
    total: int,
) -> tuple[int, int]:
    """Run task_fn over all taxa level by level, returning (completed, failed)."""
    completed = 0
    failed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for depth in levels:
            level_taxa = by_depth[depth]
            futures = {executor.submit(task_fn, node, layers): node for node in level_taxa}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                    completed += 1
                    if completed % LOG_INTERVAL == 0 or completed == total:
                        elapsed = time.monotonic() - t0
                        rate = completed / elapsed
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(
                            f"[{label}] {completed}/{total}"
                            f"  elapsed={_fmt_duration(elapsed)}"
                            f"  eta={_fmt_duration(eta)}"
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


def main() -> None:
    layers = _load_layers()
    root = get_taxon_by_id(CONFIG.plantae_key)
    if root is None:
        print(f"[process_tree] root taxon {CONFIG.plantae_key} not found")
        return

    all_taxa = list(iter_descendants(root, include_self=True))
    total = len(all_taxa)

    by_depth: dict[int, list[TaxonRecord]] = defaultdict(list)
    for t in all_taxa:
        by_depth[t["path"].count("/")].append(t)

    stats_levels = sorted(by_depth.keys(), reverse=True)   # deepest first
    rank_levels = sorted(by_depth.keys())                   # shallowest first

    print(
        f"[process_tree] {total} taxa across {len(stats_levels)} levels"
        f" — stats:{STATS_WORKERS} workers  rankings:{RANK_WORKERS} workers"
    )

    _level_pass(
        by_depth, stats_levels, _run_node, layers,
        max_workers=STATS_WORKERS, label="stats", total=total,
    )

    print("[process_tree] stats complete — starting rankings pass")

    _level_pass(
        by_depth, rank_levels, _run_rankings, layers,
        max_workers=RANK_WORKERS, label="rankings", total=total,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
