"""Compute occurrence indexes, summary stats, and relative ranks for the tree.

The index and stats passes are parallelised per node so they can crunch through
the taxonomy quickly, and they run concurrently as separate jobs so both
artifacts are always produced after enrichment. Relative rank catalogs/indexes
only run after stats complete because the ranking logic depends on those
aggregates being available.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable
import time

import util.indexing as indexing
import util.summary_stats as summary_stats
import util.taxa_navigation as taxa_navigation
from util.config import load_config


CONFIG = load_config("global")

process_tree_ranks_only = False


index_workers = 4

pending_task_multiplier = 4

stats_workers = 4

memory_high_watermark = 0.8

def _memory_usage_ratio() -> float:
    """Returns memory usage ratio based on /proc/meminfo when available."""
    try:
        total = available = None
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = float(line.split()[1])
                if total is not None and available is not None:
                    break
        if total is None or available is None or total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (available / total)))
    except OSError:
        return 0.0

def _iter_taxa(root_taxon_id: str) -> Iterable[taxa_navigation.TaxonRecord]:
    root = taxa_navigation.get_taxon_by_id(root_taxon_id)
    if root is None:
        raise ValueError(f"Unknown taxon id {root_taxon_id}")
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = taxa_navigation.get_children(node["taxon_key"])
        if children:
            stack.extend(children)


def _run_over_tree(
    task_fn: Callable[[taxa_navigation.TaxonRecord], None],
    root_taxon_id: str,
    *,
    max_workers: int,
    pending_multiplier: int,
) -> None:
    """Run `task_fn` for every node with limited outstanding futures."""
    pending = deque()
    max_pending = max_workers * max(1, pending_multiplier)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for node in _iter_taxa(root_taxon_id):
            while pending and _memory_usage_ratio() >= memory_high_watermark:
                future = pending.popleft()
                try:
                    future.result()
                except MemoryError:
                    print("[process] MemoryError while running task; continuing.")
                except Exception as exc:
                    print(f"[process] task failed: {exc}")
                time.sleep(0.05)
            pending.append(executor.submit(task_fn, node))
            if len(pending) >= max_pending:
                future = pending.popleft()
                try:
                    future.result()
                except MemoryError:
                    print("[process] MemoryError while running task; continuing.")
                except Exception as exc:
                    print(f"[process] task failed: {exc}")
        while pending:
            future = pending.popleft()
            try:
                future.result()
            except MemoryError:
                print("[process] MemoryError while running task; continuing.")
            except Exception as exc:
                print(f"[process] task failed: {exc}")


def _build_index_for_node(node: taxa_navigation.TaxonRecord) -> None:
    canonical_rank = taxa_navigation.canonical_rank(node["rank"] or "")
    if canonical_rank not in CONFIG.leaf_rank_set:
        return

    path = Path(node["path"])
    try:
        indexing.build_index_parquet(path)
        print(f"indexed {str(path).split("/")[-1]}")
    except Exception as exc:
        print(f"failed indexing {path}: {exc}")


def _compute_stats_for_node(node: taxa_navigation.TaxonRecord) -> None:
    taxon_path = Path(node["path"])
    canonical_rank = taxa_navigation.canonical_rank(node["rank"] or "")
    streaming = canonical_rank not in CONFIG.leaf_rank_set
    with summary_stats.stats_context(taxon_path):
        summary_stats.numeric_column_stats(streaming=streaming)
    print(f"built summary stats for {str(taxon_path).split("/")[-1]}")


def compute_indexes_for_tree(
    root_taxon_id: str,
    *,
    max_workers: int,
    pending_multiplier: int,
) -> None:
    """Build occurrence indexes for descendant leaves."""
    _run_over_tree(
        _build_index_for_node,
        root_taxon_id,
        max_workers=max_workers,
        pending_multiplier=pending_multiplier,
    )


def compute_stats_for_tree(
    root_taxon_id: str,
    *,
    max_workers: int,
    pending_multiplier: int,
) -> None:
    """Compute per-node numeric stats used by downstream ranking logic."""
    _run_over_tree(
        _compute_stats_for_node,
        root_taxon_id,
        max_workers=max_workers,
        pending_multiplier=pending_multiplier,
    )


def compute_relative_ranks(root_taxon_id: str) -> None:
    """Sequential DFS pass that builds catalogs + rank indexes."""
    root = taxa_navigation.get_taxon_by_id(root_taxon_id)
    if root is None:
        raise ValueError(f"Unknown taxon id {root_taxon_id}")
    stack = [root]
    while stack:
        node = stack.pop()
        try:
            indexing.build_descendant_catalogs_for_ancestor(node["taxon_key"])
            indexing.build_rank_indexes_for_ancestor(node["taxon_key"])
            print(
                "built descendant catalogs and rank indexes for "
                f"{node['scientific_name']}"
            )
        except Exception as exc:
            print(f"failed building catalogs for {node['taxon_key']}: {exc}")
        children = taxa_navigation.get_children(node["taxon_key"])
        if children:
            stack.extend(children)


def process_tree(
    root_taxon_id: str = CONFIG.root_taxon_id,
    *,
    index_workers: int = index_workers,
    stats_workers: int = stats_workers,
    pending_multiplier: int = pending_task_multiplier,
) -> None:
    """Entry point when enrichment has populated occurrence.parquet files."""
    pending_multiplier = max(1, pending_multiplier)

    if process_tree_ranks_only:
        compute_relative_ranks(root_taxon_id)
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        indexes_future = executor.submit(
            compute_indexes_for_tree,
            root_taxon_id,
            max_workers=max(1, index_workers),
            pending_multiplier=pending_multiplier,
        )
        stats_future = executor.submit(
            compute_stats_for_tree,
            root_taxon_id,
            max_workers=max(1, stats_workers),
            pending_multiplier=pending_multiplier,
        )

        # Stats must complete before relative rankings begin.
        stats_future.result()
        print("Stats finished; starting relative ranking catalogs/indexes…")
        compute_relative_ranks(root_taxon_id)
        indexes_future.result()


def main() -> None:
    process_tree()


if __name__ == "__main__":
    main()
