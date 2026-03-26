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
import util.gis_lookup as gis_lookup
import util.summary_stats as summary_stats
import util.taxa_navigation as taxa_navigation
from util.config import load_config
import pyarrow.parquet as pq


CONFIG = load_config("global")

process_tree_ranks_only = CONFIG.process_tree_ranks_only
process_tree_indexes_only = CONFIG.process_tree_indexes_only


index_workers = 4

pending_task_multiplier = 4

stats_workers = 4

rank_workers = 4

memory_high_watermark = 0.8

skip_existing_indexes = True

_layer_catalog = gis_lookup.load_layer_metadata()


def _expected_layer_targets(schema) -> dict[str, str]:
    """Resolve catalog-aware target columns expected in index/stats artifacts."""
    return dict(
        indexing.index_targets_for_columns(
            set(schema.names),
            layer_catalog=_layer_catalog,
        )
    )


def _index_is_current(node_path: Path) -> bool:
    index_path = node_path / "occurrence_index.parquet"
    if not index_path.exists():
        return False
    data_path = node_path / CONFIG.occurrence_parquet_filename
    if not data_path.exists():
        # A parent taxon may have no direct occurrence parquet while children do.
        # In that case we must rebuild so child rows can be indexed.
        taxon_key = taxa_navigation.taxon_key_from_path(node_path)
        if taxon_key:
            for child in taxa_navigation.get_children(taxon_key):
                child_data_path = Path(child["path"]) / CONFIG.occurrence_parquet_filename
                if child_data_path.exists():
                    return False
        return True
    try:
        data_schema = pq.read_schema(data_path)
    except Exception:
        return True
    expected_layers = set(_expected_layer_targets(data_schema).keys())
    if not expected_layers:
        return True
    try:
        index_schema = pq.read_schema(index_path)
    except Exception:
        return False
    return expected_layers.issubset(set(index_schema.names))


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
    if skip_existing_indexes and _index_is_current(path):
        print(f"skip indexing {str(path).split('/')[-1]} (already built)")
        return
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
        summary_stats.write_density_graph(taxon_path)
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


def _build_rank_artifacts_for_node(
    node: taxa_navigation.TaxonRecord,
    *,
    has_descendants: bool,
) -> None:
    try:
        if has_descendants:
            indexing.build_descendant_catalogs_for_ancestor(
                node["taxon_key"],
                verbose=False,
            )
            indexing.build_rank_indexes_for_ancestor(
                node["taxon_key"],
                verbose=False,
            )
            print(
                "built descendant catalogs and rank indexes for "
                f"{node['scientific_name']}"
            )
    except Exception as exc:
        print(f"failed building catalogs for {node['taxon_key']}: {exc}")


def compute_relative_ranks(
    root_taxon_id: str,
    *,
    max_workers: int = rank_workers,
) -> None:
    """Level-order pass that builds catalogs + rank indexes."""
    indexing.reset_rank_build_caches()
    root = taxa_navigation.get_taxon_by_id(root_taxon_id)
    if root is None:
        raise ValueError(f"Unknown taxon id {root_taxon_id}")
    worker_count = max(1, max_workers)
    current_level: list[taxa_navigation.TaxonRecord] = [root]
    while current_level:
        level_children = {
            node["taxon_key"]: taxa_navigation.get_children(node["taxon_key"])
            for node in current_level
        }
        if worker_count == 1 or len(current_level) == 1:
            for node in current_level:
                _build_rank_artifacts_for_node(
                    node,
                    has_descendants=bool(level_children.get(node["taxon_key"])),
                )
        else:
            with ThreadPoolExecutor(
                max_workers=min(worker_count, len(current_level))
            ) as executor:
                futures = [
                    executor.submit(
                        _build_rank_artifacts_for_node,
                        node,
                        has_descendants=bool(level_children.get(node["taxon_key"])),
                    )
                    for node in current_level
                ]
                for future in futures:
                    future.result()
        next_level: list[taxa_navigation.TaxonRecord] = []
        for node in current_level:
            children = level_children.get(node["taxon_key"])
            if children:
                next_level.extend(children)
        current_level = next_level


def process_tree(
    root_taxon_id: str = CONFIG.root_taxon_id,
    *,
    index_workers: int = index_workers,
    stats_workers: int = stats_workers,
    pending_multiplier: int = pending_task_multiplier,
) -> None:
    """Entry point when enrichment has populated occurrence.parquet files."""
    pending_multiplier = max(1, pending_multiplier)

    if process_tree_indexes_only and process_tree_ranks_only:
        raise ValueError(
            "Config conflict: both process_tree_indexes_only and "
            "process_tree_ranks_only are enabled."
        )

    if process_tree_indexes_only:
        compute_indexes_for_tree(
            root_taxon_id,
            max_workers=max(1, index_workers),
            pending_multiplier=pending_multiplier,
        )
        return

    if process_tree_ranks_only:
        compute_relative_ranks(
            root_taxon_id,
            max_workers=max(1, rank_workers),
        )
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
        compute_relative_ranks(
            root_taxon_id,
            max_workers=max(1, rank_workers),
        )
        indexes_future.result()


def main() -> None:
    process_tree()


if __name__ == "__main__":
    main()
