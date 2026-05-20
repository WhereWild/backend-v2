"""
Compute per-taxon summary statistics and density graphs for all GIS layers.

Runs after enrich_tree has populated occurrence.parquets with GIS values.
Stats are computed in parallel; leaf taxa use exact pandas/numpy stats,
non-leaf taxa stream descendant occurrence parquets with T-Digest approximations.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.config import load_config
from util.stats import compute_taxon_stats
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

CATALOG_PATH = Path("config/gis/catalog.json")
STATS_WORKERS = 4
LOG_INTERVAL = 50


def _load_layers() -> list[dict]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return [layer for category in cat["categories"] for layer in category["layers"]]


def _run_node(node: TaxonRecord, layers: list[dict]) -> str:
    compute_taxon_stats(node, layers)
    return node["taxon_key"]


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> None:
    layers = _load_layers()
    root = get_taxon_by_id(CONFIG.plantae_key)
    if root is None:
        print(f"[process_tree] root taxon {CONFIG.plantae_key} not found")
        return

    all_taxa = list(iter_descendants(root, include_self=True))
    total = len(all_taxa)

    # Group by path depth so leaves (deepest) are processed first.
    # This ensures children's occurrence_index.parquet files exist before
    # their parents try to read from them during non-leaf index building.
    by_depth: dict[int, list[TaxonRecord]] = defaultdict(list)
    for t in all_taxa:
        by_depth[t["path"].count("/")].append(t)
    levels = sorted(by_depth.keys(), reverse=True)  # deepest first

    print(f"[process_tree] {total} taxa across {len(levels)} levels — {STATS_WORKERS} workers")

    completed = 0
    failed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=STATS_WORKERS) as executor:
        for depth in levels:
            level_taxa = by_depth[depth]
            futures = {executor.submit(_run_node, node, layers): node for node in level_taxa}
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
                            f"[process_tree] {completed}/{total}"
                            f"  elapsed={_fmt_duration(elapsed)}"
                            f"  eta={_fmt_duration(eta)}"
                            f"  ({node['rank']} {node['scientific_name']})"
                        )
                except Exception as exc:
                    failed += 1
                    elapsed = time.monotonic() - t0
                    print(
                        f"[process_tree] FAIL [{elapsed:.0f}s]"
                        f"  {node['rank']} {node['scientific_name']}: {exc}"
                    )

    elapsed = time.monotonic() - t0
    print(f"[process_tree] done — {completed} ok, {failed} failed, {_fmt_duration(elapsed)} total")


if __name__ == "__main__":  # pragma: no cover
    main()
