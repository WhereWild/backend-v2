"""
Compute per-taxon summary statistics and density graphs for all GIS layers.

Runs after enrich_tree has populated occurrence.parquets with GIS values.
Stats are computed in parallel; leaf taxa use exact pandas/numpy stats,
non-leaf taxa stream descendant occurrence parquets with T-Digest approximations.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.config import load_config
from util.stats import compute_taxon_stats
from util.taxa import TaxonRecord, get_taxon_by_id, iter_descendants

CONFIG = load_config("global")

CATALOG_PATH = Path("config/gis/catalog.json")
STATS_WORKERS = 4


def _load_layers() -> list[dict]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return [layer for category in cat["categories"] for layer in category["layers"]]


def _run_node(node: TaxonRecord, layers: list[dict]) -> str:
    compute_taxon_stats(node, layers)
    return node["taxon_key"]


def main() -> None:
    layers = _load_layers()
    root = get_taxon_by_id(CONFIG.plantae_key)
    if root is None:
        print(f"[process_tree] root taxon {CONFIG.plantae_key} not found")
        return

    taxa = list(iter_descendants(root, include_self=True))
    total = len(taxa)
    print(f"[process_tree] computing stats for {total} taxa")

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=STATS_WORKERS) as executor:
        futures = {executor.submit(_run_node, node, layers): node for node in taxa}
        for future in as_completed(futures):
            node = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 1000 == 0:
                    print(f"[process_tree] {completed}/{total}")
            except Exception as exc:
                failed += 1
                print(f"[process_tree] failed {node['taxon_key']} ({node['scientific_name']}): {exc}")

    print(f"[process_tree] done — {completed} completed, {failed} failed")


if __name__ == "__main__":  # pragma: no cover
    main()
