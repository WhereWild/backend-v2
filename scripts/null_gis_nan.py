# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Convert stuck NaN → null in GIS float columns across all occurrence parquets.

NaN (float sentinel) is indistinguishable from "no coverage" by enrich_tree's
worklist scan (pc.is_null returns False for NaN), so rows with NaN won't be
re-queued for enrichment. This script converts GIS-column NaN → null so they
become visible again.

Only touches GIS float columns — skips base/coordinate columns and all
temporal columns (which use NaN legitimately as a no-coverage sentinel
and have their own re-enrichment logic).

Usage:
    python -m scripts.null_gis_nan
    DRY_RUN=1 python -m scripts.null_gis_nan   # print stats without writing
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from util.storage import atomic_write_parquet

TREE_ROOT = Path("data/taxonomy/tree")
CATALOG_PATH = Path("config/gis/catalog.json")
OCCURRENCE_FILE = "occurrence.parquet"

DRY_RUN = os.environ.get("DRY_RUN", "0") != "0"

# Columns written by populate_tree — never GIS, never touch.
_BASE_COLS = frozenset([
    "decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx",
    "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
])


def _temporal_layer_ids() -> frozenset[str]:
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    return frozenset(
        layer["id"]
        for category in cat["categories"]
        if category.get("id") == "temporal"
        for layer in category["layers"]
    )


def _is_temporal_col(col: str, temporal_ids: frozenset[str]) -> bool:
    return any(col.startswith(tid + "_") for tid in temporal_ids)


def _process_file(path: Path, temporal_ids: frozenset[str]) -> tuple[int, int]:
    """Return (nan_cells_fixed, cols_fixed). Writes file in place unless DRY_RUN."""
    table = pq.read_table(path)
    if table.num_rows == 0:
        return 0, 0

    new_columns: dict[str, pa.ChunkedArray] = {}
    for name in table.schema.names:
        if name in _BASE_COLS or _is_temporal_col(name, temporal_ids):
            continue
        col = table.column(name)
        if not pa.types.is_floating(col.type):
            continue
        n_nan = pc.sum(pc.is_nan(col)).as_py() or 0
        if n_nan == 0:
            continue
        # Replace NaN with null; leave existing nulls as null.
        new_columns[name] = pc.if_else(pc.is_nan(col), None, col)

    if not new_columns:
        return 0, 0

    total_nan = sum(
        (pc.sum(pc.is_nan(table.column(c))).as_py() or 0) for c in new_columns
    )

    if not DRY_RUN:
        updated = table
        for col_name, new_col in new_columns.items():
            idx = updated.schema.get_field_index(col_name)
            updated = updated.set_column(idx, col_name, new_col)
        atomic_write_parquet(path, updated, row_group_size=50_000)

    return total_nan, len(new_columns)


def main() -> None:
    if DRY_RUN:
        print("[null_gis_nan] DRY RUN — no files will be written")

    temporal_ids = _temporal_layer_ids()
    print(f"[null_gis_nan] temporal layer count: {len(temporal_ids)}")

    files = sorted(TREE_ROOT.rglob(OCCURRENCE_FILE))
    print(f"[null_gis_nan] scanning {len(files)} occurrence files")

    total_files_changed = 0
    total_nan_cells = 0
    total_cols_fixed = 0

    for i, path in enumerate(files, 1):
        try:
            nan_cells, cols_fixed = _process_file(path, temporal_ids)
        except Exception as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr)
            continue

        if cols_fixed > 0:
            total_files_changed += 1
            total_nan_cells += nan_cells
            total_cols_fixed += cols_fixed
            action = "would fix" if DRY_RUN else "fixed"
            print(f"[{action}] {path.parent.name}: {cols_fixed} cols, {nan_cells} NaN cells")

        if i % 5000 == 0:
            print(f"[progress] {i}/{len(files)} files, {total_files_changed} changed so far")

    verb = "would change" if DRY_RUN else "changed"
    print(
        f"[done] {verb} {total_files_changed} files, "
        f"{total_cols_fixed} col-instances, "
        f"{total_nan_cells} NaN cells → null"
    )


if __name__ == "__main__":
    main()
