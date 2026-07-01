# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Copy enrichment columns from the previous taxonomy tree into the freshly
populated one, so enrich_tree and enrich_temporal only process new or
changed observations.

Pipeline stage: runs after populate_tree, before enrich_tree.
If data/tmp/old_tree/ does not exist (first run), this is a no-op.

Observation matching is by catalogNumber at the *same tree path only*.
Re-identified observations (taxon changed week-to-week) are not found and
treated as new — they get fully re-enriched, which is correct.

Copy rules (for each matched catalogNumber):
  coords changed              → copy nothing (full re-enrich: tree + temporal)
  coords same, ts changed     → copy tree (GIS) cols; leave temporal cols null
  coords same, ts same        → copy ALL enrichment cols
  new observation             → copy nothing
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OLD_TREE_PATH = Path("data/tmp/old_tree")
TREE_ROOT = Path("data/taxonomy/tree")
CATALOG_PATH = Path("config/gis/catalog.json")
SYNC_STATE_PATH = Path("data/sync_state.json")
OCCURRENCE_FILE = "occurrence.parquet"

_BASE_COLS = frozenset([
    "decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx",
    "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
])


def _load_catalog_ids() -> tuple[frozenset[str], frozenset[str]]:
    """Return (static_layer_ids, temporal_layer_ids) from catalog."""
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    static_ids: set[str] = set()
    temporal_ids: set[str] = set()
    for category in cat["categories"]:
        is_temporal = category.get("id") == "temporal"
        for layer in category["layers"]:
            (temporal_ids if is_temporal else static_ids).add(layer["id"])
    return frozenset(static_ids), frozenset(temporal_ids)


def _is_temporal_col(col: str, temporal_ids: frozenset[str]) -> bool:
    return any(col.startswith(tid + "_") for tid in temporal_ids)


def _atomic_write(path: Path, table: pa.Table) -> None:
    from util.storage import atomic_write_parquet
    atomic_write_parquet(path, table, row_group_size=50_000)


def _carry_one(
    new_path: Path,
    old_path: Path,
    static_ids: frozenset[str],
    temporal_ids: frozenset[str],
) -> tuple[int, int, int, int]:
    """Carry enrichment columns from old_path into new_path for matching rows.

    Returns (n_carried, n_changed, n_new_obs, n_total) where:
      n_carried  = catalogNumber matched + same coords (enrichment copied)
      n_changed  = catalogNumber matched + coords/ts differ (needs re-enrich)
      n_new_obs  = catalogNumber not in old parquet (new or re-ID'd observation)
      n_total    = total rows in new parquet
    """
    new_table = pq.read_table(new_path)
    old_table = pq.read_table(old_path)

    n_total = new_table.num_rows
    if n_total == 0 or old_table.num_rows == 0:
        return 0, 0, n_total, n_total

    # Find enrichment columns present in the old parquet
    old_schema_names = old_table.schema.names
    enrich_cols = [c for c in old_schema_names if c not in _BASE_COLS]
    if not enrich_cols:
        return 0, 0, n_total, n_total

    tree_cols = [c for c in enrich_cols if not _is_temporal_col(c, temporal_ids) and c in static_ids]
    temp_cols = [c for c in enrich_cols if _is_temporal_col(c, temporal_ids)]

    new_df = new_table.to_pandas()
    old_df = old_table.to_pandas()

    # Deduplicate old on catalogNumber (GBIF is unique, but be safe)
    old_df = old_df.drop_duplicates(subset=["catalogNumber"], keep="first")

    # Build old lookup: keep only key cols + enrich cols
    old_key_cols = (
        ["catalogNumber", "decimalLatitude", "decimalLongitude", "eventTimestamp"]
        + enrich_cols
    )
    old_key_cols = [c for c in old_key_cols if c in old_df.columns]
    old_sub = old_df[old_key_cols].rename(columns={
        "decimalLatitude": "_old_lat",
        "decimalLongitude": "_old_lon",
        "eventTimestamp": "_old_ts",
        **{c: f"_old_{c}" for c in enrich_cols},
    })

    # Left-join new rows onto old by catalogNumber
    merged = new_df.merge(old_sub, on="catalogNumber", how="left")

    found = merged["_old_lat"].notna()  # True where catalogNumber existed in old
    coords_same = (
        found
        & (merged["decimalLatitude"] == merged["_old_lat"])
        & (merged["decimalLongitude"] == merged["_old_lon"])
    )
    new_ts = merged["eventTimestamp"]
    old_ts = merged["_old_ts"]
    ts_same = (new_ts == old_ts) | (new_ts.isna() & old_ts.isna())

    n_carried = int(coords_same.sum())
    n_changed = int((found & ~coords_same).sum())
    n_new_obs = int((~found).sum())

    if n_carried == 0:
        return 0, n_changed, n_new_obs, n_total

    # Build enrichment columns as Arrow arrays directly so that:
    #   matched row, valid value  → that value (float)
    #   matched row, NaN value    → NaN (no-coverage sentinel, NOT null — skip re-enrichment)
    #   unmatched row             → null (needs enrichment)
    # Going through pandas (np.where → NaN → pa.Table.from_pandas) converts all NaN
    # to Arrow null, which causes the worklist to re-queue rows that already have
    # legitimate no-coverage sentinels.
    new_cols: dict[str, pa.Array] = {}
    coords_mask = pa.array(coords_same.to_numpy(), type=pa.bool_())
    ts_mask = pa.array((coords_same & ts_same).to_numpy(), type=pa.bool_())

    for col in tree_cols:
        src = f"_old_{col}"
        if src in merged.columns:
            old_arr = pa.array(merged[src].to_numpy(dtype=np.float64, na_value=np.nan), type=pa.float64(), from_pandas=False)
            new_cols[col] = pc.if_else(coords_mask, old_arr, None)

    for col in temp_cols:
        src = f"_old_{col}"
        if src in merged.columns:
            old_arr = pa.array(merged[src].to_numpy(dtype=np.float64, na_value=np.nan), type=pa.float64(), from_pandas=False)
            new_cols[col] = pc.if_else(ts_mask, old_arr, None)

    if new_cols:
        base_table = pa.Table.from_pandas(new_df, preserve_index=False)
        for col, arr in new_cols.items():
            if col in base_table.schema.names:
                base_table = base_table.set_column(base_table.schema.get_field_index(col), col, arr)
            else:
                base_table = base_table.append_column(col, arr)
        result_table = base_table
    else:
        result_table = pa.Table.from_pandas(new_df, preserve_index=False)

    _atomic_write(new_path, result_table)
    return n_carried, n_changed, n_new_obs, n_total


def main() -> None:
    if not OLD_TREE_PATH.exists():
        print("[carry_forward] no old tree at data/tmp/old_tree/ — first run, skipping")
        return

    static_ids, temporal_ids = _load_catalog_ids()
    t0 = time.monotonic()
    total_rows = 0
    total_carried = 0
    total_changed = 0
    total_new_obs = 0
    n_taxa = 0

    for new_path in sorted(TREE_ROOT.rglob(OCCURRENCE_FILE)):
        rel = new_path.relative_to(TREE_ROOT)
        old_path = OLD_TREE_PATH / rel
        if not old_path.exists():
            # New taxon this week — all rows are new observations
            n = pq.read_metadata(new_path).num_rows
            total_rows += n
            total_new_obs += n
        else:
            carried, changed, new_obs, n_total = _carry_one(new_path, old_path, static_ids, temporal_ids)
            total_rows += n_total
            total_carried += carried
            total_changed += changed
            total_new_obs += new_obs

        n_taxa += 1
        if n_taxa % 10_000 == 0:
            elapsed = time.monotonic() - t0
            pct = total_carried / total_rows * 100 if total_rows else 0.0
            print(f"[carry_forward] {n_taxa} taxa  {total_rows:,} rows  {pct:.1f}% carried  {elapsed:.0f}s")

    elapsed = time.monotonic() - t0
    carry_pct = total_carried / total_rows * 100 if total_rows else 0.0
    print(
        f"[carry_forward] {total_carried}/{total_rows} rows carried forward "
        f"({carry_pct:.1f}%)  |  {total_new_obs} new  {total_changed} changed  "
        f"({elapsed:.1f}s)"
    )

    stats = {
        "ts": datetime.now(UTC).isoformat(),
        "total_rows": total_rows,
        "carried": total_carried,
        "carry_pct": round(carry_pct, 2),
        "new_obs": total_new_obs,
        "changed": total_changed,
        "elapsed_s": round(elapsed, 2),
    }
    try:
        state = json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}
        state["carry_forward"] = stats
        SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        print(f"[carry_forward] could not write stats to sync_state.json: {exc}")

    shutil.rmtree(OLD_TREE_PATH)
    try:
        OLD_TREE_PATH.parent.rmdir()  # removes data/tmp/ if now empty
    except OSError:
        pass  # not empty — something else lives there, leave it
    print(f"[carry_forward] cleaned up {OLD_TREE_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
