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


def _load_temporal_ids() -> frozenset[str]:
    """Return temporal layer IDs from catalog (used to classify enrichment cols)."""
    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    ids: set[str] = set()
    for category in cat["categories"]:
        if category.get("id") == "temporal":
            for layer in category["layers"]:
                ids.add(layer["id"])
    return frozenset(ids)


def _is_temporal_col(col: str, temporal_ids: frozenset[str]) -> bool:
    return any(col.startswith(tid + "_") for tid in temporal_ids)


def _atomic_write(path: Path, table: pa.Table) -> None:
    from util.storage import atomic_write_parquet
    atomic_write_parquet(path, table, row_group_size=50_000)


def _carry_one(
    new_path: Path,
    old_path: Path,
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

    tree_cols = [c for c in enrich_cols if not _is_temporal_col(c, temporal_ids)]
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

    new_cols: dict[str, np.ndarray] = {}
    for col in tree_cols:
        src = f"_old_{col}"
        if src in merged.columns:
            new_cols[col] = np.where(coords_same, merged[src].values, np.nan)

    for col in temp_cols:
        src = f"_old_{col}"
        if src in merged.columns:
            new_cols[col] = np.where(coords_same & ts_same, merged[src].values, np.nan)

    if new_cols:
        result = pd.concat(
            [new_df, pd.DataFrame(new_cols, index=new_df.index)],
            axis=1,
        )
    else:
        result = new_df

    _atomic_write(new_path, pa.Table.from_pandas(result, preserve_index=False))
    return n_carried, n_changed, n_new_obs, n_total


def main() -> None:
    if not OLD_TREE_PATH.exists():
        print("[carry_forward] no old tree at data/tmp/old_tree/ — first run, skipping")
        return

    temporal_ids = _load_temporal_ids()
    t0 = time.monotonic()
    total_rows = 0
    total_carried = 0
    total_changed = 0
    total_new_obs = 0

    for new_path in sorted(TREE_ROOT.rglob(OCCURRENCE_FILE)):
        rel = new_path.relative_to(TREE_ROOT)
        old_path = OLD_TREE_PATH / rel
        if not old_path.exists():
            # New taxon this week — all rows are new observations
            n = pq.read_metadata(new_path).num_rows
            total_rows += n
            total_new_obs += n
        else:
            carried, changed, new_obs, n_total = _carry_one(new_path, old_path, temporal_ids)
            total_rows += n_total
            total_carried += carried
            total_changed += changed
            total_new_obs += new_obs

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
