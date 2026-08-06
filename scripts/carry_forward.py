# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Copy enrichment columns from the previous occurrences.parquet into the
freshly populated one, so enrich_tree and enrich_temporal only process new
or changed observations.

Pipeline stage: runs after populate_tree, before enrich_tree.
If data/tmp/old_occurrences.parquet does not exist (first run), this is a
no-op.

Observation matching is by catalogNumber, globally (the consolidated file
has no more per-taxon-path scoping). Re-identified observations (taxon
changed week-to-week) still match by catalogNumber even if their taxon_key
changed — this is a behavior improvement over the old per-path matching,
which treated any re-identified observation as brand new. Coordinate/
timestamp changes are still detected the same way and still force
re-enrichment.

Copy rules (for each matched catalogNumber):
  coords changed              → copy nothing (full re-enrich: tree + temporal)
  coords same, ts changed     → copy tree (GIS) cols; leave temporal cols null
  coords same, ts same        → copy ALL enrichment cols
  new observation             → copy nothing

Implemented as a single DuckDB LEFT JOIN + rewrite instead of the old
per-taxon-file pandas merge loop.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb

OLD_OCCURRENCES_PATH = Path("data/tmp/old_occurrences.parquet")
OCCURRENCES_FILE = Path("data/taxonomy/occurrences.parquet")
CATALOG_PATH = Path("config/gis/catalog.json")
SYNC_STATE_PATH = Path("data/sync_state.json")

_BASE_COLS = frozenset([
    "decimalLatitude", "decimalLongitude", "catalogNumber", "hilbertIdx",
    "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
    "taxon_key", "mediaUrl", "mediaAttribution", "mediaLicense",
])

# The LEFT JOIN + ORDER BY below runs over the full new occurrences table
# (tens of millions of rows) against the old one. A bare duckdb.connect()
# has no temp_directory, so on an in-memory connection it never spills to
# disk and just keeps growing until the OS OOM-kills it — the same failure
# mode confirmed in scripts/populate_tree.py's _consolidate for a plain sort
# alone; a join on top of a sort is at least as heavy. Capping memory_limit
# well under total RAM and giving it a real temp_directory to spill into
# turns that into "slower", not "crashes".
_DUCKDB_SPILL_DIR = Path("data/tmp/duckdb_spill")
_DUCKDB_MEMORY_LIMIT = "40GB"


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    _DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    # The LEFT JOIN below is followed by our own explicit ORDER BY, which
    # already overrides whatever row order the join emits — so there's no
    # reason to pay for insertion-order preservation too (its default-on
    # per-thread buffering to reconstruct single-threaded row order was
    # confirmed in practice to be what pushed this query over memory_limit:
    # "failed to pin block... 37.2 GiB/37.2 GiB used" — DuckDB's own error
    # message names this as the first thing to try).
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


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


def _table_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0").fetchall()]


def main() -> None:
    if not OLD_OCCURRENCES_PATH.exists():
        print(f"[carry_forward] no old occurrences at {OLD_OCCURRENCES_PATH} — first run, skipping")
        return
    if not OCCURRENCES_FILE.exists():
        print("[carry_forward] no new occurrences.parquet — nothing to carry into, skipping")
        return

    static_ids, temporal_ids = _load_catalog_ids()
    t0 = time.monotonic()
    con = _duckdb_connect()

    old_cols = _table_columns(con, OLD_OCCURRENCES_PATH)
    enrich_cols = [c for c in old_cols if c not in _BASE_COLS]
    tree_cols = [c for c in enrich_cols if not _is_temporal_col(c, temporal_ids) and c in static_ids]
    temp_cols = [c for c in enrich_cols if _is_temporal_col(c, temporal_ids)]

    n_total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OCCURRENCES_FILE.as_posix()}')"
    ).fetchone()[0]

    if n_total == 0 or not enrich_cols:
        total_new_obs = n_total
        total_carried = 0
        total_changed = 0
        con.close()
    else:
        # catalogNumber is the iNaturalist observation ID, unique by
        # construction — no dedup needed here (see populate_tree._consolidate
        # for the same reasoning). A QUALIFY/row_number() pass over the old
        # side would just be paying for a guarantee the data already gives.
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW old_occ AS
            SELECT * FROM read_parquet('{OLD_OCCURRENCES_PATH.as_posix()}')
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW new_occ AS
            SELECT * FROM read_parquet('{OCCURRENCES_FILE.as_posix()}')
        """)

        coords_same = (
            'o."catalogNumber" IS NOT NULL '
            'AND n."decimalLatitude" = o."decimalLatitude" '
            'AND n."decimalLongitude" = o."decimalLongitude"'
        )
        ts_same = (
            '(n."eventTimestamp" = o."eventTimestamp" '
            'OR (n."eventTimestamp" IS NULL AND o."eventTimestamp" IS NULL))'
        )

        stats_row = con.execute(f"""
            SELECT
                SUM(CASE WHEN {coords_same} THEN 1 ELSE 0 END) AS n_carried,
                SUM(CASE WHEN o."catalogNumber" IS NOT NULL AND NOT ({coords_same}) THEN 1 ELSE 0 END) AS n_changed,
                SUM(CASE WHEN o."catalogNumber" IS NULL THEN 1 ELSE 0 END) AS n_new_obs
            FROM new_occ n LEFT JOIN old_occ o ON n."catalogNumber" = o."catalogNumber"
        """).fetchone()
        total_carried, total_changed, total_new_obs = (v or 0 for v in stats_row)

        if total_carried == 0:
            con.close()
        else:
            tree_exprs = ", ".join(
                f'CASE WHEN {coords_same} THEN o."{c}" ELSE NULL END AS "{c}"' for c in tree_cols
            )
            temp_exprs = ", ".join(
                f'CASE WHEN {coords_same} AND {ts_same} THEN o."{c}" ELSE NULL END AS "{c}"' for c in temp_cols
            )
            carried_exprs = ", ".join(e for e in (tree_exprs, temp_exprs) if e)
            select_clause = "n.*" + (f", {carried_exprs}" if carried_exprs else "")

            tmp_dest = OCCURRENCES_FILE.with_suffix(".parquet.tmp")
            con.execute(f"""
                COPY (
                    SELECT {select_clause}
                    FROM new_occ n LEFT JOIN old_occ o ON n."catalogNumber" = o."catalogNumber"
                    ORDER BY n."taxon_key"
                ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
            """)
            con.close()
            tmp_dest.replace(OCCURRENCES_FILE)

    elapsed = time.monotonic() - t0
    carry_pct = total_carried / n_total * 100 if n_total else 0.0
    print(
        f"[carry_forward] {total_carried}/{n_total} rows carried forward "
        f"({carry_pct:.1f}%)  |  {total_new_obs} new  {total_changed} changed  "
        f"({elapsed:.1f}s)"
    )

    stats = {
        "ts": datetime.now(UTC).isoformat(),
        "total_rows": n_total,
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

    OLD_OCCURRENCES_PATH.unlink(missing_ok=True)
    print(f"[carry_forward] cleaned up {OLD_OCCURRENCES_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
