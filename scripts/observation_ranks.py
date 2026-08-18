# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Precompute, per observation, the zoom level at which it should first become
visible on a taxon's map — one column per taxonomic level (KINGDOM..SPECIES,
plus INFRA for subspecies/variety/form). This is the preprocessing half of
WhereWild/backend-v2#108's map-display problem: the endpoint that actually
reads these columns to decimate a map response is separate, later work.

Why per level: density is relative to which taxon is being viewed, not
absolute — an observation can be part of a hotspot at the genus level but
the only observation of its species nearby. So the "should this be visible
yet" answer differs by level, and can't be one global column the way
hilbertIdx is (purely spatial, taxon-independent).

Why bands, not a flat hash: a pure hash(catalogNumber)-based decimation
(keep it if hash % denom(zoom) == 0) preserves true density fidelity — dense
clusters stay visibly denser — but has no coverage guarantee. A real,
isolated population can statistically vanish entirely at low zoom purely by
bad luck, which is worse than losing density fidelity. The fix: a
coarse-to-fine sequence of spatial "coverage bands" layered on top of the
hash. Within each (grouping_key, grid_cell) pair at a band's resolution, the
lowest-hash point becomes that pair's guaranteed representative at that
band's zoom; everything else in the cell waits for a finer band. Points
always render at their real coordinates — the grid decides WHEN a point is
revealed, never WHERE. This is the point-cloud analogue of a raster
mipmap/COG-overview pyramid: each band is a coarser or finer level of
detail, always built from real observations, never a synthetic centroid.

Subspecies/variety/form get their own INFRA column, separate from SPECIES:
SPECIES pools every infraspecific sibling under their shared parent (so a
species-as-a-whole view is coherent), while INFRA keeps sibling subspecies
in separate groups (so one subspecies's density can't crowd out another's
coverage guarantee at their own, finer view).

Pipeline stage: must run LAST among anything touching occurrences.parquet —
after enrich_temporal, before process_tree. Several earlier stages (notably
carry_forward) fully rewrite occurrences.parquet via `SELECT *` and would
silently drop these columns if this ran any earlier in the pipeline.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import duckdb
import pandas as pd

from config.config import load_config
from util.taxa import (
    ANCESTOR_RANK_LEVELS,
    DISPLAY_LEVELS,
    LEVEL_LABELS,
    ancestor_keys_by_rank,
    load_catalog,
    zoom_column,
)

CONFIG = load_config("global")

OCCURRENCES_FILE = Path("data/taxonomy/occurrences.parquet")
OCCURRENCES_BY_HILBERT_FILE = Path("data/taxonomy/occurrences_by_hilbert.parquet")

_DUCKDB_SPILL_DIR = Path("data/tmp/duckdb_spill")
# Deliberately conservative, not just "under total RAM": the working table
# here only ever holds catalogNumber + grouping keys + coordinates + the 8
# new columns (~19 columns), never the full ~175-column occurrence row (see
# main() — the wide data is only ever touched via a final streaming join,
# never materialized into a mutated copy), so this has a lot of headroom
# even on a box running other things concurrently.
_DUCKDB_MEMORY_LIMIT = "20GB"

# Level list and the minZoom<Label> column-naming convention live in
# util.taxa (DISPLAY_LEVELS/LEVEL_LABELS/zoom_column) — shared with main.py,
# which reads these same columns.
_ALL_LEVELS: tuple[str, ...] = DISPLAY_LEVELS

# Coarsest-first (zoom, cell size in meters). The lowest-hash point per
# (grouping_key, cell) at a band becomes visible at that band's zoom; the
# last band's zoom is also the fallback for anything left over — effectively
# "everything remaining becomes visible here, unthinned." Tunable constants,
# not yet fit against real data.
_BANDS: tuple[tuple[int, float], ...] = (
    (0, 200_000.0),
    (4, 50_000.0),
    (7, 10_000.0),
    (10, 2_000.0),
    (13, 250.0),
    (15, 1.0),
)

_MERCATOR_HALF_CIRCUMFERENCE_M = math.pi * 6378137.0


_DUCKDB_SCRATCH_DB = Path("data/tmp/observation_ranks_scratch.duckdb")


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    """Unlike scripts/carry_forward.py's connection (in-memory + temp_directory
    for spilling transient query state), this script does `CREATE TABLE occ AS
    ...` and runs UPDATE passes against it — UPDATE needs a real, persistent
    table, not a view. A bare in-memory database has no file to page actual
    table blocks into once they exceed memory_limit ("Cannot perform IO in
    in-memory database"), so this needs a real file-backed database, not just
    a temp_directory. Scratch file is deleted and recreated fresh on every
    run — nothing in it is meant to survive between runs."""
    _DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    _DUCKDB_SCRATCH_DB.parent.mkdir(parents=True, exist_ok=True)
    _DUCKDB_SCRATCH_DB.unlink(missing_ok=True)
    con = duckdb.connect(str(_DUCKDB_SCRATCH_DB))
    con.execute(f"PRAGMA memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    # Off throughout: the final write uses an explicit ORDER BY (see main()),
    # which makes insertion-order preservation redundant there too — same
    # reasoning scripts/carry_forward.py already applies.
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")
    return con


def _key_col(level: str) -> str:
    return f'"{LEVEL_LABELS[level].lower()}Key"'


def _zoom_col(level: str) -> str:
    return f'"{zoom_column(level)}"'


def build_ancestor_frame() -> pd.DataFrame:
    """One row per catalog taxon: taxon_key + its grouping key at every
    level. Catalog-sized, not occurrence-sized — cheap to build in full and
    join against occurrences afterward."""
    catalog = load_catalog()
    ancestors = ancestor_keys_by_rank()
    subspecies_equivalents = set(CONFIG.subspecies_equivalents)

    rows: list[dict[str, str | None]] = []
    for taxon_key, taxon in catalog.items():
        ranks = ancestors.get(taxon_key, {})
        row: dict[str, str | None] = {"taxon_key": taxon_key}
        for level in ANCESTOR_RANK_LEVELS:
            row[f"{LEVEL_LABELS[level].lower()}Key"] = ranks.get(level)
        row["infraKey"] = taxon_key if taxon["rank"] in subspecies_equivalents else None
        rows.append(row)
    return pd.DataFrame(rows)


def _null_count(con: duckdb.DuckDBPyConnection, zoom_col: str, key_col: str) -> int:
    return con.execute(f'SELECT COUNT(*) FROM occ WHERE {zoom_col} IS NULL AND {key_col} IS NOT NULL').fetchone()[0]


def main() -> None:
    if not OCCURRENCES_FILE.exists():
        print(f"[observation_ranks] no occurrences at {OCCURRENCES_FILE} — skipping")
        return

    t0 = time.monotonic()
    con = _duckdb_connect()

    print(f"[observation_ranks] reading {OCCURRENCES_FILE}")

    anc_df = build_ancestor_frame()
    print(f"[observation_ranks] ancestor lookup built  taxa={len(anc_df)}")
    con.register("anc", anc_df)  # noqa: F841 — referenced via SQL string below

    # Slim working table — only catalogNumber (the final join key back to the
    # wide file), the 8 grouping-key columns, mercator coords, and the 8 new
    # zoom columns. Deliberately NOT `o.*`: materializing and mutating the
    # full ~175-column row (as an earlier version of this script did) forced
    # a full second copy of the entire dataset onto disk, then a full
    # external sort over all of it at the end — several times the actual
    # working set this computation needs, and what caused an earlier OOM.
    # The wide columns are only ever touched via the streaming join in the
    # final COPY below, never copied into a mutated intermediate table.
    key_cols_sql = ", ".join(f'a.{_key_col(level)}' for level in _ALL_LEVELS)
    zoom_init_sql = ", ".join(f"CAST(NULL AS INTEGER) AS {_zoom_col(level)}" for level in _ALL_LEVELS)
    t_load = time.monotonic()
    con.execute(f"""
        CREATE TABLE occ AS
        SELECT
            o."catalogNumber",
            {key_cols_sql},
            {_MERCATOR_HALF_CIRCUMFERENCE_M} / 180.0 * o."decimalLongitude" AS "_mercX",
            LN(TAN(RADIANS(45.0 + o."decimalLatitude" / 2.0))) * {_MERCATOR_HALF_CIRCUMFERENCE_M} / PI() AS "_mercY",
            {zoom_init_sql}
        FROM read_parquet('{OCCURRENCES_FILE.as_posix()}') o
        LEFT JOIN anc a ON o."taxon_key" = a."taxon_key"
    """)
    n_rows = con.execute("SELECT COUNT(*) FROM occ").fetchone()[0]
    print(f"[observation_ranks] table built  rows={n_rows}  ({time.monotonic() - t_load:.1f}s)")

    for level_idx, level in enumerate(_ALL_LEVELS, start=1):
        key_col = _key_col(level)
        zoom_col = _zoom_col(level)
        t_level = time.monotonic()
        remaining = _null_count(con, zoom_col, key_col)
        print(f"[observation_ranks] level {level} ({level_idx}/{len(_ALL_LEVELS)})  pending={remaining}")

        for band_idx, (zoom, cell_m) in enumerate(_BANDS, start=1):
            t_band = time.monotonic()
            con.execute(f"""
                UPDATE occ SET {zoom_col} = {zoom}
                WHERE {zoom_col} IS NULL AND {key_col} IS NOT NULL
                AND "catalogNumber" IN (
                    SELECT "catalogNumber" FROM (
                        SELECT "catalogNumber",
                               RANK() OVER (
                                   PARTITION BY {key_col}, FLOOR("_mercX" / {cell_m}), FLOOR("_mercY" / {cell_m})
                                   ORDER BY HASH("catalogNumber"), "catalogNumber"
                               ) AS rnk
                        FROM occ
                        WHERE {zoom_col} IS NULL AND {key_col} IS NOT NULL
                    ) ranked WHERE rnk = 1
                )
            """)
            still_remaining = _null_count(con, zoom_col, key_col)
            updated = remaining - still_remaining
            remaining = still_remaining
            print(
                f"[observation_ranks]   band {band_idx}/{len(_BANDS)}  zoom={zoom} cell={cell_m:.0f}m  "
                f"updated={updated}  remaining={remaining}  ({time.monotonic() - t_band:.1f}s)"
            )

        # Anything left in this level's groups after the finest band becomes
        # visible at that band's zoom too — no further thinning past it.
        last_zoom = _BANDS[-1][0]
        con.execute(f"""
            UPDATE occ SET {zoom_col} = {last_zoom}
            WHERE {zoom_col} IS NULL AND {key_col} IS NOT NULL
        """)
        print(f"[observation_ranks] level {level} done  ({time.monotonic() - t_level:.1f}s)")

    # Streaming join back against the original wide file, keyed by
    # catalogNumber, with an explicit ORDER BY — same shape as
    # scripts/carry_forward.py's own proven-safe pattern at this exact data
    # scale (wide file + join + ORDER BY, straight into COPY). An earlier
    # version of this script tried to skip the sort by relying on
    # preserve_insertion_order + the source file's existing physical
    # taxon_key order, on the theory that DuckDB would carry that order
    # through the join for free. In production that produced a small number
    # of out-of-order rows (~231 out of 64.6M) at what were clearly parallel
    # join chunk boundaries — a real gap in that guarantee for JOIN
    # specifically, not something safe to keep relying on. The explicit sort
    # is the correctness-guaranteed option; it's also the one already proven
    # safe at this scale by carry_forward.py, now that `occ` itself is slim
    # (the earlier 48GB materialized-wide-copy problem is what actually made
    # the sort dangerous the first two times, not the sort itself).
    #
    # The verification query below stays regardless — cheap insurance that
    # costs nothing to keep.
    zoom_cols_sql = ", ".join(f'r.{_zoom_col(level)}' for level in _ALL_LEVELS)
    tmp_dest = OCCURRENCES_FILE.with_suffix(".parquet.tmp")
    t_write = time.monotonic()
    print(f"[observation_ranks] writing {tmp_dest}")
    con.execute(f"""
        COPY (
            SELECT o.*, {zoom_cols_sql}
            FROM read_parquet('{OCCURRENCES_FILE.as_posix()}') o
            LEFT JOIN occ r ON o."catalogNumber" = r."catalogNumber"
            ORDER BY o."taxon_key"
        ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    print(f"[observation_ranks] wrote tmp file  ({time.monotonic() - t_write:.1f}s)")

    t_verify = time.monotonic()
    print(f"[observation_ranks] verifying {tmp_dest} is taxon_key-sorted before replacing the real file")
    out_of_order = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT "taxon_key", LAG("taxon_key") OVER () AS prev_key
            FROM read_parquet('{tmp_dest.as_posix()}')
        ) WHERE prev_key IS NOT NULL AND "taxon_key" < prev_key
    """).fetchone()[0]
    con.close()
    if out_of_order > 0:
        raise RuntimeError(
            f"[observation_ranks] {tmp_dest} is NOT taxon_key-sorted ({out_of_order} out-of-order rows) — "
            f"preserve_insertion_order didn't hold as expected. Aborting WITHOUT touching {OCCURRENCES_FILE}; "
            f"{tmp_dest} left in place for inspection."
        )
    print(f"[observation_ranks] verified sorted  ({time.monotonic() - t_verify:.1f}s)")

    _DUCKDB_SCRATCH_DB.unlink(missing_ok=True)
    tmp_dest.replace(OCCURRENCES_FILE)
    print(f"[observation_ranks] wrote {OCCURRENCES_FILE}  ({time.monotonic() - t_write:.1f}s)")

    # Final phase: the hilbertIdx-sorted spatial index, built from the
    # just-written real OCCURRENCES_FILE so it's always consistent with
    # what's actually live.
    t_spatial = time.monotonic()
    build_spatial_index()
    print(f"[observation_ranks] spatial index phase done  ({time.monotonic() - t_spatial:.1f}s)")

    elapsed = time.monotonic() - t0
    print(f"[observation_ranks] done  levels={len(_ALL_LEVELS)}  rows={n_rows}  total={elapsed:.1f}s")


def build_spatial_index() -> None:
    """Slim, hilbertIdx-sorted secondary index — same established pattern as
    scripts/populate_tree.py's _build_catalog_number_index (narrow
    projection, differently sorted, for one specific access pattern), just
    keyed on hilbertIdx instead of catalogNumber, and including the
    minZoom* columns this file needs that populate_tree's index doesn't.
    Standalone (not folded into main()'s single connection) so it can be
    rebuilt/verified on its own against an already-ranked OCCURRENCES_FILE,
    without repeating the ranking passes. Plain in-memory connection is
    enough — a straight scan + sort + write over a narrow projection, no
    persistent mutated table like the ranking passes need.
    """
    print(f"[observation_ranks] building spatial index {OCCURRENCES_BY_HILBERT_FILE}")
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    zoom_cols_out_sql = ", ".join(f'"{zoom_column(level)}"' for level in _ALL_LEVELS)
    tmp_dest = OCCURRENCES_BY_HILBERT_FILE.with_suffix(".parquet.tmp")
    con.execute(f"""
        COPY (
            SELECT
                "catalogNumber", "taxon_key", "decimalLatitude", "decimalLongitude",
                "mediaUrl", "mediaAttribution", "mediaLicense", "hilbertIdx",
                {zoom_cols_out_sql}
            FROM read_parquet('{OCCURRENCES_FILE.as_posix()}')
            ORDER BY "hilbertIdx"
        ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.close()
    tmp_dest.replace(OCCURRENCES_BY_HILBERT_FILE)
    print(f"[observation_ranks] wrote {OCCURRENCES_BY_HILBERT_FILE}")


if __name__ == "__main__":  # pragma: no cover
    main()
