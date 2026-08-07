# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One-off: consolidate the fragmented pre-migration per-taxon occurrence
parquets under data/tmp/old_tree/Plantae_6/**/occurrence.parquet into the
single data/tmp/old_occurrences.parquet file that scripts/carry_forward.py
expects on its "old" side.

Safe to delete after running once.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb

SRC_GLOB = "data/tmp/old_tree/Plantae_6/**/occurrence.parquet"
DEST = Path("data/tmp/old_occurrences.parquet")


def main() -> None:
    tmp_dest = DEST.with_suffix(".parquet.tmp")

    t0 = time.monotonic()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    n_files = len(list(Path().glob(SRC_GLOB.removeprefix("./"))))
    print(f"[consolidate] found {n_files} source occurrence.parquet files")

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{SRC_GLOB}', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY "catalogNumber") = 1
        ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.close()
    tmp_dest.replace(DEST)

    elapsed = time.monotonic() - t0
    size_mb = DEST.stat().st_size / 1e6
    print(f"[consolidate] wrote {DEST} ({size_mb:.1f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
