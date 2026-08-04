# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One-off migration: consolidate per-taxon occurrence.parquet files under
data/taxonomy/tree/ into a single data/taxonomy/occurrences.parquet, stamped
with taxon_key from the catalog. Pure concatenation of already-enriched
data — no recomputation, no re-parsing of GBIF source data.

Usage:
    python -m scripts.migrate_consolidate_occurrences            # dry run (default)
    python -m scripts.migrate_consolidate_occurrences --execute  # write for real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from util.storage import atomic_write_parquet
from util.taxa import load_catalog

TREE_ROOT = Path("data") / "taxonomy" / "tree"
OCCURRENCE_FILE = "occurrence.parquet"
OUTPUT_FILE = Path("data") / "taxonomy" / "occurrences.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Write the output file (default: dry run, report only)")
    parser.add_argument("--out", type=Path, default=OUTPUT_FILE, help=f"Output path (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    catalog = load_catalog()
    path_to_key = {t["path"]: str(t["taxon_key"]) for t in catalog.values()}

    files = sorted(TREE_ROOT.rglob(OCCURRENCE_FILE))
    if not files:
        print(f"No {OCCURRENCE_FILE} files found under {TREE_ROOT}", file=sys.stderr)
        sys.exit(1)

    tables: list[pa.Table] = []
    unmatched: list[str] = []
    total_rows = 0
    for f in files:
        rel_path = f.parent.relative_to(TREE_ROOT).as_posix()
        taxon_key = path_to_key.get(rel_path)
        if taxon_key is None:
            unmatched.append(rel_path)
            continue
        table = pq.read_table(f)
        if table.num_rows == 0:
            continue
        table = table.append_column("taxon_key", pa.array([taxon_key] * table.num_rows, type=pa.string()))
        tables.append(table)
        total_rows += table.num_rows

    print(f"Scanned {len(files)} occurrence.parquet files under {TREE_ROOT}")
    print(f"  matched to catalog: {len(files) - len(unmatched)}")
    print(f"  unmatched (no catalog entry for directory path): {len(unmatched)}")
    if unmatched:
        for p in unmatched[:20]:
            print(f"    - {p}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    print(f"  total occurrence rows: {total_rows}")

    if not tables:
        print("Nothing to write.", file=sys.stderr)
        sys.exit(1)

    combined = pa.concat_tables(tables, promote_options="default")
    combined = combined.take(pc.sort_indices(combined, sort_keys=[("taxon_key", "ascending")]))

    if not args.execute:
        print(f"\nDry run — would write {combined.num_rows} rows, {len(combined.schema.names)} columns to {args.out}")
        print("Re-run with --execute to write for real.")
        return

    if args.out.exists():
        print(f"Refusing to overwrite existing file: {args.out}", file=sys.stderr)
        print("Move or remove it first if you intend to replace it.", file=sys.stderr)
        sys.exit(1)

    atomic_write_parquet(args.out, combined)
    print(f"\nWrote {combined.num_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
