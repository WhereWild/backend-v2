"""Build lookup tables linking GADM locations to canonical identifiers.

The script inspects the gadm.gpkg GeoPackage for feature layers that contain
`GID_*`/`NAME_*` columns, then writes per-level CSV files along with a
hierarchy table and GBIF region list. Afterwards it scans the taxa catalog to
build a Parquet file mapping each location GID to taxa occurrence counts.
The final table includes a taxonomy rollup pass so ancestors accumulate counts
from all descendants (e.g., family/order rows include species observations).
Downstream code can rely on a single script instead of coordinating separate
steps.
"""

from __future__ import annotations
from collections import defaultdict
import csv
import sqlite3
from pathlib import Path
from typing import Iterable
from functools import lru_cache
import pyarrow as pa
import pyarrow.parquet as pq
import util.taxa_navigation as taxa_navigation
from util.config import load_config

CONFIG = load_config("global")

gbif_column_name = "gbifRegion"

gbif_scope_name = "gbif_region"

location_catalog_batch_size = 10_000

location_level_filename_template = "level{level}.csv"

location_progress_interval = 100



def _list_feature_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
    )
    return [row[0] for row in cur.fetchall()]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return {row[1] for row in cur.fetchall()}


def _find_table_for_level(conn: sqlite3.Connection, level: int) -> str:
    gid_col = f"GID_{level}"
    name_col = f"NAME_{level}"

    for table in _list_feature_tables(conn):
        columns = _table_columns(conn, table)
        if gid_col in columns and name_col in columns:
            return table

    raise RuntimeError(
        f"Could not find a table containing {gid_col}/{name_col}. "
        "Verify the GeoPackage was downloaded correctly."
    )


def _export_level(conn: sqlite3.Connection, level: int) -> list[tuple[str, str]]:
    gid_col = f"GID_{level}"
    name_col = f"NAME_{level}"
    table = _find_table_for_level(conn, level)

    CONFIG.gis_locations_root.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.gis_locations_root / location_level_filename_template.format(
        level=level
    )
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str]] = []

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gid", "name"])

        query = f'SELECT "{gid_col}", "{name_col}" FROM "{table}"'
        for gid, name in conn.execute(query):
            key = (gid, name)
            if key in seen:
                continue
            writer.writerow([gid, name])
            seen.add(key)
            rows.append((gid, name))

    print(
        f"Wrote {output_path.relative_to(CONFIG.project_root)} from {table} "
        f"({len(seen)} unique rows)"
    )
    return rows


def _parent_gid(gid: str, level: int) -> str | None:
    if level == 0:
        return None

    base = gid
    suffix = ""
    if "_" in gid:
        base, suffix = gid.rsplit("_", 1)

    if "." not in base:
        # e.g., malformed data
        return None

    parent_base = base.rsplit(".", 1)[0]
    if level == 1:
        return parent_base

    return f"{parent_base}_{suffix}" if suffix else parent_base


def _write_hierarchy(level_rows: dict[int, list[tuple[str, str]]]) -> None:
    CONFIG.gis_locations_root.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.location_hierarchy_path
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["level", "gid", "name", "parent_gid"])
        for level in sorted(level_rows):
            for gid, name in level_rows[level]:
                parent = _parent_gid(gid, level)
                writer.writerow([level, gid, name, parent or ""])
    print(
        f"Wrote {output_path.relative_to(CONFIG.project_root)} with hierarchy"
    )


def _write_gbif_regions() -> None:
    CONFIG.gis_locations_root.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.gbif_regions_path
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gbifRegion"])
        for region in CONFIG.gbif_regions:
            writer.writerow([region])
    print(
        f"Wrote {output_path.relative_to(CONFIG.project_root)} with "
        f"{len(CONFIG.gbif_regions)} GBIF regions"
    )


def _build_location_tables() -> None:
    if not CONFIG.gadm_gpkg_path.exists():
        raise FileNotFoundError(CONFIG.gadm_gpkg_path)

    try:
        conn = sqlite3.connect(CONFIG.gadm_gpkg_path)
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"Failed to open {CONFIG.gadm_gpkg_path} ({exc}). "
            "The GeoPackage may be corrupted, re-download it from GADM."
        ) from exc
    try:
        level_rows = {}
        for level in CONFIG.location_levels:
            level_rows[level] = _export_level(conn, level)
        _write_hierarchy(level_rows)
        _write_gbif_regions()
    finally:
        conn.close()


def _iter_taxa_with_occurrences() -> Iterable[tuple[int, Path]]:
    catalog = taxa_navigation.load_catalog()
    for taxon in catalog.values():
        taxon_id = taxa_navigation.taxon_id_as_int(taxon["taxon_key"])
        if taxon_id is None:
            continue
        parquet_path = (
            taxa_navigation.normalize_taxon_path(taxon["path"])
            / CONFIG.occurrence_parquet_filename
        )
        if parquet_path.exists():
            yield taxon_id, parquet_path


def _collect_gid_counts(parquet_path: Path) -> dict[str, dict[str, int]]:
    location_columns = list(CONFIG.location_columns)
    gbif_column = (gbif_column_name, gbif_scope_name)
    per_scope_counts: dict[str, dict[str, int]] = {
        scope: defaultdict(int) for _, scope in location_columns
    }
    per_scope_counts[gbif_column[1]] = defaultdict(int)
    column_defs = location_columns + [gbif_column]
    column_names = [name for name, _ in column_defs]
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        columns=column_names, batch_size=location_catalog_batch_size
    ):
        for idx, (_, scope) in enumerate(column_defs):
            column = batch.column(idx)
            if column.null_count == len(column):
                continue
            for value in column.to_pylist():
                if value and isinstance(value, str):
                    per_scope_counts[scope][value] += 1
    return {
        scope: dict(gid_counts)
        for scope, gid_counts in per_scope_counts.items()
    }


@lru_cache(maxsize=250_000)
def _ancestor_taxon_ids(taxon_id: int) -> tuple[int, ...]:
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        return ()
    ancestors: list[int] = []
    current = taxa_navigation.get_parent_taxon(taxon)
    while current is not None:
        ancestor_id = taxa_navigation.taxon_id_as_int(str(current.get("taxon_key") or ""))
        if ancestor_id is not None:
            ancestors.append(ancestor_id)
        current = taxa_navigation.get_parent_taxon(current)
    return tuple(ancestors)


def _build_location_catalog() -> None:
    output_path = CONFIG.location_catalog_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_by_triplet: dict[tuple[str, str, int], int] = defaultdict(int)

    for idx, (taxon_id, parquet_path) in enumerate(
        _iter_taxa_with_occurrences(), start=1
    ):
        per_scope_counts = _collect_gid_counts(parquet_path)
        for scope, gid_counts in per_scope_counts.items():
            for gid, count in gid_counts.items():
                if count <= 0:
                    continue
                counts_by_triplet[(scope, gid, taxon_id)] += int(count)
        if idx % location_progress_interval == 0:
            print(f"Processed {idx} taxa…")

    if not counts_by_triplet:
        print("No location mappings created (no occurrence data found).")
        return

    # Roll up direct counts to all ancestors so ancestor taxa include descendant
    # observations for the same location.
    direct_items = list(counts_by_triplet.items())
    for idx, ((scope, gid, taxon_id), count) in enumerate(direct_items, start=1):
        for ancestor_id in _ancestor_taxon_ids(taxon_id):
            counts_by_triplet[(scope, gid, ancestor_id)] += int(count)
        if idx % (location_progress_interval * 10) == 0:
            print(f"Rolled up {idx} location-taxon rows…")

    rows_scope: list[str] = []
    rows_gid: list[str] = []
    rows_taxon: list[int] = []
    rows_count: list[int] = []
    for scope, gid, taxon_id in sorted(counts_by_triplet.keys()):
        count = int(counts_by_triplet[(scope, gid, taxon_id)])
        if count <= 0:
            continue
        rows_scope.append(scope)
        rows_gid.append(gid)
        rows_taxon.append(taxon_id)
        rows_count.append(count)

    table = pa.Table.from_pydict(
        {
            "scope": pa.array(rows_scope, type=pa.string()),
            "gid": pa.array(rows_gid, type=pa.string()),
            "taxon_id": pa.array(rows_taxon, type=pa.int64()),
            "count": pa.array(rows_count, type=pa.int64()),
        }
    )
    pq.write_table(table, output_path)
    print(
        f"Wrote {len(rows_taxon)} rows linking "
        f"{len({(scope, gid) for scope, gid, _ in counts_by_triplet.keys()})} locations "
        f"to taxa at "
        f"{output_path.relative_to(CONFIG.project_root)}"
    )


def main() -> None:
    _build_location_tables()
    _build_location_catalog()


if __name__ == "__main__":
    main()
