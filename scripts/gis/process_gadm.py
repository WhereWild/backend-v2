"""Download the GADM 4.1 GeoPackage and build location lookup tables.

Downloads the zipped GeoPackage from the UC Davis geodata mirror via aria2c,
extracts gadm.gpkg, then writes per-level CSVs, a hierarchy table, a GBIF
region list, and a location_taxa.parquet mapping each location GID to taxa
observation counts (with ancestor rollup). Re-running is a no-op if gadm.gpkg
already exists (download phase) and always rebuilds the tables/catalog.

Two phases:
  _download()            — fetch + extract gadm.gpkg (skips if already present)
  _build_tables()        — reads GADM sqlite, writes CSVs
  _build_catalog()       — reads occurrence parquets, writes location_taxa.parquet
"""

from __future__ import annotations

import csv
import shutil
import sqlite3
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config.config import load_config
from util.taxa import load_catalog

CONFIG = load_config("global")

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip"
GIS_DIR = Path("data/gis")
GADM_PATH = GIS_DIR / "gadm.gpkg"
_GADM_ZIP = GIS_DIR / "gadm_410-gpkg.zip"
LOCATIONS_DIR = Path("data/gis/locations")
TREE_ROOT = Path("data/taxonomy/tree")
OCCURRENCE_FILE = "occurrence.parquet"

_GBIF_COL = "gbifRegion"
_GBIF_SCOPE = "gbif_region"
_BATCH_SIZE = 10_000
_LOG_INTERVAL = 100


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_zip() -> None:
    print(f"[download] GADM 4.1 zip → {_GADM_ZIP}")
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={_GADM_ZIP.parent}",
            f"--out={_GADM_ZIP.name}",
            GADM_URL,
        ],
        check=True,
    )


def _extract(zip_path: Path, dest: Path) -> None:
    print(f"[extract] {zip_path.name} → {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        gpkg_name = next(n for n in zf.namelist() if n.endswith(".gpkg"))
        with zf.open(gpkg_name) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def _download() -> None:
    if GADM_PATH.exists():
        print(f"[skip] gadm — already at {GADM_PATH}")
        return
    GIS_DIR.mkdir(parents=True, exist_ok=True)
    if not _GADM_ZIP.exists():
        _download_zip()
    _extract(_GADM_ZIP, GADM_PATH)
    _GADM_ZIP.unlink()
    print(f"  Saved {GADM_PATH} ({GADM_PATH.stat().st_size // 1_000_000} MB)")


# ---------------------------------------------------------------------------
# GADM sqlite helpers
# ---------------------------------------------------------------------------

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
        cols = _table_columns(conn, table)
        if gid_col in cols and name_col in cols:
            return table
    raise RuntimeError(
        f"No feature table found containing {gid_col}/{name_col}. "
        "Verify the GeoPackage was downloaded correctly."
    )


# ---------------------------------------------------------------------------
# CSV / hierarchy writers
# ---------------------------------------------------------------------------

def _export_level(conn: sqlite3.Connection, level: int) -> list[tuple[str, str]]:
    gid_col = f"GID_{level}"
    name_col = f"NAME_{level}"
    table = _find_table_for_level(conn, level)
    LOCATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCATIONS_DIR / f"level{level}.csv"
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str]] = []
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gid", "name"])
        for gid, name in conn.execute(
            f'SELECT "{gid_col}", "{name_col}" FROM "{table}"'
        ):
            key = (gid, name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            writer.writerow([gid, name])
    print(f"Wrote {out_path} from {table} ({len(seen)} unique rows)")
    return rows


def _parent_gid(gid: str, level: int) -> str | None:
    if level == 0:
        return None
    base, _, suffix = gid.partition("_")
    if "." not in base:
        return None
    parent_base = base.rsplit(".", 1)[0]
    if level == 1:
        return parent_base
    return f"{parent_base}_{suffix}" if suffix else parent_base


def _write_hierarchy(level_rows: dict[int, list[tuple[str, str]]]) -> None:
    LOCATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCATIONS_DIR / "hierarchy.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["level", "gid", "name", "parent_gid"])
        for level in sorted(level_rows):
            for gid, name in level_rows[level]:
                parent = _parent_gid(gid, level)
                writer.writerow([level, gid, name, parent or ""])
    print(f"Wrote {out_path}")


def _write_gbif_regions() -> None:
    LOCATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCATIONS_DIR / "gbif_regions.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gbifRegion"])
        for region in CONFIG.gbif_regions:
            writer.writerow([region])
    print(f"Wrote {out_path} ({len(CONFIG.gbif_regions)} regions)")


def _build_tables() -> None:
    if not GADM_PATH.exists():
        raise FileNotFoundError(
            f"{GADM_PATH} not found. Run process_gadm first."
        )
    conn = sqlite3.connect(GADM_PATH)
    try:
        level_rows: dict[int, list[tuple[str, str]]] = {}
        for level in CONFIG.location_levels:
            level_rows[level] = _export_level(conn, level)
        _write_hierarchy(level_rows)
        _write_gbif_regions()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"Failed to read {GADM_PATH}: {exc}. "
            "The GeoPackage may be corrupted — re-download it."
        ) from exc
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Location catalog (occurrence parquet → location_taxa.parquet)
# ---------------------------------------------------------------------------

def _iter_taxa_with_occurrences() -> list[tuple[str, Path]]:
    result = []
    for key, taxon in load_catalog().items():
        occ_path = TREE_ROOT / taxon["path"] / OCCURRENCE_FILE
        if occ_path.exists():
            result.append((key, occ_path))
    return result


def _collect_gid_counts(parquet_path: Path) -> dict[str, dict[str, int]]:
    col_defs: list[tuple[str, str]] = list(CONFIG.location_columns) + [
        (_GBIF_COL, _GBIF_SCOPE)
    ]
    col_names = [col for col, _ in col_defs]
    per_scope: dict[str, dict[str, int]] = {
        scope: defaultdict(int) for _, scope in col_defs
    }
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=col_names, batch_size=_BATCH_SIZE):
        for idx, (_, scope) in enumerate(col_defs):
            column = batch.column(idx)
            if column.null_count == len(column):
                continue
            for val in column.to_pylist():
                if val and isinstance(val, str):
                    per_scope[scope][val] += 1
    return {scope: dict(counts) for scope, counts in per_scope.items()}


def _build_parent_map() -> dict[str, str]:
    catalog = load_catalog()
    path_to_key = {t["path"]: k for k, t in catalog.items()}
    parent: dict[str, str] = {}
    for key, taxon in catalog.items():
        path = taxon["path"]
        if "/" in path:
            parent_key = path_to_key.get(path.rsplit("/", 1)[0])
            if parent_key:
                parent[key] = parent_key
    return parent


def _ancestor_keys(taxon_key: str, parent_map: dict[str, str]) -> list[str]:
    ancestors: list[str] = []
    current = parent_map.get(taxon_key)
    while current is not None:
        ancestors.append(current)
        current = parent_map.get(current)
    return ancestors


def _build_catalog() -> None:
    out_path = LOCATIONS_DIR / "location_taxa.parquet"
    LOCATIONS_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    taxa = _iter_taxa_with_occurrences()
    for idx, (taxon_key, occ_path) in enumerate(taxa, start=1):
        per_scope = _collect_gid_counts(occ_path)
        for scope, gid_counts in per_scope.items():
            for gid, count in gid_counts.items():
                if count > 0:
                    counts[(scope, gid, taxon_key)] += count
        if idx % _LOG_INTERVAL == 0:
            print(f"  Processed {idx}/{len(taxa)} taxa…")

    if not counts:
        print("No location mappings found (no occurrence data).")
        return

    parent_map = _build_parent_map()
    direct = list(counts.items())
    for roll_idx, ((scope, gid, key), count) in enumerate(direct, start=1):
        for anc in _ancestor_keys(key, parent_map):
            counts[(scope, gid, anc)] += count
        if roll_idx % (_LOG_INTERVAL * 10) == 0:
            print(f"  Rolled up {roll_idx}/{len(direct)} rows…")

    rows_scope: list[str] = []
    rows_gid: list[str] = []
    rows_taxon: list[str] = []
    rows_count: list[int] = []
    for scope, gid, taxon_key in sorted(counts):
        count = counts[(scope, gid, taxon_key)]
        if count <= 0:  # pragma: no cover
            continue
        rows_scope.append(scope)
        rows_gid.append(gid)
        rows_taxon.append(taxon_key)
        rows_count.append(count)

    table = pa.table({
        "scope": pa.array(rows_scope, type=pa.string()),
        "gid": pa.array(rows_gid, type=pa.string()),
        "taxon_key": pa.array(rows_taxon, type=pa.string()),
        "count": pa.array(rows_count, type=pa.int64()),
    })
    pq.write_table(table, out_path)
    unique_locations = len({(s, g) for s, g, _ in counts})
    print(
        f"Wrote {len(rows_taxon)} rows linking "
        f"{unique_locations} locations to taxa → {out_path}"
    )


def main() -> None:
    _download()
    _build_tables()
    _build_catalog()


if __name__ == "__main__":  # pragma: no cover
    main()
