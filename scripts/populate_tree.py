# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Stream occurrence.txt (DWCA) into one consolidated occurrences.parquet.

Rows are parsed and buffered in memory, periodically flushed as unsorted
batches to a temp parquet file via ParquetWriter to bound memory use. Once
the whole input is consumed, a single DuckDB pass sorts by taxon_key so
single-taxon reads get row-group pruning, then writes the final
data/taxonomy/occurrences.parquet. No dedup on catalogNumber: it's the
iNaturalist observation ID, unique by construction — see _consolidate.

Rows carry only taxon_key, not the taxon's path — the taxonomy tree
(path, rank, ancestry) already lives in the in-memory catalog
(util.taxa.load_catalog), keyed by taxon_key, so there is no reason to
duplicate a taxon's ~100-byte path string onto every one of its occurrence
rows. Subtree-scoped reads resolve a taxon_key set from the catalog
(iter_descendants) and join/filter against that instead.

Rows are matched to a catalog entry by scientificName, not by occurrence.txt's
own taxonKey column. Confirmed against a live GBIF occurrence record
(api.gbif.org/v1/occurrence/search): even when a download's TAXON_KEY
predicate is scoped via checklistKey (Catalogue of Life Extended Release, the
alphanumeric IDs our catalog is keyed by — see config.taxonomy_roots), each
occurrence's own taxonKey/familyKey/etc. columns still report the legacy
numeric GBIF Backbone classification. checklistKey only affects how the
predicate's *filter value* is resolved, not the taxonomy baked into the
exported rows. So occurrence.txt's taxonKey can never match our catalog's
COL XR keys directly, hence the name-based lookup via
util.taxa.load_name_index() (already includes synonym scientific names —
see scripts.build_tree.update_name_index) instead of a taxonKey dict lookup.
"""

import csv
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from config.config import load_config
from scripts.build_tree import _is_usable_license, _normalize_license_url, clean_name
from util.gis import hilbert_index
from util.taxa import load_catalog, load_name_index, normalize_name

csv.field_size_limit(sys.maxsize)

CONFIG = load_config("global")

OCCURRENCE_PATH = Path("data/occurrences/occurrence.txt")
MULTIMEDIA_PATH = Path("data/occurrences/multimedia.txt")
OCCURRENCES_FILE = Path("data/taxonomy/occurrences.parquet")
CATALOG_NUMBER_INDEX_FILE = Path("data/taxonomy/catalog_number_index.parquet")

# _consolidate's dedup+sort over the full occurrences table (tens of
# millions of rows) has no natural batching point the way the streaming
# parse above does — it's one DuckDB window function + ORDER BY. A bare
# duckdb.connect() has no temp_directory, so on an in-memory connection it
# never spills to disk and just keeps growing until the OS OOM-kills it
# (confirmed in practice: a 60M-row consolidation hit 61GB RSS and got
# killed). Capping memory_limit well under total RAM and giving it a real
# temp_directory to spill into turns that into "slower", not "crashes".
_DUCKDB_SPILL_DIR = Path("data/tmp/duckdb_spill")
_DUCKDB_MEMORY_LIMIT = "28GB"


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    _DUCKDB_SPILL_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{_DUCKDB_SPILL_DIR.as_posix()}'")
    # Both call sites below have their own explicit ORDER BY, which already
    # overrides whatever row order the scan emits — so there's no reason to
    # also pay for insertion-order preservation's default-on per-thread
    # buffering. See scripts/carry_forward.py's _duckdb_connect for the
    # concrete OOM this was confirmed to cause on the equivalent join+sort.
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")
    return con


# Rows buffered in memory before a streaming flush to the unsorted temp file.
BATCH_ROWS = 500_000

OCCURRENCE_DELIMITER = "|"

SCHEMA = pa.schema([
    ("decimalLatitude",               pa.float64()),
    ("decimalLongitude",              pa.float64()),
    ("catalogNumber",                 pa.string()),
    ("hilbertIdx",                    pa.int32()),
    ("eventTimestamp",                pa.int64()),
    ("coordinateUncertaintyInMeters", pa.float64()),
    ("obscured",                      pa.string()),
    ("gbifRegion",                    pa.string()),
    ("level0Gid",                     pa.string()),
    ("level1Gid",                     pa.string()),
    ("level2Gid",                     pa.string()),
    ("dp",                            pa.string()),
    ("vitality",                      pa.string()),
    ("rcs",                           pa.string()),
    ("taxon_key",                     pa.string()),
    ("mediaUrl",                      pa.string()),
    ("mediaAttribution",              pa.string()),
    ("mediaLicense",                  pa.string()),
])


def _parse_timestamp(date: str, time: str) -> int | None:
    date = (date or "").strip()
    time = (time or "").strip()
    if not date:
        return None
    try:
        date_only = date.split("T")[0]
        if time and time.lower() != "na":
            dt = datetime.fromisoformat(f"{date_only}T{time}")
        else:
            dt = datetime.fromisoformat(date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def _parse_dp(raw: str) -> str:
    """Extract evidenceOfPresence from dynamicProperties JSON, joined by |."""
    if not raw:
        return ""
    obj = json.loads(raw)
    ev = obj.get("evidenceOfPresence", "")
    if isinstance(ev, list):
        return OCCURRENCE_DELIMITER.join(ev)
    return ev or ""


def _parse_obscured(info_withheld: str) -> str:
    if not info_withheld:
        return "No"
    return "Hidden" if info_withheld.split(" ")[-1] == "taxon" else "Obscured"


def _load_media_map(path: Path) -> dict[str, tuple[str, str, str]]:
    """Build gbifID -> (url, attribution, license_url) from multimedia.txt.

    A gbifID can have several media rows; only the first one encountered is
    considered (no scanning ahead for a "better" one), and it's kept only if
    _is_usable_license accepts its license — the same permissive-license bar
    build_tree.py already applies to GBIF backup images, so callers don't
    need to reason about two different license policies.

    Uses csv.reader (not DictReader) with precomputed column indices since
    this streams ~1.2M rows and dict construction per row is measurably
    slower at that volume.
    """
    media_map: dict[str, tuple[str, str, str]] = {}
    if not path.exists():
        return media_map

    seen: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        gbif_i = col["gbifID"]
        ident_i = col["identifier"]
        license_i = col["license"]
        rights_i = col["rightsHolder"]
        creator_i = col["creator"]
        width = len(header)

        for fields in reader:
            if len(fields) != width:
                continue
            gbif_id = fields[gbif_i]
            if gbif_id in seen:
                continue
            seen.add(gbif_id)

            license_raw = fields[license_i]
            if not _is_usable_license(license_raw):
                continue
            identifier = fields[ident_i].strip()
            if not identifier:
                continue
            attribution = fields[rights_i].strip() or fields[creator_i].strip()
            media_map[gbif_id] = (identifier, attribution, _normalize_license_url(license_raw))

    return media_map


def _flush(writer_holder: dict, tmp_path: Path, rows: list[dict]) -> None:
    """Write one unsorted batch of parsed rows to the streaming temp file."""
    if not rows:
        return

    arrays = {field.name: [] for field in SCHEMA}
    for row in rows:
        for k, v in row.items():
            arrays[k].append(v)

    table = pa.table(
        {name: pa.array(vals, type=SCHEMA.field(name).type) for name, vals in arrays.items()},
        schema=SCHEMA,
    )

    writer = writer_holder.get("writer")
    if writer is None:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(tmp_path, SCHEMA)
        writer_holder["writer"] = writer
    writer.write_table(table, row_group_size=50_000)

    rows.clear()


def _consolidate(tmp_path: Path) -> None:
    """Sort by taxon_key and write the final occurrences.parquet.

    No dedup on catalogNumber: it's the iNaturalist observation ID, unique
    by construction, so a first-seen-wins pass over every row would only be
    paying for a guarantee the data already provides — and the window
    function driving that pass was the single biggest memory cost of this
    step (confirmed: it OOM-killed a 60M-row run at 61GB RSS). A stray
    duplicate would still get caught downstream — carry_forward.py already
    dedupes its old-tree side defensively regardless of this file.
    """
    dest = OCCURRENCES_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".parquet.tmp")
    con = _duckdb_connect()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{tmp_path.as_posix()}')
            ORDER BY taxon_key
        ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.close()
    tmp_dest.replace(dest)


def _build_catalog_number_index() -> None:
    """Write catalogNumber -> (taxon_key, lat, lon), sorted by catalogNumber.

    occurrences.parquet itself is sorted by taxon_key for taxon-scoped reads
    — a catalogNumber lookup (e.g. GET /occurrence/{id}, resolving an
    inaturalist observation id to its taxon) against that file would need a
    full scan. This is a narrow, catalogNumber-sorted copy so a point lookup
    gets row-group pruning instead. lat/lon are carried along so that one
    lookup is enough to place the highlighted point on the map — no second
    read against occurrences.parquet needed. Media/timestamp deliberately
    aren't carried here: for an ingested observation, the frontend already
    has those from its normal /species/{id}/occurrences fetch once it lands
    on the species page, so returning them again here would be redundant.
    """
    dest = CATALOG_NUMBER_INDEX_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".parquet.tmp")
    con = _duckdb_connect()
    con.execute(f"""
        COPY (
            SELECT "catalogNumber", taxon_key, "decimalLatitude", "decimalLongitude"
            FROM read_parquet('{OCCURRENCES_FILE.as_posix()}')
            ORDER BY "catalogNumber"
        ) TO '{tmp_dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.close()
    tmp_dest.replace(dest)


def main() -> None:
    catalog = load_catalog()
    # Keyed by normalize_name'd scientific name (accepted + synonym names,
    # see util.taxa.load_name_index) → taxon_key list. See the module
    # docstring for why matching goes through names instead of
    # occurrence.txt's own taxonKey column.
    name_index = load_name_index()
    media_map = _load_media_map(MULTIMEDIA_PATH)
    print(f"  Loaded {len(media_map):,} usable-license media links.", flush=True)

    rows_read = 0
    rows_written = 0
    buffer: list[dict] = []
    writer_holder: dict = {}
    tmp_path = OCCURRENCES_FILE.parent / ".occurrences_unsorted.tmp.parquet"

    with open(OCCURRENCE_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows_read += 1
            if rows_read % 1_000_000 == 0:  # pragma: no cover
                print(f"  {rows_read:,} rows read, {rows_written:,} written...", flush=True)

            rank = (row.get("taxonRank") or "").strip()
            if rank not in CONFIG.leaf_rank_set:
                continue

            lat_raw = (row.get("decimalLatitude") or "").strip()
            lon_raw = (row.get("decimalLongitude") or "").strip()
            catalog_num = (row.get("catalogNumber") or "").strip()
            if not lat_raw or not lon_raw or not catalog_num:
                continue

            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except ValueError:
                continue

            sci_name = (row.get("scientificName") or "").strip()
            if not sci_name:
                continue

            name_key = normalize_name(clean_name(sci_name, rank))
            matches = name_index.get(name_key)
            # Skip unmatched and ambiguous (homonym) names rather than guess —
            # same conservative behavior as the old "taxon is None" skip.
            if not matches or len(matches) != 1:
                continue

            taxon = catalog.get(matches[0])
            if taxon is None:
                continue

            uncertainty_raw = (row.get("coordinateUncertaintyInMeters") or "").strip()
            try:
                uncertainty = float(uncertainty_raw) if uncertainty_raw else None
            except ValueError:
                uncertainty = None

            media = media_map.get((row.get("gbifID") or "").strip())

            buffer.append({
                "decimalLatitude":               lat,
                "decimalLongitude":              lon,
                "catalogNumber":                 catalog_num,
                "hilbertIdx":                    hilbert_index(lat, lon),
                "eventTimestamp":                _parse_timestamp(row.get("eventDate"), row.get("eventTime")),
                "coordinateUncertaintyInMeters": uncertainty,
                "obscured":                      _parse_obscured(row.get("informationWithheld")),
                "gbifRegion":                    (row.get("gbifRegion") or "").strip() or None,
                "level0Gid":                     (row.get("level0Gid") or "").strip() or None,
                "level1Gid":                     (row.get("level1Gid") or "").strip() or None,
                "level2Gid":                     (row.get("level2Gid") or "").strip() or None,
                "dp":                            _parse_dp(row.get("dynamicProperties") or ""),
                "vitality":                      (row.get("vitality") or "").strip().lower(),
                "rcs":                           (row.get("reproductiveCondition") or "").strip(),
                "taxon_key":                     taxon["taxon_key"],
                "mediaUrl":                      media[0] if media else None,
                "mediaAttribution":              media[1] if media else None,
                "mediaLicense":                  media[2] if media else None,
            })
            rows_written += 1

            if len(buffer) >= BATCH_ROWS:
                _flush(writer_holder, tmp_path, buffer)

    print("  Flushing remaining rows...", flush=True)
    _flush(writer_holder, tmp_path, buffer)
    writer = writer_holder.get("writer")
    if writer is not None:
        writer.close()

    print(f"Done reading. {rows_read:,} rows read, {rows_written:,} written.")

    if rows_written:
        print("  Consolidating into occurrences.parquet...", flush=True)
        _consolidate(tmp_path)
        print("  Building catalogNumber index...", flush=True)
        _build_catalog_number_index()
    tmp_path.unlink(missing_ok=True)

    # Only remove the occurrences dir when it's the real pipeline path (not a test tmp dir).
    occ_dir = OCCURRENCE_PATH.parent
    if occ_dir.name == "occurrences":
        shutil.rmtree(occ_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    main()
