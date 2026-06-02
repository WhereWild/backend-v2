# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Stream occurrence.txt (DWCA) and route each row into per-taxon parquet files.

Each leaf taxon (SPECIES / SUBSPECIES / VARIETY / FORM) gets an
occurrence.parquet written under its taxonomy tree path.  Rows are buffered
in memory and flushed in batches to reduce I/O.
"""

import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config.config import load_config
from util.gis import hilbert_index
from util.taxa import load_catalog

csv.field_size_limit(sys.maxsize)

CONFIG = load_config("global")

OCCURRENCE_PATH = Path("data/occurrences/occurrence.txt")
TREE_ROOT = Path("data/taxonomy/tree")

BUFFER_LIMIT = 5_000

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


def _flush(buffers: dict, taxon_path: str) -> None:
    rows = buffers[taxon_path]
    if not rows:
        return

    folder = TREE_ROOT / taxon_path
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / "occurrence.parquet"

    arrays = {field.name: [] for field in SCHEMA}
    for row in rows:
        for k, v in row.items():
            arrays[k].append(v)

    new_table = pa.table(
        {name: pa.array(vals, type=SCHEMA.field(name).type) for name, vals in arrays.items()},
        schema=SCHEMA,
    )

    if file_path.exists():
        existing = pq.read_table(file_path)
        if existing.schema != new_table.schema:
            existing = existing.cast(new_table.schema)
        new_table = pa.concat_tables([existing, new_table])

    from util.storage import atomic_write_parquet
    atomic_write_parquet(file_path, new_table, row_group_size=256)

    buffers[taxon_path].clear()


def main() -> None:
    catalog = load_catalog()
    buffers: dict[str, list] = defaultdict(list)
    rows_read = 0
    rows_written = 0

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

            taxon_key = (row.get("taxonKey") or "").strip()
            species_key = (row.get("speciesKey") or "").strip()
            lookup_key = taxon_key if rank in CONFIG.subspecies_equivalents else (species_key or taxon_key)
            if not lookup_key:
                continue

            taxon = catalog.get(lookup_key)
            if taxon is None:
                continue

            uncertainty_raw = (row.get("coordinateUncertaintyInMeters") or "").strip()
            try:
                uncertainty = float(uncertainty_raw) if uncertainty_raw else None
            except ValueError:
                uncertainty = None

            buffers[taxon["path"]].append({
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
            })
            rows_written += 1

            if len(buffers[taxon["path"]]) >= BUFFER_LIMIT:
                _flush(buffers, taxon["path"])

    print("  Flushing remaining buffers...", flush=True)
    for path in list(buffers):
        _flush(buffers, path)

    print(f"Done. {rows_read:,} rows read, {rows_written:,} written to tree.")
    # Only remove the occurrences dir when it's the real pipeline path (not a test tmp dir).
    occ_dir = OCCURRENCE_PATH.parent
    if occ_dir.name == "occurrences":
        shutil.rmtree(occ_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    main()
