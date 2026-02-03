'''
The purpose of this script is to take the constructed taxonomy tree from build_taxonomy_tree.py along with the pickle index from taxon ID to path, and to populate the taxonomy tree with values from the occurrence.txt from GBIF.
It does this by simply reading in rows from the txt, extracting the relevant rows, and appending them to the parquet file located at the folder belonging to the taxon ID.
Buffers are kept in memory so parquets are only written every once in a while, improving I/O throughput.
Upon completion, the tree should have a basic feature table at each terminal node, and we can add more features by indexing into GIS data at the saved lat/lon for each data point.
The occurrence.txt is sourced from the "Darwin Core Archive" option at www.gbif.org/occurrence/download. The iNat dataset has ~135 million rows.
'''

from pathlib import Path
import pandas as pd
import dateutil.parser
import json
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
import util.taxa_navigation as taxa_navigation
import util.gis_lookup as gis_lookup
from util.config import load_config

CONFIG = load_config("global")

annotations_to_kingdoms = {
            "dp": ["1"],
            "sex": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "lifeStage": ["1"],
            "rcs": ["6"],
            "vitality": ["1"],
            "gall": ["0", "2", "3", "4", "5", "6", "7", "8"],
        }

occurrence_list_delimiter = "|"

occurrence_numeric_columns = (
        "eventTimestamp",
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
    )

occurrence_read_chunksize = 1_000_000

occurrence_read_on_bad_lines = "skip"

occurrence_read_sep = "\t"

occurrence_required_fields = frozenset(
            {"decimalLatitude", "decimalLongitude", "catalogNumber"}
        )

occurrence_string_columns = (
        "dp",
        "sex",
        "lifeStage",
        "rcs",
        "vitality",
        "gall",
        "obscured",
        "catalogNumber",
        "tileId",
        "gbifRegion",
        "level0Gid",
        "level1Gid",
        "level2Gid",
    )

populate_buffer_limit = 5000


# chunk reading the input due to massive filesize, and skip malformed lines
read_kwargs = {
    "filepath_or_buffer": CONFIG.occurrence_path,
    "sep": occurrence_read_sep,
    "dtype": str,
    "chunksize": occurrence_read_chunksize,
    "on_bad_lines": occurrence_read_on_bad_lines,
}

reader = pd.read_csv(**read_kwargs)

# Fields that must be not NA in the row, otherwise we skip. We need the lat & lon for obvious reasons. catalogNumber = iNat obs ID. eventDate and eventTime allow us to construct a UNIX timestamp to save.
# coordinateUncertaintyInMeters tells us how accurate/trustworthy the location is.

# Certain annotations can only be present in certain kingdoms. 1 = animalia and many can only be present in animalia. 6 is plantae, rcs is "reproductive conditions" which is phenology, e.g. flowering info.

# The columns we write to each parquet. Some are derived from other column data in the input. The regions and levels are the location the observation was in, we save for location filtering later.

# Columns that can be lists of values, save and get indices here because we do stuff with them later.



# a dictionary of taxa to lists in-memory
buffers = defaultdict(list)

count = 0


def rows_to_table(rows: list[list[object]]) -> pa.Table:
    """Convert buffered rows into a typed PyArrow table."""
    df = pd.DataFrame(rows, columns=CONFIG.occurrence_all_columns)

    for col in occurrence_string_columns:
        df[col] = df[col].astype(str)
    for col in occurrence_numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return pa.Table.from_pandas(df)


def flush_taxon_buffer(taxon_path: str) -> None:
    """Append the current buffer for a taxon to its parquet, rewriting file atomically."""
    rows = buffers[taxon_path]
    if not rows:
        return

    folder = Path(taxon_path)
    # Make it if it does not exist. I don't think the parents argument is necessary but whatever.
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / CONFIG.occurrence_parquet_filename

    new_table = rows_to_table(rows)
    tables_to_concat = [new_table]

    if file_path.exists():
        existing_table = pq.read_table(file_path)
        # If there is a schema mismatch (probably shouldn't be) try to cast it. This can happen in strange type edge cases if all dates are null
        if existing_table.schema != new_table.schema:
            try:
                existing_table = existing_table.cast(new_table.schema)
            except pa.lib.ArrowInvalid as exc:
                raise RuntimeError(
                    f"Schema mismatch for {file_path}: {exc}"
                ) from exc
        tables_to_concat.insert(0, existing_table)

    combined = pa.concat_tables(tables_to_concat)

    # Concat the files, we use a temp operation here for an atomic write. Theoretically this could make it so the script can be stopped and resumed, but we don't quite do this as the script pretty much only needs to be run once.

    tmp_path = file_path.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp_path)
    tmp_path.replace(file_path)

    #print(f"Flushed {len(rows)} rows for {taxon_path} (total rows now {combined.num_rows})")
    buffers[taxon_path].clear()

for chunk in reader:
    print(f"processed {count} million rows")
    count += 1
    for row in chunk.itertuples():

        # skip observations not specific enough
        if row.taxonRank not in CONFIG.leaf_rank_set:
            continue

        # or observations with null values for required fields
        if any(pd.isna(getattr(row, field)) for field in occurrence_required_fields):
            continue

        # get the date and the time and convert to UNIX timestamp (if possible)
        unix_ts = None
        date = None
        if not pd.isna(row.eventDate):
            date = str(row.eventDate).split("T")[0]
        time_value = None
        if not pd.isna(row.eventTime):
            time_candidate = str(row.eventTime).strip()
            if time_candidate and time_candidate.lower() != "na":
                time_value = time_candidate
        if date and time_value:
            try:
                dt = dateutil.parser.isoparse(f"{date}T{time_value}")
                unix_ts = int(dt.timestamp())
            except (ValueError, TypeError):
                unix_ts = None

        # dynamic properties are given in a JSON compatible format, just parse it here
        dp = []
        if not pd.isna(row.dynamicProperties):
            obj = json.loads(row.dynamicProperties)
            dp = obj.get("evidenceOfPresence")
            if isinstance(dp, str):
                dp = [dp]
            elif dp is None:
                dp = []

        # reproductive conditions are given separated by |, parse them into a list here
        rcs = []
        if not pd.isna(row.reproductiveCondition):
            rcs = row.reproductiveCondition.split(occurrence_list_delimiter)

        # Not obsured unless informationWithheld is populated. If the last token is "taxon", it was obscured to protect a hidden taxon. Otherwise it was hidden by the user.
        obscured = "No"
        if not pd.isna(row.informationWithheld):
            obscured = "Hidden" if row.informationWithheld.split(" ")[-1] == "taxon" else "Obscured"

        # find the annotations we should add based on the kingdom of the observation
        annotations_to_add = [
            a for a in annotations_to_kingdoms
            if row.kingdomKey in annotations_to_kingdoms[a]
        ]

        # derive tile id for later GIS batching
        try:
            tile_id = gis_lookup.get_region_name(
                float(row.decimalLatitude),
                float(row.decimalLongitude),
            )
        except (TypeError, ValueError):
            continue

        # collect and add the columns
        mandatory_columns = [
            row.decimalLatitude,
            row.decimalLongitude,
            row.catalogNumber,
            tile_id,
            unix_ts,
            row.coordinateUncertaintyInMeters,
            obscured,
            row.gbifRegion,
            row.level0Gid,
            row.level1Gid,
            row.level2Gid
        ]

        annotation_values = {
            "dp": dp,
            "sex": row.sex,
            "lifeStage": row.lifeStage,
            "rcs": rcs,
            "vitality": row.vitality,
            "gall": ["gall"] if "gall" in dp else [],
        }

        final_extra = [
            annotation_values[col] if col in annotations_to_add else None
            for col in CONFIG.annotation_columns
        ]

        row_data = mandatory_columns + final_extra

        # turn list columns into strings separated by | since saving arrays doesn't really work
        for col in CONFIG.occurrence_list_columns:
            idx = CONFIG.occurrence_list_column_indices[col]
            if isinstance(row_data[idx], list):
                row_data[idx] = occurrence_list_delimiter.join(row_data[idx])

        lookup_key = row.taxonKey
        if row.taxonRank == "SPECIES" and not pd.isna(row.speciesKey):
            lookup_key = row.speciesKey

        taxon = taxa_navigation.get_taxon_by_id(lookup_key)
        if taxon is None:
            continue
        taxon_path = taxon["path"]

        # append the row to the buffer
        buffers[taxon_path].append(row_data)

        # if the buffer for the taxon has gotten too big, flush the buffer
        if len(buffers[taxon_path]) >= populate_buffer_limit:
            flush_taxon_buffer(taxon_path)

# final flush of all buffers after processing all rows
print("Finished processing, doing the final write")
for taxon_path in list(buffers.keys()):
    flush_taxon_buffer(taxon_path)

buffers.clear()
