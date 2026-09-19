# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Custom observation upload processing.

Normalizes a user-supplied CSV/TSV/Parquet file, enriches each observation with
static GIS layer values sampled from global COGs, computes summary statistics and
an occurrence index, then bundles everything into a downloadable ZIP archive.

Temporal enrichment is intentionally excluded: historical weather aggregates require
per-observation timestamps and the full ERA5 archive, which is not guaranteed to be
available at request time.
"""
from __future__ import annotations

import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rasterio
from fastapi import HTTPException

from config.config import METRICS_BY_TYPE, ZERO_NODATA_LAYERS, ValueType, load_config
from util import descriptions
from util.gis import (
    COMPOSITION_CLASSIFIERS,
    DERIVED_FROM_ELEVATION,
    DERIVED_FROM_SOIL,
    hilbert_index,
    sample_aspect_batch,
    sample_elevation_terrain_batch,
    sample_slope_batch,
    sample_soil_texture_batch,
    sample_vector_batch,
)
from util.rankings import (
    MIN_RANKING_SAMPLES,
    NOMINAL_SKIP_RANK_METRICS,
    ORDINAL_SKIP_RANK_METRICS,
    POSITION_FILE,
    rank_value_against_group,
    read_rank_context_groups,
    resolve_context_label,
)
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    DENSITY_GRID_FILE,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    _filter_df,
    process_observations_df,
)
from util.taxa import get_ancestors, get_taxon_by_id
from util.temporal import (
    TailBuffer,
    build_chunk_index,
    load_temporal_layers,
    map_to_worklist,
    process_chunk,
    process_chunk_mode,
    process_chunk_vpd,
    window_steps,
)
from util.ternary import build_ternary_classification_overlay, composition_group_members
from util.tiles import LAYERS_DIR, load_layers_with_category, resolve_layer_path

_LEGEND_DIR = Path("config/gis/legends")
_GADM_PATH = Path("data/gis/gadm.gpkg")
_HIERARCHY_PATH = Path("data/gis/locations/hierarchy.csv")
_CATALOG_PATH = Path("config/gis/catalog.json")

_CONFIG = load_config("global")

_gadm_gdf = None
_hierarchy: dict[str, dict] | None = None


def _load_gadm_gdf():
    global _gadm_gdf
    if _gadm_gdf is not None:
        return _gadm_gdf
    if not _GADM_PATH.exists():
        return None
    gdf = gpd.read_file(_GADM_PATH, layer="gadm_410", engine="pyogrio", columns=["GID_0", "GID_1", "GID_2"])
    _gadm_gdf = gdf
    return _gadm_gdf


def _load_hierarchy() -> dict[str, dict]:
    global _hierarchy
    if _hierarchy is not None:
        return _hierarchy
    if not _HIERARCHY_PATH.exists():
        _hierarchy = {}
        return _hierarchy
    import csv as _csv
    result: dict[str, dict] = {}
    with _HIERARCHY_PATH.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            gid = row.get("gid", "")
            if gid:
                result[gid] = {
                    "name": row.get("name", gid),
                    "level": int(row["level"]),
                    "parent_gid": row.get("parent_gid") or None,
                }
    _hierarchy = result
    return _hierarchy


def _resolve_hierarchy(gid: str, by_gid: dict[str, dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    current = by_gid.get(gid, {}).get("parent_gid")
    while current:
        if current in seen:
            break
        seen.add(current)
        rec = by_gid.get(current)
        if rec is None:
            break
        names.append(rec["name"])
        current = rec.get("parent_gid")
    names.reverse()
    return names


def build_locations_table(df: pd.DataFrame) -> pa.Table | None:
    """Build a locations table from the GID columns present in df."""
    by_gid = _load_hierarchy()
    if not by_gid:
        return None

    level_cols = [("level2Gid", 2), ("level1Gid", 1), ("level0Gid", 0)]
    seen: set[str] = set()
    rows_gid: list[str] = []
    rows_name: list[str] = []
    rows_level: list[int] = []
    rows_hierarchy: list[str] = []  # JSON-encoded list

    for col, level in level_cols:
        if col not in df.columns:
            continue
        for gid in df[col].dropna().unique():
            if not gid or gid in seen:
                continue
            seen.add(gid)
            rec = by_gid.get(gid)
            rows_gid.append(gid)
            rows_name.append(rec["name"] if rec else gid)
            rows_level.append(level)
            rows_hierarchy.append(json.dumps(_resolve_hierarchy(gid, by_gid)))

    if not rows_gid:
        return None

    return pa.table({
        "gid": pa.array(rows_gid, type=pa.string()),
        "name": pa.array(rows_name, type=pa.string()),
        "level": pa.array(rows_level, type=pa.int32()),
        "hierarchy": pa.array(rows_hierarchy, type=pa.string()),
    })


# ---------------------------------------------------------------------------
# Natural-language description (util.descriptions), for an arbitrary
# self-contained dataset -- shared by the upload path (build_archive) and the
# taxon-download path (util.download.build_species_archive), same as
# _package_archive above: both leave a set of standard-named stats files in
# work_dir first, this just reads them back.
# ---------------------------------------------------------------------------

def _load_legend_full(layer_id: str) -> dict:
    path = _LEGEND_DIR / f"{layer_id}_legend.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


class _InMemoryLocTaxaStorage:
    """Duck-types util.storage.ParquetStorage's read_table() just enough for
    descriptions.build_location_text() -- there's only ever one dataset here
    (this one upload/download job), so the path/filters it's called with are
    irrelevant; the in-memory table already IS the filtered result."""

    def __init__(self, table: pa.Table):
        self._table = table

    def read_table(self, path, columns=None, filters=None):  # noqa: ARG002
        return self._table


def _build_location_counts_table(df: pd.DataFrame) -> pa.Table | None:
    """(scope, gid, count) rows from df's own level0Gid/level1Gid/level2Gid
    columns -- the exact shape descriptions.build_location_text() expects
    from a real taxon's precomputed location_taxa.parquet, just derived
    directly from this one dataset's own rows instead of a stored,
    taxon-keyed aggregate."""
    scopes: list[str] = []
    gids: list[str] = []
    counts: list[int] = []
    for col, scope in _CONFIG.location_columns:
        if col not in df.columns:
            continue
        for gid, count in df[col].dropna().value_counts().items():
            if not gid:
                continue
            scopes.append(scope)
            gids.append(str(gid))
            counts.append(int(count))
    if not gids:
        return None
    return pa.table({
        "scope": pa.array(scopes, type=pa.string()),
        "gid": pa.array(gids, type=pa.string()),
        "count": pa.array(counts, type=pa.int64()),
    })


def build_description_profile_for_df(work_dir: Path, df: pd.DataFrame) -> dict:
    """Same descriptions.build_description_profile() the species page uses,
    fed from this one dataset's own just-computed stats (already written into
    work_dir under their standard names -- see process_observations_df/
    _copy_taxon_stats) and its own location counts, instead of a real taxon's
    precomputed global aggregates. taxon_key/loc_taxa_path below are inert
    placeholders -- _InMemoryLocTaxaStorage.read_table() ignores both, since
    there's only ever this one dataset's worth of location counts to return.
    """
    def _read_rows(filename: str) -> list[dict]:
        path = work_dir / filename
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    numerical_stats = {r["variable"]: r for r in _read_rows(NUMERICAL_STATS_FILE)}
    circular_stats = {r["variable"]: r for r in _read_rows(CIRCULAR_STATS_FILE)}
    nominal_rows = _read_rows(NOMINAL_STATS_FILE)
    ordinal_rows = _read_rows(ORDINAL_STATS_FILE)

    def _class_fractions(variable: str) -> dict[int, float]:
        return {
            int(r["metric"][6:]): float(r["value"])
            for r in nominal_rows
            if r["variable"] == variable
            and r["metric"].startswith("class_")
            and r["metric"][6:].isdigit()
            and float(r["value"] or 0) > 0
        }

    salinity_median = next(
        (float(r["value"]) for r in ordinal_rows if r["variable"] == "salinity" and r["metric"] == "median"),
        None,
    )

    locations_table = _build_location_counts_table(df)
    storage = _InMemoryLocTaxaStorage(locations_table if locations_table is not None else pa.table({
        "scope": pa.array([], type=pa.string()),
        "gid": pa.array([], type=pa.string()),
        "count": pa.array([], type=pa.int64()),
    }))

    return descriptions.build_description_profile(
        "upload",
        hierarchy=_load_hierarchy(),
        storage=storage,
        loc_taxa_path=Path("unused"),
        scope_by_level=_CONFIG.location_scope_by_level,
        kg2_class_fractions=_class_fractions("kg2") or None,
        kg2_legend_classes=_load_legend("kg2") or None,
        lc_class_fractions=_class_fractions("landcover") or None,
        lc_legend=_load_legend_full("landcover") or None,
        soil_texture_class_fractions=_class_fractions("soil_texture") or None,
        soil_texture_legend=_load_legend_full("soil_texture") or None,
        eco_class_fractions=_class_fractions("ecoregions") or None,
        eco_legend_classes=_load_legend("ecoregions") or None,
        biome_class_fractions=_class_fractions("biome") or None,
        biome_legend=_load_legend_full("biome") or None,
        salinity_median=salinity_median,
        salinity_legend_classes=_load_legend("salinity") or None,
        numerical_stats=numerical_stats or None,
        circular_stats=circular_stats or None,
    )


# ---------------------------------------------------------------------------
# Parent-taxon relative ranking ("extra options" opt-in on a raw CSV upload)
#
# A raw upload has no taxon_key and isn't in the tree, so it can't use the
# real per-taxon precomputed relative_ranks_positions.parquet the download
# path re-exports (see build_species_archive). Instead, when the user picks
# a parent taxon in the upload UI, this treats the upload's own computed
# stats as if they belonged to a new SPECIES-level child of that taxon, and
# ranks them against that parent's real, already-precomputed sibling index
# (util.rankings.read_rank_context_groups) via a cheap binary search per
# metric -- no full tree rebuild, and nothing is written back to the real
# index, so this is purely additive/read-only from the tree's perspective.
# ---------------------------------------------------------------------------

_CATEGORICAL_SAMPLE_COUNT_METRIC_PRIORITY: dict[ValueType, tuple[str, ...]] = {
    # Mirrors _write_rank_positions' own count_idx lookup: {variable}::count
    # if the value type has one (ordinal does, as a tall metric row, unlike
    # numeric/circular's wide "count" column), else {variable}::total_samples.
    ValueType.NOMINAL: ("total_samples",),
    ValueType.ORDINAL: ("count", "total_samples"),
}


def _resolve_own_categorical_rankable_values(
    work_dir: Path, layer_meta: dict[str, dict], filename: str, vtype: ValueType,
) -> dict[tuple[str, str], tuple[float, int]]:
    """(variable, metric) -> (own_value, own_sample_count) for a nominal or
    ordinal variable's rankable metrics -- the exact same METRICS_BY_TYPE
    minus skip-set vocabulary util.rankings._write_rank_positions uses (see
    NOMINAL_SKIP_RANK_METRICS/ORDINAL_SKIP_RANK_METRICS), plus its class_
    fraction handling: a taxon (here, this upload) with zero presence in a
    class gets no row at all for it, matching the real pipeline never
    writing one for a zero-valued taxon.
    """
    path = work_dir / filename
    if not path.exists():
        return {}

    skip_metrics = (
        NOMINAL_SKIP_RANK_METRICS if vtype is ValueType.NOMINAL else ORDINAL_SKIP_RANK_METRICS
    )
    rankable_metrics = set(METRICS_BY_TYPE[vtype]) - skip_metrics
    sample_count_metrics = _CATEGORICAL_SAMPLE_COUNT_METRIC_PRIORITY[vtype]

    by_variable: dict[str, dict[str, float]] = {}
    for row in pq.read_table(path).to_pylist():
        variable = row.get("variable")
        metric = row.get("metric")
        if not variable or not metric:
            continue
        by_variable.setdefault(variable, {})[metric] = row.get("value")

    result: dict[tuple[str, str], tuple[float, int]] = {}
    for variable, by_metric in by_variable.items():
        layer = layer_meta.get(variable)
        if not layer:
            continue
        try:
            if ValueType(layer.get("value_type") or "") != vtype:
                continue
        except ValueError:
            continue

        sample_count = next(
            (by_metric[m] for m in sample_count_metrics if by_metric.get(m) is not None),
            None,
        )
        if sample_count is None or sample_count < MIN_RANKING_SAMPLES:
            continue
        sample_count = int(sample_count)

        for metric, value in by_metric.items():
            if value is None or not math.isfinite(value):
                continue
            is_class_metric = metric.startswith("class_")
            if not is_class_metric and metric not in rankable_metrics:
                continue
            if is_class_metric and value == 0.0:
                continue  # no presence in this class -- matches the real writer's own row omission
            result[(variable, metric)] = (float(value), sample_count)

    return result


def _resolve_own_rankable_values(
    work_dir: Path, layer_meta: dict[str, dict],
) -> dict[tuple[str, str], tuple[float, int]]:
    """(variable, metric) -> (own_value, own_sample_count) for every
    metric this upload has >= MIN_RANKING_SAMPLES samples for -- the exact
    same metric vocabulary + per-variable sample-count gate
    util.rankings._write_rank_positions applies when building the real
    tree's ranking index, so every value here is directly comparable to
    that index. Covers numerical, circular, nominal, and ordinal variables.
    """
    def _read_rows(filename: str) -> list[dict]:
        path = work_dir / filename
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    result: dict[tuple[str, str], tuple[float, int]] = {}

    for row in _read_rows(NUMERICAL_STATS_FILE):
        variable = row.get("variable")
        layer = layer_meta.get(variable)
        if not variable or not layer:
            continue
        try:
            vtype = ValueType(layer.get("value_type") or "")
        except ValueError:
            continue
        if vtype not in (ValueType.RATIO, ValueType.INTERVAL):
            continue
        count = row.get("count")
        if count is None or count < MIN_RANKING_SAMPLES:
            continue
        for metric in METRICS_BY_TYPE[vtype]:
            value = row.get(metric)
            if value is None or not math.isfinite(value):
                continue
            result[(variable, metric)] = (float(value), int(count))

    for row in _read_rows(CIRCULAR_STATS_FILE):
        variable = row.get("variable")
        layer = layer_meta.get(variable)
        if not variable or not layer or layer.get("value_type") != ValueType.CIRCULAR:
            continue
        count = row.get("count")
        if count is None or count < MIN_RANKING_SAMPLES:
            continue
        for metric in METRICS_BY_TYPE[ValueType.CIRCULAR]:
            value = row.get(metric)
            if value is None or not math.isfinite(value):
                continue
            result[(variable, metric)] = (float(value), int(count))

    result.update(_resolve_own_categorical_rankable_values(
        work_dir, layer_meta, NOMINAL_STATS_FILE, ValueType.NOMINAL,
    ))
    result.update(_resolve_own_categorical_rankable_values(
        work_dir, layer_meta, ORDINAL_STATS_FILE, ValueType.ORDINAL,
    ))

    return result


def compute_relative_ranks_for_upload(
    work_dir: Path, layer_meta: dict[str, dict], parent_taxon_id: str,
) -> list[dict] | None:
    """Rank this upload's own computed stats against ``parent_taxon_id``'s
    real precomputed SPECIES-level sibling index, AND every one of that
    taxon's own ancestors' sibling indexes up to the root -- one context row
    per ancestor level, same as a real taxon in the tree gets ranked against
    its whole lineage (main.py's _load_relative_ranks returns one row per
    ancestor context for a given taxon_key/variable). Writes the result into
    ``work_dir / POSITION_FILE`` so _package_archive picks it up exactly
    like a species download's own (real) relative ranks. Returns the rows
    written, or None (writing nothing) if the taxon doesn't resolve or
    nothing in this upload clears the ranking sample-size threshold.

    Always ranks as a SPECIES-level entrant regardless of each context
    taxon's own rank (a genus, family, order, ... ancestor's SPECIES-rank
    descendants are all well-defined comparison cohorts) -- this mirrors
    the common case of comparing one species against its congeners, family,
    order, and so on up the tree.
    """
    selected = get_taxon_by_id(parent_taxon_id)
    if selected is None:
        return None

    own_values = _resolve_own_rankable_values(work_dir, layer_meta)
    if not own_values:
        return None

    rows: list[dict] = []
    for context_taxon in (selected, *get_ancestors(selected)):
        context_id = str(context_taxon["taxon_key"])
        context_label = resolve_context_label(context_taxon)
        groups = read_rank_context_groups(context_id, _CONFIG.species_rank)
        if not groups:
            continue
        for (variable, metric), (value, sample_count) in own_values.items():
            group = groups.get((variable, metric))
            if group is None or group.empty:
                continue
            ranked = rank_value_against_group(value, group)
            rows.append({
                "variable": variable,
                "metric": metric,
                "position": ranked["position"],
                "count": ranked["count"],
                "sampleCount": sample_count,
                "contextLabel": context_label,
            })

    if not rows:
        return None

    pq.write_table(pa.Table.from_pylist(rows), work_dir / POSITION_FILE)
    return rows


# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

_LAT_ALIASES = (
    "decimalLatitude", "decimal_latitude", "latitude", "lat",
    "lat_dd", "latitude_dd", "y",
)
_LON_ALIASES = (
    "decimalLongitude", "decimal_longitude", "longitude", "lon", "lng",
    "long", "lon_dd", "longitude_dd", "x",
)
_CATALOG_ALIASES = (
    "catalogNumber", "catalog_number",
    "occurrenceID",  "occurrence_id",
    "observationID", "observation_id",
    "recordID",      "record_id",
    "gbifID",        "gbif_id",
)
_NAME_ALIASES = ("observationName", "observation_name", "name", "title", "label")
_IMAGE_ALIASES = (
    "imageUrl", "image_url", "imageURL",
    "photoUrl", "photo_url", "photoURL",
    "mediaUrl", "media_url", "mediaURL",
    "image", "photo",
)
_DATE_ALIASES = (
    "eventDate", "event_date", "dateTime", "date_time",
    "date", "datetime", "timestamp",
    "observed_at", "observation_date", "observed_on", "observedOn",
    "recorded_at", "created_at",
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_norm = {}
    for col in columns:
        by_norm.setdefault(_norm(col), col)
    for alias in aliases:
        match = by_norm.get(_norm(alias))
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "decimalLatitude" not in df.columns:
        col = _find_column(list(df.columns), _LAT_ALIASES) or next(
            (c for c in df.columns if "latitude" in c.lower()), None
        )
        if col:
            df = df.rename(columns={col: "decimalLatitude"})
    if "decimalLongitude" not in df.columns:
        col = _find_column(list(df.columns), _LON_ALIASES) or next(
            (c for c in df.columns if "longitude" in c.lower()), None
        )
        if col:
            df = df.rename(columns={col: "decimalLongitude"})
    return df


def ensure_catalog_numbers(df: pd.DataFrame) -> pd.DataFrame:
    if "catalogNumber" in df.columns:
        df = df.copy()
        df["_catalogAutoGenerated"] = False
        return df
    df = df.copy()
    col = _find_column(list(df.columns), _CATALOG_ALIASES)
    if col:
        df = df.rename(columns={col: "catalogNumber"})
        df["_catalogAutoGenerated"] = False
        return df
    df["catalogNumber"] = [f"Observation #{i}" for i in range(1, len(df) + 1)]
    df["_catalogAutoGenerated"] = True
    return df


def ensure_observation_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "observationName" not in df.columns:
        col = _find_column(list(df.columns), _NAME_ALIASES)
        if col:
            df = df.rename(columns={col: "observationName"})
        else:
            df["observationName"] = [f"Observation #{i}" for i in range(1, len(df) + 1)]
    missing = df["observationName"].isna() | (df["observationName"].astype(str).str.strip() == "")
    if missing.any():
        fallback = pd.Series([f"Observation #{i}" for i in range(1, len(df) + 1)], index=df.index)
        # A column that's empty for every row (e.g. an empty CSV column) is
        # read as float64, which can't hold these strings -- pandas raises
        # rather than upcasting.
        df["observationName"] = df["observationName"].astype(object)
        df.loc[missing, "observationName"] = fallback[missing]
    return df


def normalize_image_column(df: pd.DataFrame) -> pd.DataFrame:
    """Recognizes an optional per-occurrence photo URL column under any of
    _IMAGE_ALIASES and renames it to the canonical `imageUrl` -- unlike
    catalogNumber/observationName above, this is genuinely optional (no
    synthesized fallback) and rides through to occurrence.parquet untouched,
    same as any other unrecognized column would; this just widens which
    column names are picked up as the intended one. See
    frontend/data/uploadLocalSpeciesDataSource.normalize.ts, which reads
    this exact column name back out as SpeciesOccurrence.mediaUrl."""
    if "imageUrl" not in df.columns:
        col = _find_column(list(df.columns), _IMAGE_ALIASES)
        if not col:
            return df
        df = df.rename(columns={col: "imageUrl"})
    df = df.copy()
    blank = df["imageUrl"].isna() | (df["imageUrl"].astype(str).str.strip() == "")
    df.loc[blank, "imageUrl"] = None
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"decimalLatitude", "decimalLongitude"} - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required coordinate columns: {', '.join(sorted(missing))}")
    df = df.copy()
    lats = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    lons = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    invalid = lats.isna() | lons.isna() | (lats < -90) | (lats > 90) | (lons < -180) | (lons > 180)
    if invalid.any():
        raise HTTPException(status_code=422, detail=f"Invalid coordinates in {int(invalid.sum())} row(s).")
    df["decimalLatitude"] = lats
    df["decimalLongitude"] = lons
    return df


def check_reserved_columns(df: pd.DataFrame, layer_ids: set[str]) -> None:
    """Reject uploads that pre-populate GIS layer columns — they'll be overwritten."""
    conflicts = sorted(set(df.columns) & layer_ids)
    if conflicts:
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded file contains columns reserved for GIS enrichment: "
                f"{', '.join(conflicts)}. Remove or rename them and try again."
            ),
        )


# ---------------------------------------------------------------------------
# GIS enrichment
# ---------------------------------------------------------------------------

def enrich_with_gadm(df: pd.DataFrame) -> pd.DataFrame:
    """Add level0Gid/level1Gid/level2Gid columns via point-in-polygon against GADM 4.1."""
    if df.empty:
        return df.copy()
    gdf = _load_gadm_gdf()
    if gdf is None:
        return df.copy()

    points = gpd.GeoDataFrame(
        {"_orig_index": df.index},
        geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, gdf[["GID_0", "GID_1", "GID_2", gdf.geometry.name]], how="left", predicate="within")
    # deduplicate in case a point lands on a shared boundary
    joined = joined[~joined.index.duplicated(keep="first")]

    result = df.copy()
    for src, dst in [("GID_0", "level0Gid"), ("GID_1", "level1Gid"), ("GID_2", "level2Gid")]:
        col = joined[src].reindex(df.index)
        result[dst] = col.where(col.notna(), other=None)

    # single most-specific GID per row for frontend filtering
    result["locationGid"] = (
        result["level2Gid"]
        .where(result["level2Gid"].notna(), result["level1Gid"])
        .where(result["level1Gid"].notna() | result["level2Gid"].notna(), result["level0Gid"])
    )
    return result


def _sample_layer(
    path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    scale: float,
    offset: float,
    nodata: float | None,
    layer_id: str = "",
) -> list[float | None]:
    coords = list(zip(lons.tolist(), lats.tolist()))
    zero_nodata = layer_id in ZERO_NODATA_LAYERS
    with rasterio.open(path) as ds:
        nd = ds.nodata if nodata is None else nodata
        results: list[float | None] = []
        for point in ds.sample(coords):
            v = float(point[0])
            if nd is not None and v == nd:
                results.append(0.0 if zero_nodata else None)
            else:
                results.append(v * scale + offset)
    return results


def _load_legend(layer_id: str) -> list[dict]:
    path = _LEGEND_DIR / f"{layer_id}_legend.json"
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    return data.get("classes", [])


def _build_layer_meta() -> dict[str, dict]:
    return {
        layer["id"]: {
            **layer,
            "category_id": cat["id"],
            "category_display_name": cat.get("display_name", cat["id"]),
        }
        for layer, cat in load_layers_with_category()
        if (layer.get("filename") or layer["id"] in DERIVED_FROM_ELEVATION or layer["id"] in DERIVED_FROM_SOIL)
        and layer.get("window_hours") is None
    }


def enrich_with_gis(df: pd.DataFrame) -> pd.DataFrame:
    """Add static GIS layer values to every observation.

    Rows are reordered by Hilbert index before sampling for COG spatial cache
    locality, then restored to their original order.
    """
    if df.empty:
        return df.copy()

    layers = [
        layer for layer in _build_layer_meta().values()
    ]

    lats = df["decimalLatitude"].to_numpy(dtype=float)
    lons = df["decimalLongitude"].to_numpy(dtype=float)
    order = np.argsort(
        [hilbert_index(float(la), float(lo)) for la, lo in zip(lats, lons)],
        kind="stable",
    )
    restore = np.argsort(order, kind="stable")
    s_lats = lats[order]
    s_lons = lons[order]

    elev_path = resolve_layer_path(LAYERS_DIR, "elevation.tif")

    result = df.copy()
    for layer in layers:
        layer_id = layer["id"]
        try:
            if layer_id in DERIVED_FROM_ELEVATION:
                if not elev_path.exists():
                    continue
                if layer_id == "aspect":
                    sorted_vals = sample_aspect_batch(s_lats, s_lons)
                else:
                    sorted_vals = sample_slope_batch(s_lats, s_lons)
                result[layer_id] = [sorted_vals[i] for i in restore]
            elif layer_id in DERIVED_FROM_SOIL:
                sorted_vals = sample_soil_texture_batch(s_lats, s_lons)
                result[layer_id] = [sorted_vals[i] for i in restore]
            elif layer.get("vector_field"):
                vec_path = resolve_layer_path(LAYERS_DIR, layer["filename"])
                if not vec_path.exists():
                    continue
                sorted_arr = sample_vector_batch(vec_path, layer["vector_field"], s_lats, s_lons)
                result[layer_id] = [None if np.isnan(sorted_arr[i]) else sorted_arr[i] for i in restore]
            else:
                cog_path = resolve_layer_path(LAYERS_DIR, layer["filename"])
                if not cog_path.exists():
                    continue
                scale  = layer.get("scale_factor") or 1.0
                offset = layer.get("add_offset")   or 0.0
                sorted_vals = _sample_layer(cog_path, s_lats, s_lons, scale, offset, nodata=None, layer_id=layer_id)
                result[layer_id] = [sorted_vals[i] for i in restore]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Temporal enrichment
# ---------------------------------------------------------------------------

def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Detect a date/time column and normalize it to eventTimestamp (Unix float, UTC).

    Returns df unchanged if no recognizable date column is found. Rows with
    unparseable timestamps get NaN. Date-only values (no time component) are
    shifted to noon UTC so hourly ERA5 lookups land in the middle of the day
    rather than at the previous day's last timestep.
    """
    import datetime as _dt
    col = _find_column(list(df.columns), _DATE_ALIASES)
    if col is None:
        return df

    utc = _dt.UTC
    noon_offset = 12 * 3600.0
    sentinel = {"", "none", "nan", "nat", "null"}

    def _parse(raw_val) -> float:
        if raw_val is None:
            return float("nan")
        if isinstance(raw_val, float) and np.isnan(raw_val):
            return float("nan")
        s = str(raw_val).strip()
        if s.lower() in sentinel:
            return float("nan")
        try:
            ts = pd.to_datetime(s)
            if ts is pd.NaT:
                return float("nan")
            if ts.tzinfo is None:
                unix = ts.replace(tzinfo=utc).timestamp()
            else:
                unix = ts.astimezone(utc).timestamp()
            if ":" not in s:
                unix += noon_offset
            return unix
        except Exception:
            return float("nan")

    result = df.copy()
    result["eventTimestamp"] = df[col].map(_parse)
    return result


def _df_to_occ_table(df: pd.DataFrame) -> pa.Table:
    """Build an occ_index Arrow table from the upload DataFrame for temporal.map_to_worklist."""
    valid = (
        df["eventTimestamp"].notna()
        & df["decimalLatitude"].notna()
        & df["decimalLongitude"].notna()
    )
    valid_idx = df.index[valid]
    lats  = df.loc[valid, "decimalLatitude"].to_numpy(dtype=np.float64)
    lons  = df.loc[valid, "decimalLongitude"].to_numpy(dtype=np.float64)
    times = df.loc[valid, "eventTimestamp"].to_numpy(dtype=np.float64)
    rows  = valid_idx.to_numpy(dtype=np.int64)
    order = np.argsort(
        [hilbert_index(float(la), float(lo)) for la, lo in zip(lats, lons)],
        kind="stable",
    )
    restore = np.argsort(order, kind="stable")
    elev_sorted = sample_elevation_terrain_batch(
        lats[order], lons[order], want_elevation=True
    ).get("elevation", [])
    elev_arr = np.array(
        [v if v is not None else np.nan for v in elev_sorted], dtype=np.float64
    )
    elevations = elev_arr[restore] if len(elev_arr) == len(rows) else np.full(len(rows), np.nan)
    return pa.table({
        "taxon_path": pa.array(["__upload__"] * len(rows), type=pa.string()),
        "row_idx":    pa.array(rows,                       type=pa.int64()),
        "latitude":   pa.array(lats,                       type=pa.float64()),
        "longitude":  pa.array(lons,                       type=pa.float64()),
        "timestamp":  pa.array(times,                      type=pa.float64()),
        "elevation":  pa.array(elevations,                 type=pa.float64()),
    })


def _apply_temporal_updates(
    df: pd.DataFrame,
    all_updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]],
) -> pd.DataFrame:
    result = df.copy()
    for col_map in all_updates.values():
        for col, pairs in col_map.items():
            if col not in result.columns:
                result[col] = np.nan
            for row_ids, values in pairs:
                result.loc[row_ids, col] = values
    return result


def _process_one_layer(
    layer,
    occ_table: pa.Table,
    raw_cache: dict | None = None,
) -> dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Process all chunks for a single temporal layer and return its updates.

    Runs sequentially within the layer (tail buffer requires chunk ordering).
    Intended to be called concurrently across layers, which are independent.
    """
    updates_out: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    primary_var = layer.sources[0] if layer.sources else layer.id

    try:
        chunk_index = build_chunk_index(layer.model, primary_var)
    except Exception:
        return updates_out

    worklist = map_to_worklist(occ_table, chunk_index, layer.grid_mode, layer.grid_step)
    if worklist.num_rows == 0:
        return updates_out

    chunk_nums_present = set(worklist.column("chunk_num").to_pylist())
    chunks_to_process = [e for e in chunk_index.ranges if e.chunk_num in chunk_nums_present]

    steps = window_steps(chunk_index.resolution, tuple(layer.windows))

    secondary_indices: dict = {}
    for src_var in layer.sources[1:]:
        try:
            secondary_indices[src_var] = build_chunk_index(layer.model, src_var)
        except Exception:
            pass

    tail_buffer: TailBuffer = {}

    for ci, chunk_entry in enumerate(chunks_to_process):
        chunk_worklist = worklist.filter(
            pc.equal(worklist.column("chunk_num"), chunk_entry.chunk_num)
        )
        if chunk_worklist.num_rows == 0:
            continue
        try:
            if layer.id == "vapor_pressure_deficit":
                chunk_updates, tail_buffer = process_chunk_vpd(
                    chunk_entry, chunk_worklist, tail_buffer,
                    layer.model, layer.sources, layer.id,
                    steps, chunk_index.resolution, "",
                    secondary_indices=secondary_indices or None,
                    range_request=True,
                    raw_cache=raw_cache,
                )
            elif layer.sources:
                chunk_updates, tail_buffer = process_chunk_mode(
                    chunk_entry, chunk_worklist, tail_buffer,
                    layer.model, layer.sources, layer.id,
                    steps, chunk_index.resolution, "",
                    secondary_indices=secondary_indices or None,
                    range_request=True,
                    raw_cache=raw_cache,
                )
            else:
                chunk_updates, tail_buffer = process_chunk(
                    chunk_entry, chunk_worklist, tail_buffer,
                    layer.model, layer.id, steps, layer.agg, "",
                    range_request=True,
                    raw_cache=raw_cache,
                )
            for tpath, col_map in chunk_updates.items():
                updates_out.setdefault(tpath, {})
                for col, pairs in col_map.items():
                    updates_out[tpath].setdefault(col, []).extend(pairs)
        except Exception:
            continue

    return updates_out


# One worker per layer — each hits a distinct S3 prefix so there's no shared
# resource contention. Workers block on network I/O, not CPU.
_UPLOAD_TEMPORAL_WORKERS = 9


def enrich_with_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Add ERA5 time-windowed statistics via HTTP range requests (no local cache).

    Skipped silently if eventTimestamp is absent or entirely null. Rows with
    null timestamps get NaN in all temporal output columns.
    """
    if "eventTimestamp" not in df.columns or df["eventTimestamp"].notna().sum() == 0:
        return df

    try:
        temporal_layers = load_temporal_layers(_CATALOG_PATH)
    except Exception:
        return df

    active_layers = [lay for lay in temporal_layers if not lay.derived]
    base_layers = [lay for lay in active_layers if not lay.sources]
    composite_layers = [lay for lay in active_layers if lay.sources]

    occ_table = _df_to_occ_table(df)
    if occ_table.num_rows == 0:
        return df

    # Shared raw cell data cache: (model, variable, chunk_num, lat_idx, lon_idx) -> array.
    # Base layers populate it; composite layers (weather_code, VPD) reuse it to avoid
    # re-fetching variables they share with base layers. Upload-only — never passed in the
    # batch enrich_temporal script where it would grow unbounded.
    raw_cell_cache: dict = {}

    all_updates: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}

    def _collect(futures: dict) -> None:
        for future in as_completed(futures):
            try:
                layer_updates = future.result()
            except Exception:
                continue
            for tpath, col_map in layer_updates.items():
                all_updates.setdefault(tpath, {})
                for col, pairs in col_map.items():
                    all_updates[tpath].setdefault(col, []).extend(pairs)

    with ThreadPoolExecutor(max_workers=min(_UPLOAD_TEMPORAL_WORKERS, len(base_layers) or 1)) as pool:
        _collect({
            pool.submit(_process_one_layer, layer, occ_table, raw_cell_cache): layer
            for layer in base_layers
        })

    if composite_layers:
        with ThreadPoolExecutor(max_workers=min(_UPLOAD_TEMPORAL_WORKERS, len(composite_layers))) as pool:
            _collect({
                pool.submit(_process_one_layer, layer, occ_table, raw_cell_cache): layer
                for layer in composite_layers
            })

    return _apply_temporal_updates(df, all_updates)


def _build_temporal_var_meta(df: pd.DataFrame) -> list[dict]:
    """Return variable_metadata rows for temporal columns present in df."""
    try:
        temporal_layers = load_temporal_layers(_CATALOG_PATH)
    except Exception:
        return []

    # Load raw catalog to get display_name, units, value_type per layer.
    category_display = "Recent Weather"
    raw_layer_meta: dict[str, dict] = {}
    try:
        with _CATALOG_PATH.open() as f:
            cat = json.load(f)
        for c in cat.get("categories", []):
            if c.get("id") != "temporal":
                continue
            category_display = c.get("display_name", category_display)
            for raw in c.get("layers", []):
                raw_layer_meta[raw["id"]] = raw
    except Exception:
        pass

    rows: list[dict] = []
    sort_offset = 10000  # place after static layer entries
    for i, layer in enumerate(temporal_layers):
        raw = raw_layer_meta.get(layer.id, {})
        display_name = raw.get("display_name") or layer.id
        units = raw.get("units") or None
        imperial_unit = raw.get("imperial_unit") or None
        value_type = raw.get("value_type") or "interval"
        domain = "discrete" if value_type in ("nominal", "ordinal") else "continuous"
        # process_chunk_mode hardcodes "mode" in column name regardless of layer.agg
        if layer.sources and layer.id != "vapor_pressure_deficit":
            col_agg = "mode"
        else:
            col_agg = layer.agg
        # For nominal/ordinal layers (e.g. weather_code), load legend from the base layer id.
        legend_json: str | None = None
        if value_type in ("nominal", "ordinal"):
            raw_classes = _load_legend(layer.id)
            if raw_classes:
                legend_json = json.dumps([
                    {
                        "id": cls["id"],
                        "name": cls.get("name", str(cls["id"])),
                        "color": cls.get("traits", {}).get("color") or None,
                    }
                    for cls in raw_classes
                ])
        for w in layer.windows:
            col = f"{layer.id}_{col_agg}_{w}h"
            if col not in df.columns:
                continue
            rows.append({
                "id":            col,
                "name":          display_name,
                "units":         units,
                "imperial_unit": imperial_unit,
                "value_type":    value_type,
                "domain":        domain,
                "category":      category_display,
                "group":         None,
                "group_label":   None,
                "sort_order":    sort_offset + i * len(layer.windows) + layer.windows.index(w),
                "render_min":    None,
                "render_max":    None,
                "legend_classes": legend_json,
                "_legend_key":   layer.id,  # base id for categorical lookup
            })
    return rows


# ---------------------------------------------------------------------------
# Custom layers ("extra options" opt-in on a raw CSV upload)
#
# A custom layer is a raster/vector file authored or edited via /gis-editor,
# never uploaded to this backend at all -- sampling happens entirely in the
# browser (see frontend/components/upload/customLayers.ts, which reuses
# /gis-editor's own raster/vector inspection and point-sampling code), and
# the frontend sends only: the already-sampled per-observation values, as
# an ordinary extra column in the raw CSV/TSV/Parquet payload, plus this
# small JSON description of what that column means. This is merged into
# layer_meta exactly like _build_temporal_var_meta's synthetic rows above,
# so process_observations_df computes real stats for it same as any built-
# in layer -- the only thing that doesn't apply is relative-rank comparison
# (compute_relative_ranks_for_upload only ever finds a sibling group for a
# variable a real taxon in the tree actually has, so a custom layer's
# variable simply never matches one; no separate skip logic is needed).
# ---------------------------------------------------------------------------

CUSTOM_LAYER_VALUE_TYPES = frozenset({"ratio", "interval", "nominal", "ordinal"})


def parse_custom_layer_metadata(raw_json: str | None) -> list[dict]:
    """Parses the frontend's JSON description of each custom layer it
    already sampled client-side into layer_meta-compatible rows. Raises
    HTTPException(422) on any malformed/invalid entry -- this drives
    request-time validation in main.py, not a background job failure.
    """
    if not raw_json:
        return []
    try:
        entries = json.loads(raw_json)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid custom_layer_metadata JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise HTTPException(status_code=422, detail="custom_layer_metadata must be a JSON array.")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise HTTPException(status_code=422, detail=f"custom_layer_metadata[{i}] must be an object.")
        layer_id = str(entry.get("id") or "").strip()
        if not layer_id:
            raise HTTPException(status_code=422, detail=f"custom_layer_metadata[{i}] is missing 'id'.")
        if layer_id in seen_ids:
            raise HTTPException(status_code=422, detail=f"Duplicate custom layer id: {layer_id!r}.")
        seen_ids.add(layer_id)

        value_type = str(entry.get("valueType") or entry.get("value_type") or "").strip()
        if value_type not in CUSTOM_LAYER_VALUE_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"custom_layer_metadata[{i}] has an unsupported valueType: {value_type!r}.",
            )

        legend_classes_raw = entry.get("legendClasses") or entry.get("legend_classes")
        legend_json: str | None = None
        render_min: float | None = None
        render_max: float | None = None
        if value_type in ("nominal", "ordinal") and legend_classes_raw:
            try:
                class_ids = [int(cls["id"]) for cls in legend_classes_raw]
                legend_json = json.dumps([
                    {
                        "id": int(cls["id"]),
                        "name": str(cls.get("name", cls["id"])),
                        "color": cls.get("color"),
                    }
                    for cls in legend_classes_raw
                ])
            except (TypeError, ValueError, KeyError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"custom_layer_metadata[{i}] has invalid legendClasses: {exc}",
                ) from exc
            if class_ids:
                # Ordinal coloring is a gradient keyed to (classId - render_min)
                # / (render_max - render_min) -- it never uses a class's own
                # color at the pixel/marker level (that's nominal's job; see
                # cogTileRenderer.ts's colorsById, only populated when
                # isNominal). Every built-in ordinal layer in catalog.json
                # sets render_min/render_max to its class-id range (e.g.
                # salinity: 0/4 for classes 0..4), not a data statistic --
                # mirror that exactly so a custom ordinal layer's gradient
                # spans its own classes instead of collapsing to a single
                # color with the range left at None.
                render_min = float(min(class_ids))
                render_max = float(max(class_ids))

        rows.append({
            "id":            layer_id,
            "name":          str(entry.get("name") or layer_id).strip(),
            "units":         entry.get("units") or None,
            "imperial_unit": None,
            "value_type":    value_type,
            "domain":        "discrete" if value_type in ("nominal", "ordinal") else "continuous",
            "category":      "Custom Layers",
            "group":         None,
            "group_label":   None,
            "sort_order":    20000 + i,  # after static + temporal layer entries
            "render_min":    render_min,
            "render_max":    render_max,
            "legend_classes": legend_json,
            "_legend_key":   layer_id,
        })
    return rows


def _add_ternary_classification_overlay(work_dir: Path, layer_meta: dict[str, dict]) -> None:
    """Augment density_grid.parquet with each compositional group's classification
    overlay (class ids + boundary lines), when a classifier is registered for it.

    Classification is static per classifier — identical for every taxon/dataset,
    since it doesn't depend on occurrence data at all (see
    util.ternary.build_ternary_classification_overlay) — so it's cheap to compute
    once here and ship it in the archive, rather than requiring the client to
    port the classification rules (e.g. USDA soil texture) and boundary-bisection
    logic itself.
    """
    path = work_dir / DENSITY_GRID_FILE
    if not path.exists():
        return
    rows = pq.read_table(path).to_pylist()
    group_members = composition_group_members(layer_meta)
    changed = False
    for row in rows:
        classifier = COMPOSITION_CLASSIFIERS.get(row.get("variable"))
        axis_columns = tuple(group_members.get(row.get("variable"), ()))
        if classifier is None or len(axis_columns) != 3:
            continue
        overlay = build_ternary_classification_overlay(row["resolution"], classifier, axis_columns)
        row["class_ids"] = overlay["class_ids"]
        row["class_boundary_a"] = overlay["boundary_a"]
        row["class_boundary_b"] = overlay["boundary_b"]
        changed = True
    if changed:
        pq.write_table(pa.Table.from_pylist(rows), path)


# ---------------------------------------------------------------------------
# Archive building
# ---------------------------------------------------------------------------

def _package_archive(
    work_dir: Path,
    df: pd.DataFrame,
    layer_meta: dict[str, dict],
    archive_name: str,
    include_csv: bool = True,
) -> Path:
    """Write occurrence.parquet + categorical_value_lookup/variable_metadata/
    locations, then zip them alongside whatever stats files the caller has
    already written into ``work_dir`` (numerical/nominal/ordinal/circular
    stats, density, density_grid, relative-rank positions) into one archive.

    Shared by the upload path (stats computed fresh via
    process_observations_df) and the taxon-download path (stats copied from
    the tree's precomputed GLOBAL_STATS_DIR) — this function only cares that
    the stats files already exist in work_dir under their standard names,
    not how they got there. Relative-rank positions are download-only (a
    custom upload has no tree ancestors to rank against), so
    ``work_dir / POSITION_FILE`` simply won't exist on that path and this
    entry is skipped, same as every other optional file here.

    ``include_csv`` also zips a .csv alongside every .parquet member — cheap
    for upload-sized data, but pandas' to_csv() on a real taxon's full
    occurrence table (hundreds of thousands of rows) costs real seconds for
    a duplicate export nobody asked for, so the download path disables it.
    """
    occ_path = work_dir / "occurrence.parquet"
    df.to_parquet(occ_path, index=False)

    lookup_rows: list[dict] = []
    for col in df.columns:
        layer = layer_meta.get(col)
        if not layer or layer.get("value_type") not in ("nominal", "ordinal"):
            continue
        # A purely synthetic layer (temporal, or a custom layer sampled
        # client-side -- see parse_custom_layer_metadata) has no on-disk
        # legend file for _load_legend() to find; its classes travel as an
        # inline JSON string on the layer dict instead, same convention
        # variable_metadata.parquet's own writer below already uses.
        if layer.get("legend_classes"):
            classes = json.loads(layer["legend_classes"])
        else:
            legend_id = layer.get("_legend_key", col)
            classes = _load_legend(legend_id)
        for cls in classes:
            lookup_rows.append({
                "variable": col,
                "code": str(cls["id"]),
                "metric": f"class_{cls['id']}",
                "label": cls.get("name", str(cls["id"])),
                "group": cls.get("group", ""),
                "groupLabel": cls.get("group_label", ""),
            })
    lookup_path = work_dir / "categorical_value_lookup.parquet"
    if lookup_rows:
        pq.write_table(pa.Table.from_pylist(lookup_rows), lookup_path)

    meta_rows = []
    for idx, layer in enumerate(layer_meta.values()):
        # Temporal rows already carry pre-built legend_classes and _legend_key.
        # Static rows need to load the legend by base layer id.
        if "legend_classes" in layer and layer["legend_classes"] is not None:
            legend_json = layer["legend_classes"]
        else:
            legend_id = layer.get("_legend_key", layer["id"])
            raw_classes = _load_legend(legend_id)
            legend_json = None
            if raw_classes:
                legend_json = json.dumps([
                    {
                        "id": cls["id"],
                        "name": cls.get("name", str(cls["id"])),
                        "color": cls.get("traits", {}).get("color") or None,
                    }
                    for cls in raw_classes
                ])
        meta_rows.append({
            "id":            layer["id"],
            "name":          layer.get("name") or layer.get("display_name") or layer["id"],
            "units":         layer.get("units") or None,
            "imperial_unit": layer.get("imperial_unit") or None,
            "value_type":    layer.get("value_type") or None,
            "domain":        layer.get("domain") or None,
            "category":      layer.get("category") or layer.get("category_display_name") or None,
            "group":         layer.get("group") or None,
            "group_label":   layer.get("group_label") or None,
            "sort_order":    idx,
            "render_min":    layer.get("render_min"),
            "render_max":    layer.get("render_max"),
            "legend_classes": legend_json,
            "composition_group": layer.get("composition_group") or None,
            "composition_axis":  layer.get("composition_axis") or None,
            "composition_label": layer.get("composition_label") or None,
        })
    meta_path = work_dir / "variable_metadata.parquet"
    if meta_rows:
        pq.write_table(pa.Table.from_pylist(meta_rows), meta_path)

    locations_path = work_dir / "locations.parquet"
    locations_table = build_locations_table(df)
    if locations_table is not None:
        pq.write_table(locations_table, locations_path)

    archive_path = work_dir / archive_name
    files_to_zip = [
        (occ_path,                              "occurrence.parquet"),
        (work_dir / NUMERICAL_STATS_FILE,       NUMERICAL_STATS_FILE),
        (work_dir / NOMINAL_STATS_FILE,         NOMINAL_STATS_FILE),
        (work_dir / ORDINAL_STATS_FILE,         ORDINAL_STATS_FILE),
        (work_dir / CIRCULAR_STATS_FILE,        CIRCULAR_STATS_FILE),
        (work_dir / DENSITY_FILE,               DENSITY_FILE),
        (work_dir / DENSITY_GRID_FILE,          DENSITY_GRID_FILE),
        (work_dir / POSITION_FILE,              POSITION_FILE),
        (lookup_path,                           "categorical_value_lookup.parquet"),
        (meta_path,                             "variable_metadata.parquet"),
        (locations_path,                        "locations.parquet"),
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for parquet_path, arcname in files_to_zip:
            if not parquet_path.exists():
                continue
            try:
                table = pq.read_table(parquet_path)
                buf = io.BytesIO()
                pq.write_table(table, buf, compression="snappy")
                zf.writestr(arcname, buf.getvalue())
                if include_csv:
                    try:
                        csv_bytes = table.to_pandas().to_csv(index=False).encode()
                        zf.writestr(arcname.replace(".parquet", ".csv"), csv_bytes)
                    except Exception:
                        pass
            except Exception:
                zf.write(parquet_path, arcname=arcname)

    return archive_path


def _add_metadata_to_archive(
    archive_path: Path,
    *,
    description_profile: dict | None = None,
    image_bytes: bytes | None = None,
    image_filename: str | None = None,
    image_url: str | None = None,
    image_license: str | None = None,
    image_license_url: str | None = None,
    image_creator: str | None = None,
    image_rights_holder: str | None = None,
    parent_taxon_id: str | None = None,
) -> None:
    """Appends upload_metadata.json (plus an embedded image, if any bytes
    were given) to an already-built archive -- a separate pass from
    _package_archive above rather than a parameter on it, since this is the
    one part of the archive that's optional and per-caller (a plain re-upload
    has none of it; a species download always has description_profile +
    image_url; a custom upload has whichever of these the user opted into).
    A no-op if the caller has nothing to add, so plain re-uploads' archives
    are byte-for-byte what they always were.

    imageFile (when present) names the zip member the actual image bytes are
    stored under, for an uploaded image -- works fully offline once
    downloaded. imageUrl alone (no imageFile) is a plain remote URL --
    convenient (no re-upload needed for e.g. a species' existing photo, or a
    custom upload pointing at one already hosted somewhere) but requires
    network access to actually display.
    """
    metadata: dict = {}
    if description_profile is not None:
        metadata["descriptionProfile"] = description_profile
    if image_bytes:
        ext = Path(image_filename or "").suffix or ".jpg"
        metadata["imageFile"] = f"taxon_image{ext}"
    if image_url:
        metadata["imageUrl"] = image_url
    if image_license:
        metadata["imageLicense"] = image_license
    if image_license_url:
        metadata["imageLicenseUrl"] = image_license_url
    if image_creator:
        metadata["imageCreator"] = image_creator
    if image_rights_holder:
        metadata["imageRightsHolder"] = image_rights_holder
    # Recorded so re-importing this ZIP and enriching it further can rank
    # against the same parent again -- the ranking itself only survives as
    # positions, never as the taxon that produced them.
    if parent_taxon_id:
        metadata["parentTaxonId"] = parent_taxon_id
    if not metadata:
        return
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("upload_metadata.json", json.dumps(metadata))
        if image_bytes:
            zf.writestr(metadata["imageFile"], image_bytes)


def _parent_taxon_filename_suffix(parent_taxon_id: str) -> str | None:
    """Same slug-taxon_key naming convention util.download._archive_filename
    uses for a species download's own filename, applied here to the user-
    selected parent taxon so a processed ZIP ranked against one is
    identifiable from its filename alone. None if the id doesn't resolve
    (build_archive already tolerates an unresolvable parent_taxon_id
    elsewhere -- see compute_relative_ranks_for_upload)."""
    taxon = get_taxon_by_id(parent_taxon_id)
    if taxon is None:
        return None
    name = taxon.get("scientific_name") or taxon.get("common_name") or str(taxon["taxon_key"])
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "taxon"
    return f"{slug}-{taxon['taxon_key']}"


def build_archive(
    df: pd.DataFrame,
    *,
    generate_description: bool = False,
    image_bytes: bytes | None = None,
    image_filename: str | None = None,
    image_url: str | None = None,
    parent_taxon_id: str | None = None,
    custom_layer_metadata: list[dict] | None = None,
) -> tuple[Path, str, Path]:
    """Compute stats and package all outputs into a ZIP archive.

    Returns ``(archive_path, archive_filename, work_dir)``. The caller is
    responsible for deleting ``work_dir`` after the response has been sent.
    """
    layer_meta = _build_layer_meta()
    for row in _build_temporal_var_meta(df):
        layer_meta[row["id"]] = row
    for row in (custom_layer_metadata or []):
        layer_meta[row["id"]] = row

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-upload-"))
    archive_name = "processed_observations.zip"
    if parent_taxon_id:
        suffix = _parent_taxon_filename_suffix(parent_taxon_id)
        if suffix:
            archive_name = f"processed_observations-{suffix}.zip"
    try:
        filtered = _filter_df(df.copy())
        process_observations_df(work_dir, filtered, layer_meta)
        _add_ternary_classification_overlay(work_dir, layer_meta)
        # Must run before _package_archive -- it checks work_dir / POSITION_FILE
        # for existence at zip time, same as every other stats file.
        if parent_taxon_id:
            compute_relative_ranks_for_upload(work_dir, layer_meta, parent_taxon_id)
        archive_path = _package_archive(work_dir, df, layer_meta, archive_name)
        description_profile = (
            build_description_profile_for_df(work_dir, df) if generate_description else None
        )
        _add_metadata_to_archive(
            archive_path,
            description_profile=description_profile,
            image_bytes=image_bytes,
            image_filename=image_filename,
            image_url=image_url,
            parent_taxon_id=parent_taxon_id,
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to build archive: {exc}") from exc

    return archive_path, archive_name, work_dir
