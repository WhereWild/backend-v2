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

from config.config import ZERO_NODATA_LAYERS
from util.gis import DERIVED_FROM_ELEVATION, hilbert_index, sample_aspect_batch, sample_elevation_terrain_batch, sample_slope_batch
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    _filter_df,
    process_observations_df,
)
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
from util.tiles import LAYERS_DIR, load_layers_with_category

_LEGEND_DIR = Path("config/gis/legends")
_GADM_PATH = Path("data/gis/gadm.gpkg")
_HIERARCHY_PATH = Path("data/gis/locations/hierarchy.csv")
_CATALOG_PATH = Path("config/gis/catalog.json")

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
        df.loc[missing, "observationName"] = fallback[missing]
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
        if (layer.get("filename") or layer["id"] in DERIVED_FROM_ELEVATION)
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

    elev_path = LAYERS_DIR / "elevation.tif"

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
            else:
                cog_path = LAYERS_DIR / layer["filename"]
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
# Archive building
# ---------------------------------------------------------------------------

def build_archive(df: pd.DataFrame) -> tuple[Path, str, Path]:
    """Compute stats and package all outputs into a ZIP archive.

    Returns ``(archive_path, archive_filename, work_dir)``. The caller is
    responsible for deleting ``work_dir`` after the response has been sent.
    """
    layer_meta = _build_layer_meta()
    for row in _build_temporal_var_meta(df):
        layer_meta[row["id"]] = row

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-upload-"))
    try:
        filtered = _filter_df(df.copy())
        process_observations_df(work_dir, filtered, layer_meta)

        occ_path = work_dir / "occurrence.parquet"
        df.to_parquet(occ_path, index=False)

        lookup_rows: list[dict] = []
        for col in df.columns:
            layer = layer_meta.get(col)
            if not layer or layer.get("value_type") not in ("nominal", "ordinal"):
                continue
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
            })
        meta_path = work_dir / "variable_metadata.parquet"
        if meta_rows:
            pq.write_table(pa.Table.from_pylist(meta_rows), meta_path)

        locations_path = work_dir / "locations.parquet"
        locations_table = build_locations_table(df)
        if locations_table is not None:
            pq.write_table(locations_table, locations_path)

        archive_name = "processed_observations.zip"
        archive_path = work_dir / archive_name
        files_to_zip = [
            (occ_path,                              "occurrence.parquet"),
            (work_dir / NUMERICAL_STATS_FILE,       NUMERICAL_STATS_FILE),
            (work_dir / NOMINAL_STATS_FILE,         NOMINAL_STATS_FILE),
            (work_dir / ORDINAL_STATS_FILE,         ORDINAL_STATS_FILE),
            (work_dir / CIRCULAR_STATS_FILE,        CIRCULAR_STATS_FILE),
            (work_dir / DENSITY_FILE,               DENSITY_FILE),
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
                    try:
                        csv_bytes = table.to_pandas().to_csv(index=False).encode()
                        zf.writestr(arcname.replace(".parquet", ".csv"), csv_bytes)
                    except Exception:
                        pass
                except Exception:
                    zf.write(parquet_path, arcname=arcname)

    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to build archive: {exc}") from exc

    return archive_path, archive_name, work_dir
