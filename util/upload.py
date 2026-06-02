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
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from fastapi import HTTPException

from util.gis import DERIVED_FROM_ELEVATION, hilbert_index, sample_aspect_batch, sample_slope_batch
from util.indexing import OCCURRENCE_INDEX_FILE
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    _filter_df,
    process_observations_df,
)
from util.tiles import LAYERS_DIR, load_layers_with_category

_LEGEND_DIR = Path("config/gis/legends")

# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

_LAT_ALIASES  = ("decimalLatitude",  "decimal_latitude",  "latitude",  "lat")
_LON_ALIASES  = ("decimalLongitude", "decimal_longitude", "longitude", "lon", "lng")
_CATALOG_ALIASES = (
    "catalogNumber", "catalog_number",
    "occurrenceID",  "occurrence_id",
    "observationID", "observation_id",
    "recordID",      "record_id",
    "gbifID",        "gbif_id",
)
_NAME_ALIASES = ("observationName", "observation_name", "name", "title", "label")


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
        return df
    df = df.copy()
    col = _find_column(list(df.columns), _CATALOG_ALIASES)
    if col:
        return df.rename(columns={col: "catalogNumber"})
    df["catalogNumber"] = [f"Observation #{i}" for i in range(1, len(df) + 1)]
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

def _sample_layer(
    path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    scale: float,
    offset: float,
    nodata: float | None,
) -> list[float | None]:
    coords = list(zip(lons.tolist(), lats.tolist()))
    with rasterio.open(path) as ds:
        nd = ds.nodata if nodata is None else nodata
        results: list[float | None] = []
        for point in ds.sample(coords):
            v = float(point[0])
            results.append(None if (nd is not None and v == nd) else v * scale + offset)
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
                sorted_vals = _sample_layer(cog_path, s_lats, s_lons, scale, offset, nodata=None)
                result[layer_id] = [sorted_vals[i] for i in restore]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Archive building
# ---------------------------------------------------------------------------

def build_archive(df: pd.DataFrame) -> tuple[Path, str, Path]:
    """Compute stats and package all outputs into a ZIP archive.

    Returns ``(archive_path, archive_filename, work_dir)``. The caller is
    responsible for deleting ``work_dir`` after the response has been sent.
    """
    layer_meta = _build_layer_meta()

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-upload-"))
    try:
        filtered = _filter_df(df.copy())
        process_observations_df(work_dir, filtered, layer_meta)

        occ_path = work_dir / "occurrence.parquet"
        df.to_parquet(occ_path, index=False)

        # occurrence_index: wide table (catalogNumber + GIS columns) for frontend
        # variable filtering. The frontend parser explodes this into per-variable
        # index rows — one entry per observation per variable.
        gis_cols = [col for col in df.columns if col in layer_meta]
        if gis_cols:
            occ_index_df = df[["catalogNumber"] + gis_cols]
            pq.write_table(
                pa.Table.from_pandas(occ_index_df, preserve_index=False),
                work_dir / OCCURRENCE_INDEX_FILE,
            )

        lookup_rows: list[dict] = []
        for col in df.columns:
            layer = layer_meta.get(col)
            if not layer or layer.get("value_type") != "nominal":
                continue
            classes = _load_legend(col)
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
            raw_classes = _load_legend(layer["id"])
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
                "id": layer["id"],
                "name": layer.get("display_name") or layer["id"],
                "units": layer.get("units") or None,
                "value_type": layer.get("value_type") or None,
                "domain": layer.get("domain") or None,
                "category": layer.get("category_display_name") or None,
                "group": layer.get("group") or None,
                "group_label": layer.get("group_label") or None,
                "sort_order": idx,
                "render_min": layer.get("render_min"),
                "render_max": layer.get("render_max"),
                "legend_classes": legend_json,
            })
        meta_path = work_dir / "variable_metadata.parquet"
        if meta_rows:
            pq.write_table(pa.Table.from_pylist(meta_rows), meta_path)

        archive_name = "processed_observations.zip"
        archive_path = work_dir / archive_name
        files_to_zip = [
            (occ_path,                              "occurrence.parquet"),
            (work_dir / NUMERICAL_STATS_FILE,       NUMERICAL_STATS_FILE),
            (work_dir / NOMINAL_STATS_FILE,         NOMINAL_STATS_FILE),
            (work_dir / CIRCULAR_STATS_FILE,        CIRCULAR_STATS_FILE),
            (work_dir / DENSITY_FILE,               DENSITY_FILE),
            (work_dir / OCCURRENCE_INDEX_FILE,      OCCURRENCE_INDEX_FILE),
            (lookup_path,                           "categorical_value_lookup.parquet"),
            (meta_path,                             "variable_metadata.parquet"),
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
