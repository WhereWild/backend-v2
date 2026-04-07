from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from rasterio.windows import Window
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from util.config import load_config
import util.gis_lookup as gis_lookup
from util.storage import get_parquet_storage_with_mode
import util.taxa_navigation as taxa_navigation


CONFIG = load_config("global")
DEM_FILENAME = "dem.tif"
DERIVED_DEM_LAYER_METRICS = {
    "slope": "slope",
    "aspect": "aspect",
    "aspect_deg": "aspect_deg",
}
LANDCOVER_WATER_CLASS_ID = 210
SEA_LEVEL_FILTER_ABS_ELEVATION_METERS = 10.0
NEGATIVE_BATCH_MAX = 250_000
NEGATIVE_WINDOW_RETRIES = 6
NEGATIVE_AUTO_GLOBAL_MAX_STEPS = 16
EVAL_THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
EVAL_QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


@dataclass(frozen=True)
class FeatureSpec:
    all_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True)
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return (lats >= self.min_lat) & (lats <= self.max_lat) & (lons >= self.min_lon) & (lons <= self.max_lon)

    def expanded(self, factor: float) -> BoundingBox:
        lat_span = max(self.max_lat - self.min_lat, CONFIG.ml_negative_min_bbox_span_degrees)
        lon_span = max(self.max_lon - self.min_lon, CONFIG.ml_negative_min_bbox_span_degrees)
        center_lat = (self.min_lat + self.max_lat) * 0.5
        center_lon = (self.min_lon + self.max_lon) * 0.5
        half_lat = (lat_span * factor) * 0.5 + CONFIG.ml_negative_base_padding_degrees
        half_lon = (lon_span * factor) * 0.5 + CONFIG.ml_negative_base_padding_degrees
        return BoundingBox(
            min_lat=max(-89.999, center_lat - half_lat),
            max_lat=min(89.999, center_lat + half_lat),
            min_lon=max(-179.999, center_lon - half_lon),
            max_lon=min(179.999, center_lon + half_lon),
        )

    def approx_equals(self, other: BoundingBox, *, tol: float = 1e-6) -> bool:
        return (
            abs(self.min_lat - other.min_lat) <= tol
            and abs(self.max_lat - other.max_lat) <= tol
            and abs(self.min_lon - other.min_lon) <= tol
            and abs(self.max_lon - other.max_lon) <= tol
        )

    def is_global(self, *, tol: float = 1e-6) -> bool:
        return (
            abs(self.min_lat - (-89.999)) <= tol
            and abs(self.max_lat - 89.999) <= tol
            and abs(self.min_lon - (-179.999)) <= tol
            and abs(self.max_lon - 179.999) <= tol
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_lat": float(self.min_lat),
            "max_lat": float(self.max_lat),
            "min_lon": float(self.min_lon),
            "max_lon": float(self.max_lon),
        }


def _configure_storage_modes() -> None:
    parquet_mode = str(CONFIG.ml_parquet_storage_mode or "").strip().lower()
    if parquet_mode not in {"local", "b2"}:
        raise ValueError(
            f"Unsupported ml_parquet_storage_mode={CONFIG.ml_parquet_storage_mode!r}; expected 'local' or 'b2'."
        )
    os.environ["WHEREWILD_PARQUET_STORAGE"] = parquet_mode

    raster_mode = str(CONFIG.ml_raster_storage_mode or "").strip().lower()
    if raster_mode not in {"auto", "local", "b2"}:
        raise ValueError(
            f"Unsupported ml_raster_storage_mode={CONFIG.ml_raster_storage_mode!r}; expected 'auto', 'local', or 'b2'."
        )
    os.environ["WHEREWILD_RASTER_STORAGE"] = raster_mode


def _parquet_storage():
    return get_parquet_storage_with_mode(
        CONFIG.data_root,
        CONFIG.project_root,
        str(CONFIG.ml_parquet_storage_mode),
    )


def _temporal_feature_names() -> list[str]:
    """Return all temporal column names derivable from config (e.g. temperature_2m_avg_1h)."""
    return gis_lookup.temporal_feature_names_from_config(CONFIG)


def _load_non_temporal_feature_spec() -> FeatureSpec:
    with _parquet_storage().open_input_file(CONFIG.gis_catalog_path) as handle:
        catalog = json.loads(handle.read())

    all_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for category in catalog.get("categories", []):
        if str(category.get("name") or "").strip().lower() == "temporal":
            continue
        for layer in category.get("layers", []):
            layer_id = str(layer.get("id") or "").strip()
            if not layer_id:
                continue
            value_type = str(layer.get("value_type") or "").strip().lower()
            all_columns.append(layer_id)
            if value_type == "categorical":
                categorical_columns.append(layer_id)
            else:
                numeric_columns.append(layer_id)

    if not all_columns:
        raise RuntimeError("No non-temporal GIS feature columns found in catalog.")

    return FeatureSpec(
        all_columns=all_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _read_subtree_occurrence_df(taxon_id: str, columns: list[str]) -> pd.DataFrame:
    """Concatenate occurrence parquets for a taxon and all its descendants."""
    taxon = taxa_navigation.get_taxon_by_id(taxon_id)
    if taxon is None:
        raise ValueError(f"Taxon not found in catalog: {taxon_id}")
    storage = _parquet_storage()
    frames: list[pd.DataFrame] = []
    for desc in taxa_navigation.iter_descendants(taxon, include_self=True):
        path = Path(desc["path"]) / CONFIG.occurrence_parquet_filename
        if not storage.exists(path):
            continue
        try:
            pf = storage.parquet_file(path)
            quality_cols = ["obscured", "coordinateUncertaintyInMeters"]
            available = [c for c in columns if c in pf.schema.names]
            if "decimalLatitude" not in available or "decimalLongitude" not in available:
                continue
            read_cols = list(dict.fromkeys(available + [c for c in quality_cols if c in pf.schema.names]))
            table = pf.read(columns=read_cols)
            table = table.filter(taxa_navigation.base_observation_mask(table))
            extra = [c for c in quality_cols if c not in available]
            if extra:
                table = table.drop([c for c in extra if c in table.column_names])
            if table.num_rows:
                frames.append(table.to_pandas())
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No occurrence parquets found for taxon {taxon_id} or its descendants.")
    df = pd.concat(frames, ignore_index=True)
    cap = int(CONFIG.ml_max_positives)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=int(CONFIG.ml_random_seed)).reset_index(drop=True)
        print(f"[train] subtree rows: {len(df)} (capped from larger set) across {len(frames)} parquet(s)")
    else:
        print(f"[train] subtree rows: {len(df)} across {len(frames)} parquet(s)")
    return df


def _read_positive_features_and_coords(
    taxon_id: str,
    feature_spec: FeatureSpec,
    *,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    all_wanted = list(feature_spec.all_columns)
    if extra_columns:
        all_wanted = list(dict.fromkeys(all_wanted + extra_columns))
    required_columns = list(dict.fromkeys(all_wanted + ["decimalLatitude", "decimalLongitude"]))

    frame = _read_subtree_occurrence_df(taxon_id, required_columns)

    for col in all_wanted:
        if col not in frame.columns:
            frame[col] = np.nan

    lats = pd.to_numeric(frame["decimalLatitude"], errors="coerce").to_numpy(dtype=np.float64)
    lons = pd.to_numeric(frame["decimalLongitude"], errors="coerce").to_numpy(dtype=np.float64)
    features = frame[all_wanted].copy()

    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    features = features.loc[valid_mask].reset_index(drop=True)
    lats = lats[valid_mask]
    lons = lons[valid_mask]

    if features.empty:
        raise RuntimeError("No positive rows had valid coordinates.")

    return features, lats, lons


def _read_phenology_split(
    taxon_id: str,
    feature_spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split occurrence parquet into reproductively-active (positive) and unannotated (negative).

    Positive = rcs column contains a known reproductive-activity value (flowers, buds, fruits).
    Negative = rcs is empty/null (unannotated occurrence).
    Rows with explicitly negative annotations (e.g. "no flowers or fruits") are excluded.
    """
    rcs_column = CONFIG.ml_phenology_rcs_column
    positive_values = set(CONFIG.ml_phenology_rcs_positive_values)

    temporal_cols = _temporal_feature_names()
    all_wanted = list(
        dict.fromkeys(feature_spec.all_columns + temporal_cols + [rcs_column, "decimalLatitude", "decimalLongitude"])
    )

    frame = _read_subtree_occurrence_df(taxon_id, all_wanted)

    # Fill missing feature columns with NaN
    for col in feature_spec.all_columns + temporal_cols:
        if col not in frame.columns:
            frame[col] = np.nan

    lats = pd.to_numeric(frame["decimalLatitude"], errors="coerce")
    lons = pd.to_numeric(frame["decimalLongitude"], errors="coerce")
    valid_coords = np.isfinite(lats.to_numpy()) & np.isfinite(lons.to_numpy())
    frame = frame.loc[valid_coords].reset_index(drop=True)

    feature_cols = list(dict.fromkeys(feature_spec.all_columns + temporal_cols))

    if rcs_column in frame.columns:
        # rcs is stored as a |-joined list; check if any token is a positive value
        def _has_positive_token(cell: object) -> bool:
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                return False
            return bool(positive_values.intersection(str(cell).strip().split("|")))

        has_positive = frame[rcs_column].map(_has_positive_token)
        # Negatives: empty/null only — exclude explicitly negative annotations
        is_negative = frame[rcs_column].isna() | (frame[rcs_column].astype(str).str.strip() == "")
    else:
        has_positive = pd.Series(False, index=frame.index)
        is_negative = pd.Series(True, index=frame.index)

    positives = frame.loc[has_positive, feature_cols].reset_index(drop=True)
    negatives = frame.loc[is_negative, feature_cols].reset_index(drop=True)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in positives.columns:
            positives[col] = np.nan
        if col not in negatives.columns:
            negatives[col] = np.nan

    return positives[feature_cols], negatives[feature_cols]


def _compute_positive_bbox(lats: np.ndarray, lons: np.ndarray) -> BoundingBox:
    finite_mask = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(finite_mask):
        raise RuntimeError("No valid coordinates available for bbox construction.")

    lat_vals = lats[finite_mask]
    lon_vals = lons[finite_mask]
    min_lat = float(np.min(lat_vals))
    max_lat = float(np.max(lat_vals))
    min_lon = float(np.min(lon_vals))
    max_lon = float(np.max(lon_vals))

    if max_lat - min_lat < CONFIG.ml_negative_min_bbox_span_degrees:
        center = (min_lat + max_lat) * 0.5
        half = CONFIG.ml_negative_min_bbox_span_degrees * 0.5
        min_lat = max(-89.999, center - half)
        max_lat = min(89.999, center + half)

    if max_lon - min_lon < CONFIG.ml_negative_min_bbox_span_degrees:
        center = (min_lon + max_lon) * 0.5
        half = CONFIG.ml_negative_min_bbox_span_degrees * 0.5
        min_lon = max(-179.999, center - half)
        max_lon = min(179.999, center + half)

    return BoundingBox(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )


def _is_nodata_value(value: float, nodata: float | None) -> bool:
    if nodata is None:
        return False
    try:
        if np.isnan(nodata):
            return bool(np.isnan(value))
    except TypeError:
        return False
    return value == nodata


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = np.deg2rad(lat_deg)
    m_per_deg_lat = (
        111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
    return float(m_per_deg_lat), float(m_per_deg_lon)


def _compute_slope_aspect(window: np.ndarray, dx_m: float, dy_m: float) -> tuple[float, float]:
    z1, z2, z3 = window[0, 0], window[0, 1], window[0, 2]
    z4, _, z6 = window[1, 0], window[1, 1], window[1, 2]
    z7, z8, z9 = window[2, 0], window[2, 1], window[2, 2]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * dx_m)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * dy_m)
    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = float(np.degrees(slope_rad))

    if dzdx == 0 and dzdy == 0:
        return slope_deg, 0.0

    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect_deg = float(90.0 - aspect)
    if aspect_deg < 0:
        aspect_deg += 360.0
    return slope_deg, aspect_deg


def _aspect_bin(aspect_deg: float) -> int:
    aspect_deg = aspect_deg % 360.0
    if aspect_deg < 22.5 or aspect_deg >= 337.5:
        return 1
    if aspect_deg < 67.5:
        return 2
    if aspect_deg < 112.5:
        return 3
    if aspect_deg < 157.5:
        return 4
    if aspect_deg < 202.5:
        return 5
    if aspect_deg < 247.5:
        return 6
    if aspect_deg < 292.5:
        return 7
    return 8


def _sample_regular_layer_values_for_tile(
    layer_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    values = np.full(len(lats), np.nan, dtype=np.float32)
    if len(lats) == 0:
        return values

    layer_meta = gis_lookup.load_layer_metadata().get(layer_id)
    if layer_meta is not None and "region_root" not in layer_meta:
        return values

    source = gis_lookup.get_cog_source(layer_id, float(lats[0]), float(lons[0]))
    if source is None:
        return values

    coords = list(zip(lons.tolist(), lats.tolist()))
    with gis_lookup.open_raster(source) as ds:
        for idx, point in enumerate(ds.sample(coords)):
            value = point[0]
            if _is_nodata_value(value, ds.nodata):
                if layer_id == "swe":
                    values[idx] = np.float32(0.0)
                continue
            if np.isfinite(value):
                values[idx] = np.float32(value)
    return values


def _sample_dem_values_for_tile(
    lats: np.ndarray,
    lons: np.ndarray,
    tile_id: str,
) -> np.ndarray:
    values = np.full(len(lats), np.nan, dtype=np.float32)
    dem_path = CONFIG.gis_regions_root / tile_id / DEM_FILENAME
    source = gis_lookup.resolve_raster_source(dem_path)
    if source is None:
        return values

    coords = list(zip(lons.tolist(), lats.tolist()))
    with gis_lookup.open_raster(source) as ds:
        for idx, point in enumerate(ds.sample(coords)):
            value = point[0]
            if _is_nodata_value(value, ds.nodata):
                continue
            if np.isfinite(value):
                values[idx] = np.float32(value)
    return values


def _sample_dem_derived_values_for_tile(
    lats: np.ndarray,
    lons: np.ndarray,
    tile_id: str,
    metrics: tuple[str, ...],
) -> dict[str, np.ndarray]:
    results = {metric: np.full(len(lats), np.nan, dtype=np.float32) for metric in metrics}
    dem_path = CONFIG.gis_regions_root / tile_id / DEM_FILENAME
    source = gis_lookup.resolve_raster_source(dem_path)
    if source is None:
        return results

    with gis_lookup.open_raster(source) as ds:
        nodata = ds.nodata
        pixel_width_deg = float(ds.transform.a)
        pixel_height_deg = abs(float(ds.transform.e))

        for idx, (lat, lon) in enumerate(zip(lats.tolist(), lons.tolist())):
            try:
                row, col = ds.index(lon, lat)
            except Exception:
                continue

            window_size = 3
            radius = window_size // 2
            if row - radius < 0 or col - radius < 0 or row + radius >= ds.height or col + radius >= ds.width:
                continue

            window = ds.read(
                1,
                window=Window(col - radius, row - radius, window_size, window_size),
                boundless=False,
            )
            if window.shape != (window_size, window_size):
                continue
            if nodata is not None and np.any(window == nodata):
                continue
            if np.any(np.isnan(window)):
                continue

            m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat)
            dx_m = pixel_width_deg * m_per_deg_lon
            dy_m = pixel_height_deg * m_per_deg_lat
            if dx_m == 0 or dy_m == 0:
                continue

            slope_deg, aspect_deg = _compute_slope_aspect(window, dx_m, dy_m)
            for metric in metrics:
                if metric == "slope":
                    results[metric][idx] = np.float32(slope_deg)
                elif metric == "aspect_deg":
                    results[metric][idx] = np.float32(aspect_deg)
                elif metric == "aspect":
                    results[metric][idx] = np.float32(_aspect_bin(aspect_deg))

    return results


def _sample_feature_matrix_for_coordinates(
    lats: np.ndarray,
    lons: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    values = {layer_id: np.full(len(lats), np.nan, dtype=np.float32) for layer_id in feature_columns}
    if len(lats) == 0:
        return pd.DataFrame(values)

    finite_mask = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(finite_mask):
        return pd.DataFrame(values)

    worklist = pd.DataFrame(
        {
            "row_id": np.arange(len(lats), dtype=np.int64),
            "decimalLatitude": lats,
            "decimalLongitude": lons,
        }
    )
    worklist = worklist.loc[finite_mask].copy()
    worklist["tileId"] = [
        gis_lookup.get_region_name(float(lat), float(lon))
        for lat, lon in zip(
            worklist["decimalLatitude"].to_numpy(dtype=float),
            worklist["decimalLongitude"].to_numpy(dtype=float),
        )
    ]
    worklist.sort_values("tileId", inplace=True)
    total_tiles = int(worklist["tileId"].nunique(dropna=True))
    print(
        f"[negatives-features] sampling {len(worklist)} coordinates across {total_tiles} tiles "
        f"and {len(feature_columns)} feature columns"
    )

    needs_elevation = "elevation" in feature_columns
    derived_layers = {
        layer_id: DERIVED_DEM_LAYER_METRICS[layer_id]
        for layer_id in feature_columns
        if layer_id in DERIVED_DEM_LAYER_METRICS
    }
    regular_layers = [
        layer_id for layer_id in feature_columns if layer_id != "elevation" and layer_id not in derived_layers
    ]

    tile_start = time.perf_counter()
    for tile_idx, (tile_id, group) in enumerate(
        worklist.groupby("tileId", sort=False, dropna=True),
        start=1,
    ):
        row_ids = group["row_id"].to_numpy(dtype=int)
        tile_lats = group["decimalLatitude"].to_numpy(dtype=np.float64)
        tile_lons = group["decimalLongitude"].to_numpy(dtype=np.float64)

        if needs_elevation:
            values["elevation"][row_ids] = _sample_dem_values_for_tile(tile_lats, tile_lons, tile_id)

        if derived_layers:
            sampled = _sample_dem_derived_values_for_tile(
                tile_lats,
                tile_lons,
                tile_id,
                tuple(sorted(set(derived_layers.values()))),
            )
            for layer_id, metric in derived_layers.items():
                values[layer_id][row_ids] = sampled[metric]

        for layer_id in regular_layers:
            values[layer_id][row_ids] = _sample_regular_layer_values_for_tile(
                layer_id,
                tile_lats,
                tile_lons,
            )

        if tile_idx == 1 or tile_idx % 25 == 0 or tile_idx == total_tiles:
            elapsed = time.perf_counter() - tile_start
            print(
                f"[negatives-features] processed tiles {tile_idx}/{total_tiles} "
                f"rows={len(group)} elapsed={elapsed:.1f}s current_tile={tile_id}"
            )

    return pd.DataFrame(values)


def _sample_ring_coordinates(
    target_rows: int,
    inner_bbox: BoundingBox | None,
    outer_bbox: BoundingBox,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    if target_rows <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), 0

    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    accepted = 0
    sampled_points = 0

    for attempt in range(1, NEGATIVE_WINDOW_RETRIES + 1):
        if accepted >= target_rows:
            break
        remaining = target_rows - accepted
        batch_size = max(
            CONFIG.ml_negative_batch_min,
            int(np.ceil(remaining * CONFIG.ml_negative_ring_oversample_factor)),
        )
        batch_size = min(batch_size, NEGATIVE_BATCH_MAX)

        batch_lats = rng.uniform(outer_bbox.min_lat, outer_bbox.max_lat, size=batch_size)
        batch_lons = rng.uniform(outer_bbox.min_lon, outer_bbox.max_lon, size=batch_size)
        sampled_points += batch_size

        if inner_bbox is not None:
            keep_mask = ~inner_bbox.contains(batch_lats, batch_lons)
            batch_lats = batch_lats[keep_mask]
            batch_lons = batch_lons[keep_mask]

        if batch_lats.size == 0:
            print(
                f"[negatives-coords] attempt={attempt}/{NEGATIVE_WINDOW_RETRIES} "
                f"batch_size={batch_size} kept=0 accepted_total={accepted}/{target_rows}"
            )
            continue

        lat_parts.append(batch_lats)
        lon_parts.append(batch_lons)
        accepted += len(batch_lats)
        print(
            f"[negatives-coords] attempt={attempt}/{NEGATIVE_WINDOW_RETRIES} "
            f"batch_size={batch_size} kept={len(batch_lats)} "
            f"accepted_total={accepted}/{target_rows}"
        )

    if not lat_parts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), sampled_points

    lats = np.concatenate(lat_parts)[:target_rows]
    lons = np.concatenate(lon_parts)[:target_rows]
    return lats, lons, sampled_points


def _prefilter_candidate_coordinates(
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(lats) == 0:
        empty = np.empty(0, dtype=bool)
        return lats, lons, empty

    mask_features = _sample_feature_matrix_for_coordinates(
        lats,
        lons,
        ["landcover", "elevation"],
    )
    landcover = pd.to_numeric(mask_features.get("landcover"), errors="coerce").to_numpy(dtype=np.float32)
    elevation = pd.to_numeric(mask_features.get("elevation"), errors="coerce").to_numpy(dtype=np.float32)
    landcover_ids = np.full(len(landcover), -9999, dtype=np.int32)
    finite_landcover = np.isfinite(landcover)
    if np.any(finite_landcover):
        landcover_ids[finite_landcover] = np.rint(landcover[finite_landcover]).astype(
            np.int32,
            copy=False,
        )
    keep_mask = (
        np.isfinite(landcover)
        & np.isfinite(elevation)
        & ~((landcover_ids == LANDCOVER_WATER_CLASS_ID) & (np.abs(elevation) <= SEA_LEVEL_FILTER_ABS_ELEVATION_METERS))
    )
    return lats[keep_mask], lons[keep_mask], keep_mask


def _sample_negative_features_raster(
    target_rows: int,
    feature_spec: FeatureSpec,
    positive_bbox: BoundingBox,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_rows <= 0:
        return pd.DataFrame(columns=feature_spec.all_columns), {"rows_collected": 0}

    total_sampled_points = 0
    windows: list[dict[str, Any]] = []
    window_plan = _build_negative_window_plan(positive_bbox)
    print(f"[negatives] planned {len(window_plan)} windows from local rings to global coverage")
    global_bbox = window_plan[-1]["outer_bbox"]
    fixed_top_up_prefilter_target = max(
        target_rows,
        int(np.ceil(target_rows * float(CONFIG.ml_negative_prefilter_oversample_factor))),
    )

    candidate_lat_parts: list[np.ndarray] = []
    candidate_lon_parts: list[np.ndarray] = []
    window_row_slices: list[tuple[int, int]] = []
    previous_bbox: BoundingBox | None = positive_bbox
    planned_rows = 0
    for idx, window in enumerate(window_plan, start=1):
        remaining = target_rows - planned_rows
        if remaining <= 0:
            break

        remaining_windows = len(window_plan) - idx + 1
        target_for_window = max(1, int(np.ceil(remaining / remaining_windows)))
        prefilter_target_for_window = max(
            target_for_window,
            int(np.ceil(target_for_window * float(CONFIG.ml_negative_prefilter_oversample_factor))),
        )
        outer_bbox = window["outer_bbox"]
        factor = window["factor"]
        source = window["source"]
        print(
            f"[negatives] starting window {idx}/{len(window_plan)} "
            f"source={source} factor={factor} "
            f"target_window_rows={target_for_window} "
            f"prefilter_target_rows={prefilter_target_for_window} "
            f"target_remaining={remaining} "
            f"outer_bbox={outer_bbox.to_dict()}"
        )
        coord_start = time.perf_counter()
        candidate_lats, candidate_lons, sampled_points = _sample_ring_coordinates(
            prefilter_target_for_window,
            previous_bbox,
            outer_bbox,
            rng,
        )
        total_sampled_points += sampled_points
        coord_elapsed = time.perf_counter() - coord_start
        print(
            f"[negatives] factor={factor} coordinate staging done in {coord_elapsed:.1f}s "
            f"candidate_points={len(candidate_lats)} sampled_points={sampled_points}"
        )
        start_idx = planned_rows
        planned_rows += len(candidate_lats)
        end_idx = planned_rows
        candidate_lat_parts.append(candidate_lats)
        candidate_lon_parts.append(candidate_lons)
        window_row_slices.append((start_idx, end_idx))

        windows.append(
            {
                "index": idx,
                "source": source,
                "factor": float(factor),
                "inner_bbox": previous_bbox.to_dict() if previous_bbox is not None else None,
                "outer_bbox": outer_bbox.to_dict(),
                "target_rows": int(target_for_window),
                "prefilter_target_rows": int(prefilter_target_for_window),
                "sampled_points": int(sampled_points),
                "candidate_points": int(len(candidate_lats)),
                "valid_rows": 0,
                "accepted_rows": 0,
                "coordinate_stage_seconds": float(coord_elapsed),
                "feature_stage_seconds": 0.0,
            }
        )
        print(
            f"[negatives] factor={factor} staged candidate_points={len(candidate_lats)} "
            f"remaining_planned={target_rows - planned_rows}"
        )
        previous_bbox = outer_bbox

    if not candidate_lat_parts:
        raise RuntimeError("Failed to stage any negative coordinates.")

    all_candidate_lats = np.concatenate(candidate_lat_parts)
    all_candidate_lons = np.concatenate(candidate_lon_parts)
    prefilter_start = time.perf_counter()
    prefiltered_lats, prefiltered_lons, prefilter_mask = _prefilter_candidate_coordinates(
        all_candidate_lats,
        all_candidate_lons,
    )
    prefilter_elapsed = time.perf_counter() - prefilter_start
    kept_after_prefilter = int(np.sum(prefilter_mask))
    print(
        f"[negatives-prefilter] kept={kept_after_prefilter}/{len(prefilter_mask)} "
        f"using !(landcover==210 and |elevation|<={SEA_LEVEL_FILTER_ABS_ELEVATION_METERS:.1f}) "
        f"elapsed={prefilter_elapsed:.1f}s"
    )
    feature_start = time.perf_counter()
    sampled_features = _sample_feature_matrix_for_coordinates(
        prefiltered_lats,
        prefiltered_lons,
        feature_spec.all_columns,
    )
    feature_elapsed = time.perf_counter() - feature_start
    accepted_frames: list[pd.DataFrame] = []
    rows_collected = 0
    prefilter_prefix = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(prefilter_mask.astype(np.int64))])
    for window, (start_idx, end_idx) in zip(windows, window_row_slices):
        window_prefilter_mask = prefilter_mask[start_idx:end_idx]
        window_prefiltered = int(np.sum(window_prefilter_mask))
        filtered_start = int(prefilter_prefix[start_idx])
        filtered_end = int(prefilter_prefix[end_idx])
        window_features = sampled_features.iloc[filtered_start:filtered_end].reset_index(drop=True)
        window["feature_stage_seconds"] = float(feature_elapsed)
        window["prefilter_stage_seconds"] = float(prefilter_elapsed)
        window["prefilter_kept_rows"] = window_prefiltered
        window["valid_rows"] = int(len(window_features))
        accepted = window_features
        target_for_window = int(window["target_rows"])
        if len(accepted) > target_for_window:
            accepted = accepted.iloc[:target_for_window].reset_index(drop=True)
        window["accepted_rows"] = int(len(accepted))
        if not accepted.empty:
            accepted_frames.append(accepted)
            rows_collected += len(accepted)
        print(
            f"[negatives] factor={window['factor']} accepted_rows={len(accepted)} "
            f"prefilter_kept={window_prefiltered} "
            f"valid_rows={window['valid_rows']} feature_seconds={feature_elapsed:.1f} "
            f"remaining={target_rows - rows_collected}"
        )

    extra_rounds = 0
    while rows_collected < target_rows:
        extra_rounds += 1
        if extra_rounds > int(CONFIG.ml_negative_global_max_extra_rounds):
            raise RuntimeError(
                "Negative sampling remained under quota after exhausting global top-up rounds. "
                "Increase ml_negative_global_max_extra_rounds, adjust the coverage threshold, "
                "or add a land-mask prefilter."
            )

        remaining = target_rows - rows_collected
        prefilter_target_rows = fixed_top_up_prefilter_target
        print(
            f"[negatives] top-up round {extra_rounds}/{CONFIG.ml_negative_global_max_extra_rounds} "
            f"source=auto_global target_remaining={remaining} "
            f"prefilter_target_rows={prefilter_target_rows}"
        )
        coord_start = time.perf_counter()
        candidate_lats, candidate_lons, sampled_points = _sample_ring_coordinates(
            prefilter_target_rows,
            None,
            global_bbox,
            rng,
        )
        total_sampled_points += sampled_points
        coord_elapsed = time.perf_counter() - coord_start
        print(
            f"[negatives] top-up round={extra_rounds} coordinate staging done in {coord_elapsed:.1f}s "
            f"candidate_points={len(candidate_lats)} sampled_points={sampled_points}"
        )
        prefilter_start = time.perf_counter()
        candidate_lats, candidate_lons, prefilter_mask = _prefilter_candidate_coordinates(
            candidate_lats,
            candidate_lons,
        )
        prefilter_elapsed = time.perf_counter() - prefilter_start
        print(
            f"[negatives] top-up round={extra_rounds} prefilter kept={int(np.sum(prefilter_mask))}/"
            f"{len(prefilter_mask)} elapsed={prefilter_elapsed:.1f}s"
        )

        feature_start = time.perf_counter()
        sampled_features = _sample_feature_matrix_for_coordinates(
            candidate_lats,
            candidate_lons,
            feature_spec.all_columns,
        )
        feature_elapsed = time.perf_counter() - feature_start
        accepted = sampled_features.reset_index(drop=True)
        if len(accepted) > remaining:
            accepted = accepted.iloc[:remaining].reset_index(drop=True)
        if not accepted.empty:
            accepted_frames.append(accepted)
            rows_collected += len(accepted)

        windows.append(
            {
                "index": len(windows) + 1,
                "source": "auto_global_top_up",
                "factor": float(window_plan[-1]["factor"]),
                "inner_bbox": None,
                "outer_bbox": global_bbox.to_dict(),
                "target_rows": int(remaining),
                "prefilter_target_rows": int(prefilter_target_rows),
                "sampled_points": int(sampled_points),
                "candidate_points": int(len(candidate_lats)),
                "prefilter_stage_seconds": float(prefilter_elapsed),
                "prefilter_kept_rows": int(len(candidate_lats)),
                "valid_rows": int(len(accepted)),
                "accepted_rows": int(len(accepted)),
                "coordinate_stage_seconds": float(coord_elapsed),
                "feature_stage_seconds": float(feature_elapsed),
            }
        )
        print(
            f"[negatives] top-up round={extra_rounds} sampled_points={sampled_points} "
            f"candidate_points={len(candidate_lats)} accepted_rows={len(accepted)} "
            f"feature_seconds={feature_elapsed:.1f} remaining={target_rows - rows_collected}"
        )

    if not accepted_frames:
        raise RuntimeError("Failed to collect any negative rows with valid GIS coverage.")

    negative_frame = pd.concat(accepted_frames, ignore_index=True)

    negative_frame = negative_frame.iloc[:target_rows].reset_index(drop=True)
    summary = {
        "strategy": "random_ring_sampling_with_global_expansion_and_grouped_tile_reads",
        "target_rows": int(target_rows),
        "rows_collected": int(len(negative_frame)),
        "sampled_points": int(total_sampled_points),
        "global_top_up_rounds": int(extra_rounds),
        "positive_bbox": positive_bbox.to_dict(),
        "windows": windows,
    }
    return negative_frame, summary


def _sample_negative_features_from_taxa(
    target_taxon_id: str,
    target_rows: int,
    feature_spec: FeatureSpec,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_rows <= 0:
        return pd.DataFrame(columns=feature_spec.all_columns), {"rows_collected": 0}

    catalog = taxa_navigation.load_catalog()
    leaf_ranks = frozenset(CONFIG.leaf_ranks)
    storage = _parquet_storage()
    target_id_str = str(target_taxon_id)

    # Collect all leaf taxa with occurrence parquets, excluding the target taxon
    candidates: list[tuple[str, Path]] = []
    for tid, taxon in catalog.items():
        if str(tid) == target_id_str:
            continue
        if str(taxon.get("rank") or "").upper() not in leaf_ranks:
            continue
        path = taxa_navigation.normalize_taxon_path(taxon["path"]) / CONFIG.occurrence_parquet_filename
        candidates.append((str(tid), path))

    if not candidates:
        raise RuntimeError(
            "No candidate taxa found for taxa-based negative sampling. "
            "Ensure leaf taxa with occurrence parquets exist in the catalog."
        )

    # Shuffle then cap pool size
    perm = rng.permutation(len(candidates))
    pool_size = min(int(CONFIG.ml_negative_taxa_candidate_pool), len(candidates))
    selected = [candidates[i] for i in perm[:pool_size]]

    max_per_taxon = int(CONFIG.ml_negative_taxa_max_per_taxon)
    print(
        f"[negatives-taxa] pool={pool_size}/{len(candidates)} target_rows={target_rows} max_per_taxon={max_per_taxon}"
    )

    collected_frames: list[pd.DataFrame] = []
    rows_collected = 0
    taxa_used = 0

    for _tid, path in selected:
        if rows_collected >= target_rows:
            break
        try:
            pf = storage.parquet_file(path)
            if pf.metadata.num_rows == 0:
                continue
            available = [c for c in feature_spec.all_columns if c in pf.schema.names]
            if not available:
                continue

            frame = pf.read(columns=available).to_pandas()
            missing = {col: np.nan for col in feature_spec.all_columns if col not in frame.columns}
            if missing:
                frame = pd.concat([frame, pd.DataFrame(missing, index=frame.index)], axis=1)
            frame = frame[feature_spec.all_columns].dropna(how="all").reset_index(drop=True)
            if frame.empty:
                continue

            to_take = min(max_per_taxon, len(frame), target_rows - rows_collected)
            if len(frame) > to_take:
                frame = frame.iloc[rng.choice(len(frame), size=to_take, replace=False)].reset_index(drop=True)

            collected_frames.append(frame)
            rows_collected += len(frame)
            taxa_used += 1
        except Exception:
            continue

    if not collected_frames:
        raise RuntimeError(
            "No negative rows could be collected from taxa parquets. "
            "Ensure occurrence parquets have GIS feature columns populated."
        )

    negative_frame = pd.concat(collected_frames, ignore_index=True)
    shuffle_idx = rng.permutation(len(negative_frame))
    negative_frame = negative_frame.iloc[shuffle_idx].iloc[:target_rows].reset_index(drop=True)

    print(f"[negatives-taxa] collected={len(negative_frame)} taxa_used={taxa_used}")
    return negative_frame, {
        "strategy": "taxa_occurrence_sampling",
        "target_rows": int(target_rows),
        "rows_collected": int(len(negative_frame)),
        "taxa_used": int(taxa_used),
        "candidate_pool_size": int(pool_size),
        "max_per_taxon": int(max_per_taxon),
    }


def _sample_negative_features(
    target_rows: int,
    feature_spec: FeatureSpec,
    positive_bbox: BoundingBox,
    rng: np.random.Generator,
    *,
    taxon_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(CONFIG.ml_negative_mode).strip().lower()
    if mode == "taxa":
        if taxon_id is None:
            raise ValueError("taxon_id is required when ml_negative_mode='taxa'")
        return _sample_negative_features_from_taxa(taxon_id, target_rows, feature_spec, rng)
    if mode == "raster":
        return _sample_negative_features_raster(target_rows, feature_spec, positive_bbox, rng)
    raise ValueError(f"Unknown ml_negative_mode={CONFIG.ml_negative_mode!r}; expected 'raster' or 'taxa'.")


def _build_negative_window_plan(positive_bbox: BoundingBox) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    last_bbox: BoundingBox | None = None

    for factor in CONFIG.ml_negative_window_factors:
        outer_bbox = positive_bbox.expanded(float(factor))
        if last_bbox is not None and outer_bbox.approx_equals(last_bbox):
            continue
        plan.append(
            {
                "source": "configured",
                "factor": float(factor),
                "outer_bbox": outer_bbox,
            }
        )
        last_bbox = outer_bbox

    growth_factor = float(CONFIG.ml_negative_global_growth_factor)
    if growth_factor <= 1.0:
        raise ValueError(f"ml_negative_global_growth_factor must be > 1.0; got {growth_factor!r}")

    next_factor = (
        float(plan[-1]["factor"])
        if plan
        else max(1.0, float(CONFIG.ml_negative_window_factors[0]) if CONFIG.ml_negative_window_factors else 1.0)
    )
    extra_steps = 0
    while last_bbox is None or not last_bbox.is_global():
        extra_steps += 1
        if extra_steps > NEGATIVE_AUTO_GLOBAL_MAX_STEPS:
            raise RuntimeError(
                "Failed to construct a global negative window plan within the safety limit. "
                "Adjust ml_negative_window_factors or ml_negative_global_growth_factor."
            )
        next_factor *= growth_factor
        outer_bbox = positive_bbox.expanded(next_factor)
        if last_bbox is not None and outer_bbox.approx_equals(last_bbox):
            if outer_bbox.is_global():
                break
            continue
        plan.append(
            {
                "source": "auto_global",
                "factor": float(next_factor),
                "outer_bbox": outer_bbox,
            }
        )
        last_bbox = outer_bbox

    if not plan:
        raise RuntimeError("Negative window plan was empty.")

    final_bbox = plan[-1]["outer_bbox"]
    print(f"[negatives] final window reaches global={final_bbox.is_global()} bbox={final_bbox.to_dict()}")
    return plan


def _prune_empty_features(
    frame: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> tuple[pd.DataFrame, FeatureSpec, list[str]]:
    kept_columns = [col for col in feature_spec.all_columns if frame[col].notna().any()]
    dropped_columns = [col for col in feature_spec.all_columns if col not in kept_columns]
    if not kept_columns:
        raise RuntimeError("All feature columns were empty after assembling training data.")

    pruned_spec = FeatureSpec(
        all_columns=kept_columns,
        numeric_columns=[col for col in feature_spec.numeric_columns if col in kept_columns],
        categorical_columns=[col for col in feature_spec.categorical_columns if col in kept_columns],
    )
    return frame[kept_columns].copy(), pruned_spec, dropped_columns


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(
    feature_spec: FeatureSpec,
    *,
    model_kind: str,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if feature_spec.numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_kind == "maxent":
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                feature_spec.numeric_columns,
            )
        )

    if feature_spec.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                feature_spec.categorical_columns,
            )
        )

    if not transformers:
        raise RuntimeError("Feature spec is empty after pruning.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def _train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    model_kind = str(CONFIG.ml_model_kind).strip().lower()
    if model_kind == "gbt":
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=CONFIG.ml_random_seed,
        )
    elif model_kind == "maxent":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=int(CONFIG.ml_random_seed),
        )
    else:
        raise ValueError(f"Unsupported ml_model_kind={CONFIG.ml_model_kind!r}; expected 'gbt' or 'maxent'.")
    model.fit(X_train, y_train)
    return model


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_pred = (y_prob >= 0.5).astype(np.int8)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "roc_auc": None,
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _score_frame(
    frame: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: Any,
) -> np.ndarray:
    transformed = preprocessor.transform(frame)
    return model.predict_proba(transformed)[:, 1]


def _quantile_summary(values: np.ndarray) -> dict[str, float] | dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0}
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "quantiles": {f"{q:.2f}": float(np.quantile(finite, q)) for q in EVAL_QUANTILES},
    }


def _threshold_metrics(
    *,
    positive_scores: np.ndarray,
    background_scores: np.ndarray | None = None,
) -> list[dict[str, float]]:
    positive_finite = positive_scores[np.isfinite(positive_scores)]
    background_finite = (
        background_scores[np.isfinite(background_scores)]
        if background_scores is not None
        else np.empty(0, dtype=np.float32)
    )
    rows: list[dict[str, float]] = []
    for threshold in EVAL_THRESHOLDS:
        row: dict[str, float] = {
            "threshold": float(threshold),
            "positive_recall": float(np.mean(positive_finite >= threshold)) if positive_finite.size else float("nan"),
        }
        if background_scores is not None:
            row["background_false_positive_rate"] = (
                float(np.mean(background_finite >= threshold)) if background_finite.size else float("nan")
            )
        rows.append(row)
    return rows


def _save_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model.pkl", "wb") as handle:
        pickle.dump(payload, handle)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _taxon_seed(taxon_id: str) -> int:
    try:
        suffix = int(taxon_id)
    except ValueError:
        suffix = sum(ord(ch) for ch in taxon_id)
    return int(CONFIG.ml_random_seed) + (suffix % 10_000_000)


def _run_training(
    taxon_id: str,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    feature_spec: FeatureSpec,
    negative_info: dict[str, Any],
    *,
    mode: str,
    positive_bbox: BoundingBox | None = None,
) -> None:
    """Shared training logic used by both standard SDM and phenology modes."""
    X = pd.concat([positives, negatives], ignore_index=True)
    y = np.concatenate(
        [
            np.ones(len(positives), dtype=np.int8),
            np.zeros(len(negatives), dtype=np.int8),
        ]
    )

    X, pruned_spec, dropped_columns = _prune_empty_features(X, feature_spec)
    if dropped_columns:
        print(f"[train] dropped empty feature columns: {len(dropped_columns)}")

    n_classes = len(np.unique(y))
    # test set needs at least one sample per class for stratification
    min_test = n_classes
    test_size = max(float(CONFIG.ml_test_size), min_test / len(y))
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(CONFIG.ml_random_seed),
        stratify=y,
    )

    model_kind = str(CONFIG.ml_model_kind).strip().lower()
    preprocessor = _build_preprocessor(pruned_spec, model_kind=model_kind)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    print(
        f"[train] preprocessed shapes: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    model = _train_model(X_train, y_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_metrics = _compute_metrics(y_train, train_prob)
    test_metrics = _compute_metrics(y_test, test_prob)
    all_positive_prob = _score_frame(positives[pruned_spec.all_columns], preprocessor, model)

    eval_background_info: dict[str, Any] | None = None
    eval_background_prob: np.ndarray | None = None
    if positive_bbox is not None and bool(CONFIG.ml_enable_background_eval):
        background_eval_target = max(4096, min(len(positives) * 2, 25_000))
        eval_rng = np.random.default_rng(_taxon_seed(f"{taxon_id}_eval_background"))
        eval_background_frame, eval_background_info = _sample_negative_features(
            background_eval_target,
            pruned_spec,
            positive_bbox,
            eval_rng,
            taxon_id=taxon_id,
        )
        eval_background_prob = _score_frame(
            eval_background_frame[pruned_spec.all_columns],
            preprocessor,
            model,
        )
    else:
        print("[train] background eval disabled; skipping broad background diagnostics")

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "all_known_positive_scores": _quantile_summary(all_positive_prob),
    }
    if eval_background_prob is not None:
        metrics["broad_background_scores"] = _quantile_summary(eval_background_prob)
        metrics["threshold_diagnostics"] = _threshold_metrics(
            positive_scores=all_positive_prob,
            background_scores=eval_background_prob,
        )
    else:
        metrics["broad_background_scores"] = {"enabled": False}
        metrics["threshold_diagnostics"] = _threshold_metrics(
            positive_scores=all_positive_prob,
            background_scores=None,
        )
    print(
        "[train] metrics "
        f"train_auc={train_metrics['roc_auc']} test_auc={test_metrics['roc_auc']} "
        f"train_logloss={train_metrics['log_loss']:.4f} test_logloss={test_metrics['log_loss']:.4f}"
    )
    positive_quantiles = metrics["all_known_positive_scores"].get("quantiles", {})
    if eval_background_prob is not None:
        background_quantiles = metrics["broad_background_scores"].get("quantiles", {})
        print(
            "[train] diagnostics "
            f"known_positive_p10={positive_quantiles.get('0.10')} "
            f"known_positive_p50={positive_quantiles.get('0.50')} "
            f"known_positive_p90={positive_quantiles.get('0.90')} "
            f"background_p95={background_quantiles.get('0.95')}"
        )
    else:
        print(
            "[train] diagnostics "
            f"known_positive_p10={positive_quantiles.get('0.10')} "
            f"known_positive_p50={positive_quantiles.get('0.50')} "
            f"known_positive_p90={positive_quantiles.get('0.90')}"
        )

    output_dir = CONFIG.models_root / f"taxon_{taxon_id}_{CONFIG.ml_model_kind}_{mode}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[train] removed old artifact: {output_dir}")
    if model_kind == "gbt":
        model_type = "sklearn_gradient_boosting"
    elif model_kind == "maxent":
        model_type = "sklearn_logistic_maxent_style"
    else:
        model_type = f"unknown_{model_kind}"
    payload = {
        "model_type": model_type,
        "model": model,
        "preprocessor": preprocessor,
        "taxon_id": taxon_id,
        "feature_columns": pruned_spec.all_columns,
        "numeric_columns": pruned_spec.numeric_columns,
        "categorical_columns": pruned_spec.categorical_columns,
        "dropped_feature_columns": dropped_columns,
        "random_seed": int(CONFIG.ml_random_seed),
        "test_size": float(CONFIG.ml_test_size),
        "training_mode": mode,
        "parquet_storage_mode": str(CONFIG.ml_parquet_storage_mode),
        "raster_storage_mode": str(CONFIG.ml_raster_storage_mode),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "taxon_id": taxon_id,
        "training_mode": mode,
        "positive_rows": int(len(positives)),
        "negative_rows": int(len(negatives)),
        "total_rows": int(len(X)),
        "feature_count_raw": int(len(feature_spec.all_columns)),
        "feature_count_used": int(len(pruned_spec.all_columns)),
        "feature_count_transformed": int(X_train.shape[1]),
        "dropped_feature_columns": dropped_columns,
        "storage": {
            "parquet_mode": str(CONFIG.ml_parquet_storage_mode),
            "raster_mode": str(CONFIG.ml_raster_storage_mode),
        },
        "negative_sampling": negative_info,
        "background_eval_enabled": positive_bbox is not None and bool(CONFIG.ml_enable_background_eval),
        "evaluation_background_sampling": eval_background_info,
    }
    _save_artifacts(output_dir, payload, metrics, summary)

    print(f"[train] saved model artifact: {output_dir / 'model.pkl'}")
    print(f"[train] saved metrics: {output_dir / 'metrics.json'}")
    print(f"[train] saved summary: {output_dir / 'summary.json'}")

    if bool(CONFIG.ml_push_model_to_b2):
        remote = os.environ.get("WW_B2_WRITER_REMOTE", "wherewild-localdev-writer")
        bucket = os.environ.get("WW_B2_BUCKET", "wherewild-data")
        prefix = os.environ.get("WW_B2_PREFIX", "data")
        rel_path = output_dir.relative_to(CONFIG.data_root)
        remote_dest = f"{remote}:{bucket}/{prefix}/{rel_path}"
        print(f"[train] pushing model to B2: {remote_dest}")
        subprocess.run(
            ["rclone", "copy", str(output_dir), remote_dest, "--transfers=4"],
            check=True,
        )
        print("[train] B2 push complete")


def main() -> None:
    _configure_storage_modes()

    taxon_id = str(CONFIG.ml_train_taxon_id)
    gis_feature_spec = _load_non_temporal_feature_spec()
    temporal_cols = _temporal_feature_names()

    print(f"[train] taxon={taxon_id}")
    print(
        f"[train] parquet_storage={CONFIG.ml_parquet_storage_mode} "
        f"raster_storage={CONFIG.ml_raster_storage_mode} "
        f"phenology_mode={CONFIG.ml_phenology_mode}"
    )

    if CONFIG.ml_phenology_mode:
        # ── Phenology mode ──────────────────────────────────────────────────
        # Both positives and negatives are real occurrences read from the parquet.
        # Positives have a rcs annotation in the configured positive values; negatives do not.
        # Since all rows come from the parquet they already have temporal enrichment,
        # so there's no negative-sampling bottleneck.
        temporal_only = bool(CONFIG.ml_phenology_temporal_only)
        print(f"[train] mode=phenology (rcs annotation as label, no synthetic negatives) temporal_only={temporal_only}")
        positives, negatives = _read_phenology_split(taxon_id, gis_feature_spec)
        all_temporal_cols = [c for c in temporal_cols if c in positives.columns]
        if temporal_only:
            if not all_temporal_cols:
                raise RuntimeError("ml_phenology_temporal_only=True but no temporal columns found in parquet.")
            full_feature_spec = FeatureSpec(
                all_columns=all_temporal_cols,
                numeric_columns=all_temporal_cols,
                categorical_columns=[],
            )
        else:
            full_feature_spec = FeatureSpec(
                all_columns=gis_feature_spec.all_columns + all_temporal_cols,
                numeric_columns=gis_feature_spec.numeric_columns + all_temporal_cols,
                categorical_columns=gis_feature_spec.categorical_columns,
            )
        print(
            f"[train] positives={len(positives)} negatives={len(negatives)} temporal_features={len(all_temporal_cols)}"
        )
        if len(positives) == 0:
            raise RuntimeError(
                f"Phenology mode: no rcs-annotated occurrences found for taxon {taxon_id}. "
                "Set ml_phenology_mode=False to train a standard SDM instead."
            )
        if len(negatives) == 0:
            raise RuntimeError(
                f"Phenology mode: all {len(positives)} occurrences for taxon {taxon_id} have "
                "a rcs annotation — no unannotated negatives available. "
                "Pick a taxon with mixed annotation coverage, or set ml_phenology_mode=False."
            )
        negative_info = {
            "strategy": "phenology_parquet_split",
            "rcs_column": CONFIG.ml_phenology_rcs_column,
            "positive_rows": int(len(positives)),
            "negative_rows": int(len(negatives)),
        }
        _run_training(taxon_id, positives, negatives, full_feature_spec, negative_info, mode="phenology")

    elif bool(CONFIG.ml_sdm_include_temporal):
        # ── Full model (GIS + temporal) ──────────────────────────────────────
        # mode="full" — GIS habitat features + temporal weather features in one model.
        # Positives: species occurrences with GIS + temporal enrichment from parquet.
        # Negatives: taxa-based occurrences with GIS + temporal columns.
        print("[train] mode=full (GIS + temporal features, taxa negatives)")
        rng = np.random.default_rng(_taxon_seed(taxon_id))

        positives, positive_lats, positive_lons = _read_positive_features_and_coords(
            taxon_id,
            gis_feature_spec,
            extra_columns=temporal_cols,
        )
        all_temporal_cols = [c for c in temporal_cols if c in positives.columns]
        if not all_temporal_cols:
            raise RuntimeError(
                "ml_sdm_include_temporal=True but no temporal columns found in parquet. "
                "Ensure the occurrence parquet has been temporally enriched."
            )
        full_feature_spec = FeatureSpec(
            all_columns=gis_feature_spec.all_columns + all_temporal_cols,
            numeric_columns=gis_feature_spec.numeric_columns + all_temporal_cols,
            categorical_columns=gis_feature_spec.categorical_columns,
        )
        positives = positives[full_feature_spec.all_columns]

        target_negative_rows = len(positives) * int(CONFIG.ml_negative_ratio)
        print(
            f"[train] positives={len(positives)} target_negatives={target_negative_rows} "
            f"ratio={CONFIG.ml_negative_ratio}:1 temporal_features={len(all_temporal_cols)}"
        )

        negatives, negative_info = _sample_negative_features_from_taxa(
            taxon_id, target_negative_rows, full_feature_spec, rng
        )
        print(f"[train] negatives collected={len(negatives)}")

        positive_bbox = _compute_positive_bbox(positive_lats, positive_lons)
        _run_training(
            taxon_id,
            positives,
            negatives,
            full_feature_spec,
            negative_info,
            mode="full",
            positive_bbox=positive_bbox,
        )

    else:
        # ── Standard SDM mode ────────────────────────────────────────────────
        # mode="sdm" — GIS features only, spatial habitat suitability.
        print("[train] mode=standard (GIS features, spatial negative sampling)")
        rng = np.random.default_rng(_taxon_seed(taxon_id))

        positives, positive_lats, positive_lons = _read_positive_features_and_coords(
            taxon_id,
            gis_feature_spec,
        )
        full_feature_spec = gis_feature_spec

        positive_bbox = _compute_positive_bbox(positive_lats, positive_lons)
        target_negative_rows = len(positives) * int(CONFIG.ml_negative_ratio)
        print(
            f"[train] positives={len(positives)} target_negatives={target_negative_rows} "
            f"ratio={CONFIG.ml_negative_ratio}:1"
        )

        negatives, negative_info = _sample_negative_features(
            target_negative_rows,
            gis_feature_spec,
            positive_bbox,
            rng,
            taxon_id=taxon_id,
        )
        print(f"[train] negatives collected={len(negatives)}")

        _run_training(
            taxon_id,
            positives,
            negatives,
            full_feature_spec,
            negative_info,
            mode="sdm",
            positive_bbox=positive_bbox,
        )


if __name__ == "__main__":
    main()
