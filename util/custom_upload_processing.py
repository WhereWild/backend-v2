from __future__ import annotations
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from fastapi import HTTPException
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scripts.enrich_tree as enrich_tree
from util.config import load_config
from util import gis_lookup, indexing, summary_stats

CONFIG = load_config("global")

def _normalize_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "decimalLatitude" not in result.columns:
        latitude_match = next(
            (
                column
                for column in result.columns
                if "latitude" in str(column).strip().lower()
            ),
            None,
        )
        if latitude_match is not None:
            result = result.rename(columns={latitude_match: "decimalLatitude"})

    if "decimalLongitude" not in result.columns:
        longitude_match = next(
            (
                column
                for column in result.columns
                if "longitude" in str(column).strip().lower()
            ),
            None,
        )
        if longitude_match is not None:
            result = result.rename(columns={longitude_match: "decimalLongitude"})

    return result


def _add_tile_ids(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"decimalLatitude", "decimalLongitude"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded file is missing required columns for tile lookup: "
                f"{missing_list}"
            ),
        )

    result = df.copy()
    latitudes = pd.to_numeric(result["decimalLatitude"], errors="coerce")
    longitudes = pd.to_numeric(result["decimalLongitude"], errors="coerce")

    invalid_coords = (
        latitudes.isna()
        | longitudes.isna()
        | (latitudes < -90)
        | (latitudes > 90)
        | (longitudes < -180)
        | (longitudes > 180)
    )
    if invalid_coords.any():
        invalid_count = int(invalid_coords.sum())
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded file has invalid coordinates in decimalLatitude/decimalLongitude "
                f"for {invalid_count} row(s)."
            ),
        )

    result["decimalLatitude"] = latitudes
    result["decimalLongitude"] = longitudes
    result["tileId"] = [
        gis_lookup.get_region_name(float(lat), float(lon))
        for lat, lon in zip(latitudes.tolist(), longitudes.tolist())
    ]
    return result


def _ensure_catalog_numbers(df: pd.DataFrame) -> pd.DataFrame:
    if "catalogNumber" in df.columns:
        return df

    result = df.copy()
    result["catalogNumber"] = [f"obs_{idx}" for idx in range(1, len(result) + 1)]
    return result


def _add_gis_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required_columns = {"tileId", "decimalLatitude", "decimalLongitude", "catalogNumber"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded file is missing required columns for GIS enrichment: "
                f"{', '.join(sorted(missing_columns))}"
            ),
        )

    layer_ids = enrich_tree._load_layer_ids()
    if not layer_ids:
        return df.copy()

    result = df.copy()
    result["decimalLatitude"] = pd.to_numeric(result["decimalLatitude"], errors="coerce")
    result["decimalLongitude"] = pd.to_numeric(result["decimalLongitude"], errors="coerce")
    result["catalogNumber"] = result["catalogNumber"].astype(str)
    result["tileId"] = result["tileId"].astype(str)

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-upload-gis_"))
    data_path = work_dir / CONFIG.occurrence_parquet_filename
    try:
        result.to_parquet(data_path, index=False)
        work_df = result[["catalogNumber", "tileId", "decimalLatitude", "decimalLongitude"]].copy()
        work_df = work_df[work_df["tileId"].astype(str).str.strip() != ""]
        if work_df.empty:
            return result
        work_df["missingLayers"] = [list(layer_ids)] * len(work_df)
        work_df["taxonKey"] = "uploaded"
        work_df["dataPath"] = str(data_path)
        worklist = pa.table(
            {
                "catalogNumber": pa.array(work_df["catalogNumber"].to_numpy(), type=pa.large_string()),
                "tileId": pa.array(work_df["tileId"].to_numpy(), type=pa.large_string()),
                "decimalLatitude": pa.array(work_df["decimalLatitude"].to_numpy(), type=pa.float64()),
                "decimalLongitude": pa.array(work_df["decimalLongitude"].to_numpy(), type=pa.float64()),
                "missingLayers": pa.array(work_df["missingLayers"].to_list(), type=pa.list_(pa.large_string())),
                "taxonKey": pa.array(work_df["taxonKey"].to_numpy(), type=pa.large_string()),
                "dataPath": pa.array(work_df["dataPath"].to_numpy(), type=pa.large_string()),
            }
        )
        enrich_tree._process_tiles(worklist)
        return pq.read_table(data_path).to_pandas()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _write_summary_artifacts_from_dataframe(df: pd.DataFrame, directory: Path) -> None:
    filtered = df.copy()
    if "obscured" in filtered.columns:
        filtered = filtered[filtered["obscured"].astype(str) == "No"]
    if "coordinateUncertaintyInMeters" in filtered.columns:
        uncertainty = pd.to_numeric(
            filtered["coordinateUncertaintyInMeters"],
            errors="coerce",
        )
        filtered = filtered[uncertainty <= 500]

    categorical_cols = [
        column
        for column in filtered.columns
        if summary_stats._layer_value_type(column) == "categorical"
    ]
    categorical_entries = summary_stats._collect_categorical_stats(filtered, categorical_cols)
    summary_stats._write_categorical_stats(
        directory,
        categorical_entries,
        merge_existing=False,
    )

    numeric_cols = [
        column
        for column in filtered.select_dtypes(include=["number"]).columns
        if (
            column not in summary_stats.excluded_numeric_columns
            and summary_stats._layer_value_type(column) != "categorical"
        )
    ]
    stats: dict[str, dict[str, Any]] = {}
    density_rows: list[dict[str, Any]] = []
    for column in numeric_cols:
        series = pd.to_numeric(filtered[column], errors="coerce").dropna()
        if series.empty:
            continue
        q10 = float(series.quantile(0.10))
        q25 = float(series.quantile(0.25))
        q50 = float(series.quantile(0.50))
        q75 = float(series.quantile(0.75))
        q90 = float(series.quantile(0.90))
        col_min = float(series.min())
        col_max = float(series.max())
        stats[column] = {
            "count": int(series.count()),
            "min": col_min,
            "10th percentile": q10,
            "25th percentile": q25,
            "median": q50,
            "75th percentile": q75,
            "90th percentile": q90,
            "max": col_max,
            "mean": float(series.mean()),
            "std": float(series.std()),
            "10-90 range": float(q90 - q10),
            "range": float(col_max - col_min),
        }

        values = series.astype(float).tolist()
        point_count = summary_stats._density_point_count(len(values))
        curve = summary_stats._build_density_curve(values, point_count)
        if curve:
            density_rows.append(
                {
                    "variable": column,
                    "count": int(len(values)),
                    "sampleCount": int(len(values)),
                    "pointCount": int(point_count),
                    "points": curve["points"],
                    "density": curve["density"],
                    "min": curve["min"],
                    "max": curve["max"],
                    "bandwidth": curve["bandwidth"],
                }
            )

    summary_stats._write_summary_stats(
        directory,
        stats,
        merge_existing=False,
    )
    density_path = directory / summary_stats.density_graph_filename
    if density_rows:
        pd.DataFrame(density_rows).to_parquet(density_path, index=False)
    else:
        density_path.unlink(missing_ok=True)


def _write_occurrence_index_from_dataframe(df: pd.DataFrame, directory: Path) -> None:
    if "catalogNumber" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is missing required column for indexing: catalogNumber",
        )

    index_path = directory / "occurrence_index.parquet"
    catalog_series = df["catalogNumber"].astype(str)
    targets = indexing.index_targets_for_columns(
        set(df.columns),
        layer_catalog=gis_lookup.load_layer_metadata(),
    )

    index_columns: dict[str, pa.Array] = {}
    column_lengths: dict[str, int] = {}
    category_offsets: dict[str, dict[str, dict[str, int | float]]] = {}
    max_len = 0

    for layer_id, value_type in targets:
        layer_series = df[layer_id]
        valid_mask = layer_series.notna()
        if "obscured" in df.columns:
            valid_mask = valid_mask & (df["obscured"].astype(str) == "No")
        if "coordinateUncertaintyInMeters" in df.columns:
            uncertainty = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
            valid_mask = valid_mask & (uncertainty <= 500)

        selected = df.loc[valid_mask, [layer_id]].copy()
        selected["catalogNumber"] = catalog_series.loc[valid_mask]
        selected[layer_id] = pd.to_numeric(selected[layer_id], errors="coerce")
        selected = selected.dropna(subset=[layer_id])
        if selected.empty:
            continue

        if value_type == "categorical":
            selected[layer_id] = selected[layer_id].astype(int)
        else:
            selected[layer_id] = selected[layer_id].astype(float)

        selected = selected.sort_values(layer_id, kind="stable")
        value_array = pa.array(
            selected[layer_id].tolist(),
            type=pa.int64() if value_type == "categorical" else pa.float64(),
        )
        catalog_array = pa.array(selected["catalogNumber"].astype(str).tolist(), type=pa.string())
        origin_array = pa.array([0] * len(selected), type=pa.int32())
        struct_array = pa.StructArray.from_arrays(
            [catalog_array, origin_array, value_array],
            fields=[
                pa.field("catalogNumber", pa.string()),
                pa.field("originId", pa.int32()),
                pa.field("value", value_array.type),
            ],
        )
        index_columns[layer_id] = struct_array
        column_lengths[layer_id] = len(struct_array)
        max_len = max(max_len, len(struct_array))

        if value_type == "categorical":
            values = selected[layer_id].tolist()
            offsets: dict[str, dict[str, int | float]] = {}
            current_value = None
            start_idx = 0
            for idx, value in enumerate(values):
                if current_value is None:
                    current_value = value
                    start_idx = idx
                    continue
                if value != current_value:
                    offsets[str(current_value)] = {
                        "value": int(current_value),
                        "start": start_idx,
                        "count": idx - start_idx,
                    }
                    current_value = value
                    start_idx = idx
            if current_value is not None:
                offsets[str(current_value)] = {
                    "value": int(current_value),
                    "start": start_idx,
                    "count": len(values) - start_idx,
                }
            if offsets:
                category_offsets[layer_id] = offsets

    if not index_columns:
        index_path.unlink(missing_ok=True)
        return

    for key, arr in list(index_columns.items()):
        if len(arr) < max_len:
            pad = pa.nulls(max_len - len(arr), type=arr.type)
            index_columns[key] = pa.concat_arrays([arr, pad])

    index_table = pa.table(index_columns)
    metadata = dict(index_table.schema.metadata or {})
    metadata[b"origin_map"] = json.dumps(
        [{"id": 0, "relative_path": ".", "taxon_key": "uploaded"}]
    ).encode("utf-8")
    metadata[b"column_lengths"] = json.dumps(column_lengths).encode("utf-8")
    metadata[b"catalog_column"] = b"catalogNumber"
    metadata[b"category_offsets"] = json.dumps(category_offsets).encode("utf-8")
    index_table = index_table.replace_schema_metadata(metadata)
    pq.write_table(index_table, index_path)


def _build_index_archive(df: pd.DataFrame, uploaded_name: str) -> tuple[Path, str, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-uploaded_0_"))
    occurrence_path = work_dir / CONFIG.occurrence_parquet_filename
    index_path = work_dir / "occurrence_index.parquet"
    archive_name = "processed_observations.zip"
    archive_path = work_dir / archive_name

    df.to_parquet(occurrence_path, index=False)

    try:
        _write_summary_artifacts_from_dataframe(df, work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build summary stat artifacts: {exc}",
        ) from exc

    try:
        _write_occurrence_index_from_dataframe(df, work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build occurrence index parquet: {exc}",
        ) from exc

    parquet_paths = [
        (occurrence_path, CONFIG.occurrence_parquet_filename),
        (work_dir / "summary_stats.parquet", "summary_stats.parquet"),
        (work_dir / "categorical_stats.parquet", "categorical_stats.parquet"),
        (work_dir / summary_stats.density_graph_filename, summary_stats.density_graph_filename),
    ]
    if index_path.exists():
        parquet_paths.append((index_path, index_path.name))

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for parquet_path, arcname in parquet_paths:
            if not parquet_path.exists():
                continue
            archive.write(parquet_path, arcname=arcname)
            csv_arcname = parquet_path.stem + ".csv"
            try:
                csv_bytes = pq.read_table(parquet_path).to_pandas().to_csv(index=False).encode()
                archive.writestr(csv_arcname, csv_bytes)
            except Exception:
                pass

    return archive_path, archive_name, work_dir