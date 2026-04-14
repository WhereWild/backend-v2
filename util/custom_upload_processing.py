from __future__ import annotations
import json
import re
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
from util import gis_lookup, indexing, summary_stats, units

CONFIG = load_config("global")

_CATALOG_NUMBER_ALIASES = (
    "catalogNumber",
    "catalog_number",
    "occurrenceID",
    "occurrence_id",
    "observationID",
    "observation_id",
    "recordID",
    "record_id",
    "gbifID",
    "gbif_id",
)

_OBSERVATION_NAME_ALIASES = (
    "observationName",
    "observation_name",
    "name",
    "title",
    "label",
)


def _variable_metadata_entry(variable_id: Any) -> tuple[str, dict[str, Any] | None]:
    variable_key = str(variable_id or "").strip()
    if not variable_key:
        return "", None
    try:
        _entries, by_id = gis_lookup.load_variable_metadata()
    except Exception:
        return variable_key, None
    entry = by_id.get(variable_key)
    if not isinstance(entry, dict):
        return variable_key, None
    return variable_key, entry


def _variable_label(variable_id: Any) -> str:
    variable_key, entry = _variable_metadata_entry(variable_id)
    if not variable_key:
        return ""
    if not entry:
        return variable_key
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return variable_key


def _normalized_column_key(column_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name or "").strip().lower())


def _find_column_by_aliases(column_names: list[Any], aliases: tuple[str, ...]) -> Any | None:
    by_normalized: dict[str, list[Any]] = {}
    for column_name in column_names:
        normalized = _normalized_column_key(column_name)
        if not normalized:
            continue
        by_normalized.setdefault(normalized, []).append(column_name)
    for alias in aliases:
        matches = by_normalized.get(_normalized_column_key(alias))
        if matches:
            return matches[0]
    return None


def _variable_category(variable_id: Any) -> str:
    variable_key, entry = _variable_metadata_entry(variable_id)
    if not variable_key:
        return ""
    if not entry:
        return ""
    category = entry.get("category")
    if isinstance(category, str):
        return category.strip()
    return ""


def _add_variable_name_column(
    frame: pd.DataFrame,
    *,
    variable_column: str = "variable",
    name_column: str = "variableName",
) -> pd.DataFrame:
    if frame.empty or variable_column not in frame.columns:
        return frame
    enriched = frame.copy()
    enriched[name_column] = [_variable_label(value) for value in enriched[variable_column].tolist()]
    return enriched


def _add_variable_category_column(
    frame: pd.DataFrame,
    *,
    variable_column: str = "variable",
    category_column: str = "variableCategory",
) -> pd.DataFrame:
    if frame.empty or variable_column not in frame.columns:
        return frame
    enriched = frame.copy()
    enriched[category_column] = [_variable_category(value) for value in enriched[variable_column].tolist()]
    return enriched


def _class_metric_to_readable_name(variable_id: Any, metric: Any) -> str:
    metric_text = str(metric or "").strip()
    if not metric_text:
        return metric_text
    if not metric_text.lower().startswith("class_"):
        return metric_text

    class_token = metric_text.split("_", 1)[1].strip()
    legend = gis_lookup.load_layer_legend(str(variable_id or ""))

    candidates = [class_token]
    try:
        numeric = float(class_token)
        if numeric.is_integer():
            candidates.append(str(int(numeric)))
        candidates.append(str(numeric))
    except (TypeError, ValueError):
        pass
    candidates.append(re.sub(r"[^a-z0-9]+", " ", class_token.lower()).strip())

    for key in candidates:
        entry = legend.get(str(key))
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    return class_token or metric_text


def _write_variable_metadata_manifest(
    directory: Path,
    *,
    rename_map: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> Path:
    rename_map = rename_map or {}
    metadata_path = output_path or (directory / "variable_metadata.parquet")
    try:
        _entries, by_id = gis_lookup.load_variable_metadata()
    except Exception:
        pd.DataFrame(columns=["id", "name", "exported_name", "category", "units", "value_type", "source_ids"]).to_parquet(
            metadata_path,
            index=False,
        )
        return metadata_path

    rows: list[dict[str, Any]] = []
    for variable_id, entry in by_id.items():
        category = _variable_category(variable_id)
        if category.strip().lower() == "recent weather":
            continue
        readable_name = _variable_label(variable_id)
        exported_name = rename_map.get(variable_id, str(variable_id))
        source_ids: list[str] = entry.get("source_ids") or []
        rows.append(
            {
                "id": str(variable_id),
                "name": readable_name,
                "exported_name": exported_name,
                "category": category,
                "units": entry.get("units"),
                "value_type": entry.get("value_type"),
                "source_ids": json.dumps(source_ids),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("id", kind="stable")
    frame.to_parquet(metadata_path, index=False)
    return metadata_path


def _variable_column_rename_map(column_names: list[str]) -> dict[str, str]:
    used_names = set(column_names)
    rename_map: dict[str, str] = {}
    for source_name in column_names:
        target_name = _variable_label(source_name)
        if not target_name or target_name == source_name:
            continue
        candidate = target_name
        suffix = 2
        while candidate in used_names and candidate != source_name:
            candidate = f"{target_name} ({suffix})"
            suffix += 1
        if candidate == source_name:
            continue
        rename_map[source_name] = candidate
        used_names.discard(source_name)
        used_names.add(candidate)
    return rename_map


def _apply_readable_variable_columns_to_parquet(parquet_path: Path) -> dict[str, str]:
    if not parquet_path.exists():
        return {}
    table = pq.read_table(parquet_path)
    original_names = list(table.schema.names)
    rename_map = _variable_column_rename_map(original_names)
    if not rename_map:
        return {}
    renamed_columns = [rename_map.get(name, name) for name in original_names]
    renamed_table = table.rename_columns(renamed_columns)
    pq.write_table(renamed_table, parquet_path)
    return rename_map


def _normalize_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "decimalLatitude" not in result.columns:
        latitude_match = _find_column_by_aliases(
            list(result.columns),
            ("decimalLatitude", "decimal_latitude", "latitude", "lat"),
        )
        if latitude_match is None:
            latitude_match = next(
                (column for column in result.columns if "latitude" in str(column).strip().lower()),
                None,
            )
        if latitude_match is not None:
            result = result.rename(columns={latitude_match: "decimalLatitude"})

    if "decimalLongitude" not in result.columns:
        longitude_match = _find_column_by_aliases(
            list(result.columns),
            ("decimalLongitude", "decimal_longitude", "longitude", "lon", "lng"),
        )
        if longitude_match is None:
            longitude_match = next(
                (column for column in result.columns if "longitude" in str(column).strip().lower()),
                None,
            )
        if longitude_match is not None:
            result = result.rename(columns={longitude_match: "decimalLongitude"})

    if "observationName" not in result.columns:
        name_match = _find_column_by_aliases(list(result.columns), _OBSERVATION_NAME_ALIASES)
        if name_match is not None:
            result = result.rename(columns={name_match: "observationName"})

    return result


def _add_tile_ids(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"decimalLatitude", "decimalLongitude"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise HTTPException(
            status_code=422,
            detail=(f"Uploaded file is missing required columns for tile lookup: {missing_list}"),
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
                f"Uploaded file has invalid coordinates in decimalLatitude/decimalLongitude for {invalid_count} row(s)."
            ),
        )

    result["decimalLatitude"] = latitudes
    result["decimalLongitude"] = longitudes
    result["tileId"] = [
        gis_lookup.get_region_name(float(lat), float(lon)) for lat, lon in zip(latitudes.tolist(), longitudes.tolist())
    ]
    return result


def _ensure_catalog_numbers(df: pd.DataFrame) -> pd.DataFrame:
    if "catalogNumber" in df.columns:
        return df
    result = df.copy()
    id_column = _find_column_by_aliases(list(result.columns), _CATALOG_NUMBER_ALIASES)
    if id_column is not None:
        return result.rename(columns={id_column: "catalogNumber"})
    result["catalogNumber"] = [f"Observation #{idx}" for idx in range(1, len(result) + 1)]
    return result


def _ensure_observation_names(df: pd.DataFrame) -> pd.DataFrame:
    if "observationName" in df.columns:
        result = df.copy()
        series = result["observationName"]
        missing_mask = series.isna() | (series.astype(str).str.strip() == "")
        if missing_mask.any():
            fallback = [f"Observation #{idx}" for idx in range(1, len(result) + 1)]
            result.loc[missing_mask, "observationName"] = pd.Series(fallback, index=result.index)[missing_mask]
        return result

    result = df.copy()
    result["observationName"] = [f"Observation #{idx}" for idx in range(1, len(result) + 1)]
    return result


def _build_internal_upload_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    conflicting_columns = []
    if "tileId" in result.columns:
        conflicting_columns.append("tileId")
    layer_catalog = _layer_catalog()
    conflicting_columns.extend(column for column in result.columns if layer_catalog.get(str(column)))
    if conflicting_columns:
        conflict_list = ", ".join(sorted({str(column) for column in conflicting_columns}))
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded file contains columns reserved for derived enrichment data: "
                f"{conflict_list}. Remove or rename those columns and try again."
            ),
        )
    return result


def _layer_catalog() -> dict[str, dict[str, Any]]:
    return gis_lookup.load_layer_metadata()


def _gis_variable_columns(column_names: list[Any]) -> list[str]:
    layer_catalog = _layer_catalog()
    return [str(column) for column in column_names if layer_catalog.get(str(column))]


def _build_export_occurrence_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    variable_columns = _gis_variable_columns(list(result.columns))
    base_columns = [column for column in result.columns if column not in variable_columns]
    selected_columns = base_columns + [column for column in variable_columns if column not in base_columns]
    return result[selected_columns].copy()


def _build_artifact_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    preserved_columns = [
        column
        for column in [
            "catalogNumber",
            "observationName",
            "decimalLatitude",
            "decimalLongitude",
            "tileId",
            "obscured",
            "coordinateUncertaintyInMeters",
        ]
        if column in result.columns
    ]
    variable_columns = _gis_variable_columns(list(result.columns))
    selected_columns = preserved_columns + [column for column in variable_columns if column not in preserved_columns]
    if not selected_columns:
        return result
    return result[selected_columns].copy()


def _add_gis_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required_columns = {"tileId", "decimalLatitude", "decimalLongitude", "catalogNumber"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Uploaded file is missing required columns for GIS enrichment: {', '.join(sorted(missing_columns))}"
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


def _apply_variable_display_scales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    scaled = df.copy()
    for column in scaled.select_dtypes(include=["number"]).columns:
        if summary_stats._layer_value_type(str(column)) == "categorical":
            continue
        factor = units.variable_display_scale(str(column))
        if factor == 1.0:
            continue
        scaled[column] = pd.to_numeric(scaled[column], errors="coerce") * factor
    return scaled


def _write_summary_artifacts_from_dataframe(df: pd.DataFrame, directory: Path) -> None:
    filtered = df.copy()
    variable_columns = set(_gis_variable_columns(list(filtered.columns)))
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
        if column in variable_columns and summary_stats._layer_value_type(column) == "categorical"
    ]
    categorical_entries = summary_stats._collect_categorical_stats(filtered, categorical_cols)
    for entry in categorical_entries:
        variable_id = entry.get("variable")
        metric = entry.get("metric")
        entry["metricLabel"] = _class_metric_to_readable_name(variable_id, metric)
        entry["variable"] = str(variable_id)
        entry["variableName"] = _variable_label(variable_id)
        entry["variableCategory"] = _variable_category(variable_id)
    summary_stats._write_categorical_stats(
        directory,
        categorical_entries,
        merge_existing=False,
    )

    numeric_cols = [
        column
        for column in filtered.select_dtypes(include=["number"]).columns
        if (
            column in variable_columns
            and column not in categorical_cols
            and column not in summary_stats.excluded_numeric_columns
            and summary_stats._layer_value_type(column) != "categorical"
        )
    ]
    stats: dict[str, dict[str, Any]] = {}
    density_rows: list[dict[str, Any]] = []
    for column in numeric_cols:
        series = pd.to_numeric(filtered[column], errors="coerce").dropna()
        if series.empty:
            continue
        stats[column] = summary_stats.summarize_layer_values_for_storage(column, series.tolist())

        values = series.astype(float).tolist()
        point_count = summary_stats._density_point_count(len(values))
        if summary_stats._is_circular_variable(column):
            curve = indexing.build_density_curve(values, point_count=point_count, circular=True)
        else:
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

    summary_path = directory / "summary_stats.parquet"
    if summary_path.exists():
        try:
            summary_frame = pd.read_parquet(summary_path)
            summary_frame = _add_variable_name_column(summary_frame)
            summary_frame = _add_variable_category_column(summary_frame)
            summary_frame.to_parquet(summary_path, index=False)
        except Exception:
            pass

    density_path = directory / summary_stats.density_graph_filename
    if density_rows:
        density_frame = pd.DataFrame(density_rows)
        density_frame = _add_variable_name_column(density_frame)
        density_frame = _add_variable_category_column(density_frame)
        density_frame.to_parquet(density_path, index=False)
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
    layer_catalog = _layer_catalog()
    targets = indexing.index_targets_for_columns(
        set(df.columns),
        layer_catalog=layer_catalog,
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
    metadata[b"origin_map"] = json.dumps([{"id": 0, "relative_path": ".", "taxon_key": "uploaded"}]).encode("utf-8")
    metadata[b"column_lengths"] = json.dumps(column_lengths).encode("utf-8")
    metadata[b"catalog_column"] = b"catalogNumber"
    metadata[b"category_offsets"] = json.dumps(category_offsets).encode("utf-8")
    index_table = index_table.replace_schema_metadata(metadata)
    pq.write_table(index_table, index_path)


def _write_categorical_value_lookup_from_dataframe(df: pd.DataFrame, directory: Path) -> Path:
    lookup_path = directory / "categorical_value_lookup.parquet"
    categorical_columns = [
        column
        for column in _gis_variable_columns(list(df.columns))
        if summary_stats._layer_value_type(column) == "categorical"
    ]
    if not categorical_columns:
        lookup_path.unlink(missing_ok=True)
        return lookup_path

    rows: list[dict[str, Any]] = []
    for variable_id in categorical_columns:
        legend = gis_lookup.load_layer_legend(variable_id)
        seen_codes: set[str] = set()
        for legend_key, entry in legend.items():
            if not isinstance(entry, dict):
                continue
            class_id = entry.get("id")
            if class_id is None:
                continue
            code_key = str(class_id)
            if code_key in seen_codes:
                continue
            seen_codes.add(code_key)
            rows.append(
                {
                    "variable": str(variable_id),
                    "variableName": _variable_label(variable_id),
                    "variableCategory": _variable_category(variable_id),
                    "code": int(class_id) if isinstance(class_id, int | bool) or str(class_id).isdigit() else class_id,
                    "metric": f"class_{class_id}",
                    "label": entry.get("name"),
                    "description": entry.get("description"),
                    "group": entry.get("group"),
                    "groupLabel": entry.get("group_label"),
                }
            )

    if not rows:
        lookup_path.unlink(missing_ok=True)
        return lookup_path

    pd.DataFrame(rows).sort_values(["variable", "code"], kind="stable").to_parquet(lookup_path, index=False)
    return lookup_path


def _build_index_archive(df: pd.DataFrame) -> tuple[Path, str, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-uploaded_0_"))
    occurrence_path = work_dir / CONFIG.occurrence_parquet_filename
    index_path = work_dir / "occurrence_index.parquet"
    archive_name = "processed_observations.zip"
    archive_path = work_dir / archive_name

    scaled_df = _apply_variable_display_scales(df)
    artifact_df = _build_artifact_dataframe(scaled_df)
    export_df = _build_export_occurrence_dataframe(scaled_df)
    export_df.to_parquet(occurrence_path, index=False)

    try:
        _write_summary_artifacts_from_dataframe(artifact_df, work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build summary stat artifacts: {exc}",
        ) from exc

    try:
        _write_occurrence_index_from_dataframe(artifact_df, work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build occurrence index parquet: {exc}",
        ) from exc

    try:
        categorical_lookup_path = _write_categorical_value_lookup_from_dataframe(artifact_df, work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build categorical value lookup parquet: {exc}",
        ) from exc

    occurrence_rename_map: dict[str, str] = {}
    variable_metadata_path = work_dir / "variable_metadata.parquet"
    staged_occurrence_path = work_dir / f"{occurrence_path.stem}.readable.parquet"
    staged_metadata_path = work_dir / "variable_metadata.readable.parquet"
    try:
        shutil.copy2(occurrence_path, staged_occurrence_path)
        occurrence_rename_map = _apply_readable_variable_columns_to_parquet(staged_occurrence_path)
        _write_variable_metadata_manifest(
            work_dir,
            rename_map=occurrence_rename_map,
            output_path=staged_metadata_path,
        )
        staged_occurrence_path.replace(occurrence_path)
        staged_metadata_path.replace(variable_metadata_path)
    except Exception:
        staged_occurrence_path.unlink(missing_ok=True)
        staged_metadata_path.unlink(missing_ok=True)
        variable_metadata_path = _write_variable_metadata_manifest(work_dir, rename_map={})

    parquet_paths = [
        (occurrence_path, CONFIG.occurrence_parquet_filename),
        (work_dir / "summary_stats.parquet", "summary_stats.parquet"),
        (work_dir / "categorical_stats.parquet", "categorical_stats.parquet"),
        (work_dir / summary_stats.density_graph_filename, summary_stats.density_graph_filename),
        (variable_metadata_path, "variable_metadata.parquet"),
    ]
    if categorical_lookup_path.exists():
        parquet_paths.append((categorical_lookup_path, categorical_lookup_path.name))
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
        try:
            data_sources = gis_lookup.load_data_sources()
            archive.writestr("data_sources.json", json.dumps(data_sources))
        except Exception:
            pass

    return archive_path, archive_name, work_dir
