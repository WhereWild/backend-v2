from __future__ import annotations
import io
import math
import shutil
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import numpy as np
import pandas as pd

from util.config import load_config
from util import custom_upload_processing, descriptions, gis_lookup, indexing, summary_stats, taxa_navigation, units, tiles
from util.storage import get_parquet_storage

CONFIG = load_config("global")

api_title = "WhereWild API"

api_version = "0.2.0"

category_sample_limit = 500

cors_allow_headers = ("*",)

cors_allow_methods = ("GET", "POST")

cors_allow_origins = ("*",)

density_points = 128

forced_categorical_variables = frozenset({"landcover"})

default_species_limit = 12

max_species_limit = 100
variable_tile_default_size = int(getattr(CONFIG, "sdm_tile_size", 256))
variable_tile_max_size = int(getattr(CONFIG, "sdm_tile_max_size", 2048))
variable_tile_cache_seconds = int(getattr(CONFIG, "sdm_tile_cache_seconds", 60))
variable_tile_default_reproject = bool(getattr(CONFIG, "sdm_tile_reproject", True))
derived_tile_variables = frozenset({"slope", "aspect", "aspect_deg"})



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        gis_lookup.preload_layer_legends()
    except FileNotFoundError:
        # Allow API to start even if GIS catalog/legends are not present yet.
        pass
    except OSError:
        # Remote/object storage might be unavailable at startup; defer to first request.
        pass
    yield


app = FastAPI(title=api_title, version=api_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cors_allow_origins),
    allow_methods=list(cors_allow_methods),
    allow_headers=list(cors_allow_headers),
)
def _path_exists(path: Path) -> bool:
    storage = get_parquet_storage(CONFIG.data_root, CONFIG.project_root)
    if storage.is_remote:
        return storage.exists(path)
    return path.exists()


@lru_cache(maxsize=1)
def _map_enabled_variables() -> frozenset[str]:
    """Return layer ids currently eligible for variable tile rendering."""
    try:
        layers = gis_lookup.load_layer_metadata()
    except Exception:
        return frozenset({"landcover", "koppen_geiger"})

    enabled: set[str] = set()
    for layer_id, meta in layers.items():
        if not layer_id:
            continue
        is_derived = bool(meta.get("derived"))
        if is_derived and layer_id not in derived_tile_variables:
            continue
        value_type = str(meta.get("value_type") or "").strip().lower()
        if value_type not in {"numeric", "categorical"}:
            continue
        region_root = str(meta.get("region_root") or "").strip()
        filename_template = str(meta.get("filename_template") or "").strip()
        if not region_root or not filename_template:
            continue
        if is_derived and filename_template != "dem.tif":
            continue
        enabled.add(str(layer_id))

    if not enabled:
        enabled.update({"landcover", "koppen_geiger"})
    return frozenset(sorted(enabled))


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

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(occurrence_path, arcname=CONFIG.occurrence_parquet_filename)
        if index_path.exists():
            archive.write(index_path, arcname=index_path.name)
        summary_stats_path = work_dir / "summary_stats.parquet"
        categorical_stats_path = work_dir / "categorical_stats.parquet"
        density_graph_path = work_dir / summary_stats.density_graph_filename
        if summary_stats_path.exists():
            archive.write(summary_stats_path, arcname=summary_stats_path.name)
        if categorical_stats_path.exists():
            archive.write(categorical_stats_path, arcname=categorical_stats_path.name)
        if density_graph_path.exists():
            archive.write(density_graph_path, arcname=density_graph_path.name)

    return archive_path, archive_name, work_dir


@app.get("/health", summary="Simple liveness probe")
def health_check() -> dict[str, str]:
    """Returns a simple liveness payload.
    
    Returns:
        A status string and UTC timestamp.
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/variables")
def list_environment_variables(
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> List[dict[str, Any]]:
    """Lists available environmental variables.
    
    Returns:
        A list of variable metadata entries.
    """
    return units.apply_unit_system_to_variables(
        gis_lookup.load_variable_metadata()[0],
        unit_system,
    )


@app.get("/api/variables/{variable_id}/tiles/{z}/{x}/{y}.png")
async def variable_tile(
    request: Request,
    variable_id: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = Query(variable_tile_default_size, ge=32, le=variable_tile_max_size),
    reproject: bool = Query(
        variable_tile_default_reproject,
        description="If true, warp to Web Mercator; if false, keep WGS84.",
    ),
    max_native_zoom: int = Query(
        10,
        ge=1,
        le=18,
        description="Max zoom to render natively. Higher zooms extract subtiles from this zoom.",
    ),
) -> Response:
    """Render a variable tile using the same overview + tile extraction flow as SDM tiles."""
    if await request.is_disconnected():
        return Response(status_code=204)

    layer_id = (variable_id or "").strip().lower()
    if not layer_id:
        raise HTTPException(status_code=400, detail="variable_id is required.")
    enabled_variables = _map_enabled_variables()
    if layer_id not in enabled_variables:
        allowed = ", ".join(sorted(enabled_variables))
        raise HTTPException(
            status_code=400,
            detail=f"Variable tiles currently support: {allowed}.",
        )
    if layer_id not in gis_lookup.load_layer_metadata():
        raise HTTPException(status_code=404, detail=f"Unknown variable '{layer_id}'.")

    if z > max_native_zoom:
        zoom_diff = z - max_native_zoom
        scale = 2 ** zoom_diff
        parent_x = x // scale
        parent_y = y // scale
        subtile_x = x % scale
        subtile_y = y % scale
        parent_tile_size = min(tile_size * scale, variable_tile_max_size)
        try:
            if await request.is_disconnected():
                return Response(status_code=204)
            parent_payload = await run_in_threadpool(
                tiles.render_variable_tile_bytes,
                variable_id=layer_id,
                z=max_native_zoom,
                x=parent_x,
                y=parent_y,
                tile_size=parent_tile_size,
                reproject=reproject,
            )
            if await request.is_disconnected():
                return Response(status_code=204)
            from PIL import Image
            import io

            parent_img = Image.open(io.BytesIO(parent_payload))
            subtile_size = parent_tile_size // scale
            left = subtile_x * subtile_size
            top = subtile_y * subtile_size
            subtile_img = parent_img.crop((left, top, left + subtile_size, top + subtile_size))
            if subtile_size != tile_size:
                subtile_img = subtile_img.resize((tile_size, tile_size), Image.LANCZOS)
            buffer = io.BytesIO()
            subtile_img.save(buffer, format="PNG")
            payload = buffer.getvalue()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        try:
            if await request.is_disconnected():
                return Response(status_code=204)
            payload = await run_in_threadpool(
                tiles.render_variable_tile_bytes,
                variable_id=layer_id,
                z=z,
                x=x,
                y=y,
                tile_size=tile_size,
                reproject=reproject,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if await request.is_disconnected():
        return Response(status_code=204)
    headers = {
        "Cache-Control": f"public, max-age={variable_tile_cache_seconds}",
    }
    return Response(content=payload, media_type="image/png", headers=headers)


@app.get("/api/species")
def list_species(
    q: str = Query(..., min_length=1, description="Search term (scientific name or common name)"),
    limit: int = Query(default_species_limit, ge=1, le=max_species_limit),
) -> List[dict[str, Any]]:
    """Searches taxa by name and returns serialized results.
    
    Args:
        q: Search term for scientific or common names.
        limit: Maximum number of matches to return.
    
    Returns:
        A list of serialized taxon payloads.
    """
    records = taxa_navigation.search_taxa_by_name(q, limit=limit)

    payloads: list[dict[str, Any]] = []
    for record, _score, matched_name in records:
        payload = taxa_navigation.serialize_taxon(record)
        if payload:
            common_names = payload.get("common_names") or []
            matched_common_name = taxa_navigation.resolve_matched_common_name(
                common_names,
                matched_name,
            )
            payload["matched_common_name"] = matched_common_name
            payloads.append(payload)
    return payloads


@app.get("/api/species/{taxon_id}")
def get_species_detail(
    taxon_id: int,
    location: Optional[str] = Query(
        None, description="Optional location GID to tailor description text."
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for description values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Loads a single taxon record by id.
    
    Args:
        taxon_id: Taxon id to look up.
        location: Optional location GID filter for location text context.
    
    Returns:
        A serialized taxon payload.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    payload = taxa_navigation.serialize_taxon(taxon) if taxon else None
    if not payload:
        raise HTTPException(
            status_code=404,
            detail=f"Species with taxon_id {taxon_id} not found",
        )
    location_gid = location.strip() if location else None
    try:
        description_profile = descriptions.build_taxon_description(
            taxon,
            location_gid=location_gid,
            unit_system=unit_system,
        )
        text = description_profile.get("text")
        if isinstance(text, str) and text.strip():
            payload["description"] = text
        payload["description_profile"] = description_profile
    except Exception as exc:
        print(f"[description] failed for taxon_id={taxon_id}: {exc}")
        traceback.print_exc()
    return payload


@app.get("/locations/search")
def search_locations_endpoint(
    q: str = Query(..., min_length=1, description="Location name or partial match"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Searches locations by name substring.
    
    Args:
        q: Search term for location names.
        limit: Maximum number of matches to return.
    
    Returns:
        A dict containing location match results.
    """
    matches = gis_lookup.search_locations(q, limit)
    return {"results": matches}


@app.get("/species/{taxon_id}/occurrences")
def species_occurrences(
    taxon_id: int,
    location: Optional[str] = Query(None, description="Filter observations by location gid"),
) -> dict[str, Any]:
    """Returns occurrence points for a taxon, optionally filtered by location.
    
    Args:
        taxon_id: Taxon id to query.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict with occurrence count and point records.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    if not _path_exists(Path(taxon["path"])):
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    normalized_location = location.strip() if location else None
    if normalized_location and not gis_lookup.is_valid_location_gid(normalized_location):
        return {
            "speciesId": taxon_id,
            "count": 0,
            "occurrences": [],
        }
    rows = taxa_navigation.load_occurrence_points(
        taxon_id,
        normalized_location,
    )
    return {
        "speciesId": taxon_id,
        "count": len(rows),
        "occurrences": rows,
    }

@app.get("/species/{taxon_id}/locations")
def species_locations(
    taxon_id: int,
    level: Optional[str] = Query(None, description="continent|country|state|county"),
    parent: Optional[str] = Query(None, description="Parent location GID (optional)"),
    limit: int = Query(500, ge=1, le=5000),
) -> List[dict[str, Any]]:
    """Returns locations where the species is present using precomputed membership."""
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    if not _path_exists(Path(taxon["path"])):
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    target_taxon_id = taxa_navigation.taxon_id_as_int(str(taxon["taxon_key"]))
    if target_taxon_id is None:
        return []

    level_map = {"continent": -1, "country": 0, "state": 1, "county": 2}
    expected_level: int | None = None
    if level is not None:
        try:
            expected_level = int(level)
        except (TypeError, ValueError):
            expected_level = level_map.get(str(level).lower())

    entries, by_gid = gis_lookup.load_location_catalog()
    if not entries:
        return []

    level_by_scope = {
        str(scope): int(level_idx)
        for level_idx, scope in CONFIG.location_scope_by_level.items()
    }
    level_by_scope["gbif_region"] = -1

    parent_tokens = [token.strip() for token in (parent or "").split("|") if token.strip()]
    records_by_lower_name: dict[str, list[gis_lookup.LocationRecord]] = {}
    for record in entries:
        records_by_lower_name.setdefault(record.name.lower(), []).append(record)

    parent_matchers: list[tuple[set[str], set[str]]] = []
    for token in parent_tokens:
        name_options = {token.lower()}
        gid_options = {token.lower()}
        by_gid_record = by_gid.get(token) or by_gid.get(token.upper())
        if by_gid_record is not None:
            name_options.add(by_gid_record.name.lower())
            gid_options.add(by_gid_record.gid.lower())
        for named_record in records_by_lower_name.get(token.lower(), []):
            name_options.add(named_record.name.lower())
            gid_options.add(named_record.gid.lower())
        parent_matchers.append((name_options, gid_options))

    ancestor_gid_cache: dict[str, set[str]] = {}

    def ancestor_gids_for(record: gis_lookup.LocationRecord) -> set[str]:
        cached = ancestor_gid_cache.get(record.gid)
        if cached is not None:
            return cached
        chain: set[str] = set()
        seen: set[str] = set()
        current = record.parent_gid
        while current:
            current_key = str(current)
            if current_key in seen:
                break
            seen.add(current_key)
            chain.add(current_key.lower())
            parent_record = by_gid.get(current_key)
            if parent_record is None:
                break
            current = parent_record.parent_gid
        ancestor_gid_cache[record.gid] = chain
        return chain

    def matches_parent(
        gid: str,
        name: str,
        hierarchy_names: list[str],
        hierarchy_gids: set[str],
    ) -> bool:
        if not parent_matchers:
            return True
        cand_gid = gid.lower()
        cand_name = name.lower()
        hierarchy_name_set = {item.lower() for item in hierarchy_names}
        for name_options, gid_options in parent_matchers:
            name_match = (
                bool(name_options & hierarchy_name_set)
                or cand_name in name_options
            )
            gid_match = cand_gid in gid_options or bool(gid_options & hierarchy_gids)
            if not (name_match or gid_match):
                return False
        return True

    location_counts = gis_lookup.location_counts_for_taxon(target_taxon_id)
    if not location_counts:
        return []

    results: list[dict[str, Any]] = []
    seen_gids: set[str] = set()
    for (scope, gid), count in location_counts.items():
        location_level = level_by_scope.get(str(scope))
        if location_level is None:
            continue
        if expected_level is not None and location_level != expected_level:
            continue
        gid_key = str(gid)
        if not gis_lookup.is_valid_location_gid(gid_key):
            continue
        if gid_key in seen_gids:
            continue
        seen_gids.add(gid_key)

        record = by_gid.get(gid_key)
        if record is not None:
            location_name = record.name
            hierarchy = gis_lookup.resolve_location_context(record, by_gid)
            hierarchy_gids = ancestor_gids_for(record)
        else:
            location_name = gid_key
            hierarchy = []
            hierarchy_gids = set()

        if not matches_parent(gid_key, location_name, hierarchy, hierarchy_gids):
            continue

        results.append(
            {
                "gid": gid_key,
                "name": location_name,
                "level": location_level,
                "hierarchy": hierarchy,
                "count": int(count),
            }
        )

    results.sort(
        key=lambda item: (
            -int(item.get("count", 0)),
            str(item.get("name", "")).lower(),
            str(item.get("gid", "")),
        )
    )
    if limit and len(results) > limit:
        return results[:limit]
    return results

@app.get("/locations/search_hierarchy")
def search_locations_by_hierarchy(
    q: str = Query("", description="Location name or partial match (optional if parent provided)"),
    level: Optional[str] = Query(None, description="continent|country|state|county or numeric level code"),
    parent: Optional[str] = Query(None, description="Parent name or gid. For counties pass 'United States|Utah' or a gid."),
    limit: int = Query(50, ge=1, le=1000),
) -> dict[str, Any]:

    q = (q or "").strip()

    level_map = {"continent": -1, "country": 0, "state": 1, "county": 2}

    expected_level = None
    if level is not None:
        try:
            expected_level = int(level)
        except Exception:
            expected_level = level_map.get(level.lower())

    parents_raw = (parent or "").strip()
    parent_tokens = [p.strip() for p in parents_raw.split("|") if p.strip()]

    resolved_parent_names: list[str] = []
    resolved_parent_gids: list[str] = []
    for tok in parent_tokens:
        resolved_name = tok
        resolved_gid = tok
        try:
            if hasattr(gis_lookup, "get_location_by_gid"):
                maybe = gis_lookup.get_location_by_gid(tok)
                if maybe:
                    resolved_name = maybe.get("name", tok)
                    resolved_gid = maybe.get("gid", tok)
        except Exception:
            pass
        resolved_parent_names.append(str(resolved_name).lower())
        resolved_parent_gids.append(str(resolved_gid).lower())

    if not q and not parent_tokens and expected_level is None:
        return {"results": []}

    candidates: list[dict[str, Any]] = []
    seen_gids = set()

    def matches_parent(cand: dict[str, Any]) -> bool:
        # if no parent requested, everything matches
        if not resolved_parent_names:
            return True
        cand_hierarchy = [str(x).lower() for x in (cand.get("hierarchy") or []) if x is not None]
        cand_name = str(cand.get("name") or "").lower()
        cand_gid = str(cand.get("gid") or "").lower()
        for pname, pgid in zip(resolved_parent_names, resolved_parent_gids):
            if pname in cand_hierarchy or pname == cand_name or pgid == cand_gid or pgid in cand_hierarchy:
                continue
            return False
        return True

    def push_candidate_if_valid(cand: dict[str, Any]):
        gid = str(cand.get("gid") or "")
        if not gid or gid in seen_gids:
            return
        # enforce parent matching here (critical fix)
        if not matches_parent(cand):
            return
        seen_gids.add(gid)
        candidates.append(cand)

    try:
        if q:
            raw = gis_lookup.search_locations(q, limit)
            for cand in raw:
                push_candidate_if_valid(cand)

        else:
            # 1) catalog-based enumeration (fast)
            if expected_level is not None and hasattr(gis_lookup, "load_location_catalog"):
                try:
                    entries, mapping = gis_lookup.load_location_catalog()
                    for rec in entries:
                        if getattr(rec, "level", None) != expected_level:
                            continue

                        # build hierarchy names
                        hierarchy = []
                        parent_gid = getattr(rec, "parent_gid", None)
                        while parent_gid:
                            parent_rec = mapping.get(parent_gid)
                            if not parent_rec:
                                break
                            hierarchy.append(parent_rec.name)
                            parent_gid = parent_rec.parent_gid

                        cand = {
                            "gid": rec.gid,
                            "name": rec.name,
                            "level": rec.level,
                            "hierarchy": list(reversed(hierarchy)),
                        }
                        push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break
                except Exception:
                    pass

            # 2) list_children if available
            if not candidates and hasattr(gis_lookup, "list_children"):
                for parent_tok in parent_tokens or []:
                    try:
                        parent_gid = None
                        if hasattr(gis_lookup, "get_location_by_gid"):
                            maybe = gis_lookup.get_location_by_gid(parent_tok)
                            if maybe:
                                parent_gid = maybe.get("gid")
                        raw = gis_lookup.list_children(parent_gid or parent_tok, level=expected_level, limit=limit * 3)
                        for cand in raw:
                            push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break
                    except Exception:
                        continue

            # 3) letter-scan fallback — keep scanning letters until we have enough valid matches
            if not candidates:
                letters = "abcdefghijklmnopqrstuvwxyz"
                per_letter_limit = max(50, min(200, limit))
                for ch in letters:
                    if len(candidates) >= limit:
                        break
                    try:
                        partial = gis_lookup.search_locations(ch, per_letter_limit)
                    except Exception:
                        continue
                    for cand in partial:
                        push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break

    except Exception:
        return {"results": []}

    # final strict filter by level (redundant but safe)
    results: list[dict[str, Any]] = []
    for cand in candidates:
        if expected_level is not None and cand.get("level") != expected_level:
            continue
        results.append({
            "gid": str(cand.get("gid") or ""),
            "name": cand.get("name") or "",
            "level": cand.get("level", -999),
            "hierarchy": cand.get("hierarchy") or [],
        })
        if len(results) >= limit:
            break

    return {"results": results}

@app.get("/species/{taxon_id}/environment/{variable_id}")
def species_environment_stats(
    taxon_id: int,
    variable_id: str,
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns environment stats for a taxon and variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Environmental variable id.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing summary stats, distributions, and rankings.
    """
    variable_id = variable_id.strip()
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable_id)
    if not variable_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        )
    raw_units = variable_entry.get("units")
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not _path_exists(taxon_dir):
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    location_gid = location.strip() if location else None
    value_type = str(variable_entry.get("value_type") or "").lower() or "numeric"
    forced_categorical = variable_id.lower() in forced_categorical_variables
    categorical_payload = None
    if forced_categorical or value_type == "categorical":
        if location_gid:
            categorical_payload = summary_stats.build_categorical_stats_for_location(
                taxon_id,
                variable_id,
                location_gid,
                sample_limit=category_sample_limit,
            )
            if categorical_payload is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No samples available for taxon {taxon_id}, "
                        f"variable '{variable_id}' and location '{location_gid}'."
                    ),
                )
            value_type = "categorical"
        else:
            categorical_payload = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
            if categorical_payload is None and forced_categorical:
                value_type = "categorical"
            elif categorical_payload is not None:
                value_type = "categorical"
    generated_at = datetime.now(timezone.utc).isoformat()

    baseline_numeric_summary = None
    baseline_categorical_distribution: list[dict[str, Any]] = []
    baseline_categorical_totals: dict[str, Any] = {}

    if categorical_payload:
        if location_gid:
            baseline_stats = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
            if baseline_stats:
                baseline_categorical_distribution = baseline_stats.get("distribution", [])
                baseline_categorical_totals = baseline_stats.get("totals", {})
        totals = categorical_payload.get("totals", {})
        total_samples = totals.get("total_samples") or 0
        summary = {
            "count": int(total_samples),
            "min": None,
            "mean": None,
            "max": None,
            "stddev": None,
            "q01": None,
            "q10": None,
            "q90": None,
            "q99": None,
        }
        if location_gid:
            ranks = []
            print(
                f"[timing][env] taxon_id={taxon_id} variable={variable_id} "
                f"location={location_gid} step=relative_ranks skipped=1 reason=location_filter"
            )
        else:
            ranks = indexing.load_relative_ranks(taxon_dir, variable_id)
        response = {
            "speciesId": taxon_id,
            "species_id": taxon_id,
            "variable": variable_id,
            "variableName": variable_entry.get("name"),
            "variable_metadata": {
                "name": variable_entry.get("name"),
                "units": raw_units,
                "value_type": "categorical",
            },
            "units": raw_units,
            "variableType": "categorical",
            "generatedAt": generated_at,
            "generated_at": generated_at,
            "summary": summary,
            "histogram": None,
            "densityCurve": None,
            "binSamples": [],
            "bin_samples": [],
            "density_curve": None,
            "categoricalDistribution": categorical_payload.get("distribution", []),
            "categorical_distribution": categorical_payload.get("distribution", []),
            "dominantCategories": categorical_payload.get("dominant", []),
            "dominant_categories": categorical_payload.get("dominant", []),
            "baselineCategoricalDistribution": baseline_categorical_distribution,
            "baseline_categorical_distribution": baseline_categorical_distribution,
            "baselineCategoricalTotals": baseline_categorical_totals,
            "baseline_categorical_totals": baseline_categorical_totals,
            "baselineSummary": baseline_numeric_summary,
            "baseline_summary": baseline_numeric_summary,
            "relativeRanks": ranks,
            "relative_ranks": ranks,
        }
        return units.apply_unit_system_to_env_response(response, unit_system, raw_units)

    if not location_gid:
        summary = summary_stats.load_numeric_summary(str(taxon_dir), variable_id)
        density_curve = summary_stats.load_density_graph(str(taxon_dir), variable_id)
        if not summary or not density_curve:
            raise HTTPException(
                status_code=503,
                # We COULD compute on-demand here but I think it's better to fail loudly as the data *should* be here for performance reasons.
                detail=(
                    f"Precomputed summary stats or KDE missing (summary={bool(summary)} "
                    f"density={bool(density_curve)}). "
                    "Rebuild summary_stats.parquet and density_graph.parquet."
                ),
            )
        ranks = indexing.load_relative_ranks(taxon_dir, variable_id)
        response = {
            "speciesId": taxon_id,
            "species_id": taxon_id,
            "variable": variable_id,
            "variableName": variable_entry.get("name"),
            "variable_metadata": {
                "name": variable_entry.get("name"),
                "units": variable_entry.get("units"),
                "value_type": value_type or "numeric",
            },
            "units": variable_entry.get("units"),
            "variableType": value_type or "numeric",
            "generatedAt": generated_at,
            "generated_at": generated_at,
            "summary": summary,
            "histogram": None,
            "densityCurve": density_curve,
            "binSamples": [],
            "bin_samples": [],
            "density_curve": density_curve,
            "baselineSummary": baseline_numeric_summary,
            "baseline_summary": baseline_numeric_summary,
            "baselineCategoricalDistribution": [],
            "baseline_categorical_distribution": [],
            "baselineCategoricalTotals": {},
            "baseline_categorical_totals": {},
            "categoricalDistribution": [],
            "categorical_distribution": [],
            "dominantCategories": [],
            "dominant_categories": [],
            "relativeRanks": ranks,
            "relative_ranks": ranks,
        }
        return units.apply_unit_system_to_env_response(response, unit_system, raw_units)

    samples = summary_stats.gather_numeric_records(
        taxon_id,
        taxon_dir,
        variable_id,
        location_gid=location_gid,
    )
    values = [sample["value"] for sample in samples]
    if not values:
        raise HTTPException(
            status_code=404,
            detail=f"No samples available for taxon {taxon_id} and variable '{variable_id}'.",
        )
    summary = summary_stats.summarize_values(values)
    density_curve = indexing.build_density_curve(values, point_count=density_points)
    ranks = []
    print(
        f"[timing][env] taxon_id={taxon_id} variable={variable_id} "
        f"location={location_gid} step=relative_ranks skipped=1 reason=location_filter"
    )
    response = {
        "speciesId": taxon_id,
        "species_id": taxon_id,
        "variable": variable_id,
        "variableName": variable_entry.get("name"),
        "variable_metadata": {
            "name": variable_entry.get("name"),
            "units": raw_units,
            "value_type": value_type or "numeric",
        },
        "units": raw_units,
        "variableType": value_type or "numeric",
        "generatedAt": generated_at,
        "generated_at": generated_at,
        "summary": summary,
        "histogram": None,
        "densityCurve": density_curve,
        "binSamples": [],
        "bin_samples": [],
        "density_curve": density_curve,
        "baselineSummary": baseline_numeric_summary,
        "baseline_summary": baseline_numeric_summary,
        "baselineCategoricalDistribution": [],
        "baseline_categorical_distribution": [],
        "baselineCategoricalTotals": {},
        "baseline_categorical_totals": {},
        "categoricalDistribution": [],
        "categorical_distribution": [],
        "dominantCategories": [],
        "dominant_categories": [],
        "relativeRanks": ranks,
        "relative_ranks": ranks,
    }
    return units.apply_unit_system_to_env_response(response, unit_system, raw_units)


@app.get("/species/{taxon_id}/environment/{variable_id}/class/{class_value}/samples")
def species_environment_class_samples(
    taxon_id: int,
    variable_id: str,
    class_value: str,
    limit: int | None = Query(None, ge=1, le=10000),
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
) -> dict[str, Any]:
    """Returns categorical class samples for a taxon and variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Categorical variable id.
        class_value: Class value to match.
        limit: Maximum number of samples to return.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing matching observation samples.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not _path_exists(taxon_dir):
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    try:
        parsed_value: float | int | str
        parsed_value = float(class_value)
        if parsed_value.is_integer():
            parsed_value = int(parsed_value)
    except ValueError:
        parsed_value = class_value
    location_gid = location.strip() if location else None
    observations: list[dict[str, Any]] = []
    if location_gid:
        observations = summary_stats.categorical_class_samples_for_location(
            taxon_id,
            variable_id,
            parsed_value,
            location_gid=location_gid,
            limit=limit,
        )
    else:
        index_path = taxon_dir / "occurrence_index.parquet"
        if not _path_exists(index_path):
            raise HTTPException(
                status_code=503,
                detail="GIS lookup utilities are unavailable on this server.",
            )
        try:
            rows = summary_stats.get_layer_records_for_class(index_path, variable_id, parsed_value)
        except Exception as exc:  # pragma: no cover - passthrough
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if limit is not None and limit > 0:
            rows = rows[:limit]
        observations = [
            {
                "catalogNumber": row[0],
                "latitude": row[1],
                "longitude": row[2],
                "value": row[3],
            }
            for row in rows
        ]
    return {
        "speciesId": taxon_id,
        "variable": variable_id,
        "classValue": parsed_value,
        "observations": observations,
        "count": len(observations),
    }


@app.get("/species/{taxon_id}/environment/{variable_id}/slice")
def species_environment_slice(
    taxon_id: int,
    variable_id: str,
    min_value: float = Query(..., alias="min"),
    max_value: float = Query(..., alias="max"),
    limit: int | None = Query(None, ge=1, le=10000),
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns numeric samples within a value range for a taxon/variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Numeric variable id.
        min_value: Minimum value to include.
        max_value: Maximum value to include.
        limit: Maximum number of samples to return.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing range parameters and matching observations.
    """
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise HTTPException(status_code=400, detail="min and max must be finite numbers")
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable_id)
    if not variable_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        )
    value_type = str(variable_entry.get("value_type") or "").lower() or "numeric"
    raw_units = variable_entry.get("units")
    min_value = units.convert_value_from_display(min_value, variable_id)
    max_value = units.convert_value_from_display(max_value, variable_id)
    resolved_unit_system = units.normalize_unit_system(unit_system)
    if resolved_unit_system and raw_units:
        min_value = units.convert_value_from_system(min_value, raw_units, resolved_unit_system)
        max_value = units.convert_value_from_system(max_value, raw_units, resolved_unit_system)
    if value_type == "categorical" or variable_id.lower() in forced_categorical_variables:
        raise HTTPException(
            status_code=400,
            detail="Categorical layers must be queried via the class samples endpoint.",
        )
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not _path_exists(taxon_dir):
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    index_path = taxon_dir / "occurrence_index.parquet"
    if not _path_exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Index parquet missing for taxon {taxon_id}",
        )
    location_gid = location.strip() if location else None
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    if location_gid:
        rows = summary_stats.numeric_range_samples_for_location(
            taxon_id,
            variable_id,
            min_value,
            max_value,
            location_gid=location_gid,
            limit=limit,
        )
    else:
        try:
            rows = summary_stats.get_sorted_layer_records_in_value_range(
                index_path,
                variable_id,
                value_min=min_value,
                value_max=max_value,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    observations: list[dict[str, Any]] = []
    for catalog, lat, lon, value in rows:
        observations.append(
            {
                "catalogNumber": catalog,
                "value": float(value) if isinstance(value, (int, float)) else value,
                "latitude": lat,
                "longitude": lon,
            }
        )
    response = {
        "speciesId": taxon_id,
        "variable": variable_id,
        "range": {"min": min_value, "max": max_value},
        "units": raw_units,
        "limit": limit,
        "count": len(observations),
        "observations": observations,
    }
    return units.apply_unit_system_to_slice_response(response, unit_system, raw_units)


@app.get("/relative-rankings/{taxon_id}")
def get_relative_rankings(
    taxon_id: int,
    rank: str = Query(..., description="Descendant rank to include (e.g., SPECIES)"),
    variable: str = Query(..., description="Environmental variable / layer id"),
    metric: str = Query(..., description="Metric to rank by (min, mean, max, std, 1-99 range)"),
    limit: int = Query(50, ge=1, le=200),
    order: str = Query("asc", description="Sort order: asc or desc"),
    min_samples: int = Query(0, ge=0, description="Minimum samples required to appear"),
    include_species_like: bool = Query(
        False, description="When rank=SPECIES, include subspecies/varieties/forms"
    ),
    include_distribution: bool = Query(
        False,
        description=(
            "Include the kernel density distribution for all eligible descendants. "
            "This can be expensive for large taxa."
        ),
    ),
    location: Optional[str] = Query(
        None,
        description="Optional location GID (GADM) or GBIF region to filter descendants by",
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns descendant rankings for a taxon by variable/metric.
    
    Args:
        taxon_id: Ancestor taxon id to rank descendants under.
        rank: Descendant rank to include.
        variable: Environmental variable id to rank by.
        metric: Metric name to rank by.
        limit: Maximum number of results to return.
        order: Sort order ("asc" or "desc").
        min_samples: Minimum sample count required to appear.
        include_species_like: Whether to include subspecies-like ranks for species.
        include_distribution: Whether to return raw values for density curves.
        location: Optional location GID to filter descendants by occurrence membership.
    
    Returns:
        A dict containing ranking entries and optional distribution data.
    """
    location_gid = location.strip() if location else None
    try:
        entries, distribution_values = indexing.child_relative_rankings(
            str(taxon_id),
            rank,
            variable,
            metric,
            limit=limit,
            order=order,
            min_samples=min_samples,
            include_species_like=include_species_like,
            return_distribution=include_distribution,
            location_gid=location_gid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total = entries[0]["count"] if entries else 0
    distribution_curve = None
    if include_distribution and distribution_values:
        distribution_curve = indexing.build_density_curve(
            distribution_values,
            point_count=density_points,
        )
    raw_units = None
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable)
    if variable_entry:
        raw_units = variable_entry.get("units")
    response = {
        "ancestor_taxon_id": taxon_id,
        "rank": rank.upper(),
        "variable": variable,
        "metric": metric,
        "units": raw_units,
        "total": total,
        "limit": limit,
        "order": order.lower(),
        "min_samples": min_samples,
        "include_species_like": include_species_like,
        "entries": entries,
        "distribution": distribution_curve,
    }
    return units.apply_unit_system_to_rankings_response(response, unit_system, raw_units)


@app.get("/relative-rankings/{taxon_id}/options")
def list_relative_ranking_options(
    taxon_id: int,
    rank: str = Query(..., description="Descendant rank to inspect (e.g., SPECIES)"),
) -> dict[str, Any]:
    """Lists available ranking metrics for an ancestor/rank.
    
    Args:
        taxon_id: Ancestor taxon id to inspect.
        rank: Descendant rank to inspect.
    
    Returns:
        A dict containing available variable/metric options.
    """
    try:
        options = indexing.list_rank_metric_options(str(taxon_id), rank)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ancestor_taxon_id": taxon_id,
        "rank": rank.upper(),
        "options": options,
    }

@app.post("/upload/raw-observations")
async def upload_raw_observations(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> FileResponse:
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    SUPPORTED = {".csv", ".tsv", ".parquet"}
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: CSV, TSV, Parquet.",
        )
    
    print("Received file, converting to parquet...")

    contents = await file.read()
    buf = io.BytesIO(contents)

    try:
        if suffix == ".parquet":
            df = pd.read_parquet(buf)
        elif suffix == ".tsv":
            df = pd.read_csv(buf, sep="\t")
        else:  # .csv
            df = pd.read_csv(buf)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc

    print("Finished converting file to parquet, normalizing fields...")

    df = custom_upload_processing._normalize_coordinate_columns(df)
    df = custom_upload_processing._ensure_catalog_numbers(df)
    print("Finished normalizing fields. adding tileID...")
    df = custom_upload_processing._add_tile_ids(df)
    print("Finished adding tileID, adding columns for GIS data, building parquet files")
    df = custom_upload_processing._add_gis_columns(df)

    archive_path, out_name, work_dir = custom_upload_processing._build_index_archive(df, filename)

    background_tasks.add_task(shutil.rmtree, work_dir, True)

    print("Finished generating, returning zip")
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=out_name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
