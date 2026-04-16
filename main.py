from __future__ import annotations
import asyncio
import hashlib
import anyio
import io
import logging
import math
import re
import shutil
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass as _dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as _np_global

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import pandas as pd
from pydantic import BaseModel, Field

from util.config import load_config
from util.request_cancellation import CancelCheck, RequestCancelledError
from util import (
    custom_upload_processing,
    descriptions,
    gis_lookup,
    indexing,
    models,
    summary_stats,
    taxa_navigation,
    units,
    tiles,
    weather_tiles,
)
from util.tile_request import log_tile_cancellation, render_species_deep_zoom_tile, run_tile_render_with_cancellation
from util.storage import get_parquet_storage

CONFIG = load_config("global")
LOGGER = logging.getLogger("uvicorn.error")

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
variable_parent_tile_max_size = int(getattr(CONFIG, "sdm_parent_tile_max_size", 1024))
species_deep_zoom_render_limit = max(1, int(getattr(CONFIG, "sdm_deep_zoom_render_limit", 1)))
variable_tile_cache_seconds = int(getattr(CONFIG, "sdm_tile_cache_seconds", 60))
variable_tile_default_reproject = bool(getattr(CONFIG, "sdm_tile_reproject", True))
derived_tile_variables = frozenset({"slope", "aspect", "aspect_deg"})
# Continuous circular variables where scalar summary stats (min/mean/max) and
# relative rankings are statistically nonsensical and should be omitted.
no_summary_stats_variables = frozenset({"aspect_deg"})
_SPECIES_DEEP_ZOOM_RENDER_SEMAPHORE = asyncio.Semaphore(species_deep_zoom_render_limit)


def _warm_taxa_name_index() -> None:
    try:
        taxa_navigation.load_name_index()
    except (FileNotFoundError, OSError) as exc:
        LOGGER.warning("[taxa.load-name-index] startup warm skipped: %s", exc)
    except Exception:
        LOGGER.exception("[taxa.load-name-index] startup warm failed")


def _build_disconnect_checker(request: Request | None, *, poll_every: int = 32) -> CancelCheck:
    if request is None:
        return lambda: None

    checks = 0
    disconnected = False

    def check() -> None:
        nonlocal checks, disconnected
        if disconnected:
            raise RequestCancelledError("Client disconnected")
        if checks == 0 or checks % poll_every == 0:
            try:
                disconnected = bool(anyio.from_thread.run(request.is_disconnected))
            except RuntimeError:
                disconnected = False
            if disconnected:
                raise RequestCancelledError("Client disconnected")
        checks += 1

    return check


def _search_locations_with_optional_cancel(
    query: str,
    limit: int,
    cancel_check: CancelCheck,
) -> list[dict[str, Any]]:
    try:
        return gis_lookup.search_locations(query, limit, cancel_check=cancel_check)
    except TypeError as exc:
        if "cancel_check" not in str(exc):
            raise
        return gis_lookup.search_locations(query, limit)


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
    import threading

    threading.Thread(target=weather_tiles.load_cache, daemon=True, name="weather-cache").start()
    threading.Thread(target=_get_homepage_cache, daemon=True, name="homepage-cache").start()
    threading.Thread(target=_warm_taxa_name_index, daemon=True, name="taxa-name-index").start()
    yield


app = FastAPI(title=api_title, version=api_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cors_allow_origins),
    allow_methods=list(cors_allow_methods),
    allow_headers=list(cors_allow_headers),
)


class TaxaQueryScope(BaseModel):
    within_taxon: int | str | None = Field(
        default=None,
        description="Resolved `within_taxon` scope as an integer id when possible, otherwise the normalized value.",
    )
    descendant_rank: str | None = Field(default=None, description="Canonical descendant rank applied to the query.")
    location: str | None = Field(default=None, description="Normalized location GID filter.")
    min_samples: int = Field(description="Minimum sample-count threshold applied to the query.")
    include_species_like: bool = Field(
        description="Whether species-like descendant ranks such as subspecies were included for species queries."
    )


class TaxaQuerySort(BaseModel):
    variable: str | None = Field(default=None, description="Variable id used for sorting ranked results.")
    metric: str | None = Field(default=None, description="Metric name used for sorting ranked results.")
    order: str = Field(description="Normalized sort order.")
    units: str | None = Field(default=None, description="Display units for ranked values when applicable.")


class TaxaQueryResult(BaseModel):
    taxon_id: int | str | None = Field(default=None, description="Canonical taxon id for the result row.")
    scientific_name: str | None = Field(default=None, description="Scientific name for the matched or ranked taxon.")
    common_name: str | None = Field(default=None, description="Preferred common name for the taxon.")
    common_names: list[str] | None = Field(default=None, description="Available localized common names.")
    rank: str | None = Field(default=None, description="Canonical taxonomic rank for the taxon.")
    slug: str | None = Field(default=None, description="Slug identifier when available.")
    description: str | None = Field(default=None, description="Localized description text when available.")
    image_url: str | None = Field(default=None, description="Primary image URL when available.")
    image_license: str | None = Field(default=None, description="Image license string when available.")
    image_creator: str | None = Field(default=None, description="Image creator attribution when available.")
    image_rights_holder: str | None = Field(default=None, description="Image rights holder when available.")
    image_references: str | None = Field(default=None, description="Primary image source reference.")
    match_score: float | None = Field(default=None, description="Text match score for search-based results.")
    sample_count: int | None = Field(default=None, description="Sample count used for eligibility and display.")
    sort_value: float | None = Field(default=None, description="Ranked metric value when sorting is applied.")
    sort_variable: str | None = Field(default=None, description="Variable id associated with `sort_value`.")
    sort_metric: str | None = Field(default=None, description="Metric name associated with `sort_value`.")
    position: int | None = Field(default=None, description="1-based rank position for ranked results.")
    percentile: float | None = Field(default=None, description="Normalized percentile for ranked results.")


class TaxaQueryResponse(BaseModel):
    query: str | None = Field(default=None, description="Normalized query string, or null when no text query was used.")
    scope: TaxaQueryScope
    sort: TaxaQuerySort
    total: int = Field(description="Total number of result rows after all ranking and filtering rules are applied.")
    matched_total: int = Field(
        description="Number of taxa that matched the text query before scoped ranked intersection or text-only filtering."
    )
    eligible_total: int = Field(
        description=(
            "Number of eligible rows before pagination. For scoped ranked queries this is the post-ranking eligible count "
            "from the leaderboard index, not a separate text-prefilter count."
        )
    )
    empty_reason: Literal["no_query", "no_text_matches", "filtered_out", "ranking_ineligible"] | None = Field(
        default=None,
        description=(
            "Null when results are present. Empty text responses may use `no_query`, `no_text_matches`, or "
            "`filtered_out`. Empty ranked responses use `ranking_ineligible`."
        ),
    )
    limit: int = Field(description="Page size requested by the client.")
    offset: int = Field(description="Pagination offset requested by the client.")
    results: list[TaxaQueryResult] = Field(description="Result rows for the current page.")


class TaxaRankingOption(BaseModel):
    variable: str = Field(description="Variable id available for ranking in the requested scope.")
    metric: str = Field(description="Metric name available for the variable in the requested scope.")
    label: str | None = Field(default=None, description="Display label for the metric.")
    count: int | None = Field(default=None, description="Indexed row count for the option when available.")
    column: str | None = Field(default=None, description="Backing parquet column name for the option.")


class TaxaRankingOptionsResponse(BaseModel):
    ancestor_taxon_id: int | str = Field(description="Resolved ancestor taxon id for the requested scope.")
    rank: str = Field(description="Canonical descendant rank used for the scope.")
    options: list[TaxaRankingOption] = Field(description="Available ranking options for the requested scope.")


def _path_exists(path: Path) -> bool:
    storage = get_parquet_storage(CONFIG.data_root, CONFIG.project_root)
    if storage.is_remote:
        return storage.exists(path)
    return path.exists()


def _with_env_observation_count(
    response: dict[str, Any],
    observation_count: int,
) -> dict[str, Any]:
    updated = dict(response)
    updated["observationCount"] = observation_count
    updated["observation_count"] = observation_count
    return updated


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
        if value_type not in {"numeric", "categorical", "circular"}:
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


@app.get("/api/weather/status", summary="Live weather cache status")
def weather_cache_status() -> dict:
    return {
        "ref_times": weather_tiles._cache_ref_times,
        "cached_variables": list(weather_tiles._cache.keys()),
        "ready": len(weather_tiles._cache) == len(weather_tiles.LIVE_WEATHER_VARIABLES),
    }


@app.get("/health", summary="Simple liveness probe")
def health_check() -> dict[str, str]:
    """Returns a simple liveness payload.

    Returns:
        A status string and UTC timestamp.
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/variables")
def list_environment_variables(
    unit_system: Optional[str] = Query(None, description="Unit system for response values (metric or imperial)"),
) -> List[dict[str, Any]]:
    """Lists available environmental variables.

    Returns:
        A list of variable metadata entries.
    """
    return units.apply_unit_system_to_variables(
        gis_lookup.load_variable_metadata()[0],
        unit_system,
    )


@app.get("/data-sources")
def list_data_sources() -> dict[str, Any]:
    """Returns structured citation metadata for all environmental data sources."""
    return gis_lookup.load_data_sources()


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

    # Live weather variables bypass the GeoTIFF pipeline entirely
    if layer_id in weather_tiles.LIVE_WEATHER_VARIABLES:
        payload = await run_in_threadpool(
            weather_tiles.render_weather_tile_bytes,
            variable_id=layer_id,
            z=z,
            x=x,
            y=y,
            tile_size=tile_size,
        )
        if payload is None:
            # Cache not yet populated — return transparent tile
            return Response(status_code=204)
        headers = {"Cache-Control": f"public, max-age={variable_tile_cache_seconds}"}
        return Response(content=payload, media_type="image/png", headers=headers)

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
        scale = 2**zoom_diff
        parent_x = x // scale
        parent_y = y // scale
        subtile_x = x % scale
        subtile_y = y % scale
        parent_tile_size = min(tile_size * scale, variable_parent_tile_max_size)
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
            payload = tiles.crop_subtile_png(
                parent_payload,
                parent_tile_size=parent_tile_size,
                scale=scale,
                subtile_x=subtile_x,
                subtile_y=subtile_y,
                tile_size=tile_size,
            )
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


@app.get("/api/species/{taxon_id}/heatmap")
def species_heatmap_metadata(
    taxon_id: int,
    model_id: str = Query(models.DEFAULT_MODEL_ID, description="Model id or artifact id."),
) -> dict[str, Any]:
    return models.describe_model(model_id, taxon_id=taxon_id)


@app.get("/api/species/{taxon_id}/heatmap/tiles/{z}/{x}/{y}.png")
async def species_heatmap_tile(
    request: Request,
    taxon_id: int,
    z: int,
    x: int,
    y: int,
    model_id: str = Query(models.DEFAULT_MODEL_ID, description="Model id or artifact id."),
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
    forecast_hours: int = Query(0, ge=0, description="GFS forecast offset in hours (0 = current)."),
    apply_phenology: bool = Query(True, description="Multiply SDM by phenology model if available."),
    phenology_only: bool = Query(False, description="Render raw phenology model output only (no SDM)."),
) -> Response:
    if await request.is_disconnected():
        log_tile_cancellation(
            "species",
            "preflight_disconnect",
            taxon_id=taxon_id,
            z=z,
            x=x,
            y=y,
            model_id=model_id,
        )
        return Response(status_code=204)

    resolved_model = models.describe_model(model_id, taxon_id=taxon_id)
    if not resolved_model.get("available"):
        raise HTTPException(
            status_code=404,
            detail=f"No heatmap model found for taxon_id {taxon_id}.",
        )

    if z > max_native_zoom:
        try:
            payload = await render_species_deep_zoom_tile(
                request,
                _SPECIES_DEEP_ZOOM_RENDER_SEMAPHORE,
                taxon_id=taxon_id,
                z=z,
                x=x,
                y=y,
                tile_size=tile_size,
                max_native_zoom=max_native_zoom,
                parent_tile_max_size=variable_parent_tile_max_size,
                model_id=model_id,
                reproject=reproject,
                forecast_hours=forecast_hours,
                apply_phenology=apply_phenology,
                phenology_only=phenology_only,
            )
            if payload is None:
                return Response(status_code=204)
        except tiles.TileRenderCancelled:
            return Response(status_code=204)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        try:
            if await request.is_disconnected():
                log_tile_cancellation(
                    "species",
                    "pre_render_disconnect",
                    taxon_id=taxon_id,
                    z=z,
                    x=x,
                    y=y,
                    model_id=model_id,
                )
                return Response(status_code=204)
            payload = await run_tile_render_with_cancellation(
                request,
                tiles.render_model_tile_bytes,
                taxon_id=taxon_id,
                z=z,
                x=x,
                y=y,
                model_id=model_id,
                tile_size=tile_size,
                reproject=reproject,
                forecast_hours=forecast_hours,
                apply_phenology=apply_phenology,
                phenology_only=phenology_only,
            )
        except tiles.TileRenderCancelled:
            log_tile_cancellation(
                "species",
                "during_render",
                taxon_id=taxon_id,
                z=z,
                x=x,
                y=y,
                model_id=model_id,
            )
            return Response(status_code=204)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if await request.is_disconnected():
        log_tile_cancellation(
            "species",
            "post_render_disconnect",
            taxon_id=taxon_id,
            z=z,
            x=x,
            y=y,
            model_id=model_id,
        )
        return Response(status_code=204)
    headers = {
        "Cache-Control": f"public, max-age={variable_tile_cache_seconds}",
    }
    return Response(content=payload, media_type="image/png", headers=headers)


@app.get("/api/heatmap/aggregate/tiles/{z}/{x}/{y}.png")
async def aggregate_heatmap_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    tile_size: int = Query(variable_tile_default_size, ge=32, le=variable_tile_max_size),
    reproject: bool = Query(variable_tile_default_reproject),
    forecast_hours: int = Query(0, ge=0, description="GFS forecast offset in hours (0 = current)."),
) -> Response:
    if await request.is_disconnected():
        log_tile_cancellation("aggregate", "preflight_disconnect", z=z, x=x, y=y)
        return Response(status_code=204)

    try:
        payload = await run_tile_render_with_cancellation(
            request,
            tiles.render_aggregate_tile_bytes,
            z=z,
            x=x,
            y=y,
            tile_size=tile_size,
            reproject=reproject,
            forecast_hours=forecast_hours,
        )
    except tiles.TileRenderCancelled:
        log_tile_cancellation(
            "aggregate",
            "during_render",
            z=z,
            x=x,
            y=y,
            forecast_hours=forecast_hours,
        )
        return Response(status_code=204)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if await request.is_disconnected():
        log_tile_cancellation(
            "aggregate",
            "post_render_disconnect",
            z=z,
            x=x,
            y=y,
            forecast_hours=forecast_hours,
        )
        return Response(status_code=204)
    headers = {
        "Cache-Control": f"public, max-age={variable_tile_cache_seconds}",
    }
    return Response(content=payload, media_type="image/png", headers=headers)


_HOMEPAGE_RASTER = CONFIG.data_root / "gis" / "temporal" / "homepage" / "aggregate_sdm.tif"
_TAXON_PROBS_PATH = _HOMEPAGE_RASTER.parent / "taxon_probs.npz"
_SCORES_CACHE_DIR = Path("/workspace/cache/scores")


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def _latest_temporal_built_at() -> str:
    import json as _json

    temporal_dir = _HOMEPAGE_RASTER.parent.parent / "rasters"
    latest = ""
    for path in temporal_dir.glob("*.meta.json"):
        try:
            payload = _json.loads(path.read_text())
        except Exception:
            continue
        built_at = payload.get("built_at")
        if isinstance(built_at, str) and built_at > latest:
            latest = built_at
    return latest


def _scores_cache_namespace() -> str:
    raw = "|".join(
        [
            str(CONFIG.data_root),
            f"taxon_probs:{_path_mtime_ns(_TAXON_PROBS_PATH)}",
            f"meta:{_path_mtime_ns(_HOMEPAGE_RASTER.parent / 'base_features_meta.json')}",
            f"base_features:{_path_mtime_ns(_HOMEPAGE_RASTER.parent / 'base_features.npz')}",
            f"temporal_built_at:{_latest_temporal_built_at()}",
        ]
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


@_dataclass
class _HomepageCache:
    meta: dict
    taxon_probs: dict  # str(taxon_id) -> np.ndarray
    gis_arrays: dict  # layer_id -> np.ndarray (only labeled layers)
    vmin: float
    vmax: float
    mtime: float


_VALID_GROUPS = {"birds", "animals", "arthropods", "fungi", "plants", "other"}

_homepage_cache: "_HomepageCache | None" = None


def _get_homepage_cache() -> "_HomepageCache | None":
    import json as _json

    global _homepage_cache
    if not _TAXON_PROBS_PATH.exists():
        return None
    mtime = _TAXON_PROBS_PATH.stat().st_mtime
    if _homepage_cache is None or _homepage_cache.mtime != mtime:
        meta_path = _HOMEPAGE_RASTER.parent / "base_features_meta.json"
        if not meta_path.exists():
            return None
        meta = _json.loads(meta_path.read_text())
        taxon_probs = dict(_np_global.load(_TAXON_PROBS_PATH))
        gis_arrays: dict = {}
        npz_path = _HOMEPAGE_RASTER.parent / "base_features.npz"
        if npz_path.exists():
            base = _np_global.load(npz_path)
            for _key in ("landcover", "elevation"):
                if _key in base:
                    gis_arrays[_key] = base[_key]
        vmin, vmax = 0.0, 1.0
        stats_path = _HOMEPAGE_RASTER.parent / "aggregate_sdm_stats.json"
        if stats_path.exists():
            import json as _json2

            _stats = _json2.loads(stats_path.read_text())
            vmin, vmax = float(_stats.get("vmin", 0.0)), float(_stats.get("vmax", 1.0))
        _homepage_cache = _HomepageCache(
            meta=meta, taxon_probs=taxon_probs, gis_arrays=gis_arrays, vmin=vmin, vmax=vmax, mtime=mtime
        )
    return _homepage_cache


def _raster_for_group(group: str | None) -> tuple[Path, float, float]:
    """Return (raster_path, vmin, vmax) for a given group (None = overall)."""
    import json as _json

    if group and group in _VALID_GROUPS:
        path = _HOMEPAGE_RASTER.parent / f"aggregate_sdm_{group}.tif"
        if path.exists():
            stats_path = _HOMEPAGE_RASTER.parent / "aggregate_sdm_group_stats.json"
            if stats_path.exists():
                all_stats = _json.loads(stats_path.read_text())
                s = all_stats.get(group, {})
                return path, float(s.get("vmin", 0.0)), float(s.get("vmax", 1.0))
            return path, 0.0, 1.0
    # Fall back to overall
    vmin, vmax = 0.0, 1.0
    stats_path = _HOMEPAGE_RASTER.parent / "aggregate_sdm_stats.json"
    if stats_path.exists():
        s = _json.loads(stats_path.read_text())
        vmin, vmax = float(s.get("vmin", 0.0)), float(s.get("vmax", 1.0))
    return _HOMEPAGE_RASTER, vmin, vmax


@app.get("/api/heatmap/homepage/tiles/{z}/{x}/{y}.png")
async def homepage_heatmap_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    tile_size: int = Query(variable_tile_default_size, ge=32, le=variable_tile_max_size),
    group: str | None = Query(None),
) -> Response:
    if await request.is_disconnected():
        log_tile_cancellation("homepage", "preflight_disconnect", z=z, x=x, y=y, group=group)
        return Response(status_code=204)

    raster_path, vmin, vmax = _raster_for_group(group)
    if not raster_path.exists():
        # Group raster not generated yet, fall back to overall
        raster_path, vmin, vmax = _raster_for_group(None)
    if not raster_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap data not yet available.")
    cache = _get_homepage_cache()
    if cache and group is None:
        vmin, vmax = cache.vmin, cache.vmax

    try:
        payload = await run_tile_render_with_cancellation(
            request,
            tiles.render_homepage_tile_bytes,
            z=z,
            x=x,
            y=y,
            raster_path=raster_path,
            tile_size=tile_size,
            vmin=vmin,
            vmax=vmax,
        )
    except tiles.TileRenderCancelled:
        log_tile_cancellation("homepage", "during_render", z=z, x=x, y=y, group=group)
        return Response(status_code=204)

    if await request.is_disconnected():
        log_tile_cancellation("homepage", "post_render_disconnect", z=z, x=x, y=y, group=group)
        return Response(status_code=204)
    headers = {"Cache-Control": "public, max-age=3600"}
    return Response(content=payload, media_type="image/png", headers=headers)


_TEMPORAL_REASON_LABELS: dict[str, tuple[str, str]] = {
    # (high-value label, low-value label)
    "temperature_2m": ("warm temperatures", "cool temperatures"),
    "precipitation": ("recent rainfall", "dry conditions"),
    "soil_moisture_0_to_7cm": ("moist soils", "dry soils"),
    "soil_moisture_0_to_10cm": ("moist soils", "dry soils"),
    "soil_temperature_0_to_7cm": ("warm soils", "cool soils"),
    "soil_temperature_0_to_10cm": ("warm soils", "cool soils"),
    "cloud_cover": ("overcast skies", "clear skies"),
    "snowfall_water_equivalent": ("snow cover", "snow-free ground"),
    "relative_humidity_2m": ("high humidity", "low humidity"),
    "vapor_pressure_deficit": ("high evaporation demand", "low evaporation demand"),
    "dew_point_2m": ("humid air", "dry air"),
}

_LANDCOVER_CLASS_TO_GROUP: dict[int, str] = {
    10: "cropland",
    11: "cropland",
    12: "cropland",
    20: "cropland",
    51: "forest",
    52: "forest",
    61: "forest",
    62: "forest",
    71: "forest",
    72: "forest",
    81: "forest",
    82: "forest",
    91: "forest",
    92: "forest",
    120: "shrubland",
    121: "shrubland",
    122: "shrubland",
    130: "grassland",
    140: "lichens_mosses",
    150: "sparse_vegetation",
    152: "sparse_vegetation",
    153: "sparse_vegetation",
    180: "wetlands",
    190: "urban",
    200: "bare_areas",
    201: "bare_areas",
    202: "bare_areas",
    210: "water",
    220: "ice_snow",
}

_LANDCOVER_GROUP_LABELS: dict[str, str] = {
    "cropland": "agricultural areas",
    "forest": "forested areas",
    "shrubland": "shrubland habitat",
    "grassland": "open grasslands",
    "lichens_mosses": "tundra habitat",
    "sparse_vegetation": "sparse vegetation",
    "wetlands": "wetland habitat",
    "urban": "urban areas",
    "bare_areas": "open bare ground",
    "water": "aquatic habitat",
    "ice_snow": "alpine habitat",
}


def _compute_single_tile_scores(z: int, x: int, y: int) -> dict:
    """Compute per-taxon scores for a single tile. Loads data locally so nothing is retained in worker memory."""
    import json as _json
    import numpy as _np
    from collections import defaultdict as _dd
    from util.tiles import TileSpec, tile_bounds_wgs84

    meta_path = _HOMEPAGE_RASTER.parent / "base_features_meta.json"
    if not meta_path.exists() or not _TAXON_PROBS_PATH.exists():
        return {"scores": {}, "reasons": {}}

    meta = _json.loads(meta_path.read_text())
    out_h, out_w = meta["shape"]
    res = meta["resolution_degrees"]
    bounds = meta["bounds"]

    min_lon, min_lat, max_lon, max_lat = tile_bounds_wgs84(TileSpec(z=z, x=x, y=y, tile_size=256))
    c_min_lon = max(min_lon, bounds["min_lon"])
    c_max_lon = min(max_lon, bounds["max_lon"])
    c_min_lat = max(min_lat, bounds["min_lat"])
    c_max_lat = min(max_lat, bounds["max_lat"])
    if c_min_lon >= c_max_lon or c_min_lat >= c_max_lat:
        return {"scores": {}, "reasons": {}}

    col0 = max(0, round((c_min_lon - bounds["min_lon"]) / res))
    col1 = min(out_w, round((c_max_lon - bounds["min_lon"]) / res))
    row0 = max(0, round((bounds["max_lat"] - c_max_lat) / res))
    row1 = min(out_h, round((bounds["max_lat"] - c_min_lat) / res))
    if col1 <= col0 or row1 <= row0:
        return {"scores": {}, "reasons": {}}

    # Load taxon probs locally — not stored in module state, GC'd on return
    taxon_npz = _np.load(_TAXON_PROBS_PATH)
    scores: dict[str, float] = {}
    probs_patches: dict[str, "_np.ndarray"] = {}
    for taxon_id_str in taxon_npz.files:
        patch = taxon_npz[taxon_id_str][row0:row1, col0:col1]
        finite = patch[_np.isfinite(patch)]
        if finite.size > 0:
            scores[taxon_id_str] = float(finite.mean())
            probs_patches[taxon_id_str] = patch
    del taxon_npz

    # Temporal reasons
    era5_res = 0.25
    er0 = max(0, round((90.0 - c_max_lat) / era5_res))
    er1_bound = round((90.0 - c_min_lat) / era5_res)
    ec0 = max(0, round((c_min_lon + 180.0) / era5_res))
    ec1_bound = round((c_max_lon + 180.0) / era5_res)
    need_h, need_w = row1 - row0, col1 - col0

    temporal_dir = _HOMEPAGE_RASTER.parent.parent / "rasters"
    temporal_patches: dict[str, "_np.ndarray"] = {}
    for p in temporal_dir.glob("*.npy"):
        full = _np.load(p).astype(_np.float32)
        er1 = min(full.shape[0], er1_bound)
        ec1 = min(full.shape[1], ec1_bound)
        sliced = full[er0:er1, ec0:ec1]
        if sliced.shape != (need_h, need_w):
            out = _np.full((need_h, need_w), _np.nan, dtype=_np.float32)
            h, w = min(sliced.shape[0], need_h), min(sliced.shape[1], need_w)
            out[:h, :w] = sliced[:h, :w]
            sliced = out
        temporal_patches[p.stem] = sliced

    lc_path = _HOMEPAGE_RASTER.parent / "base_features.npz"
    lc_arr: "_np.ndarray | None" = None
    if lc_path.exists():
        lc_npz = _np.load(lc_path)
        if "landcover" in lc_npz:
            lc_arr = lc_npz["landcover"][row0:row1, col0:col1]
        del lc_npz

    reasons: dict[str, list[str]] = {}
    for taxon_id_str, prob_patch in probs_patches.items():
        prob_flat = prob_patch.flatten()
        taxon_reasons: list[str] = []

        best_temporal: tuple[float, str] | None = None
        for feat_id, arr in temporal_patches.items():
            feat_flat = arr.flatten()
            valid = _np.isfinite(feat_flat) & _np.isfinite(prob_flat)
            if valid.sum() < 10:
                continue
            fv, pv = feat_flat[valid], prob_flat[valid]
            if fv.std() == 0 or pv.std() == 0:
                continue
            with _np.errstate(invalid="ignore"):
                corr = float(_np.corrcoef(fv, pv)[0, 1])
            if not _np.isfinite(corr):
                continue
            parsed = gis_lookup.parse_temporal_layer_id(feat_id)
            var_name = parsed[0] if parsed else re.sub(r"_\d+[hd]$", "", feat_id)
            label_pair = _TEMPORAL_REASON_LABELS.get(var_name)
            if not label_pair:
                continue
            label = label_pair[0] if corr >= 0 else label_pair[1]
            if best_temporal is None or abs(corr) > best_temporal[0]:
                best_temporal = (abs(corr), label)
        if best_temporal:
            taxon_reasons.append(best_temporal[1])

        if lc_arr is not None:
            lc_flat = lc_arr.flatten()
            valid = _np.isfinite(lc_flat) & _np.isfinite(prob_flat)
            if valid.any():
                lc_vals = _np.rint(lc_flat[valid]).astype(int)
                prob_vals = prob_flat[valid]
                group_probs: dict = _dd(list)
                for cls, p in zip(lc_vals, prob_vals):
                    group = _LANDCOVER_CLASS_TO_GROUP.get(int(cls))
                    if group:
                        group_probs[group].append(p)
                best_group = max(
                    (g for g, ps in group_probs.items() if len(ps) >= 3),
                    key=lambda g: float(_np.mean(group_probs[g])),
                    default=None,
                )
                lc_label = _LANDCOVER_GROUP_LABELS.get(best_group) if best_group else None
                if lc_label:
                    taxon_reasons.append(lc_label)

        if taxon_reasons:
            reasons[taxon_id_str] = taxon_reasons

    return {"scores": scores, "reasons": reasons}


@app.get("/api/heatmap/homepage/scores")
def homepage_tile_scores(
    z: int = Query(...),
    x0: int = Query(...),
    y0: int = Query(...),
    x1: int = Query(...),
    y1: int = Query(...),
) -> dict[str, Any]:
    """Return per-taxon average probability for the given tile range.

    Results are cached per tile on disk and shared across all workers.
    """
    import json as _json

    _SCORES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    namespace = _scores_cache_namespace()

    all_scores: dict[str, list[float]] = {}
    all_reasons: dict[str, dict[str, int]] = {}  # tid -> {reason: tile_count}

    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            cache_path = _SCORES_CACHE_DIR / f"{namespace}_z{z}_x{tx}_y{ty}.json"
            if cache_path.exists():
                try:
                    tile_result = _json.loads(cache_path.read_text())
                except Exception:
                    tile_result = _compute_single_tile_scores(z, tx, ty)
            else:
                tile_result = _compute_single_tile_scores(z, tx, ty)
                try:
                    cache_path.write_text(_json.dumps(tile_result))
                except Exception:
                    pass

            for tid, score in tile_result.get("scores", {}).items():
                all_scores.setdefault(tid, []).append(score)
            for tid, tile_reasons in tile_result.get("reasons", {}).items():
                counts = all_reasons.setdefault(tid, {})
                for r in tile_reasons:
                    counts[r] = counts.get(r, 0) + 1

    scores = {tid: float(sum(vals) / len(vals)) for tid, vals in all_scores.items()}
    # Pick the top 2 most-agreed-upon reasons across tiles
    reasons = {tid: sorted(counts, key=lambda r: -counts[r])[:2] for tid, counts in all_reasons.items() if counts}
    return {"scores": scores, "reasons": reasons}


@app.get("/api/species/with-models")
def list_species_with_models() -> List[dict[str, Any]]:
    """Returns serialized species info for every taxon that has a trained SDM artifact."""
    taxon_ids = models.get_all_sdm_taxon_ids()
    result = []
    for taxon_id in taxon_ids:
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
        if taxon is None:
            continue
        payload = taxa_navigation.serialize_taxon(taxon)
        if payload:
            result.append(payload)
    return result


def _serialize_taxon_query_result(
    taxon: taxa_navigation.TaxonRecord,
    *,
    match_score: Optional[float] = None,
    sample_count: Optional[int] = None,
    sort_value: Optional[float] = None,
    sort_variable: Optional[str] = None,
    sort_metric: Optional[str] = None,
    position: Optional[int] = None,
    percentile: Optional[float] = None,
) -> dict[str, Any] | None:
    payload = taxa_navigation.serialize_taxon(taxon)
    if payload is None:
        return None
    payload["image_references"] = taxa_navigation.normalize_image_reference(payload.get("image_references"))
    payload.update(
        {
            "match_score": match_score,
            "sample_count": sample_count,
            "sort_value": sort_value,
            "sort_variable": sort_variable,
            "sort_metric": sort_metric,
            "position": position,
            "percentile": percentile,
        }
    )
    return payload


def _serialize_taxon_query_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in rows:
        taxon = row.get("taxon")
        taxon_id = row.get("taxon_id")
        if taxon is None and taxon_id is not None:
            taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
        if taxon is None:
            continue
        payload = _serialize_taxon_query_result(
            taxon,
            match_score=float(row["match_score"]) if row.get("match_score") is not None else None,
            sample_count=row.get("sample_count"),
            sort_value=row.get("sort_value"),
            sort_variable=row.get("sort_variable"),
            sort_metric=row.get("sort_metric"),
            position=row.get("position"),
            percentile=row.get("percentile"),
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def _resolve_within_taxon_id(
    *,
    within_taxon: Optional[str],
) -> str | None:
    raw_value = str(within_taxon or "").strip()
    if not raw_value:
        return None
    if raw_value.isdigit():
        if taxa_navigation.get_taxon_by_id(raw_value) is None:
            raise HTTPException(status_code=400, detail=f"Unknown within_taxon value: {raw_value}")
        return raw_value

    resolved = taxa_navigation.resolve_taxon_reference(raw_value)
    if resolved is None:
        raise HTTPException(status_code=400, detail=f"Unknown within_taxon value: {raw_value}")
    return str(resolved.get("taxon_key") or "").strip() or None


@app.get("/api/taxa/ranking-options", response_model=TaxaRankingOptionsResponse)
def list_taxa_ranking_options(
    within_taxon: str = Query(
        ...,
        description="Ancestor taxon reference. Prefer taxon id; scientific-name slugs are a convenience path only when unambiguous.",
    ),
    descendant_rank: str = Query(..., description="Descendant rank to inspect within the ancestor scope"),
) -> TaxaRankingOptionsResponse:
    """List valid ranking variable/metric combinations for a scoped taxon context."""
    raw_within_taxon = str(within_taxon or "").strip()
    if not raw_within_taxon:
        raise HTTPException(status_code=400, detail="within_taxon is required")

    try:
        resolved_within_taxon_id = _resolve_within_taxon_id(within_taxon=raw_within_taxon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not resolved_within_taxon_id:
        raise HTTPException(status_code=400, detail="within_taxon is required")

    try:
        options = indexing.list_rank_metric_options(resolved_within_taxon_id, descendant_rank)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_taxon_id = taxa_navigation.taxon_id_as_int(resolved_within_taxon_id)
    canonical_rank = taxa_navigation.canonical_rank(descendant_rank)
    return TaxaRankingOptionsResponse(
        ancestor_taxon_id=resolved_taxon_id if resolved_taxon_id is not None else resolved_within_taxon_id,
        rank=canonical_rank,
        options=options,
    )


@app.get("/api/taxa/query", response_model=TaxaQueryResponse)
def query_taxa(
    request: Request,
    q: Optional[str] = Query(None, min_length=1, description="Optional search term"),
    within_taxon: Optional[str] = Query(
        None,
        description=(
            "Optional ancestor taxon reference. Prefer taxon id; scientific-name slugs are a convenience "
            "path only when unambiguous."
        ),
    ),
    descendant_rank: Optional[str] = Query(
        None,
        description="Descendant rank to include when using ranked results",
    ),
    sort_variable: Optional[str] = Query(
        None,
        description="Optional variable id used to sort matched taxa",
    ),
    sort_metric: Optional[str] = Query(
        None,
        description="Optional metric used to sort matched taxa",
    ),
    sort_order: str = Query("asc", description="Sort order: asc or desc"),
    limit: int = Query(default_species_limit, ge=1, le=max_species_limit),
    offset: int = Query(0, ge=0),
    min_samples: int = Query(0, ge=0),
    include_species_like: bool = Query(
        False,
        description="When descendant_rank=SPECIES, include subspecies/varieties/forms",
    ),
    location: Optional[str] = Query(
        None,
        description="Optional location GID to filter query results",
    ),
    unit_system: Optional[str] = Query(
        None,
        description="Unit system for ranked values (metric or imperial)",
    ),
) -> TaxaQueryResponse:
    """Return canonical taxon search results for both text and ranked queries.

    Text-only requests use `q` and return matched taxa with `match_score` plus
    any applicable `sample_count`. Ranked requests add `sort_variable` and
    `sort_metric`, and may also scope to a subtree via `within_taxon` and
    `descendant_rank`.

    Args:
        q: Optional scientific or common-name query string.
        within_taxon: Optional ancestor taxon reference. Prefer taxon id; scientific-name slugs are a convenience path only when unambiguous.
        descendant_rank: Optional descendant rank when scoping ranked results.
        sort_variable: Optional variable id used for ordering ranked results.
        sort_metric: Optional metric used for ordering ranked results.
        sort_order: Ranking direction, either `asc` or `desc`.
        limit: Maximum number of results to return.
        offset: Result offset for pagination.
        min_samples: Optional sample-count threshold applied to query results.
        include_species_like: Include subspecies-like ranks when ranking species.
        location: Optional location GID to filter query results.
        unit_system: Unit system for ranked values.

    Returns:
        A paginated response containing normalized query metadata and results.
        Empty ranked responses use `ranking_ineligible`; empty text responses
        may use `no_query`, `no_text_matches`, or `filtered_out`. Scoped ranked
        searches use the leaderboard index directly instead of a text-prefilter
        eligibility pass.
    """
    start = time.perf_counter()
    normalized_query = (q or "").strip() or None
    normalized_location = location.strip() if location else None
    normalized_sort_order = (sort_order or "asc").strip().lower() or "asc"
    request_mode = "ranked" if sort_variable and sort_metric else "text"
    cancel_check = _build_disconnect_checker(request)

    try:
        cancel_check()
        try:
            resolved_within_taxon_id = _resolve_within_taxon_id(
                within_taxon=within_taxon,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            query_payload = indexing.query_taxa(
                q=q,
                within_taxon_id=resolved_within_taxon_id,
                descendant_rank=descendant_rank,
                sort_variable=sort_variable,
                sort_metric=sort_metric,
                sort_order=normalized_sort_order,
                limit=limit,
                offset=offset,
                min_samples=min_samples,
                include_species_like=include_species_like,
                location_gid=location,
                cancel_check=cancel_check,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cancel_check()
        variable_entry = None
        if sort_variable:
            variable_entry = gis_lookup.load_variable_metadata()[1].get(sort_variable)
        raw_units = variable_entry.get("units") if variable_entry else None
        sort_units = raw_units

        _METRIC_UNIT_OVERRIDES: dict[str, str | None] = {
            "total_samples": "samples",
            "count": "samples",
            "unique_classes": "classes",
            "significant_unique_classes": "classes",
        }

        if sort_variable and sort_metric:
            response_rows, sort_units = units.apply_unit_system_to_query_rows(
                query_payload["results"],
                unit_system=unit_system,
                variable_id=sort_variable,
                unit=raw_units,
            )
        else:
            response_rows = query_payload["results"]

        if sort_metric:
            normalized_metric = str(sort_metric).strip().lower()
            if normalized_metric in _METRIC_UNIT_OVERRIDES:
                sort_units = _METRIC_UNIT_OVERRIDES[normalized_metric]

        cancel_check()
        results = _serialize_taxon_query_results(response_rows)

        return {
            "query": normalized_query,
            "scope": {
                "within_taxon": taxa_navigation.taxon_id_as_int(resolved_within_taxon_id) or resolved_within_taxon_id,
                "descendant_rank": taxa_navigation.canonical_rank(descendant_rank) if descendant_rank else None,
                "location": normalized_location,
                "min_samples": min_samples,
                "include_species_like": include_species_like,
            },
            "sort": {
                "variable": sort_variable,
                "metric": sort_metric,
                "order": normalized_sort_order,
                "units": sort_units,
            },
            "total": query_payload["total"],
            "matched_total": query_payload["matched_total"],
            "eligible_total": query_payload["eligible_total"],
            "empty_reason": query_payload["empty_reason"],
            "limit": limit,
            "offset": offset,
            "results": results,
        }
    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    finally:
        LOGGER.info(
            "[api.taxa.query] elapsed=%.3fs mode=%s q=%r within_taxon=%r descendant_rank=%r sort=%r/%r "
            "location=%r min_samples=%s include_species_like=%s limit=%s offset=%s",
            time.perf_counter() - start,
            request_mode,
            normalized_query,
            within_taxon,
            descendant_rank,
            sort_variable,
            sort_metric,
            normalized_location,
            min_samples,
            include_species_like,
            limit,
            offset,
        )


@app.get("/api/species/{taxon_id}")
def get_species_detail(
    taxon_id: int,
    location: Optional[str] = Query(None, description="Optional location GID to tailor description text."),
    unit_system: Optional[str] = Query(None, description="Unit system for description values (metric or imperial)"),
) -> dict[str, Any]:
    """Loads a single taxon record by id.

    Args:
        taxon_id: Taxon id to look up.
        location: Optional location GID filter for location text context.

    Returns:
        A serialized taxon payload.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(
            status_code=404,
            detail=f"Species with taxon_id {taxon_id} not found",
        )
    payload = taxa_navigation.serialize_taxon(taxon)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail=f"Species with taxon_id {taxon_id} not found",
        )
    location_gid = location.strip() if location else None
    try:
        description_profile = descriptions.build_taxon_description(
            dict(taxon),
            location_gid=location_gid,
            unit_system=unit_system,
        )
        text = description_profile.get("text")
        if isinstance(text, str) and text.strip():
            payload["description"] = text
        payload["description_profile"] = description_profile
    except Exception:
        traceback.print_exc()
    payload["heatmap"] = models.describe_model(models.DEFAULT_MODEL_ID, taxon_id=taxon_id)
    return payload


@app.get("/api/locations/search")
@app.get("/locations/search", include_in_schema=False)
def search_locations_endpoint(
    request: Request,
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
    cancel_check = _build_disconnect_checker(request)
    try:
        cancel_check()
        matches = _search_locations_with_optional_cancel(q, limit, cancel_check)
    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
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

    level_by_scope = {str(scope): int(level_idx) for level_idx, scope in CONFIG.location_scope_by_level.items()}
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
            name_match = bool(name_options & hierarchy_name_set) or cand_name in name_options
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


@app.get("/api/locations/search_hierarchy")
@app.get("/locations/search_hierarchy", include_in_schema=False)
def search_locations_by_hierarchy(
    request: Request,
    q: str = Query("", description="Location name or partial match (optional if parent provided)"),
    level: Optional[str] = Query(None, description="continent|country|state|county or numeric level code"),
    parent: Optional[str] = Query(
        None, description="Parent name or gid. For counties pass 'United States|Utah' or a gid."
    ),
    limit: int = Query(50, ge=1, le=1000),
) -> dict[str, Any]:
    cancel_check = _build_disconnect_checker(request)

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
        try:
            cancel_check()
            resolved_name = tok
            resolved_gid = tok
            if hasattr(gis_lookup, "get_location_by_gid"):
                maybe = gis_lookup.get_location_by_gid(tok)
                if maybe:
                    resolved_name = maybe.get("name", tok)
                    resolved_gid = maybe.get("gid", tok)
        except RequestCancelledError:
            raise HTTPException(status_code=499, detail="Client disconnected")
        except Exception as exc:
            LOGGER.exception("[locations.search_hierarchy] parent resolution failed token=%r", tok)
            raise HTTPException(status_code=500, detail="Location hierarchy search failed") from exc
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
        cancel_check()
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
            raw = _search_locations_with_optional_cancel(q, limit, cancel_check)
            for cand in raw:
                push_candidate_if_valid(cand)

        else:
            # 1) catalog-based enumeration (fast)
            if expected_level is not None and hasattr(gis_lookup, "load_location_catalog"):
                try:
                    entries, mapping = gis_lookup.load_location_catalog()
                    for rec in entries:
                        cancel_check()
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
                except RequestCancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "[locations.search_hierarchy] catalog enumeration failed level=%r parent=%r",
                        expected_level,
                        parent,
                    )
                    raise

            # 2) list_children if available
            if not candidates and hasattr(gis_lookup, "list_children"):
                for parent_tok in parent_tokens or []:
                    cancel_check()
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
                    except RequestCancelledError:
                        raise
                    except Exception:
                        LOGGER.exception(
                            "[locations.search_hierarchy] child enumeration failed parent=%r level=%r",
                            parent_tok,
                            expected_level,
                        )
                        raise

            # 3) letter-scan fallback — keep scanning letters until we have enough valid matches
            if not candidates:
                letters = "abcdefghijklmnopqrstuvwxyz"
                per_letter_limit = max(50, min(200, limit))
                for ch in letters:
                    cancel_check()
                    if len(candidates) >= limit:
                        break
                    try:
                        partial = _search_locations_with_optional_cancel(ch, per_letter_limit, cancel_check)
                    except RequestCancelledError:
                        raise
                    except Exception:
                        LOGGER.exception("[locations.search_hierarchy] letter scan failed prefix=%r", ch)
                        raise
                    for cand in partial:
                        push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break

    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(
            "[locations.search_hierarchy] failed q=%r level=%r parent=%r limit=%s",
            q,
            level,
            parent,
            limit,
        )
        raise HTTPException(status_code=500, detail="Location hierarchy search failed") from exc

    # final strict filter by level (redundant but safe)
    try:
        results: list[dict[str, Any]] = []
        for cand in candidates:
            cancel_check()
            if expected_level is not None and cand.get("level") != expected_level:
                continue
            results.append(
                {
                    "gid": str(cand.get("gid") or ""),
                    "name": cand.get("name") or "",
                    "level": cand.get("level", -999),
                    "hierarchy": cand.get("hierarchy") or [],
                }
            )
            if len(results) >= limit:
                break
    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc

    return {"results": results}


@app.get("/api/locations/{gid}")
def get_location_detail(gid: str) -> dict[str, Any]:
    """Return canonical hierarchy metadata for a location gid."""
    payload = gis_lookup.describe_location(gid)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown location {gid}")
    return payload


@app.get("/species/{taxon_id}/environment/{variable_id}")
def species_environment_stats(
    taxon_id: int,
    variable_id: str,
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
    unit_system: Optional[str] = Query(None, description="Unit system for response values (metric or imperial)"),
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
        observation_count = int(total_samples)
        summary = {
            "count": observation_count,
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
        else:
            ranks = indexing.load_relative_ranks(taxon_dir, variable_id)
        response = _with_env_observation_count(
            {
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
            },
            observation_count,
        )
        return units.apply_unit_system_to_env_response(response, unit_system, raw_units)

    if not location_gid:
        skip_summary = variable_id in no_summary_stats_variables
        observation_count = 0
        if value_type == "circular":
            _samples = summary_stats.gather_numeric_records(taxon_id, taxon_dir, variable_id)
            _values = [s["value"] for s in _samples]
            observation_count = len(_values)
            summary = (
                None if skip_summary else (summary_stats.summarize_values(_values, circular=True) if _values else None)
            )
            density_curve = (
                indexing.build_density_curve(_values, point_count=density_points, circular=True) if _values else None
            )
        else:
            summary = summary_stats.load_numeric_summary(str(taxon_dir), variable_id)
            if isinstance(summary, dict) and summary.get("count") is not None:
                try:
                    observation_count = int(summary.get("count") or 0)
                except (TypeError, ValueError):
                    observation_count = 0
            density_curve = summary_stats.load_density_graph(str(taxon_dir), variable_id)
        if (not skip_summary and not summary) or not density_curve:
            raise HTTPException(
                status_code=503,
                # We COULD compute on-demand here but I think it's better to fail loudly as the data *should* be here for performance reasons.
                detail=(
                    f"Precomputed summary stats or KDE missing (summary={bool(summary)} "
                    f"density={bool(density_curve)}). "
                    "Rebuild summary_stats.parquet and density_graph.parquet."
                ),
            )
        ranks = [] if skip_summary else indexing.load_relative_ranks(taxon_dir, variable_id)
        response = _with_env_observation_count(
            {
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
            },
            observation_count,
        )
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
    observation_count = len(values)
    skip_summary = variable_id in no_summary_stats_variables
    summary = None if skip_summary else summary_stats.summarize_values(values, circular=(value_type == "circular"))
    density_curve = indexing.build_density_curve(
        values, point_count=density_points, circular=(value_type == "circular")
    )
    ranks = []
    response = _with_env_observation_count(
        {
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
        },
        observation_count,
    )
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
    unit_system: Optional[str] = Query(None, description="Unit system for response values (metric or imperial)"),
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
    # For aspect_deg, min > max signals a selection that wraps through 0°/360°
    # (e.g. start=315°, end=45° means the arc through North). Preserve the order
    # so the query layer can issue two range queries and merge them.
    circular_wrap = variable_id == "aspect_deg" and max_value < min_value
    if max_value < min_value and not circular_wrap:
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
        if circular_wrap:
            rows_a = summary_stats.numeric_range_samples_for_location(
                taxon_id,
                variable_id,
                min_value,
                360.0,
                location_gid=location_gid,
                limit=limit,
            )
            rows_b = summary_stats.numeric_range_samples_for_location(
                taxon_id,
                variable_id,
                0.0,
                max_value,
                location_gid=location_gid,
                limit=limit,
            )
            seen: set[str] = set()
            for row in rows_a + rows_b:
                if row[0] not in seen:
                    seen.add(row[0])
                    rows.append(row)
            if limit:
                rows = rows[:limit]
        else:
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
            if circular_wrap:
                rows_a = summary_stats.get_sorted_layer_records_in_value_range(
                    index_path,
                    variable_id,
                    value_min=min_value,
                    value_max=360.0,
                    limit=limit,
                )
                rows_b = summary_stats.get_sorted_layer_records_in_value_range(
                    index_path,
                    variable_id,
                    value_min=0.0,
                    value_max=max_value,
                    limit=limit,
                )
                seen = set()
                for row in rows_a + rows_b:
                    if row[0] not in seen:
                        seen.add(row[0])
                        rows.append(row)
                if limit:
                    rows = rows[:limit]
            else:
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


@app.get("/gis/point")
def gis_point_value(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    variable: str = Query(..., description="GIS layer / variable id"),
    unit_system: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Returns the raster value for a variable at a lat/lon coordinate.

    This is the primitive used for both observation-pinning and
    arbitrary map-click lookups. Returns null value when the coordinate
    falls outside the layer's coverage or on a nodata pixel.


    Returns:
        A dict with variable, units, lat, lon, and value (float or null).
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=400, detail="lat and lon must be finite numbers")
    variable = variable.strip()
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable)
    if not variable_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable}' is not available.",
        )
    raw_units = variable_entry.get("units")
    if variable in gis_lookup._DERIVED_DEM_VARIABLES:
        raw_value = gis_lookup.sample_dem_derived_value(variable, lat, lon)
    else:
        cog_source = gis_lookup.get_cog_source(variable, lat, lon)
        if cog_source is None:
            return {
                "variable": variable,
                "units": raw_units,
                "lat": lat,
                "lon": lon,
                "value": None,
            }
        raw_value = gis_lookup.sample_raster_value(cog_source, lat, lon)

    display_scale = units.variable_display_scale(variable)
    if raw_value is not None and display_scale != 1.0:
        raw_value = raw_value * display_scale

    response = {
        "variable": variable,
        "units": raw_units,
        "lat": lat,
        "lon": lon,
        "value": raw_value,
        "class_name": None,
    }
    # Reuse the units machinery — wrap in a minimal env-response shape
    if raw_value is not None and unit_system and raw_units:
        wrapped = units.apply_unit_system_to_env_response(
            {"summary": {"mean": raw_value}, "units": raw_units},
            unit_system,
            raw_units,
        )
        converted_mean = (wrapped.get("summary") or {}).get("mean")
        if isinstance(converted_mean, (int, float)):
            response["value"] = converted_mean
            response["units"] = wrapped.get("units", raw_units)
    # Attach human-readable class name for categorical variables
    if raw_value is not None:
        legend = gis_lookup.load_layer_legend(variable)
        if legend:
            entry = legend.get(str(int(raw_value))) if raw_value == int(raw_value) else legend.get(str(raw_value))
            if entry:
                response["class_name"] = entry.get("name")
    return response


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
    df = custom_upload_processing._ensure_observation_names(df)
    df = custom_upload_processing._build_internal_upload_dataframe(df)
    print("Finished normalizing fields. adding tileID...")
    df = custom_upload_processing._add_tile_ids(df)
    print("Finished adding tileID, adding columns for GIS data, building parquet files...")
    df = custom_upload_processing._add_gis_columns(df)

    archive_path, out_name, work_dir = custom_upload_processing._build_index_archive(df)

    background_tasks.add_task(shutil.rmtree, work_dir, True)
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=out_name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
