# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import csv
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
import psutil
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from shapely.geometry.base import BaseGeometry
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.concurrency import run_in_threadpool

import util.rankings as rankings
from config.config import load_config
from scripts.build_tree import _is_usable_license, _normalize_license_url
from util import citations, descriptions, download, gis, taxa, tiles, units, upload
from util.rankings import POSITION_FILE, RANKINGS_FILE
from util.stats import (
    CATALOG_NUMBER_INDEX_FILE,
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    DENSITY_GRID_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    OCCURRENCES_FILE,
    ORDINAL_STATS_FILE,
    TREE_ROOT,
    apply_chained_filters,
    apply_phenology_filter,
    apply_polygon_filter,
    apply_timestamp_filter,
    collect_taxon_df,
    compute_location_filtered_stats,
    compute_phenology_counts,
    numeric_range_mask,
    parse_polygon_param,
    read_phenology_counts,
)
from util.storage import ParquetStorageProxy
from util.taxa import format_common_name, iter_descendants, normalize_name, reload_catalog, taxon_slug
from util.temporal import load_temporal_layers
from util.ternary import build_ternary_classification_overlay, composition_group_members

_CONFIG = load_config("global")
_SYNC_STATE_PATH = Path("data/sync_state.json")
_PIPELINE_STATE_PATH = Path("data/pipeline_state.json")
_TEMPORAL_STATE_PATH = Path("data/temporal_state.json")
_BASEMAP_STATE_PATH = Path("data/basemap_state.json")
# Deliberately outside data/, which gets wiped locally to clear bad state; this file is
# only ever rewritten by the deploy workflow, so a restart or a data/ wipe can't touch it.
_BUILD_DATE_PATH = Path("build_date.txt")

# ---------------------------------------------------------------------------
# Response caching — two explicit version tags rather than relying on a
# restart happening to coincide with a deploy (the box can restart for
# reasons that have nothing to do with either). Each cached endpoint below
# takes whichever version it actually depends on as an extra lru_cache key
# component, so a version bump makes old entries simply unreachable —
# correct even if reload is ever missed/delayed.
#
# _DEPLOY_VERSION: fixed once at process start. build_date.txt is
# deploy-workflow-write-only (see comment above), and a deploy always
# rebuilds+restarts the container (see .github/workflows/deploy.yml), so
# reading it once at import time is safe — it cannot change without a
# restart re-running this line anyway.
#
# _DATA_VERSION: NOT startup-time — a data rebuild can complete while this
# process keeps running (that's the whole reason /internal/reload exists
# without a restart). Bumped inside reload_data() to the wall-clock time
# reload fired, which is already the authoritative "pipeline just pushed
# fresh data" signal (scripts/rebuild.py POSTs here unconditionally after
# every sync).
#
# _TEMPORAL_VERSION: separate from both of the above — build_temporal.py's
# 30-minute cron pushes to /internal/temporal-state, not /internal/reload,
# and updates temporal-layer raster mtimes (see util.tiles.get_layer_version
# and get_layer_render_range's temporal-meta fallback) independently of both
# deploys and full data rebuilds. Anything whose response depends on a
# temporal layer's version/render-range needs this tag, not just the two
# above.
# ---------------------------------------------------------------------------


def _read_deploy_version() -> str:
    try:
        return _BUILD_DATE_PATH.read_text().strip() if _BUILD_DATE_PATH.exists() else "unknown"
    except Exception:
        return "unknown"


_DEPLOY_VERSION: str = _read_deploy_version()
_DATA_VERSION: str = "unknown"
_TEMPORAL_VERSION: str = "unknown"

_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent,
)
_LEGEND_DIR = Path("config/gis/legends")
_OCC_COLUMNS = ["catalogNumber", "decimalLatitude", "decimalLongitude", "obscured", "coordinateUncertaintyInMeters"]
_PHENOLOGY_VALUES: frozenset[str] = frozenset(_CONFIG.phenology_values)
_LARGE_TAXON_THRESHOLD = 500_000
_INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
# ArcGIS World Imagery satellite basemap — proxied so the API key never
# reaches the client (see /api/tiles/satellite below). Esri's tile path is
# z/y/x, not the z/x/y convention used everywhere else in this app.
_ARCGIS_SATELLITE_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_ARCGIS_API_KEY = os.environ.get("WW_ARCGIS_API_KEY")
# Matches the Referrer URL restriction configured on the ArcGIS API key
# itself — the key is scoped to only work for requests presenting this
# origin, so this proxy (the only thing that ever sends the real key) sends
# it explicitly rather than relying on httpx's default of no Referer at all.
_ARCGIS_REFERER = "https://wherewild.net"
# Esri returns a real 200 OK "data not available" placeholder tile (solid
# gray, near-zero JPEG entropy) instead of a 404 when a zoom level exceeds
# actual imagery coverage at that location — confirmed by hand: a known-bad
# tile came back at exactly 2521 bytes, twice, vs. tens of KB for real
# imagery. A little slack above that exact size covers minor variation
# across zoom/region without coming anywhere near a real tile's size.
_ARCGIS_NO_DATA_TILE_MAX_BYTES = 3000
# Self-hosted OpenMapTiles basemap — tileserver-gl reads the PMTiles + styles
# scripts/gis/build_basemap_tiles.py produces (see docker-compose.yml's
# `tileserver` service, docker/tileserver/config.json). Proxied through here
# rather than hit directly so it rides the same domain/Cloudflare zone as
# every other tile type, and so tileserver-gl itself never needs to be
# publicly reachable.
_TILESERVER_BASE = os.environ.get("WW_TILESERVER_BASE", "http://localhost:8791")
_BASEMAP_THEMES = frozenset(
    {
        "standard-light",
        "standard-dark",
        "standard-versatiles-light",
        "standard-versatiles-dark",
        "standard-openfreemap-light",
        "standard-openfreemap-dark",
        "variable-light",
        "labels",
    }
)
_LOCATIONS_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "gis" / "locations"
_LOC_TAXA_PATH = _LOCATIONS_DIR / "location_taxa.parquet"

# ---------------------------------------------------------------------------
# Rate limiting — per-IP, in-memory (single-host deployment, no shared state
# needed across processes). Tiers are grouped by how expensive/abusable each
# endpoint is, from an audit of every route against real data — see
# 2026-08-16 planning notes. `default_limits` is the floor every route not
# given a more specific `@limiter.limit(...)` falls back to.
# ---------------------------------------------------------------------------

_RATE_LIMIT_DEFAULT = "300/minute"
_RATE_LIMIT_STATUS = "12/minute"
_RATE_LIMIT_CHEAP = "120/minute"
_RATE_LIMIT_DETAIL = "60/minute"
_RATE_LIMIT_SEARCH = "30/minute"
_RATE_LIMIT_SUBTREE_RAW = "20/minute"
_RATE_LIMIT_DOWNLOAD = "5/minute"
_RATE_LIMIT_UPLOAD = "5/minute"
_RATE_LIMIT_TILES = "1000/minute"
_RATE_LIMIT_SATELLITE = "120/minute"
_RATE_LIMIT_TILE_RANGE = "30/minute"


def _client_key(request: Request) -> str:
    """Real client IP — this app sits behind cloudflared in prod, so the
    socket peer is always the tunnel, not the actual requester. Cloudflare
    sets cf-connecting-ip on every proxied request; falls back to
    X-Forwarded-For's first hop, then the raw socket address for direct
    local/dev access."""
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_key, default_limits=[_RATE_LIMIT_DEFAULT])


# Ranks at/above GENUS whose raw-occurrence-subtree endpoints
# (occurrences/download/observation-values/slice/class-samples/obscured/
# filtered environment) get rejected outright, regardless of observation
# count — the actual cost driver for those endpoints is collect_taxon_df's/
# _read_occurrences_scoped's subtree traversal (reading + deduping every
# descendant leaf's rows), which is slow at genus-and-above even well under
# the observation-count threshold below.
_EXPENSIVE_SUBTREE_RANKS = frozenset({"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS"})


def _taxon_observation_count(taxon: dict) -> int:
    try:
        _num_rows = _storage.read_table(
            GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE,
            filters=[("taxon_key", "=", str(taxon["taxon_key"]))],
        ).to_pylist()
        return max((int(r["count"]) for r in _num_rows if r.get("count")), default=0)
    except Exception:
        return 0


def _is_expensive_subtree_taxon(taxon: dict, observation_count: int | None = None) -> bool:
    """True if this taxon's raw-occurrence-subtree endpoints should be
    rejected: rank at/above GENUS (see _EXPENSIVE_SUBTREE_RANKS), or —
    belt-and-suspenders for a pathologically large SPECIES/SUBSPECIES-rank
    taxon — an observation count at/above _LARGE_TAXON_THRESHOLD."""
    if taxon["rank"] in _EXPENSIVE_SUBTREE_RANKS:
        return True
    if observation_count is None:
        observation_count = _taxon_observation_count(taxon)
    return observation_count >= _LARGE_TAXON_THRESHOLD


def _reject_if_large_taxon(taxon: dict) -> None:
    """Block per-observation aggregation (filtering, slicing, raw sample/value
    listing, downloading) for taxa too large/broad to do it live — matches the
    frontend's large-taxon map/filter/download disable threshold (see
    get_taxon's large_taxon field, which uses the same helper)."""
    if _is_expensive_subtree_taxon(taxon):
        raise HTTPException(status_code=400, detail="large_taxon")


def _resolve_variable_id(variable_id: str) -> str:
    """Normalise variable ids, keeping backward compat with old bio_1 → bio1 format.

    Only strips underscores when the id is not already a known layer — preserves
    temporal ids like temperature_2m_avg_24h unchanged.
    """
    known = {layer["id"] for layer in tiles.load_layers()}
    if variable_id in known:
        return variable_id
    stripped = variable_id.replace("_", "")
    return stripped


@lru_cache(maxsize=32)
def _load_legend(layer_id: str) -> list:
    if not re.fullmatch(r"[A-Za-z0-9_]+", layer_id):
        return []
    legend_root = os.path.realpath(_LEGEND_DIR)
    path = os.path.realpath(_LEGEND_DIR / f"{layer_id}_legend.json")
    if not path.startswith(legend_root + os.sep):
        return []
    if not os.path.exists(path):
        # Temporal ids like weather_code_simple_mode_24h → weather_code_simple
        base_id = re.sub(r'_(avg|sum|mode|snapshot)_\d+h$', '', layer_id, flags=re.IGNORECASE)
        if base_id != layer_id:
            if not re.fullmatch(r"[A-Za-z0-9_]+", base_id):
                return []
            path = os.path.realpath(_LEGEND_DIR / f"{base_id}_legend.json")
            if not path.startswith(legend_root + os.sep):
                return []
    if not os.path.exists(path):
        return []
    return json.loads(Path(path).read_text()).get("classes", [])


def _load_legend_full(layer_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_]+", layer_id):
        return {}
    legend_root = os.path.realpath(_LEGEND_DIR)
    path = os.path.realpath(_LEGEND_DIR / f"{layer_id}_legend.json")
    if not path.startswith(legend_root + os.sep) or not os.path.exists(path):
        return {}
    return json.loads(Path(path).read_text())


def _scope_taxon_keys(taxon: dict) -> list[str]:
    """taxon_keys in scope for occurrence-level queries: species rolls up
    subspecies/variety/form; a leaf is itself; other ranks read only their
    descendant leaves (not any stray direct-to-ancestor observations) —
    mirrors util.stats.collect_taxon_df's own scope dispatch."""
    rank = taxon["rank"]
    if rank == _CONFIG.species_rank:
        return [str(t["taxon_key"]) for t in iter_descendants(taxon, include_self=True)]
    if rank in _CONFIG.leaf_rank_set:
        return [str(taxon["taxon_key"])]
    return [str(t["taxon_key"]) for t in iter_descendants(taxon, include_self=False)]


def _read_occurrences_scoped(taxon: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """Occurrence rows for taxon's scope, read from the consolidated occurrences file."""
    try:
        schema_names = set(_storage.read_schema(OCCURRENCES_FILE).names)
    except Exception:
        return pd.DataFrame()
    cols = [c for c in columns if c in schema_names] if columns is not None else None
    keys = _scope_taxon_keys(taxon)
    try:
        table = _storage.read_table(OCCURRENCES_FILE, columns=cols, filters=[("taxon_key", "in", keys)])
    except Exception:
        return pd.DataFrame()
    return table.to_pandas()


def _lookup_index_value(taxon: dict, variable_id: str, catalog_number: str) -> float | None:
    """Read an env value for a known observation directly from the consolidated occurrences file."""
    try:
        tbl = _storage.read_table(
            OCCURRENCES_FILE,
            columns=["catalogNumber", variable_id],
            filters=[("taxon_key", "=", str(taxon["taxon_key"])), ("catalogNumber", "=", catalog_number)],
        )
        if tbl.num_rows == 0 or variable_id not in tbl.schema.names:
            return None
        val = tbl.column(variable_id)[0].as_py()
        return float(val) if val is not None else None
    except Exception:
        return None


def _filter_occ_df(df: pd.DataFrame) -> pd.DataFrame:
    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        col = df["coordinateUncertaintyInMeters"]
        df = df[col.isna() | (col <= 500)]
    return df

# ---------------------------------------------------------------------------
# Upload job queue
# ---------------------------------------------------------------------------

_MAX_UPLOAD_ROWS = 50_000
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # well above what 50k CSV/TSV/Parquet rows needs
_MAX_CONCURRENT_UPLOAD_JOBS = 20  # bounds worst-case memory (up to 50k-row DataFrame each) between TTL sweeps
_DONE_TTL_SECONDS = 3600  # archive stays available for 1 hour after completion


@dataclass
class _UploadJob:
    job_id: str
    df: pd.DataFrame
    status: str = "queued"       # queued | processing | done | error
    archive_path: Path | None = None
    archive_name: str | None = None
    work_dir: Path | None = None
    error: str | None = None
    done_at: float | None = None


_upload_queue: list[str] = []        # ordered job IDs waiting to run
_upload_jobs: dict[str, _UploadJob] = {}


async def _upload_consumer() -> None:
    while True:
        if not _upload_queue:
            await asyncio.sleep(0.2)
            continue
        job_id = _upload_queue.pop(0)
        job = _upload_jobs.get(job_id)
        if job is None:
            continue
        job.status = "processing"
        try:
            df = await run_in_threadpool(upload.enrich_with_gadm, job.df)
            df = await run_in_threadpool(upload.enrich_with_gis, df)
            df = await run_in_threadpool(upload.enrich_with_temporal, df)
            archive_path, archive_name, work_dir = await run_in_threadpool(upload.build_archive, df)
            job.archive_path = archive_path
            job.archive_name = archive_name
            job.work_dir = work_dir
            job.status = "done"
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
        finally:
            job.done_at = time.monotonic()


async def _cleanup_old_jobs() -> None:
    while True:
        await asyncio.sleep(300)
        now = time.monotonic()
        expired = [
            jid for jid, job in list(_upload_jobs.items())
            if job.done_at is not None and (now - job.done_at) > _DONE_TTL_SECONDS
        ]
        for jid in expired:
            job = _upload_jobs.pop(jid, None)
            if job and job.work_dir:
                shutil.rmtree(job.work_dir, ignore_errors=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    psutil.cpu_percent(interval=None)  # prime the delta tracker — see _status_server
    asyncio.create_task(_upload_consumer())
    asyncio.create_task(_cleanup_old_jobs())
    yield


app = FastAPI(lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Nominal-Classes", "Content-Disposition"],
)


def _license_label(url: str | None) -> str | None:
    """Derive a short display label from a canonical CC license URL.

    The catalog always stores canonical https CC URLs (normalized at build time),
    so this only needs to handle URL→label, not short codes.
    """
    if not url:
        return None
    m = re.search(r"/publicdomain/zero/([^/]+)/", url)
    if m:
        return f"CC0 {m.group(1)}"
    m = re.search(r"/licenses/([^/]+)/([^/]+)/", url)
    if m:
        parts = m.group(1).split("-")
        return "CC " + "-".join(p.upper() for p in parts) + " " + m.group(2)
    return url  # fallback: show whatever is stored


def _image_fields(taxon: dict) -> dict:
    """Return unified image_* fields, preferring iNat over GBIF backup."""
    prefix = "inat_preferred" if taxon.get("inat_preferred_image") else "gbif_backup"
    license_url = taxon.get(f"{prefix}_image_license") or None
    return {
        "image_url": taxon.get(f"{prefix}_image") or None,
        "image_license": _license_label(license_url),
        "image_license_url": license_url,
        "image_creator": taxon.get(f"{prefix}_image_creator") or None,
        "image_rights_holder": taxon.get(f"{prefix}_image_attribution") or None,
        "image_references": taxon.get(f"{prefix}_image_references") or None,
    }


# Coarse UI-rendering-mode bucket used elsewhere in this file (legend_classes
# gating, etc.) — collapses the precise measurement level down to whether the
# frontend should render it as a slider, a category picker, or a compass.
# /variables also returns the precise, unbucketed level as raw_value_type
# (nominal/ordinal/interval/ratio/circular) for anything that needs it —
# e.g. distinguishing interval from ratio — since this map is lossy.
_VALUE_TYPE_MAP = {"interval": "continuous", "ratio": "continuous", "nominal": "categorical", "ordinal": "ordinal", "circular": "circular"}


@app.get("/")
@limiter.limit(_RATE_LIMIT_CHEAP)
def root(request: Request):
    return {"status": "ok"}


@app.get("/version")
@limiter.limit(_RATE_LIMIT_CHEAP)
def version(request: Request):
    try:
        state = json.loads(_SYNC_STATE_PATH.read_text()) if _SYNC_STATE_PATH.exists() else {}
        crawl_ts = (
            state.get("gbif_occurrences", {}).get("crawl_finished")
            or state.get("gbif_taxonomy", {}).get("crawl_finished")
        )
    except Exception:
        crawl_ts = None
    try:
        api_build_date = (
            _BUILD_DATE_PATH.read_text().strip() if _BUILD_DATE_PATH.exists() else None
        )
    except Exception:
        api_build_date = None
    return {"version": crawl_ts, "api_build_date": api_build_date or None}


_STATUS_CACHE_TTL_S = 2.0
_status_cache: tuple[float, dict] | None = None


async def _compute_status() -> dict:
    pipeline = await run_in_threadpool(_status_pipeline)
    temporal = await run_in_threadpool(_status_temporal)
    basemap = await run_in_threadpool(_status_basemap)
    server = await run_in_threadpool(_status_server)
    active_job = next(
        (j for j in _upload_jobs.values() if j.status == "processing"), None
    )
    return {
        "pipeline": pipeline,
        "temporal": temporal,
        "basemap": basemap,
        "upload_queue": {
            "depth": len(_upload_queue),
            "active": active_job is not None,
        },
        "server": server,
    }


@app.get("/status")
@limiter.limit(_RATE_LIMIT_STATUS)
async def status(request: Request):
    """Short TTL cache on top of _compute_status — a burst of concurrent
    callers within the same window shares one computation instead of each
    re-running /proc reads, psutil.sensors_temperatures(), and two
    state-file reads."""
    global _status_cache
    now = time.monotonic()
    if _status_cache is not None and (now - _status_cache[0]) < _STATUS_CACHE_TTL_S:
        return _status_cache[1]
    payload = await _compute_status()
    _status_cache = (now, payload)
    return payload


@app.post("/internal/pipeline-state", status_code=200)
async def push_pipeline_state(body: dict):
    from datetime import UTC
    from datetime import datetime as _dt
    body["received_at"] = _dt.now(UTC).isoformat()
    await run_in_threadpool(
        lambda: _PIPELINE_STATE_PATH.write_text(json.dumps(body))
    )
    return {"ok": True}


@app.post("/internal/reload", status_code=200)
async def reload_data():
    """Clear all in-process caches so the next request reads fresh data from disk.
    Called after a rebuild push so the API picks up the new catalog without a restart."""
    global _DATA_VERSION
    reload_catalog()
    _load_legend.cache_clear()
    _load_hierarchy.cache_clear()
    tiles._catalog.cache_clear()
    tiles._load_nominal_colormap.cache_clear()
    build_ternary_classification_overlay.cache_clear()
    gis.clear_dataset_cache()
    _DATA_VERSION = datetime.now(UTC).isoformat()
    _clear_data_versioned_caches()
    return {"ok": True}


@app.post("/internal/temporal-state", status_code=200)
async def push_temporal_state(body: dict):
    global _TEMPORAL_VERSION
    from datetime import UTC
    from datetime import datetime as _dt
    body["received_at"] = _dt.now(UTC).isoformat()

    def _write():
        _TEMPORAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TEMPORAL_STATE_PATH.write_text(json.dumps(body))

    await run_in_threadpool(_write)
    _TEMPORAL_VERSION = _dt.now(UTC).isoformat()
    _clear_temporal_versioned_caches()
    return {"ok": True}


@app.post("/internal/basemap-state", status_code=200)
async def push_basemap_state(body: dict):
    from datetime import UTC
    from datetime import datetime as _dt
    body["received_at"] = _dt.now(UTC).isoformat()

    def _write():
        _BASEMAP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BASEMAP_STATE_PATH.write_text(json.dumps(body))

    await run_in_threadpool(_write)
    return {"ok": True}


def _status_pipeline() -> dict | None:
    # Prefer push-populated file (gambaby); fall back to local sync_state.json (GamBase)
    path = _PIPELINE_STATE_PATH if _PIPELINE_STATE_PATH.exists() else _SYNC_STATE_PATH
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        state = raw.get("pipeline", raw) if path == _SYNC_STATE_PATH else raw
    except Exception:
        return None
    from datetime import UTC
    from datetime import datetime as _dt
    now = _dt.now(UTC)
    stage = state.get("stage")
    stage_elapsed_s = None
    if state.get("status") == "in_progress" and stage:
        stage_entry = state.get("stages", {}).get(stage, {})
        started = stage_entry.get("started_at")
        if started:
            try:
                stage_elapsed_s = int((now - _dt.fromisoformat(started)).total_seconds())
            except Exception:
                pass
    return {
        "status": state.get("status"),
        "stage": stage,
        "stage_elapsed_s": stage_elapsed_s,
        "last_finished_at": state.get("finished_at"),
        "last_duration_s": state.get("duration_s"),
        "received_at": state.get("received_at"),
    }


def _status_temporal() -> dict | None:
    if not _TEMPORAL_STATE_PATH.exists():
        return None
    try:
        state = json.loads(_TEMPORAL_STATE_PATH.read_text())
    except Exception:
        return None
    from datetime import UTC
    from datetime import datetime as _dt
    elapsed_s = None
    if state.get("status") == "running":
        started = state.get("started_at")
        if started:
            try:
                elapsed_s = int((_dt.now(UTC) - _dt.fromisoformat(started)).total_seconds())
            except Exception:
                pass
    return {
        "status": state.get("status"),
        "elapsed_s": elapsed_s,
        "last_finished_at": state.get("completed_at"),
        "last_duration_s": state.get("duration_s"),
        "received_at": state.get("received_at"),
    }


def _status_basemap() -> dict | None:
    if not _BASEMAP_STATE_PATH.exists():
        return None
    try:
        state = json.loads(_BASEMAP_STATE_PATH.read_text())
    except Exception:
        return None
    from datetime import UTC
    from datetime import datetime as _dt
    elapsed_s = None
    if state.get("status") == "running":
        started = state.get("started_at")
        if started:
            try:
                elapsed_s = int((_dt.now(UTC) - _dt.fromisoformat(started)).total_seconds())
            except Exception:
                pass
    return {
        "status": state.get("status"),
        "stage": state.get("stage"),
        "elapsed_s": elapsed_s,
        "last_finished_at": state.get("completed_at"),
        "last_duration_s": state.get("duration_s"),
        "received_at": state.get("received_at"),
    }


def _status_server() -> dict:
    result: dict = {}

    # CPU usage — psutil.cpu_percent(interval=None) is non-blocking: it
    # reports the delta since the *last* call using psutil's own internal
    # timestamp, rather than this function sleeping 300ms itself. Primed
    # once at startup (see _lifespan) so the first real request doesn't
    # read 0.0. Non-blocking matters here specifically because this route
    # runs in FastAPI's shared sync threadpool — a blocking sleep here
    # occupies a worker thread doing nothing, and a burst of concurrent
    # /status calls could starve every other sync endpoint.
    try:
        result["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
    except Exception:
        result["cpu_percent"] = None

    # CPU temp
    try:
        temps = psutil.sensors_temperatures()
        cpu_temp = None
        for name, entries in temps.items():
            for entry in entries:
                label = entry.label.lower()
                if label.startswith("package id 0") or label.startswith("cpu"):
                    cpu_temp = round(entry.current, 1)
                    break
            if cpu_temp is not None:
                break
        result["cpu_temp_c"] = cpu_temp
    except Exception:
        result["cpu_temp_c"] = None

    # RAM
    try:
        mem: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    mem[k.strip()] = int(v.strip().split()[0])
        ram_total_mb = mem["MemTotal"] // 1024
        ram_used_mb = ram_total_mb - mem.get("MemAvailable", mem.get("MemFree", 0)) // 1024
        result["ram_used_mb"] = ram_used_mb
        result["ram_total_mb"] = ram_total_mb
    except Exception:
        result["ram_used_mb"] = None
        result["ram_total_mb"] = None

    # Disk
    try:
        st = os.statvfs("/")
        result["disk_used_gb"] = (st.f_blocks - st.f_bfree) * st.f_frsize // (1024 ** 3)
        result["disk_total_gb"] = st.f_blocks * st.f_frsize // (1024 ** 3)
    except Exception:
        result["disk_used_gb"] = None
        result["disk_total_gb"] = None

    # Disk 2 (overflow GIS layers dir, e.g. WHEREWILD_EXTRA_LAYERS_DIRS on prod)
    extra_dirs = tiles._extra_layers_dirs()
    if extra_dirs:
        try:
            st2 = os.statvfs(extra_dirs[0])
            result["disk2_used_gb"] = (st2.f_blocks - st2.f_bfree) * st2.f_frsize // (1024 ** 3)
            result["disk2_total_gb"] = st2.f_blocks * st2.f_frsize // (1024 ** 3)
        except Exception:
            result["disk2_used_gb"] = None
            result["disk2_total_gb"] = None
    else:
        result["disk2_used_gb"] = None
        result["disk2_total_gb"] = None

    # Uptime
    try:
        with open("/proc/uptime") as f:
            result["uptime_s"] = int(float(f.read().split()[0]))
    except Exception:
        result["uptime_s"] = None

    return result


@lru_cache(maxsize=4)
def _cached_data_sources(deploy_version: str):
    return citations.load_data_sources()


@app.get("/data-sources")
@limiter.limit(_RATE_LIMIT_CHEAP)
def data_sources(request: Request):
    return _cached_data_sources(_DEPLOY_VERSION)


@lru_cache(maxsize=32)
def _cached_list_variables(
    unit_system: str | None, forecast_h: int,
    deploy_version: str, data_version: str, temporal_version: str,
) -> list:
    forecast_suffix = f"__f{forecast_h:03d}h" if forecast_h in _VALID_FORECAST_HOURS and forecast_h > 0 else ""
    result = []
    for layer, category in tiles.load_layers_with_category():
        value_type = _VALUE_TYPE_MAP.get(layer.get("value_type", ""), "continuous")
        legend_classes = None
        if value_type in ("categorical", "ordinal"):
            raw = _load_legend(layer["id"])
            if raw:
                legend_classes = [
                    {
                        "id": cls["id"],
                        "name": cls.get("name", str(cls["id"])),
                        "color": cls.get("traits", {}).get("color") or None,
                    }
                    for cls in raw
                ]
        rmin, rmax = tiles.get_layer_render_range(layer, forecast_suffix)
        result.append({
            "id": layer["id"],
            "name": layer.get("display_name"),
            "units": units.display_units(layer, unit_system),
            "value_type": value_type,
            "raw_value_type": layer.get("value_type") or None,
            "domain": layer.get("domain") or None,
            "category": category.get("display_name", "Other"),
            "source_ids": list(dict.fromkeys(filter(None, [layer.get("source"), layer.get("model")]))) or None,
            "legend_classes": legend_classes,
            "render_min": units.convert_value(rmin, layer, unit_system),
            "render_max": units.convert_value(rmax, layer, unit_system),
            "group": layer.get("group") or None,
            "group_label": layer.get("group_label") or None,
            "agg": layer.get("agg") or None,
            "version": tiles.get_layer_version(layer, forecast_suffix),
            "composition_group": layer.get("composition_group") or None,
            "composition_axis": layer.get("composition_axis") or None,
            "composition_label": layer.get("composition_label") or None,
        })
    return result


@app.get("/variables")
@limiter.limit(_RATE_LIMIT_CHEAP)
def list_variables(request: Request, unit_system: str | None = Query(None), forecast_h: int = Query(0, ge=0)):
    # Mixes deploy-only catalog/legend content with layer version/render-range
    # values that can change on either the main data rebuild (static layers)
    # or the 30-min temporal cron (temporal layers) — needs all three tags.
    return _cached_list_variables(unit_system, forecast_h, _DEPLOY_VERSION, _DATA_VERSION, _TEMPORAL_VERSION)


@lru_cache(maxsize=4)
def _cached_list_layers(deploy_version: str):
    return tiles.load_layers()


@app.get("/api/layers")
@limiter.limit(_RATE_LIMIT_CHEAP)
def list_layers(request: Request):
    return _cached_list_layers(_DEPLOY_VERSION)


@lru_cache(maxsize=4)
def _cached_list_phenology_values(deploy_version: str):
    return [
        {"value": v, "label": v.capitalize()}
        for v in sorted(_CONFIG.phenology_values)
    ]


@app.get("/phenology_values")
@limiter.limit(_RATE_LIMIT_CHEAP)
def list_phenology_values(request: Request):
    return _cached_list_phenology_values(_DEPLOY_VERSION)


def _lookup_temporal_value_at_timestamp(
    variable: str, lat: float, lon: float, event_ts: int,
) -> float | None:
    """Live single-point historical lookup for a temporal variable at a known
    observation timestamp.

    Reuses the upload flow's per-layer processing (util.upload's
    _process_one_layer + _df_to_occ_table) with a one-row occ table — same
    HTTP-range-request path against the ERA5 .om chunks, just for one point
    instead of a batch. This is what makes "highlight this exact observation"
    show the value AT THE TIME it was made rather than gis.sample_point's
    current/live window, which is what a plain map click means.

    Scoped to base window-aggregate layers (derived temporal layers are
    computed from other already-processed columns, not fetched directly —
    see load_temporal_layers/TemporalLayer.derived) — returns None for
    anything else so the caller falls back to gis.sample_point.
    """
    try:
        layers = load_temporal_layers(tiles.CATALOG_PATH)
    except Exception:
        return None
    candidates = [lyr for lyr in layers if not lyr.derived and variable.startswith(f"{lyr.id}_")]
    if not candidates:
        return None

    occ_table = upload._df_to_occ_table(pd.DataFrame({
        "decimalLatitude": [lat],
        "decimalLongitude": [lon],
        "eventTimestamp": [float(event_ts)],
    }))

    for layer in candidates:
        try:
            updates = upload._process_one_layer(layer, occ_table)
        except Exception:
            continue
        pairs = updates.get("__upload__", {}).get(variable)
        if not pairs:
            continue
        _, values = pairs[0]
        if len(values) == 0:
            continue
        val = values[0]
        if val is None or not math.isfinite(float(val)):
            return None
        return float(val)
    return None


@app.get("/gis/point")
@limiter.limit(_RATE_LIMIT_DETAIL)
async def gis_point_value(
    request: Request,
    lat: float = Query(...),
    lon: float = Query(...),
    variable: str = Query(...),
    taxon_id: str | None = Query(None),
    catalog_number: str | None = Query(None),
    unit_system: str | None = Query(None),
    forecast_h: int = Query(0, ge=0),
    event_ts: int | None = Query(None),
    colormap: str = Query("viridis"),
):
    """Return the raster value for a variable at a lat/lon coordinate.

    If taxon_id and catalog_number are both provided the value is read from
    occurrence_index.parquet instead of the raster — ensures the returned value
    is identical to what the stats were computed from, and for temporal variables
    returns the historical aggregate at observation time rather than the current
    live window. Falls back to raster sampling when the index row is missing.

    event_ts covers the same "value at observation time, not now" case for a
    temporal variable when there's no index row to read (e.g. an observation
    not yet ingested by GBIF — see GET /occurrence/{id}'s "ingested": false):
    a live single-point historical lookup at that exact timestamp, computed
    before falling all the way through to the current/live raster.
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=400, detail="lat and lon must be finite numbers")

    variable = _resolve_variable_id(variable.strip())
    try:
        layer = tiles.get_layer(variable)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Variable '{variable}' not found")

    if forecast_h not in _VALID_FORECAST_HOURS:
        forecast_h = 0
    forecast_suffix = f"__f{forecast_h:03d}h" if forecast_h > 0 else ""

    value: float | None = None

    if taxon_id and catalog_number:
        taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
        if taxon is not None:
            value = _lookup_index_value(taxon, variable, catalog_number)

    if value is None and event_ts is not None:
        value = await run_in_threadpool(
            _lookup_temporal_value_at_timestamp, variable, lat, lon, event_ts,
        )

    if value is None:
        value = await run_in_threadpool(gis.sample_point, layer, lat, lon, forecast_suffix)

    class_name: str | None = None
    class_color: str | None = None
    value_type = layer.get("value_type")
    if value is not None and value_type in ("nominal", "ordinal"):
        legend = _load_legend(variable)
        # round(), not exact equality — a sampled/reprojected raster value
        # isn't guaranteed to land on a mathematically exact integer (e.g.
        # nearest-neighbor resampling, or float32 round-trip), so an exact
        # `value == int(value)` check can silently fail to match any class,
        # leaving class_name null and the UI falling back to showing the
        # raw numeric value instead of a class name.
        int_val = round(value)
        for entry in legend:
            if entry.get("id") == int_val:
                class_name = entry.get("name")
                if value_type == "ordinal":
                    # Ordinal has no per-class traits.color — its color is
                    # the same live-colormap-stepped lookup the tile
                    # renderer uses (see util/tiles.py's matching branch),
                    # not a hand-picked legend value.
                    rgb = tiles._cb_colormap_for_layer(variable, colormap).get(int_val)
                    class_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" if rgb else None
                else:
                    class_color = (entry.get("traits") or {}).get("color") or None
                break

    converted_value = units.convert_value(value, layer, unit_system)
    return {
        "variable": variable,
        "units": units.display_units(layer, unit_system),
        "lat": lat,
        "lon": lon,
        "value": converted_value,
        "class_name": class_name,
        "class_color": class_color,
    }


_VALID_FORECAST_HOURS = {0, 1, 8, 24, 72, 168}


def _parse_value_ranges(
    value_ranges: str | None, layer: dict, unit_system: str | None,
) -> list[tuple[float | None, float | None]] | None:
    """Parse a `value_ranges` query param (JSON list of [min, max] pairs) and
    convert each bound from the display unit system to raw/metric using
    `layer` — a layer's own filter can itself be multiple disjoint ranges
    (OR'd), not just one pair. Same per-value conversion the single
    value_min/value_max params used to get, just applied per-pair.
    """
    if not value_ranges:
        return None
    try:
        pairs = json.loads(value_ranges)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(pairs, list):
        return None
    parsed: list[tuple[float | None, float | None]] = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        value_min, value_max = pair
        if value_min is not None:
            value_min = units.convert_value_from_display(value_min, layer, unit_system)
        if value_max is not None:
            value_max = units.convert_value_from_display(value_max, layer, unit_system)
        parsed.append((value_min, value_max))
    return parsed or None


def _parse_render_range(
    render_range: str | None, layer: dict, unit_system: str | None,
) -> tuple[float, float] | None:
    """Parse the `render_range` query param (JSON `[min, max]`) — the maps
    page's "auto-adapt" mode overriding a numeric layer's colorization scale
    to the range it discovered (via GET .../tile-range/stats below) across
    the currently-visible tiles, instead of the layer's fixed catalog
    render_min/render_max. Same display-unit-system convention as
    value_ranges above — converted back to raw/metric here.
    """
    if not render_range:
        return None
    try:
        pair = json.loads(render_range)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    value_min, value_max = pair
    if not isinstance(value_min, (int, float)) or not isinstance(value_max, (int, float)):
        return None
    return (
        units.convert_value_from_display(value_min, layer, unit_system),
        units.convert_value_from_display(value_max, layer, unit_system),
    )


def _parse_and_convert_chain(chain: str | None, unit_system: str | None) -> list[dict] | None:
    """Parse the `chain` query param (JSON list of {layer_id, class_filter?, value_ranges?})
    and convert each entry's value_ranges from the display unit system to
    raw/metric using THAT entry's own layer — each chained layer can have its own
    units/scale, distinct from the primary layer's.
    """
    if not chain:
        return None
    try:
        entries = json.loads(chain)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(entries, list):
        return None
    parsed: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("layer_id"):
            continue
        try:
            chain_layer = tiles.get_layer(entry["layer_id"])
        except KeyError:
            continue
        value_ranges = []
        for pair in entry.get("value_ranges") or []:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            value_min, value_max = pair
            if value_min is not None:
                value_min = units.convert_value_from_display(value_min, chain_layer, unit_system)
            if value_max is not None:
                value_max = units.convert_value_from_display(value_max, chain_layer, unit_system)
            value_ranges.append((value_min, value_max))
        parsed.append({
            "layer_id": entry["layer_id"],
            "class_filter": entry.get("class_filter"),
            "value_ranges": value_ranges or None,
        })
    return parsed or None


@app.get("/api/variables/{variable_id}/tiles/{z}/{x}/{y}.png")
@limiter.limit(_RATE_LIMIT_TILES)
async def variable_tile_compat(
    request: Request,
    variable_id: str, z: int, x: int, y: int,
    tile_size: int = Query(256, ge=32, le=1024), colormap: str = Query("viridis"),
    cb_mode: str = Query(""), forecast_h: int = Query(0, ge=0),
    class_filter: list[int] | None = Query(None),
    value_ranges: str | None = Query(None),
    unit_system: str | None = Query(None),
    chain: str | None = Query(None),
    render_range: str | None = Query(None),
):
    """Compatibility shim for old frontend URL pattern (/api/variables/bio_1/ → bio1)."""
    layer_id = _resolve_variable_id(variable_id)
    return await layer_tile(
        request, layer_id, z, x, y, tile_size, colormap, cb_mode, forecast_h,
        class_filter, value_ranges, unit_system, chain, render_range,
    )


@app.get("/api/layers/{layer_id}/tiles/{z}/{x}/{y}.png")
@limiter.limit(_RATE_LIMIT_TILES)
async def layer_tile(
    request: Request,
    layer_id: str, z: int, x: int, y: int,
    tile_size: int = Query(256, ge=32, le=1024),
    colormap: str = Query("viridis"),
    cb_mode: str = Query(""),
    forecast_h: int = Query(0, ge=0),
    class_filter: list[int] | None = Query(None),
    value_ranges: str | None = Query(None),
    unit_system: str | None = Query(None),
    chain: str | None = Query(None),
    render_range: str | None = Query(None),
):
    if colormap not in tiles.SUPPORTED_COLORMAPS and colormap not in tiles.SUPPORTED_CIRCULAR_COLORMAPS:
        colormap = "viridis"
    if cb_mode not in tiles.SUPPORTED_CB_MODES:
        cb_mode = ""
    if forecast_h not in _VALID_FORECAST_HOURS:
        forecast_h = 0
    try:
        layer = tiles.get_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")

    # The legend's slice-selection UI displays value ranges in the user's
    # current unit system (imperial converts e.g. °C -> °F), same as
    # render_min/render_max in the variable metadata response — but tile
    # pixels (and the layer's own render_min/render_max used for coloring)
    # are always raw/metric. Without this, an imperial-unit selection gets
    # applied as if it were already metric (see the /observation-values
    # slicing endpoint, which converts back the same way).
    parsed_value_ranges = _parse_value_ranges(value_ranges, layer, unit_system)

    parsed_chain = _parse_and_convert_chain(chain, unit_system)
    parsed_render_range = _parse_render_range(render_range, layer, unit_system)

    forecast_suffix = f"__f{forecast_h:03d}h" if forecast_h > 0 else ""
    payload = await run_in_threadpool(
        tiles.render_layer_tile_bytes,
        layer_id, z, x, y, tile_size, colormap, cb_mode, forecast_suffix, class_filter,
        parsed_value_ranges, parsed_chain, parsed_render_range,
    )
    is_temporal = layer.get("window_hours") is not None
    # URLs are versioned client-side with the layer's mtime-derived version token
    # (see tiles.get_layer_version), so a given URL's content never changes —
    # safe to cache aggressively; staleness is bounded by how promptly clients
    # pick up a new version, not by this TTL.
    cache_control = "public, max-age=86400" if is_temporal else "public, max-age=31536000, immutable"
    headers: dict[str, str] = {"Cache-Control": cache_control}
    if str(layer.get("value_type") or "").lower() in ("nominal", "ordinal"):
        class_counts = await run_in_threadpool(tiles.nominal_tile_range_classes, layer_id, z, x, y, x, y)
        if class_counts:
            ordered = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
            headers["X-Nominal-Classes"] = ",".join(f"{cls}:{cnt}" for cls, cnt in ordered)
    return Response(content=payload, media_type="image/png", headers=headers)


@app.get("/api/layers/elevation/terrain-tiles/{z}/{x}/{y}.png")
@limiter.limit(_RATE_LIMIT_TILES)
async def elevation_terrain_tile(
    request: Request,
    z: int, x: int, y: int,
    tile_size: int = Query(256, ge=32, le=1024),
):
    """Terrarium-encoded raster-dem tiles for MapLibre's setTerrain()/hillshade —
    separate from /api/layers/elevation/tiles, which colorizes the same COG
    for on-map display rather than encoding it for GPU elevation sampling.
    """
    payload = await run_in_threadpool(
        tiles.render_elevation_terrain_rgb_tile_bytes, z, x, y, tile_size,
    )
    return Response(
        content=payload, media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def _fetch_satellite_tile_bytes(z: int, x: int, y: int) -> bytes:
    """Fetch one World Imagery tile from Esri using the server-side API key.

    The key (WW_ARCGIS_API_KEY) never reaches the client — only this
    function ever sees it. Esri's own billing model counts tiles returned
    per access token, the same whether the caller is a browser or this
    proxy, so relaying one request per one tile actually shown to a user
    (no server-side caching/storage of the response) stays within their
    "no bulk/volume tile exporting" restriction.
    """
    if not _ARCGIS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Satellite basemap is not configured (WW_ARCGIS_API_KEY unset)",
        )
    url = _ARCGIS_SATELLITE_TILE_URL.format(z=z, y=y, x=x)
    try:
        resp = httpx.get(
            url,
            params={"token": _ARCGIS_API_KEY},
            headers={"Referer": _ARCGIS_REFERER},
            timeout=10.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Satellite tile fetch failed: {e}") from e
    if len(resp.content) <= _ARCGIS_NO_DATA_TILE_MAX_BYTES:
        raise HTTPException(status_code=404, detail="No satellite imagery available at this zoom/location")
    return resp.content


@app.get("/api/tiles/satellite/{z}/{x}/{y}.jpg")
@limiter.limit(_RATE_LIMIT_SATELLITE)
async def satellite_tile(request: Request, z: int, x: int, y: int):
    """Proxies Esri World Imagery tiles — see _fetch_satellite_tile_bytes for why."""
    payload = await run_in_threadpool(_fetch_satellite_tile_bytes, z, x, y)
    # Shorter/mutable TTL than the DEM/derived-layer tiles above: this is
    # third-party imagery Esri can refresh over time, not our own
    # deterministically-rendered output, so it isn't safe to mark immutable.
    return Response(
        content=payload, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/api/basemap/version")
async def basemap_version():
    """Current basemap build date for the frontend's cache-busting URL
    builder (see MAP_TILE_URL_TEMPLATE_LIGHT/DARK in
    speciesOccurrenceMapHelpers.ts). No caching — same reasoning as
    /variables serving GIS layers' version tokens uncached: staleness here
    means the frontend keeps using an old (still-valid, just not-newest)
    build_date, not broken tiles, but there's no reason to add that lag.
    """
    if not _BASEMAP_STATE_PATH.exists():
        return {"build_date": None}
    try:
        state = json.loads(_BASEMAP_STATE_PATH.read_text())
    except Exception:
        return {"build_date": None}
    completed_at = state.get("completed_at")
    return {"build_date": completed_at[:10] if completed_at else None}


def _fetch_basemap_tile_bytes(theme: str, z: int, x: int, y: int) -> bytes:
    """Fetch one rendered raster tile from tileserver-gl for the given theme."""
    url = f"{_TILESERVER_BASE}/styles/{theme}/{z}/{x}/{y}.png"
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Basemap tile fetch failed: {e}") from e
    return resp.content


@app.get("/api/basemap/{theme}/{build_date}/tiles/{z}/{x}/{y}.png")
@limiter.limit(_RATE_LIMIT_TILES)
async def basemap_tile(request: Request, theme: str, build_date: str, z: int, x: int, y: int):
    """Proxies the self-hosted OpenMapTiles basemap (tileserver-gl) — see
    scripts/gis/build_basemap_tiles.py and docker-compose.yml's `tileserver`
    service.

    build_date is a pure cache-busting token (matches basemap_state.json's
    completed_at, see the frontend's basemap URL builder) — it isn't used to
    select data here, since tileserver-gl only ever has one current
    basemap.pmtiles loaded. The long immutable TTL is safe not because this
    content never changes, but because a new build produces a URL nobody has
    requested before; old build_date values just stop being requested and
    age out of cache on their own, same scheme /api/layers/{layer_id}/tiles
    already uses with its mtime-derived version token.
    """
    if theme not in _BASEMAP_THEMES:
        raise HTTPException(status_code=404, detail=f"Unknown basemap theme '{theme}'")
    payload = await run_in_threadpool(_fetch_basemap_tile_bytes, theme, z, x, y)
    return Response(
        content=payload, media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/layers/{layer_id}/tile-range/classes")
@limiter.limit(_RATE_LIMIT_TILE_RANGE)
async def layer_tile_range_classes(
    request: Request,
    layer_id: str,
    z: int = Query(...),
    x0: int = Query(...),
    y0: int = Query(...),
    x1: int = Query(...),
    y1: int = Query(...),
):
    try:
        tiles.get_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")
    class_counts = await run_in_threadpool(
        tiles.nominal_tile_range_classes, layer_id, z, x0, y0, x1, y1
    )
    ordered = sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True)
    return {"classes": ordered}


@app.get("/api/layers/{layer_id}/tile-range/stats")
@limiter.limit(_RATE_LIMIT_TILE_RANGE)
async def layer_tile_range_stats(
    request: Request,
    layer_id: str,
    z: int = Query(...),
    x0: int = Query(...),
    y0: int = Query(...),
    x1: int = Query(...),
    y1: int = Query(...),
    forecast_h: int = Query(0, ge=0),
    unit_system: str | None = Query(None),
):
    """Numeric-gradient counterpart to tile-range/classes above — the maps
    page's "auto-adapt" mode calls this to discover a fitting colorization
    range for the current viewport BEFORE requesting any colorized tiles
    (see layer_tile's render_range param), rather than discovering it from
    tiles it already rendered — deliberately skips colorizing/PNG-encoding
    each tile (see tiles.layer_tile_range_stats).
    """
    # The frontend's selected variable can be either form (e.g. "bio_1" from
    # the old compat URL pattern, or "bio1") — same resolution
    # variable_tile_compat applies for the tile endpoint itself, applied
    # here too so this endpoint accepts whichever one it's given.
    layer_id = _resolve_variable_id(layer_id)
    try:
        layer = tiles.get_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")
    if forecast_h not in _VALID_FORECAST_HOURS:
        forecast_h = 0
    forecast_suffix = f"__f{forecast_h:03d}h" if forecast_h > 0 else ""
    value_range = await run_in_threadpool(
        tiles.layer_tile_range_stats, layer_id, z, x0, y0, x1, y1, forecast_suffix,
    )
    if value_range is None:
        return {"min": None, "max": None}
    return {
        "min": units.convert_value(value_range[0], layer, unit_system),
        "max": units.convert_value(value_range[1], layer, unit_system),
    }


@lru_cache(maxsize=4096)
def _cached_get_taxon(taxon_id: str, unit_system: str | None, data_version: str) -> dict:
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    sci = taxon.get("scientific_name", "")
    preferred_raw = taxon.get("inat_preferred_common_name") or ""
    common_raw = taxon.get("common_name") or ""
    try:
        nominal_rows = _storage.read_table(
            GLOBAL_STATS_DIR / NOMINAL_STATS_FILE,
            filters=[("taxon_key", "=", str(taxon["taxon_key"]))],
        ).to_pylist()
    except FileNotFoundError:
        nominal_rows = []

    def _class_fractions(variable: str) -> dict[int, float]:
        return {
            int(r["metric"][6:]): float(r["value"])
            for r in nominal_rows
            if r["variable"] == variable
            and r["metric"].startswith("class_")
            and r["metric"][6:].isdigit()
            and float(r["value"] or 0) > 0
        }

    kg2_class_fractions = _class_fractions("kg2")
    lc_class_fractions = _class_fractions("landcover")
    soil_texture_class_fractions = _class_fractions("soil_texture")
    eco_class_fractions = _class_fractions("ecoregions")
    biome_class_fractions = _class_fractions("biome")

    try:
        numerical_rows = _storage.read_table(
            GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE,
            filters=[("taxon_key", "=", str(taxon["taxon_key"]))],
        ).to_pylist()
    except FileNotFoundError:
        numerical_rows = []
    numerical_stats = {r["variable"]: r for r in numerical_rows}

    try:
        circular_rows = _storage.read_table(
            GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE,
            filters=[("taxon_key", "=", str(taxon["taxon_key"]))],
        ).to_pylist()
    except FileNotFoundError:
        circular_rows = []
    circular_stats = {r["variable"]: r for r in circular_rows}

    try:
        ordinal_rows = _storage.read_table(
            GLOBAL_STATS_DIR / ORDINAL_STATS_FILE,
            filters=[("taxon_key", "=", str(taxon["taxon_key"]))],
        ).to_pylist()
    except FileNotFoundError:
        ordinal_rows = []
    salinity_median = next(
        (float(r["value"]) for r in ordinal_rows if r["variable"] == "salinity" and r["metric"] == "median"),
        None,
    )

    description_profile = descriptions.build_description_profile(
        taxon["taxon_key"],
        hierarchy=_load_hierarchy(),
        storage=_storage,
        loc_taxa_path=_LOC_TAXA_PATH,
        scope_by_level=_CONFIG.location_scope_by_level,
        kg2_class_fractions=kg2_class_fractions or None,
        kg2_legend_classes=_load_legend("kg2") or None,
        lc_class_fractions=lc_class_fractions or None,
        lc_legend=_load_legend_full("landcover") or None,
        soil_texture_class_fractions=soil_texture_class_fractions or None,
        soil_texture_legend=_load_legend_full("soil_texture") or None,
        eco_class_fractions=eco_class_fractions or None,
        eco_legend_classes=_load_legend("ecoregions") or None,
        biome_class_fractions=biome_class_fractions or None,
        biome_legend=_load_legend_full("biome") or None,
        salinity_median=salinity_median,
        salinity_legend_classes=_load_legend("salinity") or None,
        numerical_stats=numerical_stats or None,
        circular_stats=circular_stats or None,
        unit_system=unit_system or None,
    )
    description = next(
        (line["body"] for section in description_profile["sections"] for line in section["lines"]),
        "",
    )
    observation_count = max((int(r["count"]) for r in numerical_rows if r.get("count")), default=0)
    large_taxon = _is_expensive_subtree_taxon(taxon, observation_count)
    return {
        **taxon,
        "scientific_name": sci.replace("_", " "),
        "inat_preferred_common_name": format_common_name(preferred_raw) or None,
        "common_name": format_common_name(preferred_raw or common_raw) or None,
        **_image_fields(taxon),
        "description": description,
        "description_profile": description_profile,
        "observation_count": observation_count,
        "large_taxon": large_taxon,
    }


@app.get("/api/taxon/{taxon_id}")
@app.get("/api/species/{taxon_id}")
@limiter.limit(_RATE_LIMIT_DETAIL)
def get_taxon(request: Request, taxon_id: str, unit_system: str | None = Query(None)):
    return _cached_get_taxon(taxon_id, unit_system, _DATA_VERSION)


def _check_all_obscured(taxon: dict, location_gid: str | None) -> bool:
    """Return True when every observation in scope has obscured coordinates."""
    filter_col = _location_filter_col(location_gid) if location_gid else None
    needed = ["obscured"] + ([filter_col] if filter_col else [])
    df = _read_occurrences_scoped(taxon, columns=needed)
    if "obscured" not in df.columns:
        return False
    if filter_col and filter_col in df.columns:
        df = df[df[filter_col].astype(str) == str(location_gid)]
    if df.empty:
        return False
    return not (df["obscured"] == "No").any()


@lru_cache(maxsize=4096)
def _cached_get_species_obscured(taxon_id: str, location: str | None, data_version: str) -> dict:
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    _reject_if_large_taxon(taxon)
    location_gid = location.strip() if location else None
    all_obscured = _check_all_obscured(taxon, location_gid)
    return {
        "taxon_id": taxon_id,
        "all_obscured": all_obscured,
        "allObscured": all_obscured,
        "location_filtered": location_gid is not None,
    }


@app.get("/api/species/{taxon_id}/obscured")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_species_obscured(
    request: Request,
    taxon_id: str,
    location: str | None = Query(None, description="Optional location GID to scope the obscured check"),
):
    return _cached_get_species_obscured(taxon_id, location, _DATA_VERSION)


@lru_cache(maxsize=4096)
def _cached_get_taxon_env_stats(
    taxon_id: str, unit_system: str | None, data_version: str, deploy_version: str,
) -> dict:
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    layer_index = {layer["id"]: layer for layer in tiles.load_layers()}
    taxon_key = str(taxon["taxon_key"])
    _tk = [("taxon_key", "=", taxon_key)]

    numerical_stats: dict[str, dict] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var = row.pop("variable")
        numerical_stats[var] = row

    circular_stats: dict[str, dict] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var = row.pop("variable")
        circular_stats[var] = row

    nominal_stats: dict[str, dict] = {}
    nominal_classes: dict[str, list] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / NOMINAL_STATS_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var, metric, value = row["variable"], row["metric"], row["value"]
        if metric.startswith("class_"):
            if not value:
                continue
            class_id = int(metric[6:])
            nominal_classes.setdefault(var, []).append({"class_id": class_id, "fraction": value})
        else:
            nominal_stats.setdefault(var, {})[metric] = value
    for var in nominal_classes:
        nominal_classes[var].sort(key=lambda e: -e["fraction"])

    ordinal_stats: dict[str, dict] = {}
    ordinal_classes: dict[str, list] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / ORDINAL_STATS_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var, metric, value = row["variable"], row["metric"], row["value"]
        if metric.startswith("class_"):
            if not value:
                continue
            class_id = int(metric[6:])
            ordinal_classes.setdefault(var, []).append({"class_id": class_id, "fraction": value})
        else:
            ordinal_stats.setdefault(var, {})[metric] = value
    for var in ordinal_classes:
        ordinal_classes[var].sort(key=lambda e: e["class_id"])

    density_by_var: dict[str, dict] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / DENSITY_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var = row.pop("variable")
        density_by_var[var] = row

    all_var_ids = list(dict.fromkeys(
        list(numerical_stats) + list(circular_stats) + list(nominal_stats) + list(ordinal_stats)
    ))
    variables = []
    for var_id in all_var_ids:
        layer = layer_index.get(var_id, {})
        entry: dict = {
            "id": var_id,
            "display_name": layer.get("display_name"),
            "units": units.display_units(layer, unit_system),
            "value_type": layer.get("value_type"),
            "domain": layer.get("domain") or None,
        }
        if var_id in numerical_stats:
            entry["stats"] = units.convert_summary(numerical_stats[var_id], layer, unit_system)
            entry["density"] = units.convert_density_curve(density_by_var.get(var_id), layer, unit_system)
            entry["classes"] = None
        elif var_id in circular_stats:
            entry["stats"] = circular_stats[var_id]
            entry["density"] = density_by_var.get(var_id)
            entry["classes"] = None
        elif var_id in ordinal_stats:
            entry["stats"] = ordinal_stats[var_id]
            entry["density"] = None
            entry["classes"] = ordinal_classes.get(var_id, [])
        else:
            entry["stats"] = nominal_stats[var_id]
            entry["density"] = None
            entry["classes"] = nominal_classes.get(var_id, [])
        variables.append(entry)

    return {"variables": variables}


@app.get("/api/taxon/{taxon_id}/env-stats")
@limiter.limit(_RATE_LIMIT_DETAIL)
def get_taxon_env_stats(request: Request, taxon_id: str, unit_system: str | None = Query(None)):
    return _cached_get_taxon_env_stats(taxon_id, unit_system, _DATA_VERSION, _DEPLOY_VERSION)


# ---------------------------------------------------------------------------
# Legacy compatibility endpoints (frontend still uses these URL patterns)
# ---------------------------------------------------------------------------

def _load_relative_ranks(taxon_key: str, variable_id: str) -> list[dict]:
    """Read this taxon's ranking positions (one row per ancestor context) from
    the consolidated global positions file."""
    positions_file = GLOBAL_STATS_DIR / POSITION_FILE
    if not positions_file.exists():
        return []
    try:
        rows = _storage.read_table(
            positions_file,
            filters=[("taxon_key", "=", taxon_key), ("variable", "=", variable_id)],
        ).to_pylist()
    except Exception:
        return []

    result = []
    for row in rows:
        position = row.get("position") or 0
        count = row.get("count") or 0
        # (position + 1) / count: rank n/n = 100th percentile
        percentile = round((position + 1) / count, 3) if count > 0 else 0.0
        result.append({
            "metric": row.get("metric"),
            "position": position + 1,
            "count": count,
            "percentile": percentile,
            "sampleCount": row.get("sampleCount"),
            "context_label": row.get("contextLabel"),
            "label": row.get("contextLabel"),
        })
    return result


_GADM_LEVEL_COLS: dict[int, str] = {0: "level0Gid", 1: "level1Gid", 2: "level2Gid"}


def _location_filter_col(gid: str) -> str | None:
    """Return the occurrences.parquet column to use when filtering observations to gid."""
    rec = _load_hierarchy().get(gid)
    if rec is not None:
        return _GADM_LEVEL_COLS.get(rec["level"])
    return "gbifRegion"


def _slice_from_raw_occ(
    taxon: dict,
    variable_id: str,
    filter_col: str | None,
    gid: str | None,
    value_min: float,
    value_max: float,
    circular_wrap: bool,
    limit: int | None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    extra_filters: list[dict] | None = None,
    polygon: BaseGeometry | None = None,
) -> list[dict]:
    df = collect_taxon_df(taxon, storage=_storage)
    if df is None or variable_id not in df.columns:
        return []
    if filter_col is not None:
        if filter_col not in df.columns:
            return []
        df = df[df[filter_col].astype(str) == str(gid)]
    if phenology is not None:
        df = apply_phenology_filter(df, phenology)
    if start_ts is not None or end_ts is not None:
        df = apply_timestamp_filter(df, start_ts, end_ts)
    if polygon is not None:
        df = apply_polygon_filter(df, polygon)
    if df.empty:
        return []
    col = pd.to_numeric(df[variable_id], errors="coerce")
    mask = numeric_range_mask(col, value_min, value_max, circular_wrap)
    df = df[mask]
    df = apply_chained_filters(df, extra_filters)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    if limit is not None:
        df = df.head(limit)
    return [
        {
            "catalogNumber": str(r["catalogNumber"]),
            "latitude": r["decimalLatitude"],
            "longitude": r["decimalLongitude"],
            "value": float(r[variable_id]) if pd.notna(r[variable_id]) else None,
        }
        for r in df.to_dict("records")
    ]


def _class_samples_from_raw_occ(
    taxon: dict,
    variable_id: str,
    filter_col: str | None,
    gid: str | None,
    class_value: float,
    limit: int | None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    extra_filters: list[dict] | None = None,
    polygon: BaseGeometry | None = None,
) -> list[dict]:
    df = collect_taxon_df(taxon, storage=_storage)
    if df is None or variable_id not in df.columns:
        return []
    if filter_col is not None:
        if filter_col not in df.columns:
            return []
        df = df[df[filter_col].astype(str) == str(gid)]
    if phenology is not None:
        df = apply_phenology_filter(df, phenology)
    if start_ts is not None or end_ts is not None:
        df = apply_timestamp_filter(df, start_ts, end_ts)
    if polygon is not None:
        df = apply_polygon_filter(df, polygon)
    if df.empty:
        return []
    col = pd.to_numeric(df[variable_id], errors="coerce")
    df = df[col == class_value]
    df = apply_chained_filters(df, extra_filters)
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    if limit is not None:
        df = df.head(limit)
    return [
        {
            "catalogNumber": str(r["catalogNumber"]),
            "latitude": r["decimalLatitude"],
            "longitude": r["decimalLongitude"],
            "value": float(r[variable_id]) if pd.notna(r[variable_id]) else None,
        }
        for r in df.to_dict("records")
    ]


def _build_variable_metadata(layer: dict | None, variable_id: str, unit_system: str | None) -> dict:
    return {
        "name": layer["display_name"] if layer else variable_id,
        "units": units.display_units(layer, unit_system) if layer else None,
        "value_type": layer.get("value_type") if layer else None,
        "domain": (layer.get("domain") or None) if layer else None,
    }


@lru_cache(maxsize=4096)
def _cached_get_species_environment_base(
    taxon_id: str, variable_id: str, unit_system: str | None, data_version: str,
) -> dict:
    """Unfiltered case only — no location/phenology/timestamp/extra/polygon —
    reads precomputed global stats, bounded by (taxon, variable, unit_system).
    The filtered branch (compute_location_filtered_stats) stays live in the
    route below: extra/polygon are arbitrary JSON/WKT, an unbounded key
    space not worth caching."""
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    variable_metadata = _build_variable_metadata(layer, variable_id, unit_system)
    value_type = layer.get("value_type") if layer else None

    if value_type in ("nominal", "ordinal"):
        stats_file = ORDINAL_STATS_FILE if value_type == "ordinal" else NOMINAL_STATS_FILE
        rows = _storage.read_table(
            GLOBAL_STATS_DIR / stats_file,
            filters=[("taxon_key", "=", str(taxon["taxon_key"])), ("variable", "=", variable_id)],
        ).to_pylist()
        if not rows:
            raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
        metrics = {r["metric"]: r["value"] for r in rows}
        total_samples = int(metrics.get("total_samples", 0))
        class_index = {c["id"]: c for c in _load_legend(variable_id)}
        categorical_distribution = []
        for r in rows:
            m = r["metric"]
            if not m.startswith("class_"):
                continue
            fraction = float(r["value"])
            if not fraction:
                continue
            class_id = int(m[6:])
            info = class_index.get(class_id, {})
            categorical_distribution.append({
                "value": class_id,
                "class_name": info.get("name", str(class_id)),
                "description": "",
                "color": info.get("traits", {}).get("color") if info.get("traits") else None,
                "count": round(total_samples * fraction),
                "fraction": fraction,
            })
        if value_type == "ordinal":
            categorical_distribution.sort(key=lambda x: x["value"])
        else:
            categorical_distribution.sort(key=lambda x: -x["fraction"])
        summary: dict = {
            "count": total_samples,
            "unique_classes": int(metrics["unique_classes"]) if "unique_classes" in metrics else None,
            "entropy": float(metrics["entropy"]) if "entropy" in metrics else None,
            "mode": int(metrics["mode"]) if "mode" in metrics else None,
        }
        if value_type == "ordinal":
            for key in ("10th_percentile", "25th_percentile", "median", "75th_percentile", "90th_percentile"):
                if key in metrics:
                    summary[key] = float(metrics[key])
        else:
            summary.update({"min": None, "mean": None, "max": None})

        # Any variable id may have an associated compositional (ternary) density
        # blob — the density_grid table is keyed by whichever composition_group
        # id the catalog tags a variable's members with (see
        # util.ternary.composition_group_members), not just "soil_texture".
        # Classification (class ids + exact boundary lines) is static per
        # classifier — identical for every taxon — so it's computed once and
        # cached (util.ternary.build_ternary_classification_overlay) rather
        # than read from storage; a compositional variable with no registered
        # classifier just gets the density blob with no classes, which is a
        # valid, supported shape.
        ternary_composition_density = None
        dg_rows = []
        if _storage.exists(GLOBAL_STATS_DIR / DENSITY_GRID_FILE):
            dg_rows = _storage.read_table(
                GLOBAL_STATS_DIR / DENSITY_GRID_FILE,
                filters=[("taxon_key", "=", str(taxon["taxon_key"])), ("variable", "=", variable_id)],
            ).to_pylist()
        if dg_rows:
            dg_row = dg_rows[0]
            ternary_composition_density = {
                "resolution": dg_row["resolution"],
                "density": dg_row["density"],
                "sample_a": dg_row.get("sample_a"),
                "sample_b": dg_row.get("sample_b"),
                "sample_c": dg_row.get("sample_c"),
            }
            classifier = gis.COMPOSITION_CLASSIFIERS.get(variable_id)
            group = (layer or {}).get("composition_group")
            if classifier is not None and group:
                all_layers_by_id = {lyr["id"]: lyr for lyr in tiles.load_layers()}
                axis_columns = tuple(composition_group_members(all_layers_by_id).get(group, ()))
                if len(axis_columns) == 3:
                    overlay = build_ternary_classification_overlay(
                        dg_row["resolution"], classifier, axis_columns,
                    )
                    ternary_composition_density["class_ids"] = overlay["class_ids"]
                    ternary_composition_density["class_boundary_a"] = overlay["boundary_a"]
                    ternary_composition_density["class_boundary_b"] = overlay["boundary_b"]

        return {
            "species_id": taxon.get("taxon_key"),
            "variable": variable_id,
            "variable_metadata": variable_metadata,
            "observation_count": total_samples,
            "summary": summary,
            "density_curve": None,
            "categorical_distribution": categorical_distribution,
            "ternary_composition_density": ternary_composition_density,
            "relative_ranks": _load_relative_ranks(str(taxon.get("taxon_key", "")), variable_id),
        }

    if value_type == "circular":
        _tk_var = [("taxon_key", "=", str(taxon["taxon_key"])), ("variable", "=", variable_id)]
        rows = _storage.read_table(GLOBAL_STATS_DIR / CIRCULAR_STATS_FILE, filters=_tk_var).to_pylist()
        row = rows[0] if rows else None
        if row is None:
            raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
        count = int(row.get("count") or 0)
        summary = {
            "count": count,
            "circular_mean": row.get("circular_mean"),
            "rbar": row.get("rbar"),
            "circular_std": row.get("circular_std"),
            "circular_var": row.get("circular_var"),
            "entropy": row.get("entropy"),
            "mode": row.get("mode"),
        }
        den_rows = _storage.read_table(GLOBAL_STATS_DIR / DENSITY_FILE, filters=_tk_var).to_pylist()
        den_row = den_rows[0] if den_rows else None
        density_curve = {"points": den_row["points"], "density": den_row["density"]} if den_row else None
        return {
            "species_id": taxon.get("taxon_key"),
            "variable": variable_id,
            "variable_metadata": variable_metadata,
            "observation_count": count,
            "summary": summary,
            "density_curve": density_curve,
            "categorical_distribution": None,
            "relative_ranks": _load_relative_ranks(str(taxon.get("taxon_key", "")), variable_id),
        }

    _tk_var = [("taxon_key", "=", str(taxon["taxon_key"])), ("variable", "=", variable_id)]
    rows = _storage.read_table(GLOBAL_STATS_DIR / NUMERICAL_STATS_FILE, filters=_tk_var).to_pylist()
    row = rows[0] if rows else None
    if row is None:
        raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")

    count = int(row.get("count") or 0)
    raw_summary = {
        "count": count,
        "min": row.get("min"),
        "mean": row.get("mean"),
        "max": row.get("max"),
        "median": row.get("median"),
        "mode": row.get("mode"),
        "std": row.get("std"),
        "stddev": row.get("std"),
        "variance": row.get("variance"),
        "range": row.get("range"),
        "q10": row.get("10th_percentile"),
        "q25": row.get("25th_percentile"),
        "q75": row.get("75th_percentile"),
        "q90": row.get("90th_percentile"),
        "iqr": row.get("iqr"),
        "10_90_range": row.get("10_90_range"),
        "entropy": row.get("entropy"),
    }

    den_rows = _storage.read_table(GLOBAL_STATS_DIR / DENSITY_FILE, filters=_tk_var).to_pylist()
    den_row = den_rows[0] if den_rows else None
    density_curve = {"points": den_row["points"], "density": den_row["density"]} if den_row else None

    return {
        "species_id": taxon.get("taxon_key"),
        "variable": variable_id,
        "variable_metadata": variable_metadata,
        "observation_count": count,
        "summary": units.convert_summary(raw_summary, layer, unit_system),
        "density_curve": units.convert_density_curve(density_curve, layer, unit_system),
        "categorical_distribution": None,
        "relative_ranks": _load_relative_ranks(str(taxon.get("taxon_key", "")), variable_id),
    }


@app.get("/species/{taxon_id}/environment/{variable_id}")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_species_environment(
    request: Request,
    taxon_id: str, variable_id: str, unit_system: str | None = None,
    location: str | None = None, phenology: str | None = None,
    start_ts: int | None = None, end_ts: int | None = None,
    extra: str | None = None, polygon: str | None = None,
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")
    try:
        polygon_geom = parse_polygon_param(polygon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid polygon: {exc}") from exc

    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    variable_metadata = _build_variable_metadata(layer, variable_id, unit_system)
    extra_filters = _parse_extra_variable_filters(extra, unit_system)

    if (
        location is not None or phenology_norm is not None or start_ts is not None
        or end_ts is not None or extra_filters or polygon_geom is not None
    ) and layer is not None:
        _reject_if_large_taxon(taxon)
        filter_col = _location_filter_col(location) if location is not None else None
        if location is None or filter_col is not None:
            all_layers_by_id = {lyr["id"]: lyr for lyr in tiles.load_layers()}
            result = compute_location_filtered_stats(
                taxon, variable_id, filter_col, location, layer,
                phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
                storage=_storage, layer_meta=all_layers_by_id,
                extra_filters=extra_filters, polygon=polygon_geom,
            )
            if result is not None:
                if result["type"] == "continuous":
                    stats = result["stats"]
                    raw_summary = {
                        "count": stats["count"],
                        "min": stats.get("min"),
                        "mean": stats.get("mean"),
                        "max": stats.get("max"),
                        "median": stats.get("median"),
                        "mode": stats.get("mode"),
                        "std": stats.get("std"),
                        "stddev": stats.get("std"),
                        "variance": stats.get("variance"),
                        "range": stats.get("range"),
                        "q10": stats.get("10th_percentile"),
                        "q25": stats.get("25th_percentile"),
                        "q75": stats.get("75th_percentile"),
                        "q90": stats.get("90th_percentile"),
                        "iqr": stats.get("iqr"),
                        "10_90_range": stats.get("10_90_range"),
                        "entropy": stats.get("entropy"),
                    }
                    return {
                        "species_id": taxon.get("taxon_key"),
                        "variable": variable_id,
                        "variable_metadata": variable_metadata,
                        "observation_count": result["observation_count"],
                        "summary": units.convert_summary(raw_summary, layer, unit_system),
                        "density_curve": units.convert_density_curve(result["density_curve"], layer, unit_system),
                        "categorical_distribution": None,
                        "relative_ranks": [],
                    }
                if result["type"] == "circular":
                    stats = result["stats"]
                    return {
                        "species_id": taxon.get("taxon_key"),
                        "variable": variable_id,
                        "variable_metadata": variable_metadata,
                        "observation_count": result["observation_count"],
                        "summary": {
                            "count": stats["count"],
                            "circular_mean": stats.get("circular_mean"),
                            "rbar": stats.get("rbar"),
                            "circular_std": stats.get("circular_std"),
                            "circular_var": stats.get("circular_var"),
                            "entropy": stats.get("entropy"),
                            "mode": stats.get("mode"),
                        },
                        "density_curve": result["density_curve"],
                        "categorical_distribution": None,
                        "relative_ranks": [],
                    }
                total_samples = result["observation_count"]
                class_index = {c["id"]: c for c in _load_legend(variable_id)}
                categorical_distribution = [
                    {
                        "value": item["class_id"],
                        "class_name": class_index.get(item["class_id"], {}).get("name", str(item["class_id"])),
                        "description": "",
                        "color": (class_index.get(item["class_id"], {}).get("traits") or {}).get("color"),
                        "count": round(total_samples * item["fraction"]),
                        "fraction": item["fraction"],
                    }
                    for item in result["distribution"]
                ]
                ternary_composition_density = result.get("ternary_composition_density")
                if ternary_composition_density is not None:
                    classifier = gis.COMPOSITION_CLASSIFIERS.get(variable_id)
                    group = (layer or {}).get("composition_group")
                    if classifier is not None and group:
                        axis_columns = tuple(composition_group_members(all_layers_by_id).get(group, ()))
                        if len(axis_columns) == 3:
                            overlay = build_ternary_classification_overlay(
                                ternary_composition_density["resolution"], classifier, axis_columns,
                            )
                            ternary_composition_density["class_ids"] = overlay["class_ids"]
                            ternary_composition_density["class_boundary_a"] = overlay["boundary_a"]
                            ternary_composition_density["class_boundary_b"] = overlay["boundary_b"]
                return {
                    "species_id": taxon.get("taxon_key"),
                    "variable": variable_id,
                    "variable_metadata": variable_metadata,
                    "observation_count": total_samples,
                    "summary": {
                        "count": total_samples,
                        "min": None,
                        "mean": None,
                        "max": None,
                        "entropy": result.get("summary", {}).get("entropy"),
                        "unique_classes": result.get("summary", {}).get("unique_classes"),
                        "mode": result.get("summary", {}).get("mode"),
                    },
                    "density_curve": None,
                    "categorical_distribution": categorical_distribution,
                    "ternary_composition_density": ternary_composition_density,
                    "relative_ranks": [],
                }
            else:
                if _check_all_obscured(taxon, location):
                    return {
                        "all_obscured": True,
                        "species_id": taxon.get("taxon_key"),
                        "variable": variable_id,
                    }
                raise HTTPException(
                    status_code=404,
                    detail=f"No samples available for taxon {taxon_id} and variable '{variable_id}' with the active filters.",
                )

    return _cached_get_species_environment_base(taxon_id, variable_id, unit_system, _DATA_VERSION)


def _occurrence_response(
    catalog_number: str,
    taxon: dict,
    latitude: float | None,
    longitude: float | None,
    ingested: bool,
    extra: dict | None = None,
) -> dict:
    response = {
        "catalog_number": catalog_number,
        "taxon_id": taxon["taxon_key"],
        "scientific_name": (taxon.get("scientific_name") or "").replace("_", " "),
        "common_name": taxon.get("common_name") or None,
        "slug": taxon_slug(taxon.get("scientific_name")),
        "latitude": latitude,
        "longitude": longitude,
        "ingested": ingested,
    }
    if extra:
        response.update(extra)
    return response


def _parse_inat_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


_INAT_ATTRIBUTION_NAME_RE = re.compile(
    r"^\(c\)\s*(.+?),\s*(?:some|no) rights reserved", re.IGNORECASE,
)


def _clean_inat_attribution(raw: str) -> str:
    """Reduce iNat's "(c) Name, some rights reserved (LICENSE)" to just Name.

    The ingested path's mediaAttribution (multimedia.txt's rightsHolder) is
    already a bare name — iNat's own attribution field bakes the license
    mention into the string, which would visually duplicate the separately
    rendered license line (see the map popup / ObservationCard credit row).
    Falls back to the raw string if it doesn't match this format, rather
    than dropping it, since iNat's wording isn't 100% guaranteed uniform.
    """
    match = _INAT_ATTRIBUTION_NAME_RE.match(raw)
    return match.group(1).strip() if match else raw


def _inat_observation_photo(obs: dict) -> dict:
    """Extract the first usable-license photo from an iNat observation payload.

    Same permissive-license bar as build_tree.py's GBIF backup images and
    populate_tree.py's multimedia join (_is_usable_license), so display
    logic never has to reason about three different license policies.
    """
    for photo in obs.get("photos") or []:
        license_code = photo.get("license_code") or ""
        if not _is_usable_license(license_code):
            continue
        raw_url = photo.get("url") or ""
        if not raw_url:
            continue
        license_url = _normalize_license_url(license_code)
        raw_attribution = photo.get("attribution") or ""
        return {
            "media_url": re.sub(r"/square\.", "/original.", raw_url, count=1),
            "media_attribution": _clean_inat_attribution(raw_attribution) or None,
            "media_license_url": license_url or None,
            "media_license": _license_label(license_url) if license_url else None,
        }
    return {"media_url": None, "media_attribution": None, "media_license_url": None, "media_license": None}


def _lookup_inat_observation(catalog_number: str) -> dict | None:
    """Resolve an observation directly via iNaturalist's own API.

    Fallback for observations GBIF hasn't (or can't) ingest yet — GBIF's
    iNaturalist mirror only re-syncs periodically (see the media-join
    investigation: some research-grade observations never make it into a
    given export at all). Returns None on any failure (not found, no
    taxon, network error, or a taxon iNat has that our own catalog doesn't
    map — see util.taxa.get_taxon_by_inat_id) so the caller can fall back
    to a plain 404 uniformly.

    Also returns event_timestamp and media fields — unlike the ingested
    path, there's no other source for these (the observation isn't part of
    the taxon's normal /occurrences fetch at all), so this is the only
    place they're worth carrying.
    """
    try:
        resp = httpx.get(
            _INAT_OBSERVATIONS_URL, params={"id": catalog_number}, timeout=10.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except Exception:
        return None
    if not results:
        return None
    obs = results[0]

    inat_taxon = obs.get("taxon") or {}
    taxon = taxa.get_taxon_by_inat_id(inat_taxon.get("id"))
    if taxon is None:
        return None

    coords = (obs.get("geojson") or {}).get("coordinates")
    latitude = coords[1] if isinstance(coords, list) and len(coords) == 2 else None
    longitude = coords[0] if isinstance(coords, list) and len(coords) == 2 else None
    event_timestamp = _parse_inat_timestamp(obs.get("time_observed_at") or obs.get("observed_on"))

    return {
        "taxon": taxon,
        "latitude": latitude,
        "longitude": longitude,
        "event_timestamp": event_timestamp,
        **_inat_observation_photo(obs),
    }


@app.get("/occurrence/{catalog_number}")
@limiter.limit(_RATE_LIMIT_DETAIL)
def get_occurrence(request: Request, catalog_number: str):
    """Resolve an iNaturalist observation id (catalogNumber) to its taxon + location.

    Powers deep links like /occurrence/{id} on the frontend: given just an
    inat observation id, find which species page it belongs to and where to
    place it on that page's map — the same destination as clicking that
    observation's image in the below-map gallery, just entered by id
    instead of navigated to. Looks up CATALOG_NUMBER_INDEX_FILE (sorted by
    catalogNumber) rather than OCCURRENCES_FILE (sorted by taxon_key) so the
    lookup gets row-group pruning instead of a full scan.

    Falls back to iNaturalist's own API (_lookup_inat_observation) when the
    observation isn't in our ingested dataset at all — "ingested": false on
    that path distinguishes a live iNat-only resolution from our own data.
    """
    catalog_number = catalog_number.strip()
    if not catalog_number:
        raise HTTPException(status_code=404, detail="Observation not found")

    try:
        rows = _storage.read_table(
            CATALOG_NUMBER_INDEX_FILE,
            filters=[("catalogNumber", "=", catalog_number)],
        ).to_pylist()
    except Exception:
        rows = []

    if rows:
        taxon = taxa.get_taxon_by_id(rows[0].get("taxon_key"))
        if taxon is None:
            raise HTTPException(status_code=404, detail="Taxon not found")
        return _occurrence_response(
            catalog_number, taxon, rows[0].get("decimalLatitude"), rows[0].get("decimalLongitude"), ingested=True,
        )

    fallback = _lookup_inat_observation(catalog_number)
    if fallback is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return _occurrence_response(
        catalog_number, fallback["taxon"], fallback["latitude"], fallback["longitude"], ingested=False,
        extra={
            "event_timestamp": fallback["event_timestamp"],
            "media_url": fallback["media_url"],
            "media_attribution": fallback["media_attribution"],
            "media_license": fallback["media_license"],
            "media_license_url": fallback["media_license_url"],
        },
    )


@lru_cache(maxsize=2048)
def _cached_get_species_occurrences(
    taxon_id: str,
    location: str | None,
    phenology: str | None,
    start_ts: int | None,
    end_ts: int | None,
    data_version: str,
) -> dict:
    """Bounded LRU rather than caching every param combo unconditionally —
    start_ts/end_ts are near-continuous, so this evicts gracefully instead
    of growing unbounded; the common unfiltered "show me the map" call
    (all params None) always wins a cache slot on repeat."""
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")

    _reject_if_large_taxon(taxon)

    filter_col = _location_filter_col(location) if location is not None else None
    has_loc_or_pheno = filter_col is not None or phenology_norm is not None
    has_ts = start_ts is not None or end_ts is not None
    use_precomputed_pheno = not has_loc_or_pheno and not has_ts

    extra_cols: list[str] = []
    if filter_col:
        extra_cols.append(filter_col)
    # Always read rcs so we can fall back to live phenology counts if precomputed is missing
    extra_cols.append("rcs")
    extra_cols.append("eventTimestamp")
    extra_cols.extend(["mediaUrl", "mediaAttribution", "mediaLicense"])
    occ_columns = list(_OCC_COLUMNS) + extra_cols

    ts_min: int | None = None
    ts_max: int | None = None
    pheno_acc: Counter = Counter()

    df = _read_occurrences_scoped(taxon, columns=occ_columns)
    if df.empty:
        collected: list[dict] = []
    else:
        df = _filter_occ_df(df)
        if filter_col is not None and filter_col in df.columns:
            df = df[df[filter_col].astype(str) == str(location)]
        if phenology_norm is not None:
            df = apply_phenology_filter(df, phenology_norm)
        if "eventTimestamp" in df.columns:
            ts_col = pd.to_numeric(df["eventTimestamp"], errors="coerce").dropna()
            if len(ts_col):
                ts_min, ts_max = int(ts_col.min()), int(ts_col.max())
        if has_ts:
            df = apply_timestamp_filter(df, start_ts, end_ts)
        if not use_precomputed_pheno and "rcs" in df.columns:
            pheno_acc.update(compute_phenology_counts(df))
        media_cols = [c for c in ("mediaUrl", "mediaAttribution", "mediaLicense") if c in df.columns]
        df = df[["catalogNumber", "decimalLatitude", "decimalLongitude", *media_cols]]
        df = df.dropna(subset=["catalogNumber", "decimalLatitude", "decimalLongitude"])
        df = df.drop_duplicates(subset="catalogNumber")
        collected = []
        for r in df.to_dict("records"):
            entry = {
                "catalogNumber": str(r["catalogNumber"]),
                "latitude": r["decimalLatitude"],
                "longitude": r["decimalLongitude"],
            }
            media_url = r.get("mediaUrl")
            if isinstance(media_url, str) and media_url:
                entry["media_url"] = media_url
                attribution = r.get("mediaAttribution")
                if isinstance(attribution, str) and attribution:
                    entry["media_attribution"] = attribution
                license_url = r.get("mediaLicense")
                if isinstance(license_url, str) and license_url:
                    entry["media_license_url"] = license_url
                    entry["media_license"] = _license_label(license_url)
            collected.append(entry)

    if use_precomputed_pheno:
        pheno_counts = read_phenology_counts(TREE_ROOT / taxon["path"]) or dict(
            sorted(pheno_acc.items(), key=lambda kv: kv[1], reverse=True)
        )
    else:
        pheno_counts = dict(sorted(pheno_acc.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "occurrences": collected,
        "min_timestamp": ts_min,
        "max_timestamp": ts_max,
        "phenology_counts": pheno_counts,
    }


@app.get("/species/{taxon_id}/occurrences")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_species_occurrences(
    request: Request,
    taxon_id: str,
    location: str | None = None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
):
    return _cached_get_species_occurrences(taxon_id, location, phenology, start_ts, end_ts, _DATA_VERSION)


@lru_cache(maxsize=1)
def _load_hierarchy() -> dict[str, dict]:
    """Return gid → {name, level, parent_gid} from hierarchy.csv."""
    path = _LOCATIONS_DIR / "hierarchy.csv"
    try:
        f_ctx = _storage.open_input_file(path)
    except Exception:
        return {}
    result: dict[str, dict] = {}
    try:
        with f_ctx as raw:
            data = raw.read()
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            for row in csv.DictReader(io.StringIO(text)):
                gid = row.get("gid", "")
                if gid:
                    result[gid] = {
                        "name": row.get("name", gid),
                        "level": int(row["level"]),
                        "parent_gid": row.get("parent_gid") or None,
                    }
    except Exception:
        return {}
    return result


def _resolve_hierarchy(gid: str, by_gid: dict[str, dict]) -> list[str]:
    """Return ancestor names from top-level down to the immediate parent."""
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


def _ancestor_gids(gid: str, by_gid: dict[str, dict]) -> set[str]:
    chain: set[str] = set()
    current = by_gid.get(gid, {}).get("parent_gid")
    seen: set[str] = set()
    while current:
        if current in seen:
            break
        seen.add(current)
        chain.add(current.lower())
        rec = by_gid.get(current)
        if rec and rec.get("name"):
            chain.add(rec["name"].lower())
        current = rec.get("parent_gid") if rec else None
    return chain


@lru_cache(maxsize=4096)
def _cached_get_species_locations(
    taxon_id: str, level: int | None, parent: str | None, limit: int, data_version: str,
) -> list:
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    taxon_key = str(taxon["taxon_key"])
    try:
        table = _storage.read_table(_LOC_TAXA_PATH, filters=[("taxon_key", "=", taxon_key)])
    except Exception:
        return []

    if table.num_rows == 0:
        return []

    scope_to_level: dict[str, int] = {v: k for k, v in _CONFIG.location_scope_by_level.items()}
    scope_to_level["gbif_region"] = -1
    by_gid = _load_hierarchy()
    parent_lower = parent.strip().lower() if parent else None

    results: list[dict] = []
    seen: set[str] = set()
    for scope, gid, count in zip(
        table.column("scope").to_pylist(),
        table.column("gid").to_pylist(),
        table.column("count").to_pylist(),
    ):
        loc_level = scope_to_level.get(str(scope))
        if loc_level is None or gid in seen:
            continue
        if level is not None and loc_level != level:
            continue
        if parent_lower is not None and parent_lower not in _ancestor_gids(gid, by_gid):
            continue
        seen.add(gid)
        rec = by_gid.get(gid)
        results.append({
            "gid": gid,
            "name": rec["name"] if rec else gid,
            "level": loc_level,
            "hierarchy": _resolve_hierarchy(gid, by_gid) if rec else [],
            "count": int(count),
        })

    results.sort(key=lambda r: (-r["count"], r["name"].lower(), r["gid"]))
    return results[:limit]


@app.get("/species/{taxon_id}/locations")
@limiter.limit(_RATE_LIMIT_DETAIL)
def get_species_locations(
    request: Request, taxon_id: str, level: int | None = None, parent: str | None = None, limit: int = 500,
):
    # Was missing a rate limit entirely until now — a gap from the earlier
    # hardening pass, not a deliberate omission.
    return _cached_get_species_locations(taxon_id, level, parent, limit, _DATA_VERSION)


@app.get("/species/{taxon_id}/download")
@limiter.limit(_RATE_LIMIT_DOWNLOAD)
async def download_species_data(request: Request, background_tasks: BackgroundTasks, taxon_id: str) -> FileResponse:
    """Download a taxon's occurrence data + stats as a ZIP, same shape as the
    custom-upload archive so it can be mounted offline the same way.

    Works for any rank — non-leaf taxa aggregate every descendant leaf's
    observations (see util.download.build_species_archive), which can be
    slow for high ranks with many descendants.
    """
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    _reject_if_large_taxon(taxon)

    result = await run_in_threadpool(download.build_species_archive, taxon, _storage)
    if result is None:
        raise HTTPException(status_code=404, detail="No observations available for this taxon")
    archive_path, archive_name, work_dir = result

    background_tasks.add_task(shutil.rmtree, work_dir, True)
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=archive_name,
    )


@app.get("/species/{taxon_id}/environment/{variable_id}/observation-values")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_observation_variable_values(
    request: Request,
    taxon_id: str,
    variable_id: str,
    unit_system: str | None = None,
):
    """Return raw GIS values for all observations of a taxon for one variable."""
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    _reject_if_large_taxon(taxon)

    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")

    collected: dict[str, float] = {}
    df = _read_occurrences_scoped(taxon, columns=["catalogNumber", variable_id, "obscured", "coordinateUncertaintyInMeters"])
    if variable_id in df.columns:
        if "obscured" in df.columns:
            df = df[df["obscured"] == "No"]
        if "coordinateUncertaintyInMeters" in df.columns:
            col = df["coordinateUncertaintyInMeters"]
            df = df[col.isna() | (col <= 500)]
        for cat, val in zip(df["catalogNumber"].tolist(), df[variable_id].tolist()):
            cat = str(cat)
            if cat not in collected and val is not None and not (isinstance(val, float) and math.isnan(val)):
                converted = units.convert_value(float(val), layer, unit_system)
                if converted is not None:
                    collected[cat] = converted

    vals = list(collected.values())
    obs_min = min(vals) if vals else None
    obs_max = max(vals) if vals else None
    obs_q01: float | None = None
    obs_q99: float | None = None
    if len(vals) >= 2:
        import numpy as _np
        obs_q01, obs_q99 = _np.percentile(vals, [0.1, 99.9]).tolist()
    elif vals:
        obs_q01 = obs_min
        obs_q99 = obs_max
    return {
        "variable": variable_id,
        "min": obs_min,
        "max": obs_max,
        "q01": obs_q01,
        "q99": obs_q99,
        "observations": [{"catalogNumber": k, "value": v} for k, v in collected.items()],
    }


def _parse_extra_variable_filters(extra: str | None, unit_system: str | None) -> list[dict]:
    """Parses the `extra` query param — a JSON array of chained per-variable
    filters carried by the numeric-slice, categorical-samples, and plain
    environment-stats endpoints — into the filter-dict shape
    util.stats.apply_chained_filters expects. Each entry is one of:
    {"variable": id, "min": x, "max": y} for a continuous/circular range (in
    DISPLAY units, converted to raw here same as the primary variable);
    {"variable": id, "ranges": [{"min": x, "max": y}, ...]} for a numeric
    OR-match against multiple disjoint ranges of that one variable (e.g. two
    separately-selected slices of a histogram/KDE);
    {"variable": id, "classValue": n} for an exact categorical match; or
    {"variable": id, "classValues": [n, ...]} for a categorical OR-match
    against multiple classes of that one variable (e.g. Forest OR
    Grassland). This is what lets a client hold a slice from one variable
    active while switching to and slicing a second one."""
    if not extra:
        return []
    try:
        raw_entries = json.loads(extra)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="extra must be a JSON array")
    if not isinstance(raw_entries, list):
        raise HTTPException(status_code=400, detail="extra must be a JSON array")
    layers_by_id = {lyr["id"]: lyr for lyr in tiles.load_layers()}
    filters: list[dict] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or "variable" not in entry:
            raise HTTPException(status_code=400, detail="Each extra filter needs a 'variable'")
        variable_id = _resolve_variable_id(str(entry["variable"]))
        layer = layers_by_id.get(variable_id)
        if layer is None:
            raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")
        if "classValue" in entry:
            if layer.get("value_type") not in ("nominal", "ordinal"):
                raise HTTPException(status_code=400, detail=f"'{variable_id}' is not categorical")
            filters.append({"variable": variable_id, "class_value": float(entry["classValue"])})
            continue
        if "classValues" in entry:
            if layer.get("value_type") not in ("nominal", "ordinal"):
                raise HTTPException(status_code=400, detail=f"'{variable_id}' is not categorical")
            if not isinstance(entry["classValues"], list) or not entry["classValues"]:
                raise HTTPException(status_code=400, detail=f"'classValues' for '{variable_id}' must be a non-empty list")
            filters.append({
                "variable": variable_id,
                "class_values": [float(v) for v in entry["classValues"]],
            })
            continue
        if "ranges" in entry:
            if layer.get("value_type") == "nominal":
                raise HTTPException(status_code=400, detail=f"'{variable_id}' is categorical — use classValue")
            if not isinstance(entry["ranges"], list) or not entry["ranges"]:
                raise HTTPException(status_code=400, detail=f"'ranges' for '{variable_id}' must be a non-empty list")
            filters.append({
                "variable": variable_id,
                "ranges": [
                    _parse_display_range(variable_id, layer, r, unit_system)
                    for r in entry["ranges"]
                ],
            })
            continue
        if "min" not in entry or "max" not in entry:
            raise HTTPException(
                status_code=400,
                detail=f"Extra filter for '{variable_id}' needs min/max, ranges, classValue, or classValues",
            )
        if layer.get("value_type") == "nominal":
            raise HTTPException(status_code=400, detail=f"'{variable_id}' is categorical — use classValue")
        filters.append({
            "variable": variable_id,
            **_parse_display_range(variable_id, layer, entry, unit_system),
        })
    return filters


def _parse_display_range(variable_id: str, layer: dict, entry: dict, unit_system: str | None) -> dict:
    """Parses one {'min','max'} display-unit range entry (a single range, or
    one entry of an extra filter's 'ranges' list) into raw units, with the
    same aspect-wraparound detection as the top-level min/max case."""
    if "min" not in entry or "max" not in entry:
        raise HTTPException(status_code=400, detail=f"Range for '{variable_id}' needs min and max")
    value_min = float(entry["min"])
    value_max = float(entry["max"])
    if not math.isfinite(value_min) or not math.isfinite(value_max):
        raise HTTPException(status_code=400, detail=f"Range for '{variable_id}' must be finite")
    circular_wrap = variable_id == "aspect" and value_max < value_min
    if value_max < value_min and not circular_wrap:
        value_min, value_max = value_max, value_min
    raw_min = units.convert_value_from_display(value_min, layer, unit_system) - 1e-9
    raw_max = units.convert_value_from_display(value_max, layer, unit_system) + 1e-9
    return {"min": raw_min, "max": raw_max, "circular_wrap": circular_wrap}


@app.get("/species/{taxon_id}/environment/{variable_id}/slice")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_species_environment_slice(
    request: Request,
    taxon_id: str,
    variable_id: str,
    min_value: float = Query(..., alias="min"),
    max_value: float = Query(..., alias="max"),
    limit: int | None = Query(None, ge=1, le=10000),
    location: str | None = None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    unit_system: str | None = None,
    extra: str | None = None,
    polygon: str | None = None,
):
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise HTTPException(status_code=400, detail="min and max must be finite numbers")
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    _reject_if_large_taxon(taxon)
    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")
    try:
        polygon_geom = parse_polygon_param(polygon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid polygon: {exc}") from exc
    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")
    if layer.get("value_type") == "nominal":
        raise HTTPException(status_code=400, detail="Categorical variables must use the class samples endpoint")
    circular_wrap = variable_id == "aspect" and max_value < min_value
    if max_value < min_value and not circular_wrap:
        min_value, max_value = max_value, min_value
    # Convert display-unit min/max back to raw (metric) values for querying.
    # Add a tiny epsilon buffer to absorb float round-trip error (ft→m→ft→m loses ~1e-13).
    raw_min = units.convert_value_from_display(min_value, layer, unit_system) - 1e-9
    raw_max = units.convert_value_from_display(max_value, layer, unit_system) + 1e-9
    extra_filters = _parse_extra_variable_filters(extra, unit_system)
    filter_col = _location_filter_col(location) if location is not None else None
    if location is None or filter_col is not None:
        observations = _slice_from_raw_occ(
            taxon, variable_id, filter_col, location,
            raw_min, raw_max, circular_wrap, limit,
            phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
            extra_filters=extra_filters, polygon=polygon_geom,
        )
        observations = [
            {**obs, "value": units.convert_value(obs["value"], layer, unit_system)}
            for obs in observations
        ]
    else:
        observations = []
    return {
        "species_id": taxon.get("taxon_key"),
        "variable": variable_id,
        "range": {"min": min_value, "max": max_value},
        "count": len(observations),
        "observations": observations,
    }


@app.get("/species/{taxon_id}/environment/{variable_id}/class/{class_value}/samples")
@limiter.limit(_RATE_LIMIT_SUBTREE_RAW)
def get_species_environment_class_samples(
    request: Request,
    taxon_id: str,
    variable_id: str,
    class_value: str,
    limit: int | None = Query(None, ge=1, le=10000),
    location: str | None = None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    unit_system: str | None = None,
    extra: str | None = None,
    polygon: str | None = None,
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    _reject_if_large_taxon(taxon)
    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")
    try:
        polygon_geom = parse_polygon_param(polygon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid polygon: {exc}") from exc
    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")
    if layer.get("value_type") not in ("nominal", "ordinal"):
        raise HTTPException(status_code=400, detail="Numerical variables must use the slice endpoint")
    try:
        parsed: float | int = float(class_value)
        if parsed.is_integer():
            parsed = int(parsed)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid class value: {class_value!r}")
    extra_filters = _parse_extra_variable_filters(extra, unit_system)
    filter_col = _location_filter_col(location) if location is not None else None
    if location is None or filter_col is not None:
        observations = _class_samples_from_raw_occ(
            taxon, variable_id, filter_col, location, float(parsed), limit,
            phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
            extra_filters=extra_filters, polygon=polygon_geom,
        )
    else:
        observations = []
    return {
        "species_id": taxon.get("taxon_key"),
        "variable": variable_id,
        "class_value": parsed,
        "count": len(observations),
        "observations": observations,
    }


_METRIC_LABELS: dict[str, str] = {
    "mean": "Average",
    "median": "Median",
    "min": "Minimum",
    "max": "Maximum",
    "std": "Standard deviation",
    "circular_mean": "Directional mean",
    "mode": "Mode",
    "rbar": "Concentration (R̄)",
    "circular_std": "Circular std dev",
    "circular_var": "Circular variance",
}
_METRIC_ORDER = ["mean", "median", "min", "max", "std"]
_METRIC_RANK = {m: i for i, m in enumerate(_METRIC_ORDER)}


@lru_cache(maxsize=4096)
def _cached_list_taxa_ranking_options(within_taxon: str, descendant_rank: str, data_version: str) -> dict:
    resolved = taxa.get_taxon_by_id(within_taxon) or taxa.get_taxon_by_slug(within_taxon)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Taxon not found: {within_taxon}")

    norm_rank = descendant_rank.upper()
    rank_key = "SUBSPECIES" if norm_rank in _CONFIG.subspecies_equivalents else norm_rank

    try:
        rows = _storage.read_table(
            GLOBAL_STATS_DIR / RANKINGS_FILE,
            columns=["variable", "metric", "count"],
            filters=[
                ("contextTaxonId", "=", str(resolved["taxon_key"])),
                ("rank", "=", rank_key),
            ],
        ).to_pandas()
        # `count` is the same for every row in a (variable, metric) group (the
        # sort's full eligible population, computed once at build time — see
        # util/rankings.py::_write_rank_positions) — group just to get one
        # (variable, metric) -> count entry per option.
        column_counts = {
            (variable, metric): int(count)
            for variable, metric, count in rows.groupby(["variable", "metric"], as_index=False)["count"].max().itertuples(index=False)
        }
    except Exception:
        return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": []}

    all_layers = tiles.load_layers()
    variable_order = {v["id"]: i for i, v in enumerate(all_layers)}
    layer_value_types = {v["id"]: v.get("value_type", "") for v in all_layers}

    legend_cache: dict[str, dict[int, str]] = {}

    def _class_label(variable: str, metric: str) -> str:
        if variable not in legend_cache:
            legend_cache[variable] = {
                int(c["id"]): c.get("name", str(c["id"]))
                for c in _load_legend(variable)
                if "id" in c
            }
        try:
            class_id = int(metric[6:])
        except (ValueError, IndexError):
            return metric
        return legend_cache[variable].get(class_id, metric)

    options = []
    for (variable, metric), count in column_counts.items():
        if count <= 0:
            continue
        if metric == "mode" and layer_value_types.get(variable) == "nominal":
            continue
        if metric.startswith("class_"):
            label = _class_label(variable, metric)
            if label == metric:
                continue
        else:
            label = _METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())
        options.append({
            "variable": variable,
            "metric": metric,
            "label": label,
            "column": f"{variable}::{metric}",
            "count": count,
        })

    options.sort(key=lambda e: (
        variable_order.get(e["variable"], len(variable_order)),
        e["variable"],
        _METRIC_RANK.get(e["metric"], len(_METRIC_ORDER)),
        e["metric"],
    ))

    return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": options}


@app.get("/api/taxa/ranking-options")
@limiter.limit(_RATE_LIMIT_DETAIL)
def list_taxa_ranking_options(
    request: Request,
    within_taxon: str = Query(...),
    descendant_rank: str = Query(...),
):
    return _cached_list_taxa_ranking_options(within_taxon, descendant_rank, _DATA_VERSION)


@lru_cache(maxsize=2048)
def _cached_query_taxa(
    q: str | None,
    within_taxon: str | None,
    descendant_rank: str | None,
    sort_variable: str | None,
    sort_metric: str | None,
    sort_order: str,
    limit: int,
    offset: int,
    min_samples: int,
    include_species_like: bool,
    location: str | None,
    unit_system: str | None,
    sort_reference: float | None,
    min_rbar: float | None,
    filter_params: tuple[str, ...],
    data_version: str,
) -> dict:
    """Bounded LRU — real traffic clusters on a handful of common
    browse/sort/filter combos even though the full param space is
    combinatorial; a size cap captures the hot set without unbounded
    growth. filter_params is a tuple here (hashable cache key) rather than
    the list FastAPI hands the route — see query_taxa below."""
    normalized_q = normalize_name(q or "") or None

    resolved_taxon: taxa.TaxonRecord | None = None
    if within_taxon:
        resolved_taxon = taxa.get_taxon_by_id(within_taxon)
        if resolved_taxon is None:
            resolved_taxon = taxa.get_taxon_by_slug(within_taxon)
        if resolved_taxon is None:
            raise HTTPException(status_code=404, detail=f"Taxon not found: {within_taxon}")

    norm_rank = descendant_rank.upper() if descendant_rank else None
    norm_sort_variable = _resolve_variable_id(sort_variable) if sort_variable else None

    all_layers = tiles.load_layers()
    layer_by_id = {lyr["id"]: lyr for lyr in all_layers}
    # Each ?filter=variable:metric:op:value[:count] chains an extra summary-stat
    # predicate on top of scope/sort/text/location (e.g. "avg temp < 25C AND
    # at least 10 observations in ecoregion X") — never raw SQL from the
    # client, just a strict tuple validated against the known layer catalog.
    parsed_filters: list[rankings.StatFilter] = []
    for raw in filter_params:
        try:
            parsed = rankings.parse_stat_filter(raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        resolved_variable = _resolve_variable_id(parsed.variable)
        value = parsed.value
        # class_* filters compare a percentage/count, not a physical unit —
        # only scalar stat metrics (e.g. bio1 mean) need the same
        # display-unit-to-raw conversion /slice's min/max range already does.
        if not parsed.metric.startswith("class_"):
            layer = layer_by_id.get(resolved_variable)
            if layer is not None:
                value = units.convert_value_from_display(
                    value, layer, unit_system, metric=parsed.metric,
                )
        parsed_filters.append(
            parsed._replace(variable=resolved_variable, value=value)
        )

    result = rankings.query_taxa(
        q=normalized_q,
        within_taxon=resolved_taxon,
        descendant_rank=norm_rank,
        sort_variable=norm_sort_variable,
        sort_metric=sort_metric,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
        min_samples=min_samples,
        include_species_like=include_species_like,
        location_gid=location,
        reference_value=sort_reference,
        min_rbar=min_rbar,
        stat_filters=parsed_filters or None,
        layers=all_layers,
    )

    sort_layer = next((lyr for lyr in all_layers if lyr["id"] == norm_sort_variable), None) if norm_sort_variable else None
    is_class_metric = bool(sort_metric and sort_metric.startswith("class_"))
    serialized: list[dict] = []
    for item in result["results"]:
        taxon = item["taxon"]
        preferred = taxon.get("inat_preferred_common_name") or taxon.get("common_name") or ""
        match_name = item.get("match_name") or ""
        # Show the matched vernacular name when the query hit a non-preferred name
        # (e.g. searching "canyonlands pricklypear" shows "Canyonlands Pricklypear",
        # not the preferred "Navajo Bridge Pricklypear"). Fall back to preferred when
        # the match was against the scientific name or there was no text query.
        sci_norm = normalize_name(taxon.get("scientific_name") or "")
        use_match = bool(match_name) and normalize_name(match_name) != sci_norm
        display_name = format_common_name(match_name if use_match else preferred)
        raw_sort = item.get("sort_value")
        if is_class_metric:
            converted_sort = raw_sort * 100 if raw_sort is not None else None
        else:
            converted_sort = units.convert_value(raw_sort, sort_layer, unit_system, metric=sort_metric) if sort_layer else raw_sort
        serialized.append({
            "taxon_id": taxon["taxon_key"],
            "scientific_name": taxon.get("scientific_name", "").replace("_", " "),
            "common_name": display_name or None,
            "common_names": None,
            "rank": taxon.get("rank"),
            "slug": taxon_slug(taxon.get("scientific_name")),
            "description": None,
            **_image_fields(taxon),
            "match_score": item.get("match_score"),
            "sample_count": item.get("sample_count"),
            "sort_value": converted_sort,
            "sort_variable": sort_variable,
            "sort_metric": sort_metric,
            "location_count": item.get("location_count"),
            "position": item.get("position"),
            "percentile": item.get("percentile"),
        })

    return {
        "query": normalized_q,
        "scope": {
            "within_taxon": resolved_taxon["taxon_key"] if resolved_taxon else None,
            "descendant_rank": norm_rank,
            "location": location,
            "min_samples": min_samples,
            "include_species_like": include_species_like,
            "filters": filter_params,
        },
        "sort": {
            "variable": sort_variable,
            "metric": sort_metric,
            "order": sort_order,
            "units": "%" if is_class_metric else (units.display_units(sort_layer, unit_system, metric=sort_metric) if sort_layer else None),
        },
        "total": result["total"],
        "matched_total": result["matched_total"],
        "eligible_total": result["eligible_total"],
        "empty_reason": result["empty_reason"],
        "limit": limit,
        "offset": offset,
        "results": serialized,
    }


@app.get("/api/taxa/query")
@limiter.limit(_RATE_LIMIT_SEARCH)
def query_taxa(
    request: Request,
    q: str | None = Query(None, min_length=1),
    within_taxon: str | None = Query(None),
    descendant_rank: str | None = Query(None),
    sort_variable: str | None = Query(None),
    sort_metric: str | None = Query(None),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    min_samples: int = Query(10, ge=0),
    include_species_like: bool = Query(False),
    location: str | None = Query(None),
    unit_system: str | None = Query(None),
    sort_reference: float | None = Query(None),
    min_rbar: float | None = Query(None, ge=0.0, le=1.0),
    filter_params: list[str] = Query([], alias="filter"),
):
    return _cached_query_taxa(
        q, within_taxon, descendant_rank, sort_variable, sort_metric, sort_order,
        limit, offset, min_samples, include_species_like, location, unit_system,
        sort_reference, min_rbar, tuple(filter_params), _DATA_VERSION,
    )


# ---------------------------------------------------------------------------
# Response-cache invalidation — called from reload_data()/push_temporal_state()
# whenever their respective version tag bumps, purely to drop now-unreachable
# stale-version entries before LRU pressure would otherwise evict them.
# Not load-bearing for correctness: each cached function's version tag is
# already baked into its lru_cache key, so a version bump alone makes old
# entries unreachable even if these clears are ever skipped.
# ---------------------------------------------------------------------------

def _clear_data_versioned_caches() -> None:
    _cached_get_taxon.cache_clear()
    _cached_get_species_obscured.cache_clear()
    _cached_get_taxon_env_stats.cache_clear()
    _cached_get_species_environment_base.cache_clear()
    _cached_get_species_occurrences.cache_clear()
    _cached_get_species_locations.cache_clear()
    _cached_list_taxa_ranking_options.cache_clear()
    _cached_query_taxa.cache_clear()
    _cached_list_variables.cache_clear()  # also temporal-tagged, see below


def _clear_temporal_versioned_caches() -> None:
    _cached_list_variables.cache_clear()


@app.post("/upload/raw-observations")
@limiter.limit(_RATE_LIMIT_UPLOAD)
async def upload_raw_observations(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Accept a CSV, TSV, or Parquet file and queue it for processing.

    Returns a job ID immediately. Poll /upload/status/{job_id} for progress,
    then fetch the result from /upload/download/{job_id} when status is 'done'.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".tsv", ".parquet"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: CSV, TSV, Parquet.",
        )

    # Reject on the client-declared size before buffering anything into
    # memory — the row-count check below only runs after a full
    # `await file.read()`, which would otherwise buffer an arbitrarily
    # large body first.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB size limit.",
                )
        except ValueError:
            pass

    if len(_upload_jobs) >= _MAX_CONCURRENT_UPLOAD_JOBS:
        raise HTTPException(
            status_code=429,
            detail="Too many uploads are already being processed. Try again shortly.",
        )

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB size limit.",
        )
    buf = io.BytesIO(contents)
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(buf)
        elif suffix == ".tsv":
            df = pd.read_csv(buf, sep="\t")
        else:
            df = pd.read_csv(buf)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc

    if len(df) > _MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Upload exceeds the {_MAX_UPLOAD_ROWS:,}-row limit ({len(df):,} rows).",
        )

    static_layer_ids = {
        layer["id"] for layer in tiles.load_layers()
        if layer.get("filename") and layer.get("window_hours") is None
    }

    df = upload.normalize_coordinate_columns(df)
    df = upload.normalize_timestamp_column(df)
    df = upload.ensure_catalog_numbers(df)
    df = upload.ensure_observation_names(df)
    df = upload.validate_coordinates(df)
    upload.check_reserved_columns(df, static_layer_ids)

    job_id = str(uuid.uuid4())
    _upload_jobs[job_id] = _UploadJob(job_id=job_id, df=df)
    _upload_queue.append(job_id)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "position": len(_upload_queue), "status": "queued"},
    )


@app.get("/upload/status/{job_id}")
async def upload_job_status(job_id: str):
    """Return the current status and queue position of an upload job."""
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    position = _upload_queue.index(job_id) + 1 if job_id in _upload_queue else 0
    return {"job_id": job_id, "status": job.status, "position": position, "error": job.error}


@app.get("/upload/download/{job_id}")
async def upload_job_download(background_tasks: BackgroundTasks, job_id: str) -> FileResponse:
    """Download the processed archive for a completed upload job.

    The archive is removed from the server after this call.
    """
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "Processing failed.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready (status: {job.status}).")
    if not job.archive_path or not job.archive_path.exists():
        raise HTTPException(status_code=410, detail="Archive has expired or was removed.")
    if job.work_dir:
        background_tasks.add_task(shutil.rmtree, job.work_dir, True)
    _upload_jobs.pop(job_id, None)
    return FileResponse(
        path=job.archive_path,
        media_type="application/zip",
        filename=job.archive_name or "processed_observations.zip",
    )
