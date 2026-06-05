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
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

import util.rankings as rankings
from config.config import load_config
from util import citations, gis, taxa, tiles, units, upload
from util.rankings import TREE_ROOT as RANKINGS_TREE_ROOT
from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    TREE_ROOT,
    apply_phenology_filter,
    apply_timestamp_filter,
    collect_taxon_df,
    compute_location_filtered_stats,
    compute_phenology_counts,
    read_phenology_counts,
)
from util.storage import ParquetStorageProxy
from util.taxa import format_common_name, iter_descendants, normalize_name, taxon_slug

_CONFIG = load_config("global")
_SYNC_STATE_PATH = Path("data/sync_state.json")
_PIPELINE_STATE_PATH = Path("data/pipeline_state.json")
_TEMPORAL_STATE_PATH = Path("data/temporal_state.json")
_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent,
)
_LEGEND_DIR = Path("config/gis/legends")
_OCC_FILE = "occurrence.parquet"
_OCC_COLUMNS = ["catalogNumber", "decimalLatitude", "decimalLongitude", "obscured", "coordinateUncertaintyInMeters"]
_PHENOLOGY_VALUES: frozenset[str] = frozenset(_CONFIG.phenology_values)
_LOCATIONS_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "gis" / "locations"
_LOC_TAXA_PATH = _LOCATIONS_DIR / "location_taxa.parquet"


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


def _lookup_index_value(taxon: dict, variable_id: str, catalog_number: str) -> float | None:
    """Read an env value for a known observation directly from occurrence.parquet."""
    occ_path = TREE_ROOT / taxon["path"] / "occurrence.parquet"
    try:
        import pyarrow.parquet as _pq
        tbl = _pq.read_table(occ_path, columns=["catalogNumber", variable_id])
        df = tbl.to_pandas()
        row = df[df["catalogNumber"] == catalog_number]
        if row.empty or variable_id not in row.columns:
            return None
        val = row.iloc[0][variable_id]
        return float(val) if val is not None and pd.notna(val) else None
    except Exception:
        return None


def _filter_occ_df(df: pd.DataFrame) -> pd.DataFrame:
    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[df["coordinateUncertaintyInMeters"] <= 500]
    return df

# ---------------------------------------------------------------------------
# Upload job queue
# ---------------------------------------------------------------------------

_MAX_UPLOAD_ROWS = 50_000
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
            df = await run_in_threadpool(upload.enrich_with_gis, job.df)
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
    asyncio.create_task(_upload_consumer())
    asyncio.create_task(_cleanup_old_jobs())
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"], expose_headers=["X-Nominal-Classes"])


def _image_fields(taxon: dict) -> dict:
    """Return unified image_* fields, preferring iNat over GBIF backup."""
    prefix = "inat_preferred" if taxon.get("inat_preferred_image") else "gbif_backup"
    return {
        "image_url": taxon.get(f"{prefix}_image") or None,
        "image_license": taxon.get(f"{prefix}_image_license") or None,
        "image_creator": taxon.get(f"{prefix}_image_creator") or None,
        "image_rights_holder": taxon.get(f"{prefix}_image_attribution") or None,
        "image_references": taxon.get(f"{prefix}_image_references") or None,
    }


_VALUE_TYPE_MAP = {"interval": "continuous", "ratio": "continuous", "nominal": "categorical"}


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/version")
def version():
    try:
        state = json.loads(_SYNC_STATE_PATH.read_text()) if _SYNC_STATE_PATH.exists() else {}
        crawl_ts = (
            state.get("gbif_occurrences", {}).get("crawl_finished")
            or state.get("gbif_taxonomy", {}).get("crawl_finished")
        )
    except Exception:
        crawl_ts = None
    return {"version": crawl_ts}


@app.get("/status")
async def status():
    pipeline = await run_in_threadpool(_status_pipeline)
    temporal = await run_in_threadpool(_status_temporal)
    server = await run_in_threadpool(_status_server)
    active_job = next(
        (j for j in _upload_jobs.values() if j.status == "processing"), None
    )
    return {
        "pipeline": pipeline,
        "temporal": temporal,
        "upload_queue": {
            "depth": len(_upload_queue),
            "active": active_job is not None,
        },
        "server": server,
    }


@app.post("/internal/pipeline-state", status_code=200)
async def push_pipeline_state(body: dict):
    from datetime import UTC
    from datetime import datetime as _dt
    body["received_at"] = _dt.now(UTC).isoformat()
    await run_in_threadpool(
        lambda: _PIPELINE_STATE_PATH.write_text(json.dumps(body))
    )
    return {"ok": True}


@app.post("/internal/temporal-state", status_code=200)
async def push_temporal_state(body: dict):
    from datetime import UTC
    from datetime import datetime as _dt
    body["received_at"] = _dt.now(UTC).isoformat()
    await run_in_threadpool(
        lambda: _TEMPORAL_STATE_PATH.write_text(json.dumps(body))
    )
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


def _status_server() -> dict:
    import time as _time
    result: dict = {}

    # CPU usage — two samples 300ms apart
    try:
        def _read_cpu():
            with open("/proc/stat") as f:
                parts = f.readline().split()
            vals = list(map(int, parts[1:8]))
            return vals[3] + vals[4], sum(vals)  # idle, total

        i1, t1 = _read_cpu()
        _time.sleep(0.3)
        i2, t2 = _read_cpu()
        result["cpu_percent"] = round((1 - (i2 - i1) / (t2 - t1)) * 100, 1)
    except Exception:
        result["cpu_percent"] = None

    # CPU temp
    try:
        import psutil
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

    # Uptime
    try:
        with open("/proc/uptime") as f:
            result["uptime_s"] = int(float(f.read().split()[0]))
    except Exception:
        result["uptime_s"] = None

    return result


@app.get("/data-sources")
def data_sources():
    return citations.load_data_sources()


@app.get("/variables")
def list_variables(unit_system: str | None = Query(None)):
    result = []
    for layer, category in tiles.load_layers_with_category():
        value_type = _VALUE_TYPE_MAP.get(layer.get("value_type", ""), "continuous")
        legend_classes = None
        if value_type == "categorical":
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
        rmin, rmax = tiles.get_layer_render_range(layer)
        result.append({
            "id": layer["id"],
            "name": layer.get("display_name"),
            "units": units.display_units(layer, unit_system),
            "value_type": value_type,
            "domain": layer.get("domain") or None,
            "category": category.get("display_name", "Other"),
            "source_ids": [layer["source"]] if layer.get("source") else None,
            "legend_classes": legend_classes,
            "render_min": units.convert_value(rmin, layer, unit_system),
            "render_max": units.convert_value(rmax, layer, unit_system),
            "group": layer.get("group") or None,
            "group_label": layer.get("group_label") or None,
        })
    return result


@app.get("/api/layers")
def list_layers():
    return tiles.load_layers()


@app.get("/phenology_values")
def list_phenology_values():
    return [
        {"value": v, "label": v.capitalize()}
        for v in sorted(_CONFIG.phenology_values)
    ]


@app.get("/gis/point")
async def gis_point_value(
    lat: float = Query(...),
    lon: float = Query(...),
    variable: str = Query(...),
    taxon_id: str | None = Query(None),
    catalog_number: str | None = Query(None),
    unit_system: str | None = Query(None),
):
    """Return the raster value for a variable at a lat/lon coordinate.

    If taxon_id and catalog_number are both provided the value is read from
    occurrence_index.parquet instead of the raster — ensures the returned value
    is identical to what the stats were computed from, and for temporal variables
    returns the historical aggregate at observation time rather than the current
    live window. Falls back to raster sampling when the index row is missing.
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=400, detail="lat and lon must be finite numbers")

    variable = _resolve_variable_id(variable.strip())
    try:
        layer = tiles.get_layer(variable)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Variable '{variable}' not found")

    value: float | None = None

    if taxon_id and catalog_number:
        taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
        if taxon is not None:
            value = _lookup_index_value(taxon, variable, catalog_number)

    if value is None:
        value = await run_in_threadpool(gis.sample_point, layer, lat, lon)

    class_name: str | None = None
    class_color: str | None = None
    if value is not None and layer.get("value_type") == "nominal":
        legend = _load_legend(variable)
        int_val = int(value) if value == int(value) else None
        for entry in legend:
            if entry.get("id") == int_val:
                class_name = entry.get("name")
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


@app.get("/api/variables/{variable_id}/tiles/{z}/{x}/{y}.png")
async def variable_tile_compat(variable_id: str, z: int, x: int, y: int, tile_size: int = Query(256, ge=32, le=1024)):
    """Compatibility shim for old frontend URL pattern (/api/variables/bio_1/ → bio1)."""
    layer_id = _resolve_variable_id(variable_id)
    return await layer_tile(layer_id, z, x, y, tile_size)


@app.get("/api/layers/{layer_id}/tiles/{z}/{x}/{y}.png")
async def layer_tile(layer_id: str, z: int, x: int, y: int, tile_size: int = Query(256, ge=32, le=1024)):
    try:
        layer = tiles.get_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")

    payload = await run_in_threadpool(
        tiles.render_layer_tile_bytes,
        layer_id, z, x, y, tile_size,
    )
    is_temporal = layer.get("window_hours") is not None
    cache_max_age = 300 if is_temporal else 604800
    headers: dict[str, str] = {"Cache-Control": f"public, max-age={cache_max_age}"}
    if str(layer.get("value_type") or "").lower() == "nominal":
        class_counts = await run_in_threadpool(tiles.nominal_tile_range_classes, layer_id, z, x, y, x, y)
        if class_counts:
            ordered = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
            headers["X-Nominal-Classes"] = ",".join(f"{cls}:{cnt}" for cls, cnt in ordered)
    return Response(content=payload, media_type="image/png", headers=headers)


@app.get("/api/layers/{layer_id}/tile-range/classes")
async def layer_tile_range_classes(
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


@app.get("/api/taxon/{taxon_id}")
@app.get("/api/species/{taxon_id}")
def get_taxon(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    sci = taxon.get("scientific_name", "")
    def _title_name(s: str) -> str:
        def cap_word(w: str) -> str:
            return re.sub(r"(^|(?<=-))[a-z]", lambda m: m.group().upper(), w.lower())
        return " ".join(cap_word(w) for w in s.split())
    vnames = [_title_name(n) for n in taxon.get("vernacular_names") or []]
    return {**taxon, "scientific_name": sci.replace("_", " "), "vernacular_names": vnames, **_image_fields(taxon)}


@app.get("/api/species/{taxon_id}/obscured")
def get_species_obscured(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return {"allObscured": False}


@app.get("/api/taxon/{taxon_id}/env-stats")
def get_taxon_env_stats(taxon_id: str, unit_system: str | None = Query(None)):
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

    density_by_var: dict[str, dict] = {}
    for row in _storage.read_table(GLOBAL_STATS_DIR / DENSITY_FILE, filters=_tk).to_pylist():
        row.pop("taxon_key", None)
        var = row.pop("variable")
        density_by_var[var] = row

    all_var_ids = list(dict.fromkeys(list(numerical_stats) + list(circular_stats) + list(nominal_stats)))
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
        else:
            entry["stats"] = nominal_stats[var_id]
            entry["density"] = None
            entry["classes"] = nominal_classes.get(var_id, [])
        variables.append(entry)

    return {"variables": variables}


# ---------------------------------------------------------------------------
# Legacy compatibility endpoints (frontend still uses these URL patterns)
# ---------------------------------------------------------------------------

def _load_relative_ranks(taxon_key: str, variable_id: str) -> list[dict]:
    """Read per-context {rank}_positions.parquet files for one taxon+variable."""
    taxon = taxa.get_taxon_by_id(taxon_key)
    if not taxon:
        return []
    path = taxon.get("path", "")
    rank = (taxon.get("rank") or "").lower()
    if not path or not rank:
        return []

    parts = path.split("/")
    result = []
    cumulative = ""
    for i, part in enumerate(parts[:-1]):
        cumulative = part if i == 0 else f"{cumulative}/{part}"
        positions_file = RANKINGS_TREE_ROOT / cumulative / f"{rank}_positions.parquet"
        if not positions_file.exists():
            continue
        try:
            rows = _storage.read_table(
                positions_file,
                filters=[("taxon_key", "=", taxon_key), ("variable", "=", variable_id)],
            ).to_pylist()
        except Exception:
            continue
        for row in rows:
            position = row.get("position") or 0
            count = row.get("count") or 0
            percentile = round(position / count, 3) if count > 0 else 0.0
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


def _timestamp_range_from_metadata(path: Path) -> tuple[int, int] | None:
    """Read min/max eventTimestamp from parquet footer stats — no row scan required."""
    try:
        meta = _storage.read_metadata(path)
        if meta.num_row_groups == 0:
            return None
        col_idx = None
        rg0 = meta.row_group(0)
        for j in range(rg0.num_columns):
            if rg0.column(j).path_in_schema == "eventTimestamp":
                col_idx = j
                break
        if col_idx is None:
            return None
        ts_min: int | None = None
        ts_max: int | None = None
        for i in range(meta.num_row_groups):
            stats = meta.row_group(i).column(col_idx).statistics
            if stats and stats.has_statistics and stats.min is not None:
                ts_min = stats.min if ts_min is None else min(ts_min, stats.min)
                ts_max = stats.max if ts_max is None else max(ts_max, stats.max)
        return (int(ts_min), int(ts_max)) if ts_min is not None and ts_max is not None else None
    except Exception:
        return None


def _location_filter_col(gid: str) -> str | None:
    """Return the occurrence.parquet column to use when filtering observations to gid."""
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
    if df.empty:
        return []
    col = pd.to_numeric(df[variable_id], errors="coerce")
    if circular_wrap:
        mask = col.between(value_min, 360.0, inclusive="both") | col.between(0.0, value_max, inclusive="both")
    else:
        mask = col.between(value_min, value_max, inclusive="both")
    df = df[mask].dropna(subset=["decimalLatitude", "decimalLongitude"])
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
    if df.empty:
        return []
    col = pd.to_numeric(df[variable_id], errors="coerce")
    df = df[col == class_value].dropna(subset=["decimalLatitude", "decimalLongitude"])
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


@app.get("/species/{taxon_id}/environment/{variable_id}")
def get_species_environment(
    taxon_id: str, variable_id: str, unit_system: str | None = None,
    location: str | None = None, phenology: str | None = None,
    start_ts: int | None = None, end_ts: int | None = None,
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")

    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    variable_metadata = {
        "name": layer["display_name"] if layer else variable_id,
        "units": units.display_units(layer, unit_system) if layer else None,
        "value_type": layer.get("value_type") if layer else None,
        "domain": (layer.get("domain") or None) if layer else None,
    }
    value_type = layer.get("value_type") if layer else None

    if (location is not None or phenology_norm is not None or start_ts is not None or end_ts is not None) and layer is not None:
        filter_col = _location_filter_col(location) if location is not None else None
        if location is None or filter_col is not None:
            result = compute_location_filtered_stats(
                taxon, variable_id, filter_col, location, layer,
                phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
                storage=_storage,
            )
            if result is not None:
                if result["type"] == "continuous":
                    stats = result["stats"]
                    raw_summary = {
                        "count": stats["count"],
                        "min": stats.get("min"),
                        "mean": stats.get("mean"),
                        "max": stats.get("max"),
                        "stddev": stats.get("std"),
                        "q10": stats.get("10th_percentile"),
                        "q90": stats.get("90th_percentile"),
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
                return {
                    "species_id": taxon.get("taxon_key"),
                    "variable": variable_id,
                    "variable_metadata": variable_metadata,
                    "observation_count": total_samples,
                    "summary": {"count": total_samples, "min": None, "mean": None, "max": None},
                    "density_curve": None,
                    "categorical_distribution": categorical_distribution,
                    "relative_ranks": [],
                }

    if value_type == "nominal":
        rows = _storage.read_table(
            GLOBAL_STATS_DIR / NOMINAL_STATS_FILE,
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
        categorical_distribution.sort(key=lambda x: -x["fraction"])
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
                "unique_classes": int(metrics["unique_classes"]) if "unique_classes" in metrics else None,
                "entropy": float(metrics["entropy"]) if "entropy" in metrics else None,
                "mode": int(metrics["mode"]) if "mode" in metrics else None,
            },
            "density_curve": None,
            "categorical_distribution": categorical_distribution,
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
        "stddev": row.get("std"),
        "q10": row.get("10th_percentile"),
        "q90": row.get("90th_percentile"),
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


@app.get("/species/{taxon_id}/occurrences")
def get_species_occurrences(
    taxon_id: str,
    location: str | None = None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")

    is_leaf = taxon["rank"] in _CONFIG.leaf_rank_set
    filter_col = _location_filter_col(location) if location is not None else None
    has_loc_or_pheno = filter_col is not None or phenology_norm is not None
    has_ts = start_ts is not None or end_ts is not None
    use_precomputed_pheno = not has_loc_or_pheno and not has_ts

    extra_cols: list[str] = []
    if filter_col:
        extra_cols.append(filter_col)
    # Always read rcs so we can fall back to live phenology counts if precomputed is missing
    extra_cols.append("rcs")
    # Need eventTimestamp in data when filtering by it, or when computing range
    # from row-filtered data (loc/pheno active).
    if has_ts or has_loc_or_pheno:
        extra_cols.append("eventTimestamp")
    occ_columns = list(_OCC_COLUMNS) + extra_cols

    collected: list[dict] = []
    seen: set[str] = set()
    ts_min: int | None = None
    ts_max: int | None = None
    pheno_acc: Counter = Counter()

    def _read_occ(path: Path) -> None:
        nonlocal ts_min, ts_max
        # Fast path: parquet footer stats when no row-level filters change the range
        if not has_loc_or_pheno:
            result = _timestamp_range_from_metadata(path)
            if result:
                lo, hi = result
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)
        try:
            schema_names = set(_storage.read_schema(path).names)
            cols_to_read = [c for c in occ_columns if c in schema_names]
            table = _storage.read_table(path, columns=cols_to_read)
        except Exception:
            return
        if table.num_rows == 0:
            return
        df = _filter_occ_df(table.to_pandas())
        if filter_col is not None:
            df = df[df[filter_col].astype(str) == str(location)]
        if phenology_norm is not None:
            df = apply_phenology_filter(df, phenology_norm)
        # Range from actual data when loc/pheno filters are active (before ts filter)
        if has_loc_or_pheno and "eventTimestamp" in df.columns:
            ts_col = pd.to_numeric(df["eventTimestamp"], errors="coerce").dropna()
            if len(ts_col):
                lo, hi = int(ts_col.min()), int(ts_col.max())
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)
        if has_ts:
            df = apply_timestamp_filter(df, start_ts, end_ts)
        if not use_precomputed_pheno and "rcs" in df.columns:
            pheno_acc.update(compute_phenology_counts(df))
        df = df[["catalogNumber", "decimalLatitude", "decimalLongitude"]].dropna()
        for r in df.to_dict("records"):
            cid = str(r["catalogNumber"])
            if cid in seen:
                continue
            seen.add(cid)
            collected.append({"catalogNumber": cid, "latitude": r["decimalLatitude"], "longitude": r["decimalLongitude"]})

    if taxon["rank"] == _CONFIG.species_rank:
        for desc in iter_descendants(taxon, include_self=True):
            _read_occ(TREE_ROOT / desc["path"] / _OCC_FILE)
    elif is_leaf:
        _read_occ(TREE_ROOT / taxon["path"] / _OCC_FILE)
    else:
        for desc in iter_descendants(taxon, include_self=False):
            _read_occ(TREE_ROOT / desc["path"] / _OCC_FILE)

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


@app.get("/species/{taxon_id}/locations")
def get_species_locations(taxon_id: str, level: int | None = None, parent: str | None = None, limit: int = 500):
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



@app.get("/species/{taxon_id}/environment/{variable_id}/observation-values")
def get_observation_variable_values(
    taxon_id: str,
    variable_id: str,
    unit_system: str | None = None,
):
    """Return raw GIS values for all observations of a taxon for one variable."""
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")

    collected: dict[str, float] = {}

    def _read_occ(path: Path) -> None:
        if not path.exists():
            return
        try:
            import pyarrow.parquet as _pq
            tbl = _pq.read_table(path, columns=["catalogNumber", variable_id])
            for cat, val in zip(tbl["catalogNumber"].to_pylist(), tbl[variable_id].to_pylist()):
                if cat not in collected and val is not None:
                    converted = units.convert_value(float(val), layer, unit_system)
                    if converted is not None:
                        collected[cat] = converted
        except Exception:
            pass

    is_leaf = taxon["rank"] in _CONFIG.leaf_rank_set
    if taxon["rank"] == _CONFIG.species_rank:
        for desc in iter_descendants(taxon, include_self=True):
            _read_occ(TREE_ROOT / desc["path"] / "occurrence.parquet")
    elif is_leaf:
        _read_occ(TREE_ROOT / taxon["path"] / "occurrence.parquet")
    else:
        for desc in iter_descendants(taxon, include_self=False):
            _read_occ(TREE_ROOT / desc["path"] / "occurrence.parquet")

    vals = collected.values()
    obs_min = min(vals) if collected else None
    obs_max = max(vals) if collected else None
    return {
        "variable": variable_id,
        "min": obs_min,
        "max": obs_max,
        "observations": [{"catalogNumber": k, "value": v} for k, v in collected.items()],
    }


@app.get("/species/{taxon_id}/environment/{variable_id}/slice")
def get_species_environment_slice(
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
):
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise HTTPException(status_code=400, detail="min and max must be finite numbers")
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")
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
    raw_min = units.convert_value_from_display(min_value, layer, unit_system)
    raw_max = units.convert_value_from_display(max_value, layer, unit_system)
    filter_col = _location_filter_col(location) if location is not None else None
    if location is None or filter_col is not None:
        observations = _slice_from_raw_occ(
            taxon, variable_id, filter_col, location,
            raw_min, raw_max, circular_wrap, limit,
            phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
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
def get_species_environment_class_samples(
    taxon_id: str,
    variable_id: str,
    class_value: str,
    limit: int | None = Query(None, ge=1, le=10000),
    location: str | None = None,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    phenology_norm = phenology.strip().lower() if phenology else None
    if phenology_norm is not None and phenology_norm not in _PHENOLOGY_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid phenology value: {phenology!r}")
    variable_id = _resolve_variable_id(variable_id)
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")
    if layer.get("value_type") != "nominal":
        raise HTTPException(status_code=400, detail="Numerical variables must use the slice endpoint")
    try:
        parsed: float | int = float(class_value)
        if parsed.is_integer():
            parsed = int(parsed)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid class value: {class_value!r}")
    filter_col = _location_filter_col(location) if location is not None else None
    if location is None or filter_col is not None:
        observations = _class_samples_from_raw_occ(
            taxon, variable_id, filter_col, location, float(parsed), limit, phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
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


@app.get("/api/taxa/ranking-options")
def list_taxa_ranking_options(
    within_taxon: str = Query(...),
    descendant_rank: str = Query(...),
):
    resolved = taxa.get_taxon_by_id(within_taxon) or taxa.get_taxon_by_slug(within_taxon)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Taxon not found: {within_taxon}")

    norm_rank = descendant_rank.upper()
    rank_lower = "subspecies" if norm_rank in _CONFIG.subspecies_equivalents else norm_rank.lower()
    index_path = rankings.TREE_ROOT / resolved["path"] / f"{rank_lower}_index.parquet"

    try:
        schema = _storage.read_schema(index_path)
        raw_lengths = (schema.metadata or {}).get(b"column_lengths")
        column_lengths = {k: int(v) for k, v in json.loads(raw_lengths).items() if int(v) > 0} if raw_lengths else {}
    except Exception:
        return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": []}

    variable_order = {v["id"]: i for i, v in enumerate(tiles.load_layers())}

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
    for col in schema.names:
        if "::" not in col:
            continue
        count = int(column_lengths.get(col, 0) or 0)
        if count <= 0:
            continue
        variable, metric = col.split("::", 1)
        if metric.startswith("class_"):
            label = _class_label(variable, metric)
        else:
            label = _METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())
        options.append({
            "variable": variable,
            "metric": metric,
            "label": label,
            "column": col,
            "count": count,
        })

    options.sort(key=lambda e: (
        variable_order.get(e["variable"], len(variable_order)),
        e["variable"],
        _METRIC_RANK.get(e["metric"], len(_METRIC_ORDER)),
        e["metric"],
    ))

    return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": options}


@app.get("/api/taxa/query")
def query_taxa(
    q: str | None = Query(None, min_length=1),
    within_taxon: str | None = Query(None),
    descendant_rank: str | None = Query(None),
    sort_variable: str | None = Query(None),
    sort_metric: str | None = Query(None),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    min_samples: int = Query(0, ge=0),
    include_species_like: bool = Query(False),
    location: str | None = Query(None),
    unit_system: str | None = Query(None),
    sort_reference: float | None = Query(None, ge=0.0, lt=360.0),
    min_rbar: float | None = Query(None, ge=0.0, le=1.0),
):
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
    )

    sort_layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == norm_sort_variable), None) if norm_sort_variable else None
    serialized: list[dict] = []
    for item in result["results"]:
        taxon = item["taxon"]
        preferred = taxon.get("inat_preferred_common_name") or taxon.get("common_name") or ""
        raw_sort = item.get("sort_value")
        converted_sort = units.convert_value(raw_sort, sort_layer, unit_system, metric=sort_metric) if sort_layer else raw_sort
        serialized.append({
            "taxon_id": taxon["taxon_key"],
            "scientific_name": taxon.get("scientific_name", "").replace("_", " "),
            "common_name": format_common_name(preferred) or None,
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
        },
        "sort": {
            "variable": sort_variable,
            "metric": sort_metric,
            "order": sort_order,
            "units": units.display_units(sort_layer, unit_system) if sort_layer else None,
        },
        "total": result["total"],
        "matched_total": result["matched_total"],
        "eligible_total": result["eligible_total"],
        "empty_reason": result["empty_reason"],
        "limit": limit,
        "offset": offset,
        "results": serialized,
    }


@app.post("/upload/raw-observations")
async def upload_raw_observations(
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

    contents = await file.read()
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
