import csv
import io
import json
import math
import re
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

import util.rankings as rankings
from config.config import load_config
from util import citations, gis, taxa, tiles, upload
from util.rankings import POSITION_FILE
from util.stats import (
    CIRCULAR_STATS_FILE,
    NOMINAL_STATS_FILE,
    DENSITY_FILE,
    NUMERICAL_STATS_FILE,
    OCCURRENCE_INDEX_FILE,
    TREE_ROOT,
    apply_phenology_filter,
    apply_timestamp_filter,
    collect_taxon_df,
    compute_location_filtered_stats,
    compute_phenology_counts,
    read_phenology_counts,
)
from util.taxa import format_common_name, iter_descendants, normalize_name, taxon_slug

_CONFIG = load_config("global")
_LEGEND_DIR = Path("config/gis/legends")
_OCC_FILE = "occurrence.parquet"
_OCC_COLUMNS = ["catalogNumber", "decimalLatitude", "decimalLongitude", "obscured", "coordinateUncertaintyInMeters"]
_PHENOLOGY_VALUES: frozenset[str] = frozenset(_CONFIG.phenology_values)
_LOCATIONS_DIR = Path("data/gis/locations")
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
    path = _LEGEND_DIR / f"{layer_id}_legend.json"
    if not path.exists():
        # Temporal ids like weather_code_simple_mode_24h → weather_code_simple
        base_id = re.sub(r'_(avg|sum|mode|snapshot)_\d+h$', '', layer_id, flags=re.IGNORECASE)
        if base_id != layer_id:
            path = _LEGEND_DIR / f"{base_id}_legend.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("classes", [])


def _lookup_index_value(taxon: dict, variable_id: str, catalog_number: str) -> float | None:
    """Read a precomputed env value for a known observation from occurrence_index.parquet.

    Preferred over raster sampling for static variables: avoids the FP sensitivity
    that can cause mismatches on derived variables like aspect near flat terrain.
    Returns None if the index doesn't exist, the column is absent, or the row is missing.
    """
    index_path = TREE_ROOT / taxon["path"] / OCCURRENCE_INDEX_FILE
    if not index_path.exists():
        return None
    schema = pq.read_schema(index_path)
    if variable_id not in schema.names:
        return None
    try:
        df = pq.read_table(
            index_path,
            columns=["catalogNumber", variable_id],
            filters=[("catalogNumber", "=", catalog_number)],
        ).to_pandas()
    except Exception:
        return None
    if df.empty:
        return None
    val = df.iloc[0][variable_id]
    return float(val) if pd.notna(val) else None


def _filter_occ_df(df: pd.DataFrame) -> pd.DataFrame:
    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[df["coordinateUncertaintyInMeters"] <= 500]
    return df

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


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


@app.get("/data-sources")
def data_sources():
    return citations.load_data_sources()


@app.get("/variables")
def list_variables():
    return [
        {
            "id": layer["id"],
            "name": layer.get("display_name"),
            "units": layer.get("units") or None,
            "value_type": _VALUE_TYPE_MAP.get(layer.get("value_type", ""), "continuous"),
            "domain": layer.get("domain") or None,
            "category": category.get("display_name", "Other"),
            "source_ids": [layer["source"]] if layer.get("source") else None,
        }
        for layer, category in tiles.load_layers_with_category()
    ]


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
):
    """Return the raster value for a variable at a lat/lon coordinate.

    For static layers, if taxon_id and catalog_number are both provided the value
    is read from occurrence_index.parquet instead of the raster — this avoids FP
    mismatches on sensitive derived variables (e.g. aspect on flat terrain) and
    ensures the returned value is identical to what the stats were computed from.
    Falls back to raster sampling when the index row is missing.

    For temporal layers the index is ignored; always returns the current
    (no-forecast-offset) aggregate for the requested window.
    """
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=400, detail="lat and lon must be finite numbers")

    variable = _resolve_variable_id(variable.strip())
    try:
        layer = tiles.get_layer(variable)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Variable '{variable}' not found")

    is_temporal = layer.get("window_hours") is not None
    value: float | None = None

    if not is_temporal and taxon_id and catalog_number:
        taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
        if taxon is not None:
            value = _lookup_index_value(taxon, variable, catalog_number)

    if value is None:
        value = await run_in_threadpool(gis.sample_point, layer, lat, lon)

    class_name: str | None = None
    if value is not None and layer.get("value_type") == "nominal":
        legend = _load_legend(variable)
        int_val = int(value) if value == int(value) else None
        for entry in legend:
            if entry.get("id") == int_val:
                class_name = entry.get("name")
                break

    return {
        "variable": variable,
        "units": layer.get("units") or None,
        "lat": lat,
        "lon": lon,
        "value": value,
        "class_name": class_name,
    }


@app.get("/api/variables/{variable_id}/tiles/{z}/{x}/{y}.png")
async def variable_tile_compat(variable_id: str, z: int, x: int, y: int, tile_size: int = Query(256, ge=32, le=1024)):
    """Compatibility shim for old frontend URL pattern (/api/variables/bio_1/ → bio1)."""
    layer_id = _resolve_variable_id(variable_id)
    return await layer_tile(layer_id, z, x, y, tile_size)


@app.get("/api/layers/{layer_id}/tiles/{z}/{x}/{y}.png")
async def layer_tile(layer_id: str, z: int, x: int, y: int, tile_size: int = Query(256, ge=32, le=1024)):
    try:
        tiles.get_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")

    payload = await run_in_threadpool(
        tiles.render_layer_tile_bytes,
        layer_id, z, x, y, tile_size,
    )
    return Response(content=payload, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/taxon/{taxon_id}")
@app.get("/api/species/{taxon_id}")
def get_taxon(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return {**taxon, **_image_fields(taxon)}


@app.get("/api/species/{taxon_id}/obscured")
def get_species_obscured(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return {"allObscured": False}


@app.get("/api/taxon/{taxon_id}/env-stats")
def get_taxon_env_stats(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    taxon_dir = TREE_ROOT / taxon["path"]
    layer_index = {layer["id"]: layer for layer in tiles.load_layers()}

    numerical_stats: dict[str, dict] = {}
    num_path = taxon_dir / NUMERICAL_STATS_FILE
    if num_path.exists():
        for row in pq.read_table(num_path).to_pylist():
            var = row.pop("variable")
            numerical_stats[var] = row

    circular_stats: dict[str, dict] = {}
    circ_path = taxon_dir / CIRCULAR_STATS_FILE
    if circ_path.exists():
        for row in pq.read_table(circ_path).to_pylist():
            var = row.pop("variable")
            circular_stats[var] = row

    nominal_stats: dict[str, dict] = {}
    nominal_classes: dict[str, list] = {}
    nom_path = taxon_dir / NOMINAL_STATS_FILE
    if nom_path.exists():
        for row in pq.read_table(nom_path).to_pylist():
            var, metric, value = row["variable"], row["metric"], row["value"]
            if metric.startswith("class_"):
                class_id = int(metric[6:])
                nominal_classes.setdefault(var, []).append({"class_id": class_id, "fraction": value})
            else:
                nominal_stats.setdefault(var, {})[metric] = value
        for var in nominal_classes:
            nominal_classes[var].sort(key=lambda e: -e["fraction"])

    density_by_var: dict[str, dict] = {}
    den_path = taxon_dir / DENSITY_FILE
    if den_path.exists():
        for row in pq.read_table(den_path).to_pylist():
            var = row.pop("variable")
            density_by_var[var] = row

    all_var_ids = list(dict.fromkeys(list(numerical_stats) + list(circular_stats) + list(nominal_stats)))
    variables = []
    for var_id in all_var_ids:
        layer = layer_index.get(var_id, {})
        entry: dict = {
            "id": var_id,
            "display_name": layer.get("display_name"),
            "units": layer.get("units") or None,
            "value_type": layer.get("value_type"),
            "domain": layer.get("domain") or None,
        }
        if var_id in numerical_stats:
            entry["stats"] = numerical_stats[var_id]
            entry["density"] = density_by_var.get(var_id)
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

def _load_relative_ranks(taxon_dir: Path, variable_id: str) -> list[dict]:
    """Read relative_ranks_positions.parquet and return rank rows for one variable."""
    pos_path = taxon_dir / POSITION_FILE
    if not pos_path.exists():
        return []
    try:
        rows = pq.read_table(pos_path).to_pylist()
    except Exception:
        return []
    result = []
    for row in rows:
        if row.get("variable") != variable_id:
            continue
        position = row.get("position") or 0
        count = row.get("count") or 0
        percentile = round(position / count, 3) if count > 0 else 0.0
        result.append({
            "metric": row.get("metric"),
            "position": position + 1,          # 1-based rank
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
        meta = pq.read_metadata(str(path))
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
    df = collect_taxon_df(taxon)
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
    df = collect_taxon_df(taxon)
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
    taxon_dir = TREE_ROOT / taxon["path"]
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    variable_metadata = {
        "name": layer["display_name"] if layer else variable_id,
        "units": (layer.get("units") or None) if layer else None,
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
            )
            if result is not None:
                if result["type"] == "continuous":
                    stats = result["stats"]
                    return {
                        "species_id": taxon.get("taxon_key"),
                        "variable": variable_id,
                        "variable_metadata": variable_metadata,
                        "observation_count": result["observation_count"],
                        "summary": {
                            "count": stats["count"],
                            "min": stats.get("min"),
                            "mean": stats.get("mean"),
                            "max": stats.get("max"),
                            "stddev": stats.get("std"),
                            "q10": stats.get("10th_percentile"),
                            "q90": stats.get("90th_percentile"),
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
        nom_path = taxon_dir / NOMINAL_STATS_FILE
        if not nom_path.exists():
            raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
        rows = [r for r in pq.read_table(nom_path).to_pylist() if r["variable"] == variable_id]
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
            class_id = int(m[6:])
            fraction = float(r["value"])
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
            "summary": {"count": total_samples, "min": None, "mean": None, "max": None},
            "density_curve": None,
            "categorical_distribution": categorical_distribution,
            "relative_ranks": _load_relative_ranks(taxon_dir, variable_id),
        }

    if value_type == "circular":
        circ_path = taxon_dir / CIRCULAR_STATS_FILE
        if not circ_path.exists():
            raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
        row = next((r for r in pq.read_table(circ_path).to_pylist() if r["variable"] == variable_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
        count = int(row.get("count") or 0)
        summary = {
            "count": count,
            "circular_mean": row.get("circular_mean"),
            "rbar": row.get("rbar"),
            "circular_std": row.get("circular_std"),
        }
        density_curve = None
        den_path = taxon_dir / DENSITY_FILE
        if den_path.exists():
            den_row = next((r for r in pq.read_table(den_path).to_pylist() if r["variable"] == variable_id), None)
            if den_row:
                density_curve = {"points": den_row["points"], "density": den_row["density"]}
        return {
            "species_id": taxon.get("taxon_key"),
            "variable": variable_id,
            "variable_metadata": variable_metadata,
            "observation_count": count,
            "summary": summary,
            "density_curve": density_curve,
            "categorical_distribution": None,
            "relative_ranks": [],
        }

    num_path = taxon_dir / NUMERICAL_STATS_FILE
    if not num_path.exists():
        raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")
    row = next((r for r in pq.read_table(num_path).to_pylist() if r["variable"] == variable_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No stats for {variable_id}")

    count = int(row.get("count") or 0)
    summary = {
        "count": count,
        "min": row.get("min"),
        "mean": row.get("mean"),
        "max": row.get("max"),
        "stddev": row.get("std"),
        "q10": row.get("10th_percentile"),
        "q90": row.get("90th_percentile"),
    }

    density_curve = None
    den_path = taxon_dir / DENSITY_FILE
    if den_path.exists():
        den_row = next((r for r in pq.read_table(den_path).to_pylist() if r["variable"] == variable_id), None)
        if den_row:
            density_curve = {"points": den_row["points"], "density": den_row["density"]}

    return {
        "species_id": taxon.get("taxon_key"),
        "variable": variable_id,
        "variable_metadata": variable_metadata,
        "observation_count": count,
        "summary": summary,
        "density_curve": density_curve,
        "categorical_distribution": None,
        "relative_ranks": _load_relative_ranks(taxon_dir, variable_id),
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
    # When no filters at all, use precomputed phenology counts from process_tree
    use_precomputed_pheno = not has_loc_or_pheno and not has_ts

    extra_cols: list[str] = []
    if filter_col:
        extra_cols.append(filter_col)
    # rcs needed for phenology filter OR for live phenology counting
    if phenology_norm or not use_precomputed_pheno:
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
        if not path.exists():
            return
        # Fast path: parquet footer stats when no row-level filters change the range
        if not has_loc_or_pheno:
            result = _timestamp_range_from_metadata(path)
            if result:
                lo, hi = result
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)
        try:
            schema_names = set(pq.read_schema(path).names)
            cols_to_read = [c for c in occ_columns if c in schema_names]
        except Exception:
            cols_to_read = list(_OCC_COLUMNS)
        table = pq.read_table(path, columns=cols_to_read)
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
        pheno_counts = read_phenology_counts(TREE_ROOT / taxon["path"])
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
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row.get("gid", "")
            if gid:
                result[gid] = {
                    "name": row.get("name", gid),
                    "level": int(row["level"]),
                    "parent_gid": row.get("parent_gid") or None,
                }
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

    if not _LOC_TAXA_PATH.exists():
        return []

    taxon_key = str(taxon["taxon_key"])
    try:
        table = pq.read_table(_LOC_TAXA_PATH, filters=[("taxon_key", "=", taxon_key)])
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


def _read_index_for_slice(
    index_path: Path,
    variable_id: str,
    *,
    value_min: float | None = None,
    value_max: float | None = None,
    circular_wrap: bool = False,
    class_value: float | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Filter occurrence_index.parquet by value range or class and return observations."""
    schema = pq.read_schema(index_path)
    if variable_id not in schema.names:
        return []
    cols = [c for c in ["catalogNumber", "decimalLatitude", "decimalLongitude", variable_id] if c in schema.names]
    df = pq.read_table(index_path, columns=cols).to_pandas()
    if class_value is not None:
        col = pd.to_numeric(df[variable_id], errors="coerce")
        mask = col == float(class_value)
    elif circular_wrap:
        col = pd.to_numeric(df[variable_id], errors="coerce")
        mask = col.between(value_min, 360.0, inclusive="both") | col.between(0.0, value_max, inclusive="both")
    else:
        col = pd.to_numeric(df[variable_id], errors="coerce")
        mask = col.between(value_min, value_max, inclusive="both")
    df = df[mask].dropna(subset=["decimalLatitude", "decimalLongitude"])
    if limit is not None:
        df = df.head(limit)
    return [
        {
            "catalogNumber": str(r["catalogNumber"]),
            "latitude": r["decimalLatitude"],
            "longitude": r["decimalLongitude"],
            "value": (float(r[variable_id]) if pd.notna(r[variable_id]) else None),
        }
        for r in df.to_dict("records")
    ]


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
    if location is not None or phenology_norm is not None or start_ts is not None or end_ts is not None:
        filter_col = _location_filter_col(location) if location is not None else None
        if location is None or filter_col is not None:
            observations = _slice_from_raw_occ(
                taxon, variable_id, filter_col, location,
                min_value, max_value, circular_wrap, limit,
                phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
            )
            return {
                "species_id": taxon.get("taxon_key"),
                "variable": variable_id,
                "range": {"min": min_value, "max": max_value},
                "count": len(observations),
                "observations": observations,
            }
    index_path = TREE_ROOT / taxon["path"] / OCCURRENCE_INDEX_FILE
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Occurrence index not built for this taxon")
    observations = _read_index_for_slice(
        index_path, variable_id,
        value_min=min_value, value_max=max_value, circular_wrap=circular_wrap,
        limit=limit,
    )
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
    if location is not None or phenology_norm is not None or start_ts is not None or end_ts is not None:
        filter_col = _location_filter_col(location) if location is not None else None
        if location is None or filter_col is not None:
            observations = _class_samples_from_raw_occ(
                taxon, variable_id, filter_col, location, float(parsed), limit, phenology=phenology_norm, start_ts=start_ts, end_ts=end_ts,
            )
            return {
                "species_id": taxon.get("taxon_key"),
                "variable": variable_id,
                "class_value": parsed,
                "count": len(observations),
                "observations": observations,
            }
    index_path = TREE_ROOT / taxon["path"] / OCCURRENCE_INDEX_FILE
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Occurrence index not built for this taxon")
    observations = _read_index_for_slice(
        index_path, variable_id, class_value=float(parsed), limit=limit,
    )
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

    if not index_path.exists():
        return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": []}

    column_lengths = rankings._load_column_lengths(index_path)
    try:
        schema = pq.read_schema(index_path)
    except Exception:
        return {"ancestor_taxon_id": resolved["taxon_key"], "rank": norm_rank, "options": []}

    variable_order = {v["id"]: i for i, v in enumerate(tiles.load_layers())}

    options = []
    for col in schema.names:
        if "::" not in col:
            continue
        count = int(column_lengths.get(col, 0) or 0)
        if count <= 0:
            continue
        variable, metric = col.split("::", 1)
        if metric.startswith("class_"):
            continue
        options.append({
            "variable": variable,
            "metric": metric,
            "label": _METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize()),
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
    norm_sort_variable = sort_variable.replace("_", "") if sort_variable else None

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
    )

    serialized: list[dict] = []
    for item in result["results"]:
        taxon = item["taxon"]
        preferred = taxon.get("inat_preferred_common_name") or taxon.get("common_name") or ""
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
            "sort_value": item.get("sort_value"),
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
            "units": unit_system,
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> FileResponse:
    """Accept a CSV, TSV, or Parquet file of observations and return a ZIP archive.

    The archive contains the original observations enriched with all static GIS
    layer values, pre-computed summary statistics, density curves, and a flat
    occurrence index — all in both Parquet and CSV formats.

    Temporal enrichment is not included: historical weather aggregates require
    per-observation timestamps and the full ERA5 archive.
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

    static_layer_ids = {
        layer["id"] for layer in tiles.load_layers()
        if layer.get("filename") and layer.get("window_hours") is None
    }

    df = upload.normalize_coordinate_columns(df)
    df = upload.ensure_catalog_numbers(df)
    df = upload.ensure_observation_names(df)
    df = upload.validate_coordinates(df)
    upload.check_reserved_columns(df, static_layer_ids)

    df = await run_in_threadpool(upload.enrich_with_gis, df)
    archive_path, archive_name, work_dir = await run_in_threadpool(upload.build_archive, df)

    background_tasks.add_task(shutil.rmtree, work_dir, True)
    return FileResponse(path=archive_path, media_type="application/zip", filename=archive_name)
