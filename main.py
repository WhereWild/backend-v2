import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from config.config import load_config
from util import citations, taxa, tiles
from util.rankings import POSITION_FILE
from util.stats import NOMINAL_STATS_FILE, NUMERICAL_DENSITY_FILE, NUMERICAL_STATS_FILE, OCCURRENCE_INDEX_FILE, TREE_ROOT
from util.taxa import format_common_name, iter_descendants, normalize_name, taxon_slug

_CONFIG = load_config("global")
_LEGEND_DIR = Path("config/gis/legends")
_OCC_FILE = "occurrence.parquet"
_OCC_COLUMNS = ["catalogNumber", "decimalLatitude", "decimalLongitude", "obscured", "coordinateUncertaintyInMeters"]


@lru_cache(maxsize=32)
def _load_legend(layer_id: str) -> list:
    path = _LEGEND_DIR / f"{layer_id}_legend.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("classes", [])


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


@app.get("/api/variables/{variable_id}/tiles/{z}/{x}/{y}.png")
async def variable_tile_compat(variable_id: str, z: int, x: int, y: int, tile_size: int = Query(256, ge=32, le=1024)):
    """Compatibility shim for old frontend URL pattern (/api/variables/bio_1/ → bio1)."""
    layer_id = variable_id.replace("_", "")
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
    den_path = taxon_dir / NUMERICAL_DENSITY_FILE
    if den_path.exists():
        for row in pq.read_table(den_path).to_pylist():
            var = row.pop("variable")
            density_by_var[var] = row

    all_var_ids = list(dict.fromkeys(list(numerical_stats) + list(nominal_stats)))
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


@app.get("/species/{taxon_id}/environment/{variable_id}")
def get_species_environment(taxon_id: str, variable_id: str, unit_system: str | None = None):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    variable_id = variable_id.replace("_", "")  # bio_1 → bio1
    taxon_dir = TREE_ROOT / taxon["path"]
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    variable_metadata = {
        "name": layer["display_name"] if layer else variable_id,
        "units": (layer.get("units") or None) if layer else None,
        "value_type": layer.get("value_type") if layer else None,
        "domain": (layer.get("domain") or None) if layer else None,
    }
    value_type = layer.get("value_type") if layer else None

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
                "description": info.get("description"),
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
    den_path = taxon_dir / NUMERICAL_DENSITY_FILE
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
def get_species_occurrences(taxon_id: str, location: str | None = None):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")

    is_leaf = taxon["rank"] in _CONFIG.leaf_rank_set
    collected: list[dict] = []

    seen: set[str] = set()

    def _read_occ(path: Path) -> None:
        if not path.exists():
            return
        df = _filter_occ_df(pq.read_table(path, columns=_OCC_COLUMNS).to_pandas())
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

    return {"occurrences": collected}


@app.get("/species/{taxon_id}/locations")
def get_species_locations(taxon_id: str, level: int | None = None, limit: int = 500):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return []


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
):
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise HTTPException(status_code=400, detail="min and max must be finite numbers")
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    variable_id = variable_id.replace("_", "")
    layer = next((lyr for lyr in tiles.load_layers() if lyr["id"] == variable_id), None)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_id}' not found")
    if layer.get("value_type") == "nominal":
        raise HTTPException(status_code=400, detail="Categorical variables must use the class samples endpoint")
    circular_wrap = variable_id == "aspect_deg" and max_value < min_value
    if max_value < min_value and not circular_wrap:
        min_value, max_value = max_value, min_value
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
):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    variable_id = variable_id.replace("_", "")
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


@app.get("/api/taxa/query")
def query_taxa(
    q: str | None = Query(None, min_length=1),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    min_samples: int = Query(0, ge=0),
    unit_system: str | None = Query(None),
):
    normalized_query = normalize_name(q or "")

    if not normalized_query:
        return {
            "query": None,
            "scope": {"within_taxon": None, "descendant_rank": None, "location": None,
                      "min_samples": min_samples, "include_species_like": False},
            "sort": {"variable": None, "metric": None, "order": "asc", "units": None},
            "total": 0,
            "matched_total": 0,
            "eligible_total": 0,
            "empty_reason": "no_query",
            "limit": limit,
            "offset": offset,
            "results": [],
        }

    matches = taxa.search_taxa_by_name(normalized_query, limit=limit + offset)
    page = matches[offset:]
    matched_total = len(matches)

    results = []
    for taxon, score, matched_name in page:
        preferred = taxon.get("inat_preferred_common_name") or taxon.get("common_name") or ""
        sci_normalized = normalize_name(taxon.get("scientific_name", ""))
        display_name = preferred if matched_name == sci_normalized else (matched_name or preferred)
        results.append({
            "taxon_id": taxon["taxon_key"],
            "scientific_name": taxon.get("scientific_name", "").replace("_", " "),
            "common_name": format_common_name(display_name) or None,
            "common_names": None,
            "rank": taxon.get("rank"),
            "slug": taxon_slug(taxon.get("scientific_name")),
            "description": None,
            **_image_fields(taxon),
            "match_score": score,
            "sample_count": None,
            "sort_value": None,
            "sort_variable": None,
            "sort_metric": None,
            "position": None,
            "percentile": None,
        })

    return {
        "query": normalized_query,
        "scope": {"within_taxon": None, "descendant_rank": None, "location": None,
                  "min_samples": min_samples, "include_species_like": False},
        "sort": {"variable": None, "metric": None, "order": "asc", "units": None},
        "total": len(results),
        "matched_total": matched_total,
        "eligible_total": matched_total,
        "empty_reason": None if results else "no_text_matches",
        "limit": limit,
        "offset": offset,
        "results": results,
    }
