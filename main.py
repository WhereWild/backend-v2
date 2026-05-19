from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from util import citations, taxa, tiles
from util.taxa import format_common_name, normalize_name, taxon_slug

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
def get_taxon(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return {**taxon, **_image_fields(taxon)}


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
