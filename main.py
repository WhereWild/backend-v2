from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from util import taxa
from util.taxa import normalize_name, taxon_slug

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/api/taxon/{taxon_id}")
def get_taxon(taxon_id: str):
    taxon = taxa.get_taxon_by_id(taxon_id) or taxa.get_taxon_by_slug(taxon_id)
    if taxon is None:
        raise HTTPException(status_code=404, detail="Taxon not found")
    return taxon


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
    for taxon, score in page:
        results.append({
            "taxon_id": taxon["taxon_key"],
            "scientific_name": taxon.get("scientific_name", "").replace("_", " "),
            "common_name": taxon.get("common_name") or None,
            "common_names": None,
            "rank": taxon.get("rank"),
            "slug": taxon_slug(taxon.get("scientific_name")),
            "description": None,
            "image_url": None,
            "image_license": None,
            "image_creator": None,
            "image_rights_holder": None,
            "image_references": None,
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
