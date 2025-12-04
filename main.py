import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from wherewild.stats_repository import (
    SpeciesStatsNotFoundError,
    VariableNotFoundError,
    get_species_stats,
    get_variable_leaderboard,
)

app = FastAPI(title="WhereWild API", version="0.1.0")


@app.get("/health", summary="Simple liveness probe")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/species/{species_id}/environment/{variable}",
    summary="Retrieve precomputed environmental stats for a species/variable pair",
)
def species_environment_stats(species_id: int, variable: str) -> dict:
    variable_id = variable.strip()
    try:
        return get_species_stats(species_id=species_id, variable_id=variable_id)
    except VariableNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        ) from None
    except SpeciesStatsNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No stats found for species {
                species_id} and variable '{variable_id}'.",
        ) from None


@app.get(
    "/variables/{variable}/leaderboard",
    summary="Retrieve the leaderboard for a GIS variable",
)
def variable_leaderboard(variable: str) -> dict:
    variable_id = variable.strip()
    try:
        return get_variable_leaderboard(variable_id=variable_id)
    except VariableNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        ) from None
    except SpeciesStatsNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No leaderboard found for variable '{variable_id}'.",
        ) from None


logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parent
SPECIES_DIR = Path(os.environ.get(
    "SPECIES_DIR", PROJECT_ROOT / "processed" / "species")).resolve()
CATALOG_PATH = SPECIES_DIR / "species_catalog.json"

if not CATALOG_PATH.exists():
    logging.error("species_catalog.json not found at %s", CATALOG_PATH)
    raise SystemExit(f"Missing species_catalog.json at {CATALOG_PATH}")

CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
if isinstance(CATALOG, dict):
    CATALOG = [CATALOG]


def normalize_taxon_id(value: int | str | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.isdigit():
            return int(candidate)
    return None


def build_taxon_index(entries: list[dict]) -> dict[int, dict]:
    index: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        taxon_id = normalize_taxon_id(entry.get("taxon_id"))
        if taxon_id is None:
            continue
        if taxon_id in index:
            first = index[taxon_id].get("scientific_name") or index[taxon_id].get(
                "common_name") or "unknown"
            duplicate = entry.get("scientific_name") or entry.get(
                "common_name") or "unknown"
            raise ValueError(
                f"Duplicate taxon_id {taxon_id} detected between '{
                    first}' and '{duplicate}'."
            )
        index[taxon_id] = entry
    return index


TAXON_ID_INDEX = build_taxon_index(CATALOG)

logging.info("Running file: %s", Path(__file__).resolve())
logging.info("SPECIES_DIR = %s", SPECIES_DIR)
logging.info("catalog_path = %s", CATALOG_PATH)
logging.info("loaded %d species; sample common_names: %s", len(
    CATALOG), [s.get("common_name") for s in CATALOG[:8]])

app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["GET"], allow_headers=["*"])

if (SPECIES_DIR / "images").exists():
    app.mount("/static/species_images",
              StaticFiles(directory=str(SPECIES_DIR / "images")), name="species_images")


def image_url(request: Request, fname: str):
    if not fname:
        return None
    base = str(request.base_url).rstrip("/")
    filename = fname.replace("images/", "")
    return f"{base}/static/species_images/{filename}"


def serialize_species_brief(species: dict, request: Request) -> dict:
    """Return the subset of species fields used by the list endpoint."""
    fields = ("taxon_id", "slug", "common_name", "scientific_name")
    payload = {field: species.get(field) for field in fields}
    payload["image_url"] = image_url(request, species.get("image_file"))
    return payload


@app.get("/api/species")
def list_species(request: Request, q: str | None = None, limit: int | None = None):
    items = CATALOG
    if q:
        ql = q.lower()
        items = [i for i in items if ql in (i.get("common_name", "").lower(
        ) + i.get("scientific_name", "").lower() + i.get("slug", ""))]
    if limit:
        items = items[:limit]
    response: list[dict] = []
    for species in items:
        response.append(serialize_species_brief(species, request))
    return response


@app.get("/api/species/{taxon_id}")
def get_species(taxon_id: int, request: Request):
    it = TAXON_ID_INDEX.get(taxon_id)
    if not it:
        raise HTTPException(status_code=404, detail=f"Species with taxon_id {
                            taxon_id} not found")
    out = dict(it)
    out["image_url"] = image_url(request, it.get("image_file"))
    return out


@app.get("/_debug/species_info")
def debug_species_info():
    return {
        "SPECIES_DIR": str(SPECIES_DIR),
        "catalog_path": str(CATALOG_PATH),
        "count": len(CATALOG),
        "sample_common_names": [s.get("common_name") for s in CATALOG[:10]]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
