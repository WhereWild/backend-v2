from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json, os

SPECIES_DIR = Path(os.environ.get("SPECIES_DIR", Path(__file__).resolve().parent / "processed" / "species"))
CATALOG = json.loads((SPECIES_DIR / "species_catalog.json").read_text(encoding="utf-8"))
BY_SLUG = {s["slug"]: s for s in CATALOG}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

if (SPECIES_DIR / "images").exists():
    app.mount("/static/species_images", StaticFiles(directory=str(SPECIES_DIR / "images")), name="species_images")

def image_url(request: Request, fname: str):
    base = str(request.base_url).rstrip("/")
    return f"{base}/static/species_images/{fname}" if fname else None

@app.get("/api/species")
def list_species(request: Request, q: str | None = None, limit: int | None = None):
    items = CATALOG
    if q:
        ql = q.lower()
        items = [i for i in items if ql in (i.get("common_name","").lower() + i.get("scientific_name","").lower() + i.get("slug",""))]
    if limit: items = items[:limit]
    return [{**{"image_url": image_url(request, it.get("image_file"))}, **{k: it[k] for k in ("taxon_id","slug","common_name","scientific_name")}} for it in items]

@app.get("/api/species/{slug}")
def get_species(slug: str, request: Request):
    it = BY_SLUG.get(slug)
    if not it: raise HTTPException(404)
    out = dict(it)
    out["image_url"] = image_url(request, it.get("image_file"))
    return out
@app.get("/api/species/by_name")
def get_species_by_name(request: Request, name: str):
    name_lower = name.lower()
    for it in CATALOG:
        if it.get("common_name","").lower() == name_lower:
            return {**{"image_url": image_url(request, it.get("image_file"))}, **it}
    raise HTTPException(404, detail=f"Species with common name {name} not found")