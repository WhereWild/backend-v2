"""
Download iNat DWC-A and build offline GBIF->iNat ID mapping.

Downloads inaturalist-taxonomy.dwca.zip fresh each run, extracts taxa.csv,
matches scientific names against the GBIF catalog, writes inat_gbif_mapping.csv,
and updates taxon_catalog.pkl with inat_id on each matched entry.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG_DIR = Path("data/taxonomy/catalog")
CATALOG_PATH = CATALOG_DIR / "taxon_catalog.pkl"
MAPPING_PATH = CATALOG_DIR / "inat_gbif_mapping.csv"
CACHE_DIR = Path("data/taxonomy/cache")
INAT_DWCA_CACHE = CACHE_DIR / "inat_dwca.zip"
SYNC_STATE_PATH = Path("data/sync_state.json")

INAT_DWCA_URL = "https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip"
INAT_TAXA_FILENAME = "taxa.csv"
_UA = "wherewild-build-id-maps/1.0"

MAPPING_RANKS = frozenset({"SPECIES", "SUBSPECIES", "VARIETY", "FORM"})
INFRA_RANKS = frozenset({"SUBSPECIES", "VARIETY", "FORM"})
INFRA_MARKERS = frozenset({"var.", "subsp.", "f.", "nothosubsp.", "nothovar."})


def normalize_name(value: str) -> str:
    if not value:
        return ""
    cleaned = value.replace("_", " ").replace("×", "x").lower()
    return " ".join(cleaned.split())


def strip_infra_markers(value: str) -> str:
    tokens = [t for t in value.split() if t not in INFRA_MARKERS]
    return " ".join(tokens)


def load_catalog() -> dict:
    with open(CATALOG_PATH, "rb") as f:
        return pickle.load(f)


def save_catalog(payload: dict) -> None:
    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_gbif_indexes(
    catalog: dict,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]]]:
    exact: dict[tuple[str, str], list[str]] = defaultdict(list)
    stripped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for taxon_key, taxon in catalog.items():
        rank = (taxon.get("rank") or "").strip().upper()
        name = normalize_name(taxon.get("scientific_name") or "")
        if not name or not rank:
            continue
        exact[(rank, name)].append(taxon_key)
        s = strip_infra_markers(name)
        if s:
            stripped[(rank, s)].append(taxon_key)
    return exact, stripped


def download_dwca() -> bytes:
    state = json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}
    cached_etag = state.get("inat_taxonomy", {}).get("etag", "")

    print(f"Checking {INAT_DWCA_URL} ...")
    req = Request(INAT_DWCA_URL, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        remote_etag = r.headers.get("ETag", "")

    if remote_etag and remote_etag == cached_etag and INAT_DWCA_CACHE.exists():
        print("  iNat DWC-A: cache up to date")
        return INAT_DWCA_CACHE.read_bytes()

    print(f"Downloading {INAT_DWCA_URL} ...")
    req = Request(INAT_DWCA_URL, headers={"User-Agent": _UA})
    with urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data) / 1_048_576:.1f} MB")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INAT_DWCA_CACHE.write_bytes(data)
    if remote_etag:
        state.setdefault("inat_taxonomy", {})["etag"] = remote_etag
        SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))

    return data


def extract_taxa_csv(dwca_bytes: bytes) -> io.TextIOWrapper:
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    return io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="")


def build_mapping(catalog: dict, dwca_bytes: bytes) -> None:
    exact_index, stripped_index = build_gbif_indexes(catalog)
    print(f"  GBIF catalog taxa: {len(catalog):,}")

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    # GBIF-side totals: how many of our catalog entries are eligible per rank
    gbif_totals: Counter[str] = Counter()
    for taxon in catalog.values():
        rank = (taxon.get("rank") or "").strip().upper()
        if rank in MAPPING_RANKS:
            gbif_totals[rank] += 1

    matched: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    used_gbif: dict[str, str] = {}

    fieldnames = [
        "gbif_taxon_key", "inat_id", "inat_taxon_url", "rank", "scientific_name", "match_type",
    ]

    with open(MAPPING_PATH, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        reader = csv.DictReader(extract_taxa_csv(dwca_bytes))
        for row in reader:
            rank = (row.get("taxonRank") or "").strip().upper()
            if rank not in MAPPING_RANKS:
                continue

            name = normalize_name(row.get("scientificName") or "")
            if not name:
                continue

            gbif_keys = exact_index.get((rank, name), [])
            match_type = "exact"

            if not gbif_keys and rank in INFRA_RANKS:
                s = strip_infra_markers(name)
                if s:
                    gbif_keys = stripped_index.get((rank, s), [])
                    match_type = "stripped"

            if len(gbif_keys) == 1:
                gbif_key = gbif_keys[0]
                inat_id = str(row.get("id") or "").strip()
                inat_taxon_url = str(row.get("taxonID") or "").strip()
                if not inat_id:
                    continue
                existing = used_gbif.get(gbif_key)
                if existing and existing != inat_id:
                    conflicts[rank] += 1
                    continue
                used_gbif[gbif_key] = inat_id
                matched[rank] += 1
                writer.writerow({
                    "gbif_taxon_key": gbif_key,
                    "inat_id": inat_id,
                    "inat_taxon_url": inat_taxon_url,
                    "rank": rank,
                    "scientific_name": name,
                    "match_type": match_type,
                })

    print(f"\nSaved mapping to {MAPPING_PATH}")
    print("\nMatch summary (GBIF catalog coverage):")
    for rank in sorted(gbif_totals):
        total = gbif_totals[rank]
        rate = matched[rank] / total * 100 if total else 0
        missing = total - matched[rank] - conflicts[rank]
        print(
            f"  {rank:<12} catalog={total:>9,}  matched={matched[rank]:>9,} "
            f"conflicts={conflicts[rank]:>9,}  missing={missing:>9,}  rate={rate:5.1f}%"
        )


def apply_mapping(catalog: dict) -> int:
    updated = 0
    with open(MAPPING_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gbif_key = str(row.get("gbif_taxon_key") or "").strip()
            inat_id = str(row.get("inat_id") or "").strip()
            if not gbif_key or not inat_id:
                continue
            taxon = catalog.get(gbif_key)
            if not taxon:
                continue
            taxon["inat_id"] = inat_id
            inat_taxon_url = str(row.get("inat_taxon_url") or "").strip()
            if inat_taxon_url:
                taxon["inat_taxon_url"] = inat_taxon_url
            updated += 1
    return updated


def main() -> None:
    dwca_bytes = download_dwca()

    payload = load_catalog()
    catalog = payload["catalog"]

    build_mapping(catalog, dwca_bytes)

    updated = apply_mapping(catalog)
    save_catalog(payload)
    print(f"Updated {updated:,} catalog entries with inat_id.")


if __name__ == "__main__":  # pragma: no cover
    main()
