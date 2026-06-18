# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build the taxonomy catalog, ID maps, and enriched name/image data.

Runs in three sequential phases after sync_gbif has produced species_list.csv:

Phase 1 — Catalog construction  (formerly build_tree.py)
  - Parse species_list.csv, build catalog + name index, write taxon_catalog.pkl
  - Create per-taxon directory tree under data/taxonomy/tree/

Phase 2 — ID mapping  (formerly build_id_maps.py)
  - Download iNat DWC-A zip (ETag-cached), extract taxa.csv
  - Match scientific names against GBIF catalog, write inat_gbif_mapping.csv
  - Apply inat_id to taxon_catalog.pkl

Phase 3 — Name & image enrichment  (formerly polish_tree.py)
  - iNat DWC-A VernacularNames-*.csv  (matched via inat_id, same zip as Phase 2)
  - GBIF backbone VernacularName.tsv  (matched via GBIF taxon key, range requests)
  - iNat API /v1/taxa                 (preferred_common_name + default_photo)
  - GBIF occurrence DWCA multimedia   (backup images)

File sources are ETag-cached in data/taxonomy/cache/ via data/sync_state.json.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from remotezip import RemoteZip

from config.config import load_config

CONFIG = load_config("global")

CATALOG_DIR = Path("data/taxonomy/catalog")
CATALOG_PATH = CATALOG_DIR / "taxon_catalog.pkl"
MAPPING_PATH = CATALOG_DIR / "inat_gbif_mapping.csv"
TREE_ROOT = Path("data/taxonomy/tree")
CACHE_DIR = Path("data/taxonomy/cache")
INAT_DWCA_CACHE = CACHE_DIR / "inat_dwca.zip"
BACKBONE_VERNACULAR_CACHE = CACHE_DIR / "gbif_vernacular.tsv"
OCCURRENCE_PATH = Path("data/occurrences/occurrence.txt")
MULTIMEDIA_PATH = Path("data/occurrences/multimedia.txt")
SYNC_STATE_PATH = Path("data/sync_state.json")

INAT_DWCA_URL = "https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip"
BACKBONE_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
BACKBONE_VERNACULAR_FILENAME = "VernacularName.tsv"
INAT_TAXA_FILENAME = "taxa.csv"

HYBRID_MARKER = "×"
INFRASPECIFIC_MARKERS = ("var.", "subsp.", "f.", "nothosubsp.", "nothovar.")
TAXONOMY_LEVELS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
TSV_DELIMITER = "\t"

MAPPING_RANKS = frozenset({"SPECIES", "SUBSPECIES", "VARIETY", "FORM"})
NAME_MATCH_RANKS = MAPPING_RANKS | frozenset({"GENUS", "FAMILY", "ORDER", "CLASS", "PHYLUM", "KINGDOM"})
INFRA_RANKS = frozenset({"SUBSPECIES", "VARIETY", "FORM"})
INFRA_MARKERS = frozenset({"var.", "subsp.", "f.", "nothosubsp.", "nothovar."})

_UA = "wherewild-build-tree/1.0"

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    return json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}


def _save_state(state: dict) -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Phase 1: Catalog construction
# ---------------------------------------------------------------------------

def _normalize_index_key(value: str) -> str:
    """Lowercase + collapse whitespace, used for name search index keys."""
    return " ".join(value.replace("_", " ").lower().split())


def normalize_name(value: str) -> str:
    """Normalize a scientific name for matching (also maps × to x)."""
    if not value:
        return ""
    cleaned = value.replace("_", " ").replace("×", "x").lower()
    return " ".join(cleaned.split())


def clean_name(name: str, rank: str) -> str:
    if not name:
        return ""
    parts = name.split()

    if rank == CONFIG.species_rank:
        if len(parts) >= 3 and parts[1] == HYBRID_MARKER:
            return "_".join(parts[:3])
        return "_".join(parts[:2])

    if rank in CONFIG.subspecies_equivalents:
        if len(parts) >= 3 and parts[2].lower() in INFRASPECIFIC_MARKERS:
            return "_".join(parts[:4])
        return "_".join(parts[:3])

    return name.replace(" ", "_")


def _collect_synonymy(csv_path: Path) -> tuple[
    dict[str, set[str]],   # genus_key → {old_genus_names}
    dict[str, list[str]],  # accepted_entry_key → [old_taxon_keys]
    dict[str, list[str]],  # accepted_entry_key → [old_sci_names (including genus-renamed aliases)]
]:
    """First CSV pass: collect all synonym relationships.

    Separating collection from catalog construction means ACCEPTED rows that
    overwrite catalog entries can never wipe synonym data collected from earlier
    SYNONYM rows — the two concerns no longer race.

    Also synthesises genus-renamed aliases: since we learn "Escobaria → Pelecyphora"
    from the genus-merge rows, we can additionally index "Pelecyphora chlorantha"
    as a search alias for entries whose synonym name is "Escobaria chlorantha".
    """
    genus_synonym_map: dict[str, set[str]] = defaultdict(set)
    old_to_new_genus: dict[str, str] = {}   # "Escobaria" → "Pelecyphora"
    pending_syn_keys: dict[str, list[str]] = defaultdict(list)
    pending_syn_names_raw: dict[str, list[str]] = defaultdict(list)

    _filters_ok = _csv_row_filters  # local alias for the shared filter logic

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=TSV_DELIMITER)
        for row in reader:
            if not _filters_ok(row):
                continue

            row_taxon_key = str(row["taxonKey"])

            if row["taxonRank"] == CONFIG.species_rank:
                species_key = str(row.get("speciesKey") or "")
                accepted_taxon_key = str(row.get("acceptedTaxonKey") or "")
                sci_parts = row["scientificName"].split()
                accepted_genus = row["genus"]

                if sci_parts and sci_parts[0] != accepted_genus and row.get("genusKey"):
                    old_genus = sci_parts[0]
                    genus_synonym_map[row["genusKey"]].add(old_genus)
                    old_to_new_genus[old_genus] = accepted_genus

                if row_taxon_key != species_key:
                    syn_name = clean_name(row["scientificName"], row["taxonRank"])
                    # Primary: parent species entry
                    pending_syn_keys[species_key].append(row_taxon_key)
                    if syn_name:
                        pending_syn_names_raw[species_key].append(syn_name)
                    # Also the subspecies when acceptedTaxonKey ≠ speciesKey
                    # (e.g. Escobaria chlorantha → Pelecyphora dasyacantha subsp. dasyacantha)
                    if accepted_taxon_key and accepted_taxon_key != species_key:
                        pending_syn_keys[accepted_taxon_key].append(row_taxon_key)
                        if syn_name:
                            pending_syn_names_raw[accepted_taxon_key].append(syn_name)

            elif row["taxonRank"] in CONFIG.subspecies_equivalents:
                sci_parts = row["scientificName"].split()
                accepted_genus = row["genus"]
                if sci_parts and sci_parts[0] != accepted_genus:
                    old_name = clean_name(row["scientificName"], row["taxonRank"])
                    if old_name:
                        pending_syn_names_raw[row_taxon_key].append(old_name)

    # Synthesise genus-renamed aliases using the genus map we just built.
    # "Escobaria_chlorantha" → also add "Pelecyphora_chlorantha" so searching
    # the new-genus form of an old name still finds the accepted taxon.
    pending_syn_names: dict[str, list[str]] = defaultdict(list)
    for ekey, raw_names in pending_syn_names_raw.items():
        seen: set[str] = set()
        for name in raw_names:
            if name not in seen:
                pending_syn_names[ekey].append(name)
                seen.add(name)
            parts = name.split("_")
            if parts and parts[0] in old_to_new_genus:
                renamed = "_".join([old_to_new_genus[parts[0]]] + parts[1:])
                if renamed not in seen:
                    pending_syn_names[ekey].append(renamed)
                    seen.add(renamed)

    return dict(genus_synonym_map), dict(pending_syn_keys), dict(pending_syn_names)


def _csv_row_filters(row: dict) -> bool:
    """Shared row-filter predicate used by both CSV passes in build_catalog."""
    if row["taxonRank"] not in CONFIG.leaf_rank_set:
        return False
    if row["taxonRank"] == CONFIG.species_rank and len(row["scientificName"].split()) < 2:
        return False
    if not row.get("genus") or not row.get("genusKey"):
        return False
    if row["taxonRank"] in CONFIG.subspecies_equivalents:
        if not row.get("species") or not row.get("speciesKey"):
            return False
    return True


def build_catalog(csv_path: Path, write_dirs: bool = False) -> tuple[dict, dict]:
    """Parse species list CSV and return (catalog, combined_name_index).

    Uses two passes over the CSV: the first collects all synonym relationships
    so that row ordering cannot cause ACCEPTED rows to silently discard synonym
    data; the second builds the catalog entries using that complete picture.
    """
    genus_synonym_map, pending_syn_keys, pending_syn_names = _collect_synonymy(csv_path)

    catalog: dict = {}
    scientific_index: dict = {}
    common_index: defaultdict = defaultdict(set)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=TSV_DELIMITER)
        for row in reader:
            if not _csv_row_filters(row):
                continue

            path_parts = []
            rel_path = ""
            for level in TAXONOMY_LEVELS:
                name = row.get(level)
                key = row.get(level + "Key")
                if name and key:
                    cleaned = clean_name(name, level.upper())
                    path_parts.append(f"{cleaned}_{key}")
                    rel_path = "/".join(path_parts)
                    if key not in catalog:
                        catalog[str(key)] = {
                            "taxon_key": str(key),
                            "path": rel_path,
                            "scientific_name": cleaned,
                            "common_name": "",
                            "rank": level.upper(),
                        }

            cleaned_name = clean_name(row["acceptedScientificName"], row["taxonRank"])
            taxon_key_to_write = row["taxonKey"]

            if row["taxonRank"] in CONFIG.subspecies_equivalents:
                sci_parts = row["scientificName"].split()
                accepted_genus = row["genus"]
                if sci_parts and sci_parts[0] != accepted_genus:
                    # Rename variety to use accepted genus in both name and path
                    sci_parts[0] = accepted_genus
                    cleaned_name = clean_name(" ".join(sci_parts), row["taxonRank"])
                else:
                    cleaned_name = clean_name(row["scientificName"], row["taxonRank"])
                path_parts.append(f"{cleaned_name}_{row['taxonKey']}")
                rel_path = "/".join(path_parts)
            elif row["taxonRank"] == CONFIG.species_rank:
                taxon_key_to_write = row["speciesKey"]

            if write_dirs:
                (TREE_ROOT / Path(*path_parts)).mkdir(parents=True, exist_ok=True)

            common_name = row.get("commonName", "")
            entry_key = str(taxon_key_to_write)
            catalog[entry_key] = {
                "taxon_key": entry_key,
                "path": rel_path,
                "scientific_name": cleaned_name,
                "common_name": common_name,
                "rank": row["taxonRank"],
            }

            scientific_name_key = _normalize_index_key(cleaned_name)
            if scientific_name_key:
                scientific_index[scientific_name_key] = taxon_key_to_write

            common_name_key = _normalize_index_key(common_name)
            if common_name_key:
                common_index[common_name_key].add(taxon_key_to_write)

    # Apply all synonym data collected in pass 1.
    for ekey, keys in pending_syn_keys.items():
        entry = catalog.get(ekey)
        if not entry:
            continue
        existing_keys: list = entry.setdefault("gbif_synonym_keys", [])
        for k in keys:
            if k not in existing_keys:
                existing_keys.append(k)
    for ekey, names in pending_syn_names.items():
        entry = catalog.get(ekey)
        if not entry:
            continue
        existing_names: list = entry.setdefault("gbif_synonym_names", [])
        for n in names:
            if n not in existing_names:
                existing_names.append(n)

    # Apply genus synonymy to genus catalog entries.
    for genus_key, old_names in genus_synonym_map.items():
        genus_entry = catalog.get(str(genus_key))
        if genus_entry:
            existing: set[str] = set(genus_entry.get("genus_synonym_names") or [])
            genus_entry["genus_synonym_names"] = sorted(existing | old_names)

    # Propagate species synonym names down to infra-specific children so that
    # searching an old species name (e.g. "Escobaria chlorantha") also surfaces
    # all subspecies/varieties under the accepted species.
    path_to_key = {t["path"]: k for k, t in catalog.items()}
    for key, taxon in catalog.items():
        if (taxon.get("rank") or "").upper() not in INFRA_RANKS:
            continue
        path = taxon.get("path") or ""
        if "/" not in path:
            continue
        parent_key = path_to_key.get(path.rsplit("/", 1)[0])
        if not parent_key:
            continue
        parent_syn_names = catalog[parent_key].get("gbif_synonym_names") or []
        if not parent_syn_names:
            continue
        existing_names: list = taxon.setdefault("gbif_synonym_names", [])
        for syn_name in parent_syn_names:
            if syn_name not in existing_names:
                existing_names.append(syn_name)

    common_index_sorted = {k: sorted(v) for k, v in common_index.items()}
    combined_index: dict = {k: {v} for k, v in scientific_index.items()}
    for name, keys in common_index_sorted.items():
        combined_index.setdefault(name, set()).update(keys)
    combined_index = {k: sorted(v) for k, v in combined_index.items()}

    return catalog, combined_index


# ---------------------------------------------------------------------------
# iNat DWC-A download (shared by Phases 2 and 3)
# ---------------------------------------------------------------------------

def fetch_inat_dwca() -> bytes:
    state = _load_state()
    cached_etag = state.get("inat_taxonomy", {}).get("etag", "")

    print(f"Checking {INAT_DWCA_URL} ...")
    req = Request(INAT_DWCA_URL, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        remote_etag = r.headers.get("ETag", "")

    if remote_etag and remote_etag == cached_etag and INAT_DWCA_CACHE.exists():
        print("  iNat DWC-A: cache up to date")
        return INAT_DWCA_CACHE.read_bytes()

    print(f"Downloading {INAT_DWCA_URL} ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={INAT_DWCA_CACHE.parent}",
            f"--out={INAT_DWCA_CACHE.name}",
            INAT_DWCA_URL,
        ],
        check=True,
    )
    data = INAT_DWCA_CACHE.read_bytes()
    print(f"  Downloaded {len(data) / 1_048_576:.1f} MB")

    if remote_etag:
        state.setdefault("inat_taxonomy", {})["etag"] = remote_etag
        _save_state(state)

    return data


# ---------------------------------------------------------------------------
# Phase 2: ID mapping (DWC-A taxa.csv → inat_id on each catalog entry)
# ---------------------------------------------------------------------------

def strip_infra_markers(value: str) -> str:
    tokens = [t for t in value.split() if t not in INFRA_MARKERS]
    return " ".join(tokens)


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


def extract_taxa_csv(dwca_bytes: bytes) -> io.TextIOWrapper:
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    return io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="")


def build_mapping(catalog: dict, dwca_bytes: bytes) -> None:
    exact_index, stripped_index = build_gbif_indexes(catalog)
    print(f"  GBIF catalog taxa: {len(catalog):,}")

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

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
            if rank not in NAME_MATCH_RANKS:
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


def infer_species_inat_ids(catalog: dict, dwca_bytes: bytes) -> int:
    """Infer missing inat_id for species whose children all agree on an iNat parent.

    Handles genus-synonym mismatches (e.g. GBIF Pelecyphora vivipara vs iNat
    Escobaria vivipara) where exact name matching in build_mapping fails but the
    child varieties/subspecies were already matched and share a common iNat parent.
    """
    # Build iNat child -> parent ID map from DWC-A parentNameUsageID
    inat_parent: dict[str, str] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    with io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            child_id = (row.get("id") or "").strip()
            parent_url = (row.get("parentNameUsageID") or "").strip()
            if child_id and parent_url:
                inat_parent[child_id] = parent_url.rsplit("/", 1)[-1]

    # Build GBIF parent path -> list of child catalog keys
    path_to_key = {taxon["path"]: key for key, taxon in catalog.items()}
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    for key, taxon in catalog.items():
        path = taxon["path"]
        if "/" in path:
            parent_path = path.rsplit("/", 1)[0]
            parent_key = path_to_key.get(parent_path)
            if parent_key:
                parent_to_children[parent_key].append(key)

    updated = 0
    for parent_key, child_keys in parent_to_children.items():
        parent_entry = catalog.get(parent_key)
        if not parent_entry or _clean(parent_entry.get("inat_id")):
            continue  # Already has inat_id or doesn't exist

        inferred: set[str] = set()
        for child_key in child_keys:
            child = catalog.get(child_key)
            if not child:
                continue
            child_inat_id = _clean(child.get("inat_id"))
            if not child_inat_id:
                continue
            parent_inat_id = inat_parent.get(child_inat_id)
            if parent_inat_id:
                inferred.add(parent_inat_id)

        if len(inferred) == 1:
            parent_entry["inat_id"] = inferred.pop()
            parent_entry["inat_taxon_url"] = f"https://www.inaturalist.org/taxa/{parent_entry['inat_id']}"
            updated += 1

    return updated


# ---------------------------------------------------------------------------
# Phase 2b: Genus synonym iNat ID resolution
# ---------------------------------------------------------------------------

def resolve_genus_synonym_ids(catalog: dict, dwca_bytes: bytes) -> int:
    """For genus nodes with genus_synonym_names, find iNat IDs for synonym genera.

    Stored as inat_synonym_ids (not inat_id) so the primary iNat page is always
    preferred for names/images, with synonym IDs used only to pull additional
    vernacular names and as fallback preferred data.
    """
    inat_genus_ids: dict[str, str] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    with io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("taxonRank") or "").strip().upper() != "GENUS":
                continue
            name = normalize_name(row.get("scientificName") or "")
            inat_id = (row.get("id") or "").strip()
            if name and inat_id:
                inat_genus_ids[name] = inat_id

    updated = 0
    for taxon_key, taxon in catalog.items():
        syn_names = taxon.get("genus_synonym_names") or []
        if not syn_names:
            continue
        primary_inat_id = _clean(taxon.get("inat_id"))
        existing_syn_ids: set[str] = set(taxon.get("inat_synonym_ids") or [])
        for syn_name in syn_names:
            syn_inat_id = inat_genus_ids.get(normalize_name(syn_name))
            if syn_inat_id and syn_inat_id != primary_inat_id and syn_inat_id not in existing_syn_ids:
                existing_syn_ids.add(syn_inat_id)
                updated += 1
        if existing_syn_ids:
            taxon["inat_synonym_ids"] = sorted(existing_syn_ids)
    return updated


# ---------------------------------------------------------------------------
# Phase 2b-2: Species synonym iNat ID resolution
# ---------------------------------------------------------------------------

def resolve_species_synonym_ids(catalog: dict, dwca_bytes: bytes) -> int:
    """For species with gbif_synonym_names and no inat_id, try synonym name matching.

    Unlike genus synonyms (stored as inat_synonym_ids), species synonyms get
    inat_id directly — the iNat page under the old name IS the taxon's page
    (e.g. iNat still uses Escobaria vivipara while GBIF says Pelecyphora vivipara).
    """
    inat_species_ids: dict[str, str] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    with io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("taxonRank") or "").strip().upper() != "SPECIES":
                continue
            name = normalize_name(row.get("scientificName") or "")
            inat_id = (row.get("id") or "").strip()
            if name and inat_id:
                inat_species_ids[name] = inat_id

    updated = 0
    for taxon_key, taxon in catalog.items():
        if _clean(taxon.get("inat_id")):
            continue
        if (taxon.get("rank") or "").upper() != CONFIG.species_rank:
            continue
        for syn_name in (taxon.get("gbif_synonym_names") or []):
            syn_inat_id = inat_species_ids.get(normalize_name(syn_name.replace("_", " ")))
            if syn_inat_id:
                taxon["inat_id"] = syn_inat_id
                updated += 1
                break
    return updated


# ---------------------------------------------------------------------------
# Phase 2b-3: Infra-specific synonym iNat ID resolution
# ---------------------------------------------------------------------------

def resolve_subspecies_synonym_ids(catalog: dict, dwca_bytes: bytes) -> int:
    """For varieties/subspecies with gbif_synonym_names and no inat_id, try synonym matching.

    Genus-renamed infra-specific taxa (e.g. Pelecyphora vivipara var. neomexicana,
    renamed from Escobaria vivipara var. neomexicana) can't be found by build_mapping
    because iNat still uses the old genus name.  gbif_synonym_names always lists the
    old-genus name first (before the synthesised new-genus alias), so the first hit
    will be the one iNat actually knows about.
    """
    inat_infra_ids: dict[str, str] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    with io.TextIOWrapper(zf.open(INAT_TAXA_FILENAME), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rank = (row.get("taxonRank") or "").strip().upper()
            if rank not in INFRA_RANKS:
                continue
            name = normalize_name(row.get("scientificName") or "")
            inat_id = (row.get("id") or "").strip()
            if name and inat_id:
                inat_infra_ids[name] = inat_id

    updated = 0
    for taxon_key, taxon in catalog.items():
        if _clean(taxon.get("inat_id")):
            continue
        if (taxon.get("rank") or "").upper() not in INFRA_RANKS:
            continue
        for syn_name in (taxon.get("gbif_synonym_names") or []):
            syn_inat_id = inat_infra_ids.get(normalize_name(syn_name.replace("_", " ")))
            if syn_inat_id:
                taxon["inat_id"] = syn_inat_id
                updated += 1
                break
    return updated


# ---------------------------------------------------------------------------
# Phase 2c: Observation-based iNat ID mapping
# ---------------------------------------------------------------------------

def _extract_observation_id(values: list) -> str | None:
    """Return first valid iNat observation ID from a list of catalogNumber values."""
    import re as _re
    for value in values:
        if value is None:
            continue
        if isinstance(value, int) and value >= 0:
            return str(value)
        if isinstance(value, float) and value.is_integer() and value >= 0:
            return str(int(value))
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            return text
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        matches = _re.findall(r"\d+", text)
        if len(matches) == 1:
            return matches[0]
    return None


def _resolve_obs_inat_ids(catalog: dict, obs_by_taxon: dict[str, str]) -> int:
    """Given taxon_key → obs_id, batch-query iNat observations API and apply inat_ids."""
    if not obs_by_taxon:
        return 0
    obs_to_taxon = {obs_id: taxon_key for taxon_key, obs_id in obs_by_taxon.items()}
    obs_ids = list(obs_to_taxon.keys())
    updated = 0
    errors = 0
    for i in range(0, len(obs_ids), INAT_BATCH_SIZE):
        batch = obs_ids[i: i + INAT_BATCH_SIZE]
        try:
            params = {"id": ",".join(batch), "per_page": str(len(batch))}
            url = f"{INAT_OBS_ENDPOINT}?{urlencode(params)}"
            req = Request(url, headers={"User-Agent": _UA})
            with urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode("utf-8"))
            results = payload.get("results") or []
        except Exception as exc:
            print(f"  Batch error: {exc}", flush=True)
            errors += 1
            time.sleep(1.0 / INAT_RATE_LIMIT)
            continue
        for obs in results:
            obs_id = str(obs.get("id") or "").strip()
            inat_taxon = obs.get("taxon") or {}
            inat_id = str(inat_taxon.get("id") or "").strip()
            if not obs_id or not inat_id:
                continue
            gbif_key = obs_to_taxon.get(obs_id)
            entry = catalog.get(gbif_key) if gbif_key else None
            if not entry:
                continue
            entry["inat_id"] = inat_id
            entry["inat_taxon_url"] = f"https://www.inaturalist.org/taxa/{inat_id}"
            updated += 1
        time.sleep(1.0 / INAT_RATE_LIMIT)
    print(f"  Observation mapping: {updated:,} resolved, {errors:,} errors", flush=True)
    return updated


# ---------------------------------------------------------------------------
# GBIF backbone VernacularName.tsv via remote ZIP range requests
# ---------------------------------------------------------------------------

def fetch_backbone_vernacular() -> bytes:
    state = _load_state()
    cached_etag = state.get("gbif_backbone", {}).get("etag", "")

    print(f"Checking {BACKBONE_URL} ...")
    req = Request(BACKBONE_URL, method="HEAD", headers={"User-Agent": _UA})
    with urlopen(req, timeout=30) as r:
        remote_etag = r.headers.get("ETag", "")

    if remote_etag and remote_etag == cached_etag and BACKBONE_VERNACULAR_CACHE.exists():
        print("  GBIF backbone VernacularName.tsv: cache up to date")
        return BACKBONE_VERNACULAR_CACHE.read_bytes()

    print(f"Fetching {BACKBONE_VERNACULAR_FILENAME} from GBIF backbone...")
    with RemoteZip(BACKBONE_URL) as rz:
        data = rz.read(BACKBONE_VERNACULAR_FILENAME)
    print(f"  {len(data) / 1_048_576:.1f} MB")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_VERNACULAR_CACHE.write_bytes(data)
    if remote_etag:
        state.setdefault("gbif_backbone", {})["etag"] = remote_etag
        _save_state(state)

    return data


# ---------------------------------------------------------------------------
# Phase 3: Name loading
# ---------------------------------------------------------------------------

def load_inat_vernacular(dwca_bytes: bytes) -> dict[str, list[str]]:
    """Return inat_id -> all English vernacular names found."""
    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    zf = zipfile.ZipFile(io.BytesIO(dwca_bytes))
    for entry in sorted(zf.namelist()):
        if not (entry.startswith("VernacularNames") and entry.endswith(".csv")):
            continue
        with io.TextIOWrapper(zf.open(entry), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                taxon_id = (row.get("id") or "").strip()
                name = (row.get("vernacularName") or "").strip()
                lang = (row.get("language") or "").strip().lower()
                if not taxon_id or not name or lang not in ("en", "eng"):
                    continue
                if name not in seen.setdefault(taxon_id, set()):
                    result.setdefault(taxon_id, []).append(name)
                    seen[taxon_id].add(name)
    return result


def load_gbif_vernacular(tsv_bytes: bytes) -> dict[str, list[str]]:
    """Return gbif_taxon_key -> all English vernacular names (preferred names first)."""
    preferred: dict[str, list[str]] = {}
    others: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    reader = csv.DictReader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t")
    for row in reader:
        taxon_id = (row.get("taxonID") or "").strip()
        name = (row.get("vernacularName") or "").strip()
        lang = (row.get("language") or "").strip().lower()
        is_preferred = (row.get("isPreferredName") or "").strip() in ("1", "true")
        if not taxon_id or not name or lang not in ("en", "eng"):
            continue
        if "/" in taxon_id:
            taxon_id = taxon_id.rsplit("/", 1)[-1]
        if name in seen.setdefault(taxon_id, set()):
            continue
        seen[taxon_id].add(name)
        if is_preferred:
            preferred.setdefault(taxon_id, []).append(name)
        else:
            others.setdefault(taxon_id, []).append(name)
    result = {**others}
    for tid, names in preferred.items():
        result[tid] = names + result.get(tid, [])
    return result


# ---------------------------------------------------------------------------
# Catalog update
# ---------------------------------------------------------------------------

def apply_names(
    catalog: dict,
    inat_map: dict[str, list[str]],
    gbif_map: dict[str, list[str]],
) -> int:
    updated = 0
    for taxon_key, taxon in catalog.items():
        inat_id = str(taxon.get("inat_id") or "").strip()
        inat_names = (inat_map.get(inat_id) if inat_id else None) or []
        # Also pull vernacular names from synonym iNat IDs (e.g. Escobaria names
        # for a genus node whose primary iNat page is Pelecyphora).
        for syn_inat_id in (taxon.get("inat_synonym_ids") or []):
            for name in (inat_map.get(syn_inat_id) or []):
                if name not in inat_names:
                    inat_names = list(inat_names) + [name]
        # Look up GBIF vernacular under the primary key AND any synonym keys (e.g. a
        # reclassified species still has its old taxon key in the backbone TSV).
        gbif_keys = [taxon_key] + list(taxon.get("gbif_synonym_keys") or [])
        seen_gbif: set[str] = set()
        gbif_names: list[str] = []
        for gk in gbif_keys:
            for name in (gbif_map.get(gk) or []):
                if name not in seen_gbif:
                    gbif_names.append(name)
                    seen_gbif.add(name)
        seen = set(inat_names)
        merged = list(inat_names) + [n for n in gbif_names if n not in seen]
        if merged:
            taxon["common_name"] = merged[0]
            taxon["vernacular_names"] = merged
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# iNat preferred names and images
# ---------------------------------------------------------------------------

INAT_TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa"
INAT_OBS_ENDPOINT = "https://api.inaturalist.org/v1/observations"
INAT_PHOTO_BASE_URL = "https://www.inaturalist.org/photos"
INAT_BATCH_SIZE = 200
INAT_RATE_LIMIT = 1.0  # requests per second
OCCURRENCE_PARQUET_FILENAME = "occurrence.parquet"


def fetch_taxa_batch(ids: list[str], timeout: int = 30) -> list[dict]:
    params = {"id": ",".join(ids), "locale": "en", "per_page": str(len(ids))}
    url = f"{INAT_TAXA_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    results = payload.get("results") or []
    return results if isinstance(results, list) else []


def _clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null"} else text


def extract_preferred_image_metadata(taxon_payload: dict) -> dict[str, str]:
    default_photo = taxon_payload.get("default_photo")
    if not isinstance(default_photo, dict):
        return {}
    image_url = ""
    for field in ("original_url", "large_url", "medium_url", "url", "square_url"):
        v = _clean(default_photo.get(field))
        if v:
            image_url = v
            break
    if not image_url:
        return {}
    photo_id = _clean(default_photo.get("id"))
    return {
        "inat_preferred_image": image_url,
        "inat_preferred_image_license": _clean(default_photo.get("license_code")),
        "inat_preferred_image_creator": _clean(default_photo.get("attribution_name")),
        "inat_preferred_image_attribution": _clean(default_photo.get("attribution")),
        "inat_preferred_image_references": f"{INAT_PHOTO_BASE_URL}/{photo_id}" if photo_id else "",
    }


def apply_inat_preferred(
    catalog: dict,
    inat_to_taxa: dict[str, list[str]],
    results: list[dict],
) -> tuple[int, int]:
    names_updated = 0
    images_updated = 0
    for taxon in results:
        inat_id = _clean(taxon.get("id"))
        preferred_name = _clean(taxon.get("preferred_common_name"))
        image_meta = extract_preferred_image_metadata(taxon)
        if not inat_id:
            continue
        for taxon_key in inat_to_taxa.get(inat_id, []):
            entry = catalog.get(taxon_key)
            if not entry:
                continue
            if preferred_name and not _clean(entry.get("inat_preferred_common_name")):
                entry["inat_preferred_common_name"] = preferred_name
                names_updated += 1
            if image_meta and not _clean(entry.get("inat_preferred_image")):
                entry.update(image_meta)
                images_updated += 1
    return names_updated, images_updated


def run_inat_preferred(catalog: dict) -> tuple[int, int]:
    # Primary pass: taxa with a direct inat_id missing name or image.
    inat_to_taxa: dict[str, list[str]] = {}
    for taxon_key, taxon in catalog.items():
        inat_id = _clean(taxon.get("inat_id"))
        if not inat_id:
            continue
        has_name = bool(_clean(taxon.get("inat_preferred_common_name")))
        has_image = bool(_clean(taxon.get("inat_preferred_image")))
        if has_name and has_image:
            continue
        inat_to_taxa.setdefault(inat_id, []).append(taxon_key)

    # Synonym fallback pass: taxa with inat_synonym_ids still missing name or image
    # after the primary pass (e.g. Pelecyphora genus has an image from its own iNat
    # page but no preferred name — fall back to Escobaria's iNat page).
    inat_synonym_to_taxa: dict[str, list[str]] = {}
    for taxon_key, taxon in catalog.items():
        syn_ids = taxon.get("inat_synonym_ids") or []
        if not syn_ids:
            continue
        has_name = bool(_clean(taxon.get("inat_preferred_common_name")))
        has_image = bool(_clean(taxon.get("inat_preferred_image")))
        if has_name and has_image:
            continue
        for syn_id in syn_ids:
            inat_synonym_to_taxa.setdefault(syn_id, []).append(taxon_key)

    all_ids = list(dict.fromkeys(list(inat_to_taxa.keys()) + list(inat_synonym_to_taxa.keys())))
    combined_map = {**inat_synonym_to_taxa, **inat_to_taxa}  # primary takes precedence

    total_batches = (len(all_ids) + INAT_BATCH_SIZE - 1) // INAT_BATCH_SIZE
    eta_min = total_batches / INAT_RATE_LIMIT / 60
    print(
        f"  Taxa needing iNat preferred metadata: {len(inat_to_taxa):,} primary + "
        f"{len(inat_synonym_to_taxa):,} synonym fallback "
        f"({total_batches} batches, ~{eta_min:.0f} min)",
        flush=True,
    )
    if not all_ids:
        print("  Nothing to do.")
        return 0, 0

    names_updated = 0
    images_updated = 0
    errors = 0

    for i in range(0, len(all_ids), INAT_BATCH_SIZE):
        batch = all_ids[i : i + INAT_BATCH_SIZE]
        try:
            results = fetch_taxa_batch(batch)
        except Exception as exc:
            print(f"  Batch error: {exc}", flush=True)
            errors += 1
            time.sleep(1.0 / INAT_RATE_LIMIT)
            continue
        n, im = apply_inat_preferred(catalog, combined_map, results)
        names_updated += n
        images_updated += im
        request_num = i // INAT_BATCH_SIZE + 1
        if request_num % 10 == 0:
            remaining = max(total_batches - request_num, 0)
            print(
                f"  [{request_num:,}/{total_batches:,}] names={names_updated:,} "
                f"images={images_updated:,} errors={errors:,} remaining={remaining:,}",
                flush=True,
            )
        time.sleep(1.0 / INAT_RATE_LIMIT)

    return names_updated, images_updated


def update_name_index(payload: dict) -> int:
    """Add all vernacular/preferred name entries missing from the index."""
    catalog = payload["catalog"]
    index = payload["combined_name_index"]
    added = 0
    for taxon_key, taxon in catalog.items():
        candidates: list[str] = []
        # Scientific name must always be searchable, including for non-leaf taxa
        # (genera, families, etc.) that build_catalog never adds to the index.
        sci = str(taxon.get("scientific_name") or "").replace("_", " ").strip()
        if sci:
            candidates.append(sci)
        for field in ("common_name", "inat_preferred_common_name"):
            raw = str(taxon.get(field) or "").strip()
            if raw:
                candidates.append(raw)
        for name in taxon.get("vernacular_names") or []:
            raw = str(name).strip()
            if raw:
                candidates.append(raw)
        # Synonym scientific names (old genus or species names) must be searchable
        # so searching "Escobaria vivipara" or "Escobaria" still finds the accepted taxon.
        for name in (taxon.get("gbif_synonym_names") or []) + (taxon.get("genus_synonym_names") or []):
            raw = str(name).replace("_", " ").strip()
            if raw:
                candidates.append(raw)
        for raw in candidates:
            key = _normalize_index_key(raw)
            if not key:
                continue
            existing = set(index.get(key, []))
            if taxon_key not in existing:
                existing.add(taxon_key)
                index[key] = sorted(existing)
                added += 1
    return added


# ---------------------------------------------------------------------------
# GBIF backup images from occurrence DWCA
# ---------------------------------------------------------------------------

_LICENSE_PRIORITY = [
    ("publicdomain", 0), ("cc0", 0),
    ("/by/4", 1), ("/by/3", 1), ("/by/2", 1), ("cc by ", 1),
    ("/by-sa/", 2), ("cc by-sa", 2),
    ("/by-nc/", 3), ("cc by-nc ", 3),
    ("/by-nc-sa/", 4), ("cc by-nc-sa", 4),
]
_USABLE_LICENSES = {
    "cc0", "cc by", "cc-by", "/by/", "/by-sa/", "/by-nc/", "/by-nc-sa/",
    "publicdomain", "public domain",
}
_BAD_EVIDENCE = {"track", "scat", "feather", "bone", "molt", "hair"}
_OKAY_EVIDENCE = {"gall", "egg", "construction", "leafmine"}
_SUBSPECIES_RANKS = {"SUBSPECIES", "VARIETY", "FORM"}


def _license_score(s: str) -> int:
    n = s.strip().lower()
    for pattern, score in _LICENSE_PRIORITY:
        if pattern in n:
            return score
    return 99


def _is_usable_license(s: str) -> bool:
    n = s.strip().lower()
    return any(p in n for p in _USABLE_LICENSES)


def _image_quality(license_str: str, vitality: str, evidence: str, rcs: str) -> tuple:
    v = 0 if vitality == "alive" else (2 if vitality == "dead" else 1)
    if evidence == "organism":
        e = 0
    elif not evidence:
        e = 1
    elif any(b in evidence for b in _BAD_EVIDENCE):
        e = 3
    elif any(o in evidence for o in _OKAY_EVIDENCE):
        e = 2
    else:
        e = 1
    r = (0 if rcs == "flowers" else
         1 if rcs in {"fruits or seeds", "fruits", "seeds"} else
         2 if rcs == "flower buds" else 3)
    return (v, e, r, _license_score(license_str))


def _build_gbif_to_taxon(
    catalog_keys: set[str],
    synonym_to_catalog: dict[str, str] | None = None,
    unmatched_inat_keys: set[str] | None = None,
    obs_id_out: dict[str, str] | None = None,
) -> dict[str, tuple]:
    """Stream occurrence.txt → gbifID: (taxon_key, vitality, evidence, rcs).

    synonym_to_catalog maps old synonym taxon keys to their current accepted catalog
    key so that occurrence records submitted under the old name are still matched.

    If unmatched_inat_keys and obs_id_out are provided, also collects the first
    iNat catalogNumber per unmatched taxon in the same pass (used by
    run_observation_mapping so the file is only streamed once).
    """
    mapping: dict[str, tuple] = {}
    rows = 0
    with open(OCCURRENCE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows += 1
            if rows % 1_000_000 == 0:  # pragma: no cover
                print(f"  {rows:,} rows, {len(mapping):,} matched...", flush=True)
            gbif_id = (row.get("gbifID") or "").strip()
            if not gbif_id:
                continue
            rank = (row.get("taxonRank") or "").upper()
            taxon_key = (row.get("taxonKey") or "").strip()
            species_key = (row.get("speciesKey") or "").strip()
            key = taxon_key if rank in _SUBSPECIES_RANKS else (species_key or taxon_key)
            if not key:
                continue
            if key not in catalog_keys and synonym_to_catalog:
                key = synonym_to_catalog.get(key, key)
            if not key or key not in catalog_keys:
                continue
            # Opportunistically collect an obs ID for iNat ID resolution.
            if unmatched_inat_keys and obs_id_out is not None and key in unmatched_inat_keys and key not in obs_id_out:
                catalog_num = (row.get("catalogNumber") or "").strip()
                obs_id = _extract_observation_id([catalog_num])
                if obs_id:
                    obs_id_out[key] = obs_id
            vitality = (row.get("vitality") or "").strip().lower()
            rcs = (row.get("reproductiveCondition") or "").strip().lower()
            evidence = ""
            dp = row.get("dynamicProperties") or ""
            if dp:
                try:
                    obj = json.loads(dp)
                    ev = obj.get("evidenceOfPresence", "")
                    evidence = (",".join(ev) if isinstance(ev, list) else (ev or "")).lower()
                except (json.JSONDecodeError, TypeError):
                    pass
            mapping[gbif_id] = (key, vitality, evidence, rcs)
    print(f"  {rows:,} rows scanned, {len(mapping):,} gbifIDs mapped", flush=True)
    return mapping


def _build_gbif_images(gbif_to_taxon: dict[str, tuple]) -> dict[str, dict]:
    """Stream multimedia.txt → taxon_key: best image record."""
    best: dict[str, tuple] = {}
    rows = 0
    with open(MULTIMEDIA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows += 1
            if rows % 5_000_000 == 0:  # pragma: no cover
                print(f"  {rows:,} rows, {len(best):,} taxa covered...", flush=True)
            gbif_id = (row.get("gbifID") or "").strip()
            info = gbif_to_taxon.get(gbif_id)
            if not info:
                continue
            taxon_key, vitality, evidence, rcs = info
            license_str = row.get("license") or ""
            if not _is_usable_license(license_str):
                continue
            media_type = (row.get("type") or "").strip().lower()
            media_format = (row.get("format") or "").strip().lower()
            if not (media_type in {"stillimage", "image"} or media_format.startswith("image/")):
                continue
            url = (row.get("identifier") or "").strip()
            if not url:
                continue
            score = _image_quality(license_str, vitality, evidence, rcs)
            if taxon_key in best and score >= best[taxon_key][0]:
                continue
            best[taxon_key] = (score, {
                "gbif_backup_image": url,
                "gbif_backup_image_license": license_str,
                "gbif_backup_image_creator": _clean(row.get("creator")),
                "gbif_backup_image_attribution": _clean(row.get("rightsHolder")),
                "gbif_backup_image_references": _clean(row.get("references")),
            })
    print(f"  {rows:,} rows scanned, {len(best):,} taxa with images", flush=True)
    return {k: v[1] for k, v in best.items()}


def run_gbif_backup(catalog: dict) -> tuple[int, int]:
    """Stream occurrence.txt once: collect backup image candidates AND obs IDs for
    any taxa still lacking inat_id, then apply both results.

    Returns (images_updated, obs_ids_resolved).
    """
    if not OCCURRENCE_PATH.exists() or not MULTIMEDIA_PATH.exists():
        print("  Occurrence data not yet downloaded, skipping.", flush=True)
        return 0, 0
    synonym_to_catalog: dict[str, str] = {
        syn_key: catalog_key
        for catalog_key, taxon in catalog.items()
        for syn_key in (taxon.get("gbif_synonym_keys") or [])
    }
    # Taxa that still need an inat_id resolved via an observation lookup.
    unmatched_inat_keys: set[str] = {
        key for key, taxon in catalog.items()
        if not _clean(taxon.get("inat_id"))
        and (taxon.get("rank") or "").upper() in MAPPING_RANKS
    }
    obs_id_out: dict[str, str] = {}
    gbif_to_taxon = _build_gbif_to_taxon(
        set(catalog.keys()), synonym_to_catalog,
        unmatched_inat_keys, obs_id_out,
    )
    print(f"  Collected obs IDs for {len(obs_id_out):,} unmatched taxa", flush=True)
    obs_resolved = _resolve_obs_inat_ids(catalog, obs_id_out)
    taxon_images = _build_gbif_images(gbif_to_taxon)
    updated = 0
    for taxon_key, fields in taxon_images.items():
        entry = catalog.get(taxon_key)
        if entry is not None:
            entry.update(fields)
            updated += 1
    return updated, obs_resolved


# ---------------------------------------------------------------------------
# Image propagation
# ---------------------------------------------------------------------------

def propagate_images(catalog: dict) -> int:
    """Bottom-up pass: give imageless ancestor nodes an image from a direct child.

    Processes deepest nodes first so each parent can inherit from a child that
    may itself have just inherited from its own children — one row at a time up
    the tree. Prefers inat_preferred_image over gbif_backup_image.
    """
    path_to_key = {taxon["path"]: key for key, taxon in catalog.items()}
    children: dict[str, list[str]] = defaultdict(list)
    for key, taxon in catalog.items():
        path = taxon["path"]
        if "/" in path:
            parent_path = path.rsplit("/", 1)[0]
            parent_key = path_to_key.get(parent_path)
            if parent_key:
                children[parent_key].append(key)

    by_depth = sorted(catalog.keys(), key=lambda k: catalog[k]["path"].count("/"), reverse=True)

    _INAT_IMG_FIELDS = (
        "inat_preferred_image",
        "inat_preferred_image_license",
        "inat_preferred_image_creator",
        "inat_preferred_image_attribution",
        "inat_preferred_image_references",
    )
    _GBIF_IMG_FIELDS = (
        "gbif_backup_image",
        "gbif_backup_image_license",
        "gbif_backup_image_creator",
        "gbif_backup_image_attribution",
        "gbif_backup_image_references",
    )

    updated = 0
    for key in by_depth:
        taxon = catalog[key]
        if _clean(taxon.get("inat_preferred_image")) or _clean(taxon.get("gbif_backup_image")):
            continue
        inherited: dict | None = None
        child_keys = children.get(key, [])
        for child_key in child_keys:
            child = catalog[child_key]
            if _clean(child.get("inat_preferred_image")):
                inherited = {f: child.get(f, "") for f in _INAT_IMG_FIELDS}
                break
        if inherited is None:
            for child_key in child_keys:
                child = catalog[child_key]
                if _clean(child.get("gbif_backup_image")):
                    inherited = {f: child.get(f, "") for f in _GBIF_IMG_FIELDS}
                    break
        if inherited:
            taxon.update(inherited)
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Phase 1: Build catalog from GBIF species list
    csv_path = CATALOG_DIR / "species_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Species list not found: {csv_path} — run sync_gbif first")
    print(f"Building catalog from {csv_path}...")
    catalog, combined_index = build_catalog(csv_path, write_dirs=True)
    payload: dict = {"catalog": catalog, "combined_name_index": combined_index}
    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(catalog)} taxa to {CATALOG_PATH}")
    csv_path.unlink(missing_ok=True)

    # Phase 2: ID mapping — download DWC-A once, reuse in Phase 3
    print("\nBuilding iNat ID mapping...")
    dwca_bytes = fetch_inat_dwca()
    build_mapping(catalog, dwca_bytes)
    id_updated = apply_mapping(catalog)
    print(f"Applied inat_id to {id_updated:,} catalog entries.")
    MAPPING_PATH.unlink(missing_ok=True)
    inferred = infer_species_inat_ids(catalog, dwca_bytes)
    print(f"Inferred inat_id for {inferred:,} additional species from children.")
    species_syn_resolved = resolve_species_synonym_ids(catalog, dwca_bytes)
    print(f"Resolved inat_id for {species_syn_resolved:,} species via synonym name matching.")
    infra_syn_resolved = resolve_subspecies_synonym_ids(catalog, dwca_bytes)
    print(f"Resolved inat_id for {infra_syn_resolved:,} infra-specific taxa via synonym name matching.")
    syn_ids_resolved = resolve_genus_synonym_ids(catalog, dwca_bytes)
    print(f"Resolved {syn_ids_resolved:,} inat_synonym_ids for genus nodes with synonym genera.")

    # Phase 3: Common names, preferred images, GBIF backup images.
    # run_gbif_backup also streams occurrence.txt to collect obs IDs for any taxa
    # still lacking inat_id, resolving them via the iNat observations API in the
    # same pass — no extra file scan needed.
    # Load vernacular name maps now (no inat_id dependency), but defer apply_names
    # until after run_gbif_backup so that obs-based inat_id resolutions also get
    # their iNat vernacular names applied.
    print("\nFetching GBIF backbone vernacular names...")
    vernacular_bytes = fetch_backbone_vernacular()
    print("Loading vernacular names...")
    inat_map = load_inat_vernacular(dwca_bytes)
    print(f"  iNat: {len(inat_map):,} English names")
    gbif_map = load_gbif_vernacular(vernacular_bytes)
    print(f"  GBIF: {len(gbif_map):,} English names")
    print(f"  Catalog: {len(catalog):,} taxa")

    print("\nFetching GBIF backup images from occurrence data...")
    backup_n, obs_resolved = run_gbif_backup(catalog)
    print(f"  Resolved inat_id for {obs_resolved:,} additional taxa via observations.")

    updated = apply_names(catalog, inat_map, gbif_map)

    # Now that observation mapping is done, fetch iNat preferred names and images
    # for any newly resolved taxa (plus anything still pending).
    print("\nFetching iNat preferred names and images...")
    names_n, images_n = run_inat_preferred(catalog)

    print("\nPropagating images to imageless ancestor nodes...")
    inherited_n = propagate_images(catalog)
    print(f"  Propagated images to {inherited_n:,} ancestor nodes.")

    index_added = update_name_index(payload)
    print(f"Added {index_added:,} new entries to name search index.")

    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated {updated:,} catalog entries with common names.")
    print(
        f"Updated {names_n:,} preferred common names, "
        f"{images_n:,} preferred images, {backup_n:,} GBIF backup images."
    )


def rebuild_index() -> None:
    with open(CATALOG_PATH, "rb") as f:
        payload = pickle.load(f)
    added = update_name_index(payload)
    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Added {added:,} new entries to name search index.")


if __name__ == "__main__":  # pragma: no cover
    main()
