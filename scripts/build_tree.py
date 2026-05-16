"""
Build a taxonomy catalog from a GBIF species list CSV.

Reads data/taxonomy/catalog/species_list.csv (produced by sync_gbif.py) and writes
a taxon_catalog.pkl containing a catalog of all taxa and a combined name index.
"""

import csv
import pickle
from collections import defaultdict
from pathlib import Path

from config.config import load_config

CONFIG = load_config("global")

CATALOG_DIR = Path("data/taxonomy/catalog")
TREE_ROOT = Path("data/taxonomy/tree")

HYBRID_MARKER = "×"
INFRASPECIFIC_MARKERS = ("var.", "subsp.", "f.", "nothosubsp.", "nothovar.")
TAXONOMY_LEVELS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
TSV_DELIMITER = "\t"


def normalize_name(value: str) -> str:
    if not value:
        return ""
    return " ".join(value.replace("_", " ").lower().split())


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


def build_catalog(csv_path: Path) -> tuple[dict, dict]:
    """Parse species list CSV and return (catalog, combined_name_index)."""
    catalog: dict = {}
    scientific_index: dict = {}
    common_index: defaultdict = defaultdict(set)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=TSV_DELIMITER)
        for row in reader:
            if row["taxonRank"] not in CONFIG.leaf_rank_set:
                continue
            if row["taxonRank"] == CONFIG.species_rank and len(row["scientificName"].split()) < 2:
                continue
            if not row.get("genus") or not row.get("genusKey"):
                continue
            if row["taxonRank"] in CONFIG.subspecies_equivalents:
                if not row.get("species") or not row.get("speciesKey"):
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
                cleaned_name = clean_name(row["scientificName"], row["taxonRank"])
                path_parts.append(f"{cleaned_name}_{row['taxonKey']}")
            elif row["taxonRank"] == CONFIG.species_rank:
                taxon_key_to_write = row["speciesKey"]

            if CONFIG.do_write_dirs:
                (TREE_ROOT / Path(*path_parts)).mkdir(parents=True, exist_ok=True)

            common_name = row.get("commonName", "")
            catalog[str(taxon_key_to_write)] = {
                "taxon_key": str(taxon_key_to_write),
                "path": rel_path,
                "scientific_name": cleaned_name,
                "common_name": common_name,
                "rank": row["taxonRank"],
            }

            scientific_name_key = normalize_name(cleaned_name)
            if scientific_name_key:
                scientific_index[scientific_name_key] = taxon_key_to_write

            common_name_key = normalize_name(common_name)
            if common_name_key:
                common_index[common_name_key].add(taxon_key_to_write)

    common_index_sorted = {k: sorted(v) for k, v in common_index.items()}
    combined_index: dict = {k: {v} for k, v in scientific_index.items()}
    for name, keys in common_index_sorted.items():
        combined_index.setdefault(name, set()).update(keys)
    combined_index = {k: sorted(v) for k, v in combined_index.items()}

    return catalog, combined_index


def main() -> None:
    csv_path = CATALOG_DIR / "species_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Species list not found: {csv_path} — run sync_gbif first")

    print(f"Building catalog from {csv_path}...")
    CONFIG.do_write_dirs = True
    catalog, combined_index = build_catalog(csv_path)

    out_path = CATALOG_DIR / "taxon_catalog.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"catalog": catalog, "combined_name_index": combined_index},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Wrote {len(catalog)} taxa to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
