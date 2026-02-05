'''
The purpose of this script is to build a taxonomy tree given an input tsv file obtained from GBIF that simply contains a list of species for a given dataset. The data contains everything we need to construct a tree
with the 7 major taxonomic ranks, along with subspecies and the like. It creates a nested file structure, and also a lookup table saved to a pickle file that goes from taxon ID to the filepath.
The script is short and simple. It simply reads in rows from the TSV and constructs paths using the given values.
The input taxa.csv is sourced as a species list download from www.gbif.org/occurrence/download
'''

import csv
import pickle
from collections import defaultdict
from pathlib import Path
from util.config import load_config

CONFIG = load_config("global")

taxonomy_hybrid_marker = "\u00d7"

taxonomy_infraspecific_markers = (
        "var.",
        "subsp.",
        "f.",
        "nothosubsp.",
        "nothovar.",
    )

taxonomy_levels = (
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    )

taxonomy_tsv_delimiter = "\t"

def normalize_name(value):
    if not value:
        return ""
    # Replace underscores so stored scientific names are searchable with spaces
    return " ".join(value.replace("_", " ").lower().split())

def clean_name(name, rank):
    '''
    Simple helper method designed to clean the scientific name of species and subspecies. Why? Because the provided data often includes ugly extra info like citations of the author who proposed the species.
    We only want the binomial name (or other appropriate variations), so this helper cleans names to enforce this.
    '''
    if not name:
        return ""
    parts = name.split()

    if rank == CONFIG.species_rank:
        # Possible example: Opuntia × columbiana (represents a hybrid). We simply slice the first 3 tokens. If not, take the first 2.
        if len(parts) >= 3 and parts[1] == taxonomy_hybrid_marker:
            return "_".join(parts[:3])
        return "_".join(parts[:2])

    elif rank in CONFIG.subspecies_equivalents:
        # Example: Opuntia polyacantha var. erinacea represents a variety, a level below a species. There can also be subspecies, forms, and rarely, nothosubspecies and nothovarieties (hybrids I believe)
        if len(parts) >= 3 and parts[2].lower() in taxonomy_infraspecific_markers:
            return "_".join(parts[:4])
        # Otherwise, since it was a subspecies, it means there was no marker, and we just take the first 3, e.g. Echinocereus triglochidiatus mojavensis
        else:
            return "_".join(parts[:3])

    return name.replace(" ", "_")


catalog = {}  # taxonKey → tuple(path, scientific_name, common_name, rank)
scientific_index = {}
common_index = defaultdict(set)

with open(CONFIG.taxa_csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=taxonomy_tsv_delimiter)
    for row in reader:
        # Observations SHOULD only be at the species level or lower if research grade. And we only care about these observations anyways. So skip rows not at the proper rank.
        if row["taxonRank"] not in CONFIG.leaf_rank_set:
            continue

        # Skip fake species rows that have names that don't conform
        if row["taxonRank"] == CONFIG.species_rank and len(row["scientificName"].split()) < 2:
            continue

        # Iterate through taxonomy levels and build the path. We also build a map of taxonKey to path
        path_parts = []
        for level in taxonomy_levels:
            name = row.get(level)
            key = row.get(level + "Key")
            if name and key:
                cleaned_name = clean_name(name, level.upper())
                path_parts.append(f"{cleaned_name}_{key}")
                path = CONFIG.taxonomy_root / Path(*path_parts)
                rel_path = path.relative_to(CONFIG.taxonomy_root).as_posix()

                # Keep catalog entries light for fast serialization/deserialization
                if key not in catalog:
                    catalog[str(key)] = {
                        "taxon_key": str(key),
                        "path": rel_path,
                        "scientific_name": cleaned_name,
                        "common_name": "",
                        "rank": level.upper(),
                    }

        # Finally, we need to add the key for the terminal node at the row, e.g. the taxonKey of this row.
        cleaned_name = clean_name(row["acceptedScientificName"], row["taxonRank"])

        taxon_key_to_write = row['taxonKey']

        # It should already be added by speciesKey UNLESS the row is a subspecies or similar, in which case we add it here.
        if row["taxonRank"] in CONFIG.subspecies_equivalents:
            # Use the scientific name for subspecies as that's what contains the actual subspecies info
            cleaned_name = clean_name(row["scientificName"], row["taxonRank"])
            path_parts.append(f"{cleaned_name}_{row['taxonKey']}")
        # If the rank is a species, we want to write with the species key, not the taxonKey which can be different
        elif row["taxonRank"] == CONFIG.species_rank:
            taxon_key_to_write = row['speciesKey']

        # We make the path here. It should create non terminal folders as well since they are already present on the path.
        path = CONFIG.taxonomy_root / Path(*path_parts)
        if CONFIG.do_write_dirs:
            path.mkdir(parents=True, exist_ok=True)

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

# Convert sets to sorted lists for stable serialization
common_index = {name: sorted(keys) for name, keys in common_index.items()}

# Build a combined name index to avoid doing it at load time.
combined_index = {name: {key} for name, key in scientific_index.items()}
for name, keys in common_index.items():
    combined_index.setdefault(name, set()).update(keys)
combined_index = {name: sorted(keys) for name, keys in combined_index.items()}

# Save catalog and indexes to pickle for fast Python loading.
payload = {
    "catalog": catalog,
    "combined_name_index": combined_index,
}

with open(CONFIG.taxon_catalog_path, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
