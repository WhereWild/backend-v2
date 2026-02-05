"""
Update taxon_catalog.pkl with vernacular names from iNaturalist DWCA + GBIF TSV.

Reads VernacularNames-*.csv from inaturalist-taxonomy.dwca and VernacularName.tsv
from the GBIF dataset, storing {name, language, lexicon, source} for each name.
Filters out names containing profanity or slurs.
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from better_profanity import profanity
from util.config import load_config

# Increase CSV field size limit for large fields
csv.field_size_limit(sys.maxsize)

CONFIG = load_config("global")

# Initialize profanity filter with slurs enabled
profanity.load_censor_words()

# Add additional slurs and offensive terms to filter
# These are racial, ethnic, and other offensive slurs that may appear in historical data
ADDITIONAL_SLURS = []
profanity.add_censor_words(ADDITIONAL_SLURS)


def load_vernacular_names_from_inat(
    vernacular_paths: list[Path],
) -> dict[str, list[dict[str, str]]]:
    """
    Load vernacular names from iNat DWCA CSV files and group by iNat taxon ID.

    Returns:
        Dictionary mapping iNat taxon ID -> list of vernacular names
    """
    print("Loading vernacular names from iNat DWCA files...")
    vernacular_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    rows_processed = 0
    rows_matched = 0
    rows_filtered = 0

    for vernacular_path in vernacular_paths:
        print(f"  Reading {vernacular_path.name}...")
        with open(vernacular_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if not header:
                continue
            index = {name: idx for idx, name in enumerate(header)}
            id_idx = index.get("id")
            name_idx = index.get("vernacularName")
            lang_idx = index.get("language")
            lex_idx = index.get("lexicon")
            if id_idx is None or name_idx is None:
                continue

            for row in reader:
                rows_processed += 1

                if rows_processed % 200_000 == 0:
                    print(
                        f"  Processed {rows_processed:,} rows, "
                        f"matched {rows_matched:,}, filtered {rows_filtered:,}..."
                    )

                taxon_id = (row[id_idx] if id_idx < len(row) else "").strip()
                vernacular_name = (row[name_idx] if name_idx < len(row) else "").strip()
                language = (row[lang_idx] if lang_idx is not None and lang_idx < len(row) else "").strip()
                lexicon = (row[lex_idx] if lex_idx is not None and lex_idx < len(row) else "").strip()
                if not taxon_id or not vernacular_name:
                    continue

                if profanity.contains_profanity(vernacular_name):
                    rows_filtered += 1
                    continue

                key = (vernacular_name, language, lexicon)
                if key not in seen[taxon_id]:
                    vernacular_map[taxon_id].append(
                        {
                            "name": vernacular_name,
                            "language": language,
                            "lexicon": lexicon,
                            "source": "inat",
                        }
                    )
                    seen[taxon_id].add(key)
                    rows_matched += 1

    print(f"\n  Total rows: {rows_processed:,}")
    print(f"  Matched rows: {rows_matched:,}")
    print(f"  Filtered (profanity): {rows_filtered:,}")
    print(f"  Taxa with vernacular names: {len(vernacular_map):,}")

    return dict(vernacular_map)


def load_vernacular_names_from_gbif(
    vernacular_path: Path,
) -> dict[str, list[dict[str, str]]]:
    """
    Load vernacular names from GBIF VernacularName.tsv and group by GBIF taxonID.
    """
    print(f"Loading vernacular names from {vernacular_path}...")
    vernacular_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    rows_processed = 0
    rows_matched = 0
    rows_filtered = 0

    with open(vernacular_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
        if not header:
            return dict(vernacular_map)
        index = {name: idx for idx, name in enumerate(header)}
        taxon_idx = index.get("taxonID")
        name_idx = index.get("vernacularName")
        lang_idx = index.get("language")
        source_idx = index.get("source")
        if taxon_idx is None or name_idx is None:
            return dict(vernacular_map)
        for row in reader:
            rows_processed += 1
            if rows_processed % 200_000 == 0:
                print(
                    f"  Processed {rows_processed:,} rows, "
                    f"matched {rows_matched:,}, filtered {rows_filtered:,}..."
                )
            taxon_id = (row[taxon_idx] if taxon_idx < len(row) else "").strip()
            vernacular_name = (row[name_idx] if name_idx < len(row) else "").strip()
            language = (row[lang_idx] if lang_idx is not None and lang_idx < len(row) else "").strip()
            lexicon = (row[source_idx] if source_idx is not None and source_idx < len(row) else "").strip()

            if not taxon_id or not vernacular_name:
                continue

            if profanity.contains_profanity(vernacular_name):
                rows_filtered += 1
                continue

            key = (vernacular_name, language, lexicon)
            if key not in seen[taxon_id]:
                vernacular_map[taxon_id].append(
                    {
                        "name": vernacular_name,
                        "language": language,
                        "lexicon": lexicon,
                        "source": "gbif",
                    }
                )
                seen[taxon_id].add(key)
                rows_matched += 1

    print(f"\n  Total rows: {rows_processed:,}")
    print(f"  Matched rows: {rows_matched:,}")
    print(f"  Filtered (profanity): {rows_filtered:,}")
    print(f"  Taxa with vernacular names: {len(vernacular_map):,}")

    return dict(vernacular_map)


def update_catalog_with_vernacular_names() -> None:
    """Update the taxon catalog with iNat vernacular names."""
    vernacular_dir = CONFIG.species_dir / CONFIG.inat_dwca_dirname
    gbif_vernacular_path = CONFIG.species_dir / "VernacularName.tsv"
    catalog_path = CONFIG.taxon_catalog_path

    if not vernacular_dir.exists():
        raise FileNotFoundError(f"iNat DWCA not found at {vernacular_dir}")
    if not gbif_vernacular_path.exists():
        raise FileNotFoundError(f"VernacularName.tsv not found at {gbif_vernacular_path}")

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found at {catalog_path}")

    vernacular_paths = sorted(vernacular_dir.glob("VernacularNames-*.csv"))
    if not vernacular_paths:
        raise FileNotFoundError(f"No vernacular files found in {vernacular_dir}")

    inat_map = load_vernacular_names_from_inat(vernacular_paths)
    gbif_map = load_vernacular_names_from_gbif(gbif_vernacular_path)

    # Load catalog
    print(f"\nLoading catalog from {catalog_path}...")
    with open(catalog_path, "rb") as f:
        payload = pickle.load(f)

    catalog = payload["catalog"]
    print(f"  Catalog has {len(catalog):,} taxa")

    # Clear existing common names so updates are fresh
    print("\nClearing existing common names...")
    cleared = 0
    for taxon in catalog.values():
        if "common_name" in taxon:
            taxon["common_name"] = []
            cleared += 1
    print(f"  Cleared {cleared:,} taxa with existing common names")

    # Update catalog with vernacular names (using inat_id)
    print("\nUpdating catalog with vernacular names...")
    updated_count = 0
    totals_by_rank = defaultdict(int)
    updated_by_rank = defaultdict(int)
    
    for taxon_key, taxon in catalog.items():
        rank = (taxon.get("rank") or "").strip().upper()
        totals_by_rank[rank] += 1
        inat_id = str(taxon.get("inat_id") or "").strip()
        names: list[dict[str, str]] = []
        inat_id = str(taxon.get("inat_id") or "").strip()
        if inat_id and inat_id in inat_map:
            names.extend(inat_map[inat_id])
        if taxon_key in gbif_map:
            names.extend(gbif_map[taxon_key])
        if names:
            taxon["common_name"] = names  # Store as list of {name, language, lexicon, source}
            updated_count += 1
            updated_by_rank[rank] += 1

    print(f"  Updated {updated_count:,} taxa with vernacular names")

    print("\nCoverage by rank:")
    for rank in sorted(totals_by_rank.keys()):
        total = totals_by_rank[rank]
        updated = updated_by_rank.get(rank, 0)
        rate = (updated / total * 100) if total else 0
        print(f"  {rank or 'UNKNOWN':<12} {updated:>9,}/{total:>9,} ({rate:5.1f}%)")

    # Save updated catalog
    backup_path = catalog_path.with_suffix(".pkl.backup")
    print(f"\nBacking up original catalog to {backup_path}...")
    import shutil
    shutil.copy(catalog_path, backup_path)

    print(f"Saving updated catalog to {catalog_path}...")
    with open(catalog_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")

    # Print some examples
    print("\nExample updated taxa:")
    examples_shown = 0
    for taxon_key, taxon in catalog.items():
        names = taxon.get("common_name")
        if isinstance(names, list) and len(names) > 1:
            print(f"  {taxon_key} ({taxon.get('scientific_name', '')}):")
            print(f"    {names[:5]}")
            examples_shown += 1
            if examples_shown >= 5:
                break


if __name__ == "__main__":
    update_catalog_with_vernacular_names()
