"""
Build a taxon -> media URL mapping from GBIF occurrence download files.

Run in two steps:
1. First run builds gbif_taxon_lookup.txt (gbifID -> taxon key mapping)
2. Second run (after lookup exists) processes multimedia.txt

Expects occurrence.txt and multimedia.txt in species_dir (from GBIF DWCA download).
"""

import csv
import pickle

from util.config import load_config

CONFIG = load_config("global")

gbif_row_limit = None


# Column positions in occurrence.txt (0-indexed)
GBIF_ID_COL = 0
DYNAMIC_PROPERTIES_COL = 20
VITALITY_COL = 34
# NOTE: reproductiveCondition column index varies, we'll find it dynamically
TAXON_RANK_COL = 172
TAXON_KEY_COL = 193
SPECIES_KEY_COL = 202

# Patterns for usable licenses (permissive + non-commercial, since WhereWild is non-commercial)
USABLE_LICENSE_PATTERNS = {
    "cc0",
    "cc by",
    "cc-by",
    "/by/",
    "/by-sa/",
    "/by-nc/",
    "/by-nc-sa/",
    "publicdomain",
    "public domain",
}

# License priority (lower = better, more permissive)
LICENSE_PRIORITY = [
    ("publicdomain", 0),
    ("cc0", 0),
    ("/by/4", 1),
    ("/by/3", 1),
    ("/by/2", 1),
    ("cc by ", 1),
    ("/by-sa/", 2),
    ("cc by-sa", 2),
    ("/by-nc/", 3),
    ("cc by-nc ", 3),
    ("/by-nc-sa/", 4),
    ("cc by-nc-sa", 4),
]


def get_license_score(license_str: str) -> int:
    """Return a score for license (lower = more permissive)."""
    if not license_str:
        return 99
    normalized = license_str.strip().lower()
    for pattern, score in LICENSE_PRIORITY:
        if pattern in normalized:
            return score
    return 99  # unknown license


def is_usable_license(license_str: str) -> bool:
    """Check if a license is usable (permissive or NC for non-commercial use)."""
    if not license_str:
        return False
    normalized = license_str.strip().lower()
    return any(pattern in normalized for pattern in USABLE_LICENSE_PATTERNS)


def get_image_quality_score(license_str: str, vitality: str, evidence: str, rcs: str = "") -> tuple[int, int, int, int]:
    """
    Return a composite quality score (lower = better).

    Returns (vitality_score, evidence_score, rcs_score, license_score) tuple for sorting.
    Priority: vitality > evidence > rcs (flowering/fruiting) > license (quality-first, license as tiebreaker)
    """
    # Vitality score (0 = alive, 1 = empty/undetermined, 2 = dead)
    vitality_score = 0 if vitality == "alive" else (2 if vitality == "dead" else 1)

    # Evidence score:
    # 0 = organism (definitely the live organism)
    # 1 = empty/undetermined (we don't know what it is - could be anything)
    # 2 = egg/gall/construction (secondary evidence, not primary organism)
    # 3 = track/scat/feather/bone/molt/hair (NOT the organism)
    bad_evidence = {"track", "scat", "feather", "bone", "molt", "hair"}
    okay_evidence = {"gall", "egg", "construction", "leafmine"}

    if evidence == "organism":
        evidence_score = 0  # explicit organism annotation
    elif not evidence:
        evidence_score = 1  # missing/unknown - could be anything, should be deprioritized
    elif any(bad in evidence for bad in bad_evidence):
        evidence_score = 3  # definitely not the organism
    elif any(okay in evidence for okay in okay_evidence):
        evidence_score = 2  # secondary evidence
    else:
        evidence_score = 1  # unknown annotation, treat as undetermined

    # Reproductive Condition score (plant phenology):
    # 0 = flowering (best - shows reproductive structures)
    # 1 = fruiting (good - shows reproductive structures)
    # 2 = flower budding (good - shows upcoming flowers)
    # 3 = empty/no evidence/other (less informative)
    rcs_lower = rcs.lower().strip()
    if rcs_lower == "flowers":
        rcs_score = 0
    elif rcs_lower in {"fruits or seeds", "fruits", "seeds"}:
        rcs_score = 1
    elif rcs_lower == "flower buds":
        rcs_score = 2
    else:
        rcs_score = 3

    license_score = get_license_score(license_str)

    return (vitality_score, evidence_score, rcs_score, license_score)


def build_lookup_file():
    """Extract needed columns from occurrence.txt into a smaller lookup file."""
    occurrence_path = CONFIG.gbif_occurrence_path
    lookup_path = CONFIG.gbif_taxon_lookup_path

    if not occurrence_path.exists():
        raise FileNotFoundError(f"occurrence.txt not found at {occurrence_path}")

    if lookup_path.exists():
        print(f"Lookup file already exists: {lookup_path}")
        print("  Delete it to rebuild, or proceed to multimedia processing.")
        return False

    print(f"Building lookup from {occurrence_path}...")
    print(f"  Output: {lookup_path}")
    if gbif_row_limit:
        print(f"  Row limit: {gbif_row_limit:,} (for testing)")
    print()

    rows_processed = 0
    rows_written = 0

    with open(occurrence_path, "r", encoding="utf-8") as infile, \
         open(lookup_path, "w", encoding="utf-8") as outfile:

        reader = csv.reader(infile, delimiter="\t")
        header = next(reader)  # read header

        # Find reproductiveCondition column index
        rcs_col_index = header.index("reproductiveCondition") if "reproductiveCondition" in header else -1

        # Write header
        outfile.write("gbifID\ttaxonRank\ttaxonKey\tspeciesKey\tvitality\tdynamicProperties\treproductiveCondition\n")

        for row in reader:
            rows_processed += 1

            # Stop if we hit the row limit
            if gbif_row_limit and rows_processed > gbif_row_limit:
                break

            if rows_processed % 1_000_000 == 0:
                print(f"  Processed {rows_processed:,} rows, written {rows_written:,}...")

            if len(row) <= SPECIES_KEY_COL:
                continue

            gbif_id = row[GBIF_ID_COL]
            taxon_rank = row[TAXON_RANK_COL]
            taxon_key = row[TAXON_KEY_COL]
            species_key = row[SPECIES_KEY_COL]
            vitality = row[VITALITY_COL] if len(row) > VITALITY_COL else ""
            dynamic_props = row[DYNAMIC_PROPERTIES_COL] if len(row) > DYNAMIC_PROPERTIES_COL else ""
            rcs = row[rcs_col_index] if rcs_col_index >= 0 and len(row) > rcs_col_index else ""

            if not gbif_id:
                continue

            outfile.write(f"{gbif_id}\t{taxon_rank}\t{taxon_key}\t{species_key}\t{vitality}\t{dynamic_props}\t{rcs}\n")
            rows_written += 1

    print()
    print(f"  Done! Processed {rows_processed:,} rows, wrote {rows_written:,}")
    print(f"  Saved to: {lookup_path}")
    return True


def load_gbif_to_taxon_mapping() -> dict[str, tuple[str, str, str, str]]:
    """
    Load the lookup file and build gbifID -> (taxon_key, vitality, evidence, rcs) mapping.

    Uses speciesKey for SPECIES rank, taxonKey for subspecies/variety/form.
    Returns vitality, evidenceOfPresence, and reproductiveCondition for quality scoring.
    """
    lookup_path = CONFIG.gbif_taxon_lookup_path

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_path}")

    print(f"Loading taxon lookup from {lookup_path}...")

    mapping = {}
    subspecies_ranks = {"SUBSPECIES", "VARIETY", "FORM"}
    rows_loaded = 0

    with open(lookup_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header

        for row in reader:
            rows_loaded += 1

            if rows_loaded % 10_000_000 == 0:
                print(f"  Loaded {rows_loaded:,} rows...")

            if len(row) < 7:
                continue

            gbif_id, taxon_rank, taxon_key, species_key, vitality, dynamic_props, rcs = row[0], row[1], row[2], row[3], row[4], row[5], row[6]

            if not gbif_id:
                continue

            # Use speciesKey for species, taxonKey for subspecies
            if taxon_rank.upper() in subspecies_ranks:
                key = taxon_key
            else:
                key = species_key if species_key else taxon_key

            # Extract evidenceOfPresence from dynamicProperties JSON
            evidence = ""
            if dynamic_props:
                try:
                    import json
                    obj = json.loads(dynamic_props)
                    ev = obj.get("evidenceOfPresence", "")
                    if isinstance(ev, list):
                        evidence = ",".join(ev) if ev else ""
                    else:
                        evidence = ev or ""
                except (json.JSONDecodeError, TypeError):
                    pass

            if key:
                mapping[gbif_id] = (key, vitality.lower(), evidence.lower(), rcs.lower())

    print(f"  Loaded {len(mapping):,} gbifID -> taxon mappings")
    return mapping


def build_taxon_media_mapping(gbif_to_taxon: dict[str, tuple[str, str, str]]) -> dict[str, dict]:
    """
    Read multimedia.txt and build taxon_key -> best media record.

    Keeps only the best (most permissive license, alive, organism) image per taxon.
    """
    multimedia_path = CONFIG.gbif_multimedia_path

    if not multimedia_path.exists():
        raise FileNotFoundError(f"multimedia.txt not found at {multimedia_path}")

    print(f"Processing {multimedia_path}...")
    print("  (Keeping only best quality image per taxon to save memory)")
    print("  Quality priority: alive/dead > organism/evidence > flowering/fruiting > license")

    # taxon_key -> (quality_score, media_record)
    best_media = {}
    total_rows = 0
    matched_rows = 0
    usable_rows = 0
    replacements = 0
    dead_skipped = 0
    evidence_skipped = 0

    with open(multimedia_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_rows += 1

            if total_rows % 5_000_000 == 0:
                print(f"  Processed {total_rows:,} rows, {len(best_media):,} taxa covered...")

            gbif_id = row.get("gbifID", "")
            taxon_info = gbif_to_taxon.get(gbif_id)

            if not taxon_info:
                continue

            taxon_key, vitality, evidence, rcs = taxon_info
            matched_rows += 1

            license_str = row.get("license", "")
            if not is_usable_license(license_str):
                continue

            usable_rows += 1

            media_type = (row.get("type") or "").strip().lower()
            media_format = (row.get("format") or "").strip().lower()
            is_image_type = media_type in {"stillimage", "image"}
            is_image_format = media_format.startswith("image/")
            if not (is_image_type or is_image_format):
                continue

            url = row.get("identifier", "")
            if not url:
                continue

            # Calculate quality score (vitality, evidence, rcs, license)
            new_score = get_image_quality_score(license_str, vitality, evidence, rcs)

            # Track what we're filtering
            if new_score[0] == 2:  # dead
                dead_skipped += 1
            if new_score[1] == 3:  # bad evidence (track/scat/feather)
                evidence_skipped += 1

            # Check if we already have an image for this taxon
            if taxon_key in best_media:
                current_score = best_media[taxon_key][0]
                # Only replace if new image has better (lower) score
                # Tuple comparison: license first, then vitality, then evidence
                if new_score >= current_score:
                    continue
                replacements += 1

            media_record = {
                "url": url,
                "license": license_str,
                "creator": row.get("creator", ""),
                "rightsHolder": row.get("rightsHolder", ""),
                "references": row.get("references", ""),
            }
            best_media[taxon_key] = (new_score, media_record)

    print(f"\n  Total rows: {total_rows:,}")
    print(f"  Matched to taxa: {matched_rows:,}")
    print(f"  Usable license: {usable_rows:,}")
    print(f"  Dead organisms seen: {dead_skipped:,}")
    print(f"  Track/scat/feather evidence seen: {evidence_skipped:,}")
    print(f"  Unique taxa with media: {len(best_media):,}")
    print(f"  Better quality found (replacements): {replacements:,}")

    # Strip the score, return just the records
    return {k: v[1] for k, v in best_media.items()}


def load_catalog() -> dict:
    """Load the taxon catalog."""
    with open(CONFIG.taxon_catalog_path, "rb") as f:
        payload = pickle.load(f)
    return payload["catalog"]


def print_coverage_stats(taxon_media: dict[str, dict]):
    """Print coverage statistics against the catalog."""
    print("\nLoading catalog for coverage stats...")
    catalog = load_catalog()

    catalog_keys = set(catalog.keys())
    media_keys = set(taxon_media.keys())

    covered = catalog_keys & media_keys
    not_covered = catalog_keys - media_keys

    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"  Taxa in catalog: {len(catalog_keys):,}")
    print(f"  Taxa with images: {len(covered):,} ({100*len(covered)/len(catalog_keys):.1f}%)")
    print(f"  Taxa without images: {len(not_covered):,} ({100*len(not_covered)/len(catalog_keys):.1f}%)")

    # License distribution of selected images
    from collections import Counter
    license_scores = Counter(get_license_score(v.get("license", "")) for v in taxon_media.values())
    print("\n  License quality of selected images:")
    score_labels = {0: "Public Domain/CC0", 1: "CC-BY", 2: "CC-BY-SA", 3: "CC-BY-NC", 4: "CC-BY-NC-SA", 99: "Unknown"}
    for score in sorted(license_scores.keys()):
        label = score_labels.get(score, f"Score {score}")
        count = license_scores[score]
        print(f"    {label}: {count:,}")


def main():
    # Step 1: Build lookup file if needed (continue in same run)
    build_lookup_file()

    output_path = CONFIG.taxon_media_path
    if output_path.exists():
        print(f"\nTaxon media already exists: {output_path}")
        print("  Delete it to rebuild, or proceed to coverage stats.")
        with open(output_path, "rb") as f:
            taxon_media = pickle.load(f)
    else:
        # Step 2: Load gbifID -> taxon mapping
        gbif_to_taxon = load_gbif_to_taxon_mapping()

        # Step 3: Build taxon -> media mapping (best image per taxon)
        taxon_media = build_taxon_media_mapping(gbif_to_taxon)

        # Step 4: Save to pickle
        print(f"\nSaving to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(taxon_media, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Done.")

    # Step 5: Print coverage stats
    print_coverage_stats(taxon_media)


if __name__ == "__main__":
    main()
