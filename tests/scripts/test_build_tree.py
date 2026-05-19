import csv
import pickle
from pathlib import Path

import pytest

import scripts.build_tree as build_tree

COLUMNS = [
    "taxonRank", "taxonKey", "speciesKey", "acceptedScientificName", "scientificName", "commonName",
    "kingdom", "kingdomKey", "phylum", "phylumKey", "class", "classKey",
    "order", "orderKey", "family", "familyKey", "genus", "genusKey", "species",
]


def _make_csv(rows: list[dict], tmp_path: Path) -> Path:
    p = tmp_path / "species_list.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})
    return p


def _base_row(**kwargs) -> dict:
    return {
        "kingdom": "Plantae", "kingdomKey": "6",
        "phylum": "Tracheophyta", "phylumKey": "7707728",
        "class": "Magnoliopsida", "classKey": "220",
        "order": "Caryophyllales", "orderKey": "793",
        "family": "Cactaceae", "familyKey": "2519",
        "genus": "Opuntia", "genusKey": "2923968",
        **kwargs,
    }


SPECIES_ROW = _base_row(
    taxonRank="SPECIES",
    taxonKey="2923970",
    speciesKey="2923970",
    scientificName="Opuntia humifusa Raf.",
    acceptedScientificName="Opuntia humifusa Raf.",
    commonName="devil's tongue",
    species="Opuntia humifusa",
)

HYBRID_ROW = _base_row(
    taxonRank="SPECIES",
    taxonKey="5643900",
    speciesKey="5643900",
    scientificName="Opuntia × columbiana Griffiths",
    acceptedScientificName="Opuntia × columbiana Griffiths",
    commonName="",
    species="Opuntia × columbiana",
)

VARIETY_ROW = _base_row(
    taxonRank="VARIETY",
    taxonKey="7263189",
    speciesKey="2923970",
    scientificName="Opuntia polyacantha var. erinacea (Engelm. & J.M.Bigelow) B.D.Parfitt",
    acceptedScientificName="Opuntia polyacantha var. erinacea",
    commonName="",
    species="Opuntia polyacantha",
)

SUBSPECIES_NO_MARKER_ROW = _base_row(
    taxonRank="SUBSPECIES",
    taxonKey="9999001",
    speciesKey="2923970",
    scientificName="Echinocereus triglochidiatus mojavensis",
    acceptedScientificName="Echinocereus triglochidiatus mojavensis",
    commonName="",
    species="Echinocereus triglochidiatus",
)

SKIPPED_RANK_ROW = _base_row(
    taxonRank="GENUS",
    taxonKey="2923968",
    speciesKey="",
    scientificName="Opuntia",
    acceptedScientificName="Opuntia",
)

MALFORMED_SPECIES_ROW = _base_row(
    taxonRank="SPECIES",
    taxonKey="0000001",
    speciesKey="0000001",
    scientificName="Opuntia",
    acceptedScientificName="Opuntia",
)

NO_GENUS_ROW = {
    "taxonRank": "SPECIES",
    "taxonKey": "0000002",
    "speciesKey": "0000002",
    "scientificName": "Unknown plantae",
    "acceptedScientificName": "Unknown plantae",
    "commonName": "",
    "kingdom": "Plantae", "kingdomKey": "6",
    "phylum": "Tracheophyta", "phylumKey": "7707728",
    "class": "Magnoliopsida", "classKey": "220",
    "order": "Caryophyllales", "orderKey": "422",
    "family": "", "familyKey": "",
    "genus": "", "genusKey": "",
    "species": "",
}

SUBSPECIES_NO_SPECIES_ROW = {
    **_base_row(
        taxonRank="VARIETY",
        taxonKey="0000003",
        speciesKey="",
        scientificName="Opuntia humifusa var. austrina",
        acceptedScientificName="Opuntia humifusa var. austrina",
        commonName="",
    ),
    "species": "",
}


# --- normalize_name ---

def test_normalize_name_basic():
    assert build_tree.normalize_name("Opuntia humifusa") == "opuntia humifusa"


def test_normalize_name_underscores():
    assert build_tree.normalize_name("opuntia_humifusa") == "opuntia humifusa"


def test_normalize_name_empty():
    assert build_tree.normalize_name("") == ""


# --- clean_name ---

def test_clean_name_species():
    assert build_tree.clean_name("Opuntia humifusa Raf.", "SPECIES") == "Opuntia_humifusa"


def test_clean_name_hybrid():
    result = build_tree.clean_name("Opuntia × columbiana Griffiths", "SPECIES")
    assert result == "Opuntia_×_columbiana"


def test_clean_name_variety_with_marker():
    result = build_tree.clean_name("Opuntia polyacantha var. erinacea Foo", "VARIETY")
    assert result == "Opuntia_polyacantha_var._erinacea"


def test_clean_name_subspecies_no_marker():
    result = build_tree.clean_name("Echinocereus triglochidiatus mojavensis", "SUBSPECIES")
    assert result == "Echinocereus_triglochidiatus_mojavensis"


def test_clean_name_other_rank():
    assert build_tree.clean_name("Opuntia Mill.", "GENUS") == "Opuntia_Mill."


def test_clean_name_empty():
    assert build_tree.clean_name("", "SPECIES") == ""


# --- build_catalog ---

def test_build_catalog_species(tmp_path):
    csv_path = _make_csv([SPECIES_ROW], tmp_path)
    catalog, index = build_tree.build_catalog(csv_path)
    assert "2923970" in catalog
    entry = catalog["2923970"]
    assert entry["scientific_name"] == "Opuntia_humifusa"
    assert entry["common_name"] == "devil's tongue"
    assert entry["rank"] == "SPECIES"


def test_build_catalog_hybrid(tmp_path):
    csv_path = _make_csv([HYBRID_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "5643900" in catalog
    assert catalog["5643900"]["scientific_name"] == "Opuntia_×_columbiana"


def test_build_catalog_variety(tmp_path):
    csv_path = _make_csv([VARIETY_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "7263189" in catalog
    assert catalog["7263189"]["rank"] == "VARIETY"


def test_build_catalog_subspecies_no_marker(tmp_path):
    csv_path = _make_csv([SUBSPECIES_NO_MARKER_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "9999001" in catalog


def test_build_catalog_subspecies_path_includes_subspecies_dir(tmp_path):
    csv_path = _make_csv([SUBSPECIES_NO_MARKER_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    path = catalog["9999001"]["path"]
    assert "9999001" in path
    assert path.endswith("Echinocereus_triglochidiatus_mojavensis_9999001")


def test_build_catalog_skips_non_leaf_rank(tmp_path):
    csv_path = _make_csv([SKIPPED_RANK_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "2923968" not in catalog


def test_build_catalog_skips_malformed_species(tmp_path):
    csv_path = _make_csv([MALFORMED_SPECIES_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "0000001" not in catalog


def test_build_catalog_skips_missing_genus(tmp_path):
    csv_path = _make_csv([NO_GENUS_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "0000002" not in catalog


def test_build_catalog_skips_subspecies_missing_species(tmp_path):
    csv_path = _make_csv([SUBSPECIES_NO_SPECIES_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "0000003" not in catalog


def test_build_catalog_intermediate_nodes(tmp_path):
    csv_path = _make_csv([SPECIES_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    # Genus, family etc. should also be in catalog as intermediate nodes
    assert "2923968" in catalog  # Opuntia genus
    assert catalog["2923968"]["rank"] == "GENUS"


def test_build_catalog_name_index(tmp_path):
    csv_path = _make_csv([SPECIES_ROW], tmp_path)
    _, index = build_tree.build_catalog(csv_path)
    assert "opuntia humifusa" in index
    assert "devil's tongue" in index


def test_build_catalog_path_structure(tmp_path):
    csv_path = _make_csv([SPECIES_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    path = catalog["2923970"]["path"]
    assert path.startswith("Plantae_6/")
    assert "Opuntia_2923968" in path


# --- main ---

def test_main_writes_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "TREE_ROOT", tmp_path / "tree")
    _make_csv([SPECIES_ROW], tmp_path)

    build_tree.main()

    pkl = tmp_path / "taxon_catalog.pkl"
    assert pkl.exists()
    payload = pickle.loads(pkl.read_bytes())
    assert "catalog" in payload
    assert "combined_name_index" in payload


def test_main_missing_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="sync_gbif"):
        build_tree.main()


def test_build_catalog_do_write_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "TREE_ROOT", tmp_path / "tree")
    csv_path = _make_csv([SPECIES_ROW], tmp_path)
    build_tree.build_catalog(csv_path, write_dirs=True)
    assert any((tmp_path / "tree").rglob("Plantae_6"))
