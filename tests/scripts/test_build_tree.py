# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import csv
import io
import json
import pickle
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.build_tree as build_tree

COLUMNS = [
    "taxonRank", "taxonKey", "speciesKey", "acceptedScientificName", "scientificName", "commonName",
    "taxonomicStatus", "kingdom", "kingdomKey", "phylum", "phylumKey", "class", "classKey",
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
        "taxonomicStatus": "ACCEPTED",
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
    "taxonomicStatus": "ACCEPTED",
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

# Regression fixture for a real COL data gap: a SYNONYM infra row whose
# accepted target is itself infraspecific (e.g. "Escobaria sneedii var.
# sneedii" -> "Pelecyphora sneedii subsp. sneedii") can have blank
# species/speciesKey in COL's GBIF checklist-scoped export, even though the
# row itself carries real occurrence data. Unlike SUBSPECIES_NO_SPECIES_ROW
# (an ACCEPTED row, which should still be dropped — there's no parent
# species to nest it under), this SYNONYM row should be kept and placed
# directly under its genus, same as any other synonym that can't cleanly
# nest under an accepted species.
SYNONYM_VARIETY_NO_SPECIES_ROW = {
    **_base_row(
        taxonRank="VARIETY",
        taxonKey="0000004",
        speciesKey="",
        scientificName="Escobaria sneedii var. sneedii",
        acceptedScientificName="Pelecyphora sneedii subsp. sneedii",
        commonName="",
        taxonomicStatus="SYNONYM",
        genus="Pelecyphora",
        genusKey="6000000",
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
    assert path.endswith("Opuntia_triglochidiatus_mojavensis_9999001")


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


def test_build_catalog_keeps_synonym_infra_missing_species(tmp_path):
    csv_path = _make_csv([SYNONYM_VARIETY_NO_SPECIES_ROW], tmp_path)
    catalog, _ = build_tree.build_catalog(csv_path)
    assert "0000004" in catalog
    entry = catalog["0000004"]
    assert entry["rank"] == "VARIETY"
    # Renamed to the accepted genus and placed directly under it (no parent
    # species node to nest under, since COL's export omits that context).
    assert entry["scientific_name"] == "Pelecyphora_sneedii_var._sneedii"
    assert entry["path"].endswith("Pelecyphora_6000000/Pelecyphora_sneedii_var._sneedii_0000004")
    assert entry["gbif_synonym_names"] == ["Escobaria sneedii var. sneedii"]


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
    monkeypatch.setattr(build_tree, "CATALOG_PATH", tmp_path / "taxon_catalog.pkl")
    monkeypatch.setattr(build_tree, "fetch_inat_dwca", lambda: b"")
    monkeypatch.setattr(build_tree, "build_mapping", lambda catalog, dwca_bytes: None)
    monkeypatch.setattr(build_tree, "apply_mapping", lambda catalog: 0)
    monkeypatch.setattr(build_tree, "load_inat_vernacular", lambda b: {})
    monkeypatch.setattr(build_tree, "fetch_col_vernacular", lambda catalog: {})
    monkeypatch.setattr(build_tree, "apply_names", lambda catalog, im, gm: 0)
    monkeypatch.setattr(build_tree, "run_inat_preferred", lambda catalog: (0, 0))
    monkeypatch.setattr(build_tree, "run_gbif_backup", lambda catalog: (0, 0))
    monkeypatch.setattr(build_tree, "infer_species_inat_ids", lambda catalog, b: 0)
    monkeypatch.setattr(build_tree, "resolve_genus_synonym_ids", lambda catalog, b: 0)
    monkeypatch.setattr(build_tree, "propagate_images", lambda catalog: 0)
    monkeypatch.setattr(build_tree, "update_name_index", lambda payload: 0)
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


# ===========================================================================
# Helpers shared by Phase 2 / Phase 3 tests
# ===========================================================================

def _make_cm(inner):
    """Wrap an object in a context manager mock."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=inner)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _head_urlopen(etag="etag-abc", content_length=None):
    """Return a urlopen mock whose response has ETag / Content-Length headers."""
    resp = MagicMock()
    def _get(k, d=""):
        if k == "ETag":
            return etag
        if k == "Content-Length" and content_length is not None:
            return str(content_length)
        return d
    resp.headers.get = _get
    return _make_cm(resp)


def _make_dwca_bytes(taxa_rows: list[dict], fieldnames=None) -> bytes:
    if fieldnames is None:
        fieldnames = ["id", "taxonID", "taxonRank", "scientificName"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in taxa_rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})
        zf.writestr("taxa.csv", csv_buf.getvalue())
        # Placeholder vernacular file so load_inat_vernacular doesn't crash
        zf.writestr("VernacularNames-0.csv", "id,vernacularName,language\n")
    return buf.getvalue()


def _make_occurrence_tsv(rows: list[dict]) -> str:
    fieldnames = [
        "gbifID", "taxonKey", "speciesKey", "taxonRank", "scientificName",
        "vitality", "reproductiveCondition", "dynamicProperties",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    for row in rows:
        w.writerow({f: row.get(f, "") for f in fieldnames})
    return buf.getvalue()


def _make_multimedia_tsv(rows: list[dict]) -> str:
    fieldnames = ["gbifID", "type", "format", "identifier", "license",
                  "creator", "rightsHolder", "references"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    for row in rows:
        w.writerow({f: row.get(f, "") for f in fieldnames})
    return buf.getvalue()


# ===========================================================================
# Sync state helpers
# ===========================================================================

def test_load_state_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    assert build_tree._load_state() == {}


def test_load_state_exists(tmp_path, monkeypatch):
    p = tmp_path / "sync_state.json"
    p.write_text('{"inat_taxonomy": {"etag": "abc"}}')
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", p)
    assert build_tree._load_state() == {"inat_taxonomy": {"etag": "abc"}}


def test_save_state_creates_parents(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "sync_state.json"
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", p)
    build_tree._save_state({"foo": "bar"})
    assert json.loads(p.read_text()) == {"foo": "bar"}


# ===========================================================================
# fetch_inat_dwca
# ===========================================================================

def test_fetch_inat_dwca_cache_hit(tmp_path, monkeypatch):
    cache = tmp_path / "inat_dwca.zip"
    valid_zip = _make_dwca_bytes([])
    cache.write_bytes(valid_zip)
    state_path = tmp_path / "sync_state.json"
    state_path.write_text(json.dumps({"inat_taxonomy": {"etag": "etag-abc"}}))
    monkeypatch.setattr(build_tree, "INAT_DWCA_CACHE", cache)
    monkeypatch.setattr(build_tree, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", state_path)

    with patch("scripts.build_tree.urlopen", return_value=_head_urlopen("etag-abc")):
        result = build_tree.fetch_inat_dwca()

    assert result == valid_zip


def test_fetch_inat_dwca_download(tmp_path, monkeypatch):
    cache = tmp_path / "inat_dwca.zip"
    state_path = tmp_path / "sync_state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(build_tree, "INAT_DWCA_CACHE", cache)
    monkeypatch.setattr(build_tree, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", state_path)

    valid_zip = _make_dwca_bytes([])

    def fake_run(cmd, **kw):
        cache.write_bytes(valid_zip)

    with patch("scripts.build_tree.urlopen", return_value=_head_urlopen("etag-new")), \
         patch("scripts.build_tree.subprocess.run", side_effect=fake_run):
        result = build_tree.fetch_inat_dwca()

    assert result == valid_zip
    assert json.loads(state_path.read_text())["inat_taxonomy"]["etag"] == "etag-new"


def test_fetch_inat_dwca_no_etag_saved(tmp_path, monkeypatch):
    cache = tmp_path / "inat_dwca.zip"
    state_path = tmp_path / "sync_state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(build_tree, "INAT_DWCA_CACHE", cache)
    monkeypatch.setattr(build_tree, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "SYNC_STATE_PATH", state_path)

    def fake_run(cmd, **kw):
        cache.write_bytes(_make_dwca_bytes([]))

    with patch("scripts.build_tree.urlopen", return_value=_head_urlopen("")), \
         patch("scripts.build_tree.subprocess.run", side_effect=fake_run):
        build_tree.fetch_inat_dwca()

    assert "inat_taxonomy" not in json.loads(state_path.read_text())


# ===========================================================================
# strip_infra_markers / build_gbif_indexes / extract_taxa_csv
# ===========================================================================

def test_strip_infra_markers_removes_var():
    assert build_tree.strip_infra_markers("opuntia polyacantha var. erinacea") == "opuntia polyacantha erinacea"


def test_strip_infra_markers_removes_subsp():
    assert build_tree.strip_infra_markers("pinus sylvestris subsp. sylvestris") == "pinus sylvestris sylvestris"


def test_strip_infra_markers_no_markers():
    assert build_tree.strip_infra_markers("opuntia humifusa") == "opuntia humifusa"


def test_build_gbif_indexes_exact():
    catalog = {"123": {"rank": "SPECIES", "scientific_name": "Opuntia_humifusa"}}
    exact, _ = build_tree.build_gbif_indexes(catalog)
    assert ("SPECIES", "opuntia humifusa", "") in exact
    assert exact[("SPECIES", "opuntia humifusa", "")] == ["123"]


def test_build_gbif_indexes_parent_disambiguates_homonym():
    catalog = {
        "1": {"rank": "GENUS", "scientific_name": "Townsendia", "path": "Plantae_1/Asteraceae_2/Townsendia_1"},
        "2": {"rank": "GENUS", "scientific_name": "Townsendia", "path": "Animalia_9/Asilidae_8/Townsendia_2"},
    }
    exact, _ = build_tree.build_gbif_indexes(catalog)
    assert exact[("GENUS", "townsendia", "asteraceae")] == ["1"]
    assert exact[("GENUS", "townsendia", "asilidae")] == ["2"]


def test_build_gbif_indexes_stripped_for_variety():
    catalog = {"456": {"rank": "VARIETY", "scientific_name": "Opuntia_polyacantha_var._erinacea"}}
    _, stripped = build_tree.build_gbif_indexes(catalog)
    assert ("VARIETY", "opuntia polyacantha erinacea") in stripped


def test_build_gbif_indexes_skips_missing_rank():
    catalog = {"789": {"rank": "", "scientific_name": "Something"}}
    exact, stripped = build_tree.build_gbif_indexes(catalog)
    assert len(exact) == 0


def test_extract_taxa_csv_returns_readable():
    dwca = _make_dwca_bytes([
        {"id": "1", "taxonID": "https://www.inaturalist.org/taxa/1",
         "taxonRank": "SPECIES", "scientificName": "Opuntia humifusa"},
    ])
    f = build_tree.extract_taxa_csv(dwca)
    rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["scientificName"] == "Opuntia humifusa"


# ===========================================================================
# build_mapping
# ===========================================================================

def _catalog_entry(key="123", rank="SPECIES", sci="Opuntia_humifusa"):
    return {key: {"taxon_key": key, "rank": rank, "scientific_name": sci, "common_name": ""}}


def test_build_mapping_exact_match(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "99", "taxonID": "https://www.inaturalist.org/taxa/99",
         "taxonRank": "SPECIES", "scientificName": "Opuntia humifusa"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["inat_id"] == "99"
    assert rows[0]["match_type"] == "exact"


def test_build_mapping_stripped_match(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry("456", "VARIETY", "Opuntia_polyacantha_var._erinacea")
    dwca = _make_dwca_bytes([
        {"id": "77", "taxonID": "", "taxonRank": "VARIETY",
         "scientificName": "Opuntia polyacantha erinacea"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["match_type"] == "stripped"


def test_build_mapping_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "55", "taxonID": "", "taxonRank": "SPECIES", "scientificName": "Completely different"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


def test_build_mapping_conflict_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "11", "taxonID": "", "taxonRank": "SPECIES", "scientificName": "Opuntia humifusa"},
        {"id": "22", "taxonID": "", "taxonRank": "SPECIES", "scientificName": "Opuntia humifusa"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["inat_id"] == "11"


def test_build_mapping_genus_homonym_disambiguated_by_family(tmp_path, monkeypatch):
    """Regression test for #111: plant genus Townsendia (Asteraceae) must not
    match the unrelated robberfly genus Townsendia (Asilidae) just because
    both share a rank+name — the immediate parent (family) must agree too."""
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = {
        "1": {
            "taxon_key": "1", "rank": "GENUS", "scientific_name": "Townsendia", "common_name": "",
            "path": "Plantae_100/Asteraceae_200/Townsendia_1",
        },
    }
    dwca = _make_dwca_bytes(
        [
            {"id": "501", "taxonID": "", "taxonRank": "GENUS", "scientificName": "Townsendia", "family": "Asilidae"},
            {"id": "502", "taxonID": "", "taxonRank": "GENUS", "scientificName": "Townsendia", "family": "Asteraceae"},
        ],
        fieldnames=["id", "taxonID", "taxonRank", "scientificName", "family"],
    )
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["inat_id"] == "502"


def test_build_mapping_skips_unmapped_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "11", "taxonID": "", "taxonRank": "GENUS", "scientificName": "Opuntia humifusa"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


def test_build_mapping_skips_empty_inat_id(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "", "taxonID": "", "taxonRank": "SPECIES", "scientificName": "Opuntia humifusa"},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


# ===========================================================================
# apply_mapping
# ===========================================================================

def test_apply_mapping_sets_inat_id(tmp_path, monkeypatch):
    mp = tmp_path / "mapping.csv"
    mp.write_text(
        "gbif_taxon_key,inat_id,inat_taxon_url,rank,scientific_name,match_type\n"
        "123,99,https://www.inaturalist.org/taxa/99,SPECIES,opuntia humifusa,exact\n"
    )
    monkeypatch.setattr(build_tree, "MAPPING_PATH", mp)
    catalog = _catalog_entry()
    count = build_tree.apply_mapping(catalog)
    assert count == 1
    assert catalog["123"]["inat_id"] == "99"
    assert catalog["123"]["inat_taxon_url"] == "https://www.inaturalist.org/taxa/99"


def test_apply_mapping_skips_missing_catalog_key(tmp_path, monkeypatch):
    mp = tmp_path / "mapping.csv"
    mp.write_text(
        "gbif_taxon_key,inat_id,inat_taxon_url,rank,scientific_name,match_type\n"
        "999,55,,SPECIES,something,exact\n"
    )
    monkeypatch.setattr(build_tree, "MAPPING_PATH", mp)
    count = build_tree.apply_mapping(_catalog_entry())
    assert count == 0


def test_apply_mapping_skips_empty_ids(tmp_path, monkeypatch):
    mp = tmp_path / "mapping.csv"
    mp.write_text(
        "gbif_taxon_key,inat_id,inat_taxon_url,rank,scientific_name,match_type\n"
        ",,,,\n"
    )
    monkeypatch.setattr(build_tree, "MAPPING_PATH", mp)
    count = build_tree.apply_mapping(_catalog_entry())
    assert count == 0


def test_apply_mapping_no_inat_taxon_url(tmp_path, monkeypatch):
    mp = tmp_path / "mapping.csv"
    mp.write_text(
        "gbif_taxon_key,inat_id,inat_taxon_url,rank,scientific_name,match_type\n"
        "123,99,,SPECIES,opuntia humifusa,exact\n"
    )
    monkeypatch.setattr(build_tree, "MAPPING_PATH", mp)
    catalog = _catalog_entry()
    build_tree.apply_mapping(catalog)
    assert "inat_taxon_url" not in catalog["123"]


# ===========================================================================
# _gbif_species_search / _resolve_col_highertaxon_key / fetch_col_vernacular
# ===========================================================================

def _json_urlopen(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return _make_cm(resp)


def test_gbif_species_search_returns_parsed_json():
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen({"count": 1, "results": []})):
        result = build_tree._gbif_species_search({"q": "Opuntia"})
    assert result == {"count": 1, "results": []}


def test_gbif_species_search_retries_transient_network_error():
    from urllib.error import URLError

    with patch(
        "scripts.build_tree.urlopen",
        side_effect=[URLError("SSL: UNEXPECTED_EOF_WHILE_READING"), _json_urlopen({"count": 1, "results": []})],
    ), patch("scripts.build_tree.time.sleep"):
        result = build_tree._gbif_species_search({"q": "Opuntia"})
    assert result == {"count": 1, "results": []}


def test_gbif_species_search_gives_up_after_max_retries(monkeypatch):
    from urllib.error import URLError

    monkeypatch.setattr(build_tree, "COL_API_MAX_RETRIES", 2)
    with patch(
        "scripts.build_tree.urlopen",
        side_effect=[URLError("boom"), URLError("boom again")],
    ), patch("scripts.build_tree.time.sleep"):
        with pytest.raises(URLError):
            build_tree._gbif_species_search({"q": "Opuntia"})


def test_gbif_species_search_does_not_retry_http_error():
    from urllib.error import HTTPError

    http_err = HTTPError("url", 400, "Bad Request", {}, None)
    with patch("scripts.build_tree.urlopen", side_effect=[http_err, _json_urlopen({"count": 1, "results": []})]), \
         patch("scripts.build_tree.time.sleep") as mock_sleep:
        with pytest.raises(HTTPError):
            build_tree._gbif_species_search({"q": "Opuntia"})
    mock_sleep.assert_not_called()


def test_resolve_col_highertaxon_key_matches_taxon_id():
    payload = {"results": [
        {"taxonID": "OTHER", "key": 111},
        {"taxonID": "7HS", "key": 295935017},
    ]}
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(payload)):
        result = build_tree._resolve_col_highertaxon_key("Cactaceae", "7HS")
    assert result == 295935017


def test_resolve_col_highertaxon_key_no_match_returns_none():
    payload = {"results": [{"taxonID": "OTHER", "key": 111}]}
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(payload)):
        assert build_tree._resolve_col_highertaxon_key("Cactaceae", "7HS") is None


def test_fetch_col_vernacular_missing_root_returns_empty():
    assert build_tree.fetch_col_vernacular({}) == {}


def test_fetch_col_vernacular_unresolvable_root_returns_empty(monkeypatch):
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: None)
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("7HS",))
    catalog = {"7HS": {"scientific_name": "Cactaceae"}}
    assert build_tree.fetch_col_vernacular(catalog) == {}


def test_fetch_col_vernacular_single_page(monkeypatch):
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: 295935017)
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("7HS",))
    page = {
        "count": 1,
        "endOfRecords": True,
        "results": [
            {
                "taxonID": "3BHGW",
                "vernacularNames": [
                    {"vernacularName": "Ball Cactus", "language": "eng"},
                    {"vernacularName": "biznaga", "language": "spa"},
                ],
            },
        ],
    }
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(page)), \
         patch("scripts.build_tree.time.sleep"):
        catalog = {"7HS": {"scientific_name": "Cactaceae"}}
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"3BHGW": ["Ball Cactus"]}


def test_fetch_col_vernacular_skips_taxa_with_no_english_names(monkeypatch):
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: 295935017)
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("7HS",))
    page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "3BHGW", "vernacularNames": [
            {"vernacularName": "biznaga", "language": "spa"},
        ]}],
    }
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(page)), \
         patch("scripts.build_tree.time.sleep"):
        catalog = {"7HS": {"scientific_name": "Cactaceae"}}
        assert build_tree.fetch_col_vernacular(catalog) == {}


def test_fetch_col_vernacular_paginates_until_end_of_records(monkeypatch):
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: 295935017)
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("7HS",))
    monkeypatch.setattr(build_tree, "COL_VERNACULAR_PAGE_SIZE", 1)
    page1 = {
        "count": 2,
        "endOfRecords": False,
        "results": [{"taxonID": "A1", "vernacularNames": [{"vernacularName": "Name A", "language": "en"}]}],
    }
    page2 = {
        "count": 2,
        "endOfRecords": True,
        "results": [{"taxonID": "A2", "vernacularNames": [{"vernacularName": "Name B", "language": "en"}]}],
    }
    with patch(
        "scripts.build_tree.urlopen",
        side_effect=[_json_urlopen(page1), _json_urlopen(page2)],
    ), patch("scripts.build_tree.time.sleep"):
        catalog = {"7HS": {"scientific_name": "Cactaceae"}}
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"A1": ["Name A"], "A2": ["Name B"]}


def test_fetch_col_vernacular_splits_oversized_subtree(monkeypatch):
    # GBIF's /v1/species/search rejects offset > 100,000 — a root whose
    # accepted-usage count exceeds COL_SAFE_SUBTREE_LIMIT must be split into
    # its catalog children instead of paginated as one flat scan.
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("P",))
    monkeypatch.setattr(
        build_tree,
        "_resolve_col_highertaxon_key",
        lambda name, tid: {"Plantae": 1, "GenusA": 2, "GenusB": 3}[name],
    )
    catalog = {
        "P": {"scientific_name": "Plantae", "path": "Plantae_P"},
        "A": {"scientific_name": "GenusA", "path": "Plantae_P/GenusA_A"},
        "B": {"scientific_name": "GenusB", "path": "Plantae_P/GenusB_B"},
    }
    root_page = {
        "count": 200_000,
        "endOfRecords": True,
        # Must never be parsed into the result — the root page is discarded
        # once a split is triggered.
        "results": [{"taxonID": "SHOULD_NOT_APPEAR", "vernacularNames": [{"vernacularName": "nope", "language": "en"}]}],
    }
    child_a_page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "CA1", "vernacularNames": [{"vernacularName": "Child A Name", "language": "en"}]}],
    }
    child_b_page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "CB1", "vernacularNames": [{"vernacularName": "Child B Name", "language": "en"}]}],
    }
    with patch(
        "scripts.build_tree.urlopen",
        side_effect=[_json_urlopen(root_page), _json_urlopen(child_a_page), _json_urlopen(child_b_page)],
    ), patch("scripts.build_tree.time.sleep"):
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"CA1": ["Child A Name"], "CB1": ["Child B Name"]}


def test_fetch_col_vernacular_merges_two_independent_roots(monkeypatch):
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("P", "F"))
    monkeypatch.setattr(
        build_tree,
        "_resolve_col_highertaxon_key",
        lambda name, tid: {"Plantae": 1, "Fungi": 2}[name],
    )
    catalog = {
        "P": {"scientific_name": "Plantae", "path": "Plantae_P"},
        "F": {"scientific_name": "Fungi", "path": "Fungi_F"},
    }
    plantae_page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "PL1", "vernacularNames": [{"vernacularName": "A Plant", "language": "en"}]}],
    }
    fungi_page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "FU1", "vernacularNames": [{"vernacularName": "A Fungus", "language": "en"}]}],
    }
    with patch(
        "scripts.build_tree.urlopen",
        side_effect=[_json_urlopen(plantae_page), _json_urlopen(fungi_page)],
    ), patch("scripts.build_tree.time.sleep"):
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"PL1": ["A Plant"], "FU1": ["A Fungus"]}


def test_fetch_col_vernacular_one_root_missing_still_returns_other(monkeypatch, capsys):
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("P", "MISSING"))
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: 1)
    catalog = {"P": {"scientific_name": "Plantae", "path": "Plantae_P"}}
    page = {
        "count": 1,
        "endOfRecords": True,
        "results": [{"taxonID": "PL1", "vernacularNames": [{"vernacularName": "A Plant", "language": "en"}]}],
    }
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(page)), \
         patch("scripts.build_tree.time.sleep"):
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"PL1": ["A Plant"]}
    assert "MISSING" in capsys.readouterr().out


def test_fetch_col_vernacular_scans_anyway_when_oversized_with_no_children(monkeypatch):
    # No catalog children to split by (e.g. a leaf-ish node) — falls back to
    # scanning the oversized subtree directly rather than silently dropping it.
    monkeypatch.setattr(build_tree.CONFIG, "taxonomy_roots", ("P",))
    monkeypatch.setattr(build_tree, "_resolve_col_highertaxon_key", lambda name, tid: 1)
    catalog = {"P": {"scientific_name": "Plantae", "path": "Plantae_P"}}
    page = {
        "count": 200_000,
        "endOfRecords": True,
        "results": [{"taxonID": "X1", "vernacularNames": [{"vernacularName": "Name X", "language": "en"}]}],
    }
    with patch("scripts.build_tree.urlopen", return_value=_json_urlopen(page)), \
         patch("scripts.build_tree.time.sleep"):
        result = build_tree.fetch_col_vernacular(catalog)
    assert result == {"X1": ["Name X"]}


# ===========================================================================
# load_inat_vernacular
# ===========================================================================

def _make_inat_vernacular_dwca(entries: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_buf = io.StringIO()
        w = csv.DictWriter(csv_buf, fieldnames=["id", "vernacularName", "language"])
        w.writeheader()
        for e in entries:
            w.writerow({f: e.get(f, "") for f in ["id", "vernacularName", "language"]})
        zf.writestr("VernacularNames-0.csv", csv_buf.getvalue())
    return buf.getvalue()


def test_load_inat_vernacular_english():
    dwca = _make_inat_vernacular_dwca([
        {"id": "99", "vernacularName": "Eastern Prickly Pear", "language": "en"},
        {"id": "99", "vernacularName": "Devil's Tongue", "language": "en"},
    ])
    result = build_tree.load_inat_vernacular(dwca)
    assert "Eastern Prickly Pear" in result["99"]
    assert "Devil's Tongue" in result["99"]


def test_load_inat_vernacular_filters_non_english():
    dwca = _make_inat_vernacular_dwca([
        {"id": "99", "vernacularName": "Kakteen", "language": "de"},
    ])
    assert "99" not in build_tree.load_inat_vernacular(dwca)


def test_load_inat_vernacular_deduplicates():
    dwca = _make_inat_vernacular_dwca([
        {"id": "99", "vernacularName": "Prickly Pear", "language": "en"},
        {"id": "99", "vernacularName": "Prickly Pear", "language": "en"},
    ])
    result = build_tree.load_inat_vernacular(dwca)
    assert result["99"].count("Prickly Pear") == 1


def test_load_inat_vernacular_ignores_non_vernacular_files():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("taxa.csv", "id,scientificName\n99,Opuntia\n")
        zf.writestr("VernacularNames-0.csv", "id,vernacularName,language\n99,Pear,en\n")
    result = build_tree.load_inat_vernacular(buf.getvalue())
    assert "99" in result


# ===========================================================================
# apply_names
# ===========================================================================

def test_apply_names_inat_priority():
    catalog = {"123": {"inat_id": "99", "common_name": ""}}
    inat_map = {"99": ["Prickly Pear", "Devil's Tongue"]}
    gbif_map = {"123": ["Eastern Prickly Pear"]}
    count = build_tree.apply_names(catalog, inat_map, gbif_map)
    assert count == 1
    assert catalog["123"]["common_name"] == "Prickly Pear"
    names = catalog["123"]["vernacular_names"]
    assert "Eastern Prickly Pear" in names
    assert "Prickly Pear" in names


def test_apply_names_gbif_only():
    catalog = {"123": {"common_name": ""}}
    count = build_tree.apply_names(catalog, {}, {"123": ["Oak"]})
    assert count == 1
    assert catalog["123"]["common_name"] == "Oak"


def test_apply_names_no_names():
    catalog = {"123": {"common_name": ""}}
    assert build_tree.apply_names(catalog, {}, {}) == 0


def test_apply_names_deduplicates():
    catalog = {"123": {"inat_id": "99", "common_name": ""}}
    build_tree.apply_names(catalog, {"99": ["Oak"]}, {"123": ["Oak"]})
    assert catalog["123"]["vernacular_names"].count("Oak") == 1


def test_apply_names_gbif_synonym_key_fallback():
    # A reclassified species still carries its old COL XR taxon key in
    # gbif_synonym_keys — vernacular names attached to that old key should
    # still be found.
    catalog = {"123": {"common_name": "", "gbif_synonym_keys": ["456"]}}
    count = build_tree.apply_names(catalog, {}, {"456": ["Old Name"]})
    assert count == 1
    assert catalog["123"]["common_name"] == "Old Name"


# ===========================================================================
# _clean / fetch_taxa_batch / extract_preferred_image_metadata / apply_inat_preferred
# ===========================================================================

def test_clean_none_string():
    assert build_tree._clean("none") == ""


def test_clean_null_string():
    assert build_tree._clean("null") == ""


def test_clean_normal():
    assert build_tree._clean("  Hello  ") == "Hello"


def test_clean_empty():
    assert build_tree._clean("") == ""


def test_clean_none_value():
    assert build_tree._clean(None) == ""


def test_fetch_taxa_batch_returns_results():
    resp = MagicMock()
    resp.read.return_value = json.dumps({"results": [{"id": "1"}]}).encode()
    with patch("scripts.build_tree.urlopen", return_value=_make_cm(resp)):
        result = build_tree.fetch_taxa_batch(["1"])
    assert result[0]["id"] == "1"


def test_fetch_taxa_batch_empty_results():
    resp = MagicMock()
    resp.read.return_value = json.dumps({"results": []}).encode()
    with patch("scripts.build_tree.urlopen", return_value=_make_cm(resp)):
        assert build_tree.fetch_taxa_batch(["99"]) == []


def test_extract_preferred_image_metadata_no_photo():
    assert build_tree.extract_preferred_image_metadata({}) == {}


def test_extract_preferred_image_metadata_full():
    payload = {"default_photo": {
        "id": "12345",
        "original_url": "https://example.com/photo.jpg",
        "license_code": "cc-by",
        "attribution_name": "Alice",
        "attribution": "© Alice",
    }}
    result = build_tree.extract_preferred_image_metadata(payload)
    assert result["inat_preferred_image"] == "https://example.com/photo.jpg"
    assert result["inat_preferred_image_license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "12345" in result["inat_preferred_image_references"]


def test_extract_preferred_image_metadata_fallback_url():
    payload = {"default_photo": {"id": "99", "large_url": "https://example.com/large.jpg", "license_code": "cc0"}}
    result = build_tree.extract_preferred_image_metadata(payload)
    assert result["inat_preferred_image"] == "https://example.com/large.jpg"


def test_extract_preferred_image_metadata_no_url():
    assert build_tree.extract_preferred_image_metadata({"default_photo": {"id": "99"}}) == {}


def test_extract_preferred_image_metadata_no_photo_id():
    payload = {"default_photo": {"original_url": "https://example.com/img.jpg", "license_code": "cc0"}}
    result = build_tree.extract_preferred_image_metadata(payload)
    assert result["inat_preferred_image_references"] == ""


def test_apply_inat_preferred_updates_name_and_image():
    catalog = {"k1": {"inat_id": "99", "common_name": ""}}
    inat_to_taxa = {"99": ["k1"]}
    results = [{"id": "99", "preferred_common_name": "Prickly Pear", "default_photo": {
        "id": "111", "original_url": "https://example.com/img.jpg",
        "license_code": "cc0", "attribution_name": "Alice", "attribution": "© Alice",
    }}]
    names, images = build_tree.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert names == 1
    assert images == 1
    assert catalog["k1"]["inat_preferred_common_name"] == "Prickly Pear"


def test_apply_inat_preferred_skips_already_set():
    catalog = {"k1": {
        "inat_id": "99",
        "inat_preferred_common_name": "Already set",
        "inat_preferred_image": "https://example.com/img.jpg",
    }}
    inat_to_taxa = {"99": ["k1"]}
    results = [{"id": "99", "preferred_common_name": "New Name",
                "default_photo": {"id": "X", "original_url": "https://example.com/other.jpg"}}]
    names, images = build_tree.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert names == 0
    assert images == 0


def test_apply_inat_preferred_no_inat_id_in_result():
    names, images = build_tree.apply_inat_preferred({}, {}, [{"preferred_common_name": "Oak"}])
    assert names == 0
    assert images == 0


def test_apply_inat_preferred_taxon_key_not_in_catalog():
    catalog = {}
    inat_to_taxa = {"99": ["missing_key"]}
    results = [{"id": "99", "preferred_common_name": "Oak", "default_photo": None}]
    names, images = build_tree.apply_inat_preferred(catalog, inat_to_taxa, results)
    assert names == 0
    assert images == 0


# ===========================================================================
# run_inat_preferred
# ===========================================================================

def test_run_inat_preferred_empty_catalog():
    assert build_tree.run_inat_preferred({}) == (0, 0)


def test_run_inat_preferred_skips_complete_entries():
    catalog = {"k1": {
        "inat_id": "99",
        "inat_preferred_common_name": "Set",
        "inat_preferred_image": "https://example.com/img.jpg",
    }}
    with patch("scripts.build_tree.time.sleep"):
        names, images = build_tree.run_inat_preferred(catalog)
    assert names == 0
    assert images == 0


def test_run_inat_preferred_fetches_and_applies(monkeypatch):
    catalog = {"k1": {"inat_id": "99", "common_name": ""}}
    monkeypatch.setattr(build_tree, "fetch_taxa_batch",
                        lambda ids, **kw: [{"id": "99", "preferred_common_name": "Pear",
                                            "default_photo": None}])
    monkeypatch.setattr(build_tree, "INAT_RATE_LIMIT", 10000.0)
    with patch("scripts.build_tree.time.sleep"):
        names, _ = build_tree.run_inat_preferred(catalog)
    assert names == 1


def test_run_inat_preferred_skips_no_inat_id():
    catalog = {"k1": {"common_name": ""}}  # no inat_id key
    with patch("scripts.build_tree.time.sleep"):
        names, images = build_tree.run_inat_preferred(catalog)
    assert names == 0
    assert images == 0


def test_run_inat_preferred_handles_batch_error(monkeypatch):
    catalog = {"k1": {"inat_id": "99", "common_name": ""}}
    monkeypatch.setattr(build_tree, "fetch_taxa_batch",
                        lambda ids, **kw: (_ for _ in ()).throw(RuntimeError("err")))
    monkeypatch.setattr(build_tree, "INAT_RATE_LIMIT", 10000.0)
    with patch("scripts.build_tree.time.sleep"):
        names, images = build_tree.run_inat_preferred(catalog)
    assert names == 0


def test_run_inat_preferred_progress_print(monkeypatch, capsys):
    catalog = {str(i): {"inat_id": str(i), "common_name": ""} for i in range(10)}
    monkeypatch.setattr(build_tree, "fetch_taxa_batch", lambda ids, **kw: [])
    monkeypatch.setattr(build_tree, "INAT_BATCH_SIZE", 1)
    monkeypatch.setattr(build_tree, "INAT_RATE_LIMIT", 10000.0)
    with patch("scripts.build_tree.time.sleep"):
        build_tree.run_inat_preferred(catalog)
    assert "10/" in capsys.readouterr().out


# ===========================================================================
# update_name_index
# ===========================================================================

def test_update_name_index_adds_common_name():
    payload = {
        "catalog": {"123": {"common_name": "Oak"}},
        "combined_name_index": {},
    }
    added = build_tree.update_name_index(payload)
    assert added == 1
    assert "oak" in payload["combined_name_index"]


def test_update_name_index_adds_preferred_name():
    payload = {
        "catalog": {"123": {"inat_preferred_common_name": "White Oak"}},
        "combined_name_index": {},
    }
    build_tree.update_name_index(payload)
    assert "white oak" in payload["combined_name_index"]


def test_update_name_index_adds_vernacular_names():
    payload = {
        "catalog": {"123": {"vernacular_names": ["Sessile Oak", "Durmast Oak"]}},
        "combined_name_index": {},
    }
    build_tree.update_name_index(payload)
    assert "sessile oak" in payload["combined_name_index"]
    assert "durmast oak" in payload["combined_name_index"]


def test_update_name_index_no_duplicate():
    payload = {
        "catalog": {"123": {"common_name": "Oak"}},
        "combined_name_index": {"oak": ["123"]},
    }
    assert build_tree.update_name_index(payload) == 0


# ===========================================================================
# _license_score / _is_usable_license / _image_quality
# ===========================================================================

def test_license_score_cc0():
    assert build_tree._license_score("CC0") == 0


def test_license_score_publicdomain():
    assert build_tree._license_score("publicdomain") == 0


def test_license_score_by():
    assert build_tree._license_score("https://creativecommons.org/licenses/by/4.0/") == 1


def test_license_score_by_sa():
    assert build_tree._license_score("cc by-sa") == 2


def test_license_score_by_nc():
    assert build_tree._license_score("https://creativecommons.org/licenses/by-nc/4.0/") == 3


def test_license_score_by_nc_sa():
    assert build_tree._license_score("cc by-nc-sa") == 4


def test_license_score_unknown():
    assert build_tree._license_score("all rights reserved") == 99


def test_is_usable_license_cc0():
    assert build_tree._is_usable_license("CC0") is True


def test_is_usable_license_cc_by():
    assert build_tree._is_usable_license("cc by") is True


def test_is_usable_license_proprietary():
    assert build_tree._is_usable_license("all rights reserved") is False


def test_image_quality_best():
    assert build_tree._image_quality("cc0", "alive", "organism", "flowers") == (0, 0, 0, 0)


def test_image_quality_dead():
    score = build_tree._image_quality("cc0", "dead", "organism", "flowers")
    assert score[0] == 2


def test_image_quality_unknown_vitality():
    score = build_tree._image_quality("cc0", "", "organism", "flowers")
    assert score[0] == 1


def test_image_quality_bad_evidence():
    assert build_tree._image_quality("cc0", "alive", "track", "")[1] == 3


def test_image_quality_okay_evidence():
    assert build_tree._image_quality("cc0", "alive", "gall", "")[1] == 2


def test_image_quality_other_evidence():
    assert build_tree._image_quality("cc0", "alive", "cast", "")[1] == 1


def test_image_quality_empty_evidence():
    assert build_tree._image_quality("cc0", "alive", "", "")[1] == 1


def test_image_quality_rcs_fruits():
    assert build_tree._image_quality("cc0", "alive", "", "fruits")[2] == 1


def test_image_quality_rcs_flower_buds():
    assert build_tree._image_quality("cc0", "alive", "", "flower buds")[2] == 2


def test_image_quality_rcs_other():
    assert build_tree._image_quality("cc0", "alive", "", "vegetative")[2] == 3


# ===========================================================================
# _build_gbif_to_taxon / _build_gbif_images / run_gbif_backup
# ===========================================================================

SPECIES_CATALOG = {"123": {"scientific_name": "Foo_bar"}}
SUBSPECIES_CATALOG = {"456": {"scientific_name": "Foo_bar_subsp._baz"}}


def test_build_gbif_to_taxon_species(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar", "taxonRank": "SPECIES"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    result = build_tree._build_gbif_to_taxon(SPECIES_CATALOG)
    assert "g1" in result
    assert result["g1"][0] == "123"


def test_build_gbif_to_taxon_subspecies_uses_taxon_key(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g2", "scientificName": "Foo bar subsp. baz", "taxonRank": "SUBSPECIES"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    result = build_tree._build_gbif_to_taxon(SUBSPECIES_CATALOG)
    assert result["g2"][0] == "456"


def test_build_gbif_to_taxon_skips_uncatalogued(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g3", "scientificName": "Unmatched species", "taxonRank": "SPECIES"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    assert "g3" not in build_tree._build_gbif_to_taxon(SPECIES_CATALOG)


def test_build_gbif_to_taxon_skips_empty_gbifid(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "", "scientificName": "Foo bar", "taxonRank": "SPECIES"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    assert len(build_tree._build_gbif_to_taxon(SPECIES_CATALOG)) == 0


def test_build_gbif_to_taxon_skips_ambiguous_homonym(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar", "taxonRank": "SPECIES"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    ambiguous_catalog = {
        "123": {"scientific_name": "Foo_bar"},
        "789": {"scientific_name": "Foo_bar"},
    }
    assert "g1" not in build_tree._build_gbif_to_taxon(ambiguous_catalog)


def test_build_gbif_to_taxon_parses_dynamic_properties(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    dp = json.dumps({"evidenceOfPresence": "organism"})
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar",
         "taxonRank": "SPECIES", "dynamicProperties": dp},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    result = build_tree._build_gbif_to_taxon(SPECIES_CATALOG)
    assert result["g1"][2] == "organism"


def test_build_gbif_to_taxon_parses_evidence_list(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    dp = json.dumps({"evidenceOfPresence": ["organism", "track"]})
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar",
         "taxonRank": "SPECIES", "dynamicProperties": dp},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    result = build_tree._build_gbif_to_taxon(SPECIES_CATALOG)
    assert "organism" in result["g1"][2]


def test_build_gbif_images_basic(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage",
         "identifier": "https://img.example.com/1.jpg", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    result = build_tree._build_gbif_images({"g1": ("123", "alive", "organism", "flowers")})
    assert "123" in result
    assert result["123"]["gbif_backup_image"] == "https://img.example.com/1.jpg"


def test_build_gbif_images_skips_unusable_license(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage",
         "identifier": "https://img.example.com/1.jpg", "license": "all rights reserved"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    assert "123" not in build_tree._build_gbif_images({"g1": ("123", "", "", "")})


def test_build_gbif_images_skips_non_image(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "Sound",
         "identifier": "https://audio.example.com/1.mp3", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    assert "123" not in build_tree._build_gbif_images({"g1": ("123", "", "", "")})


def test_build_gbif_images_accepts_image_format(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "", "format": "image/jpeg",
         "identifier": "https://img.example.com/1.jpg", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    result = build_tree._build_gbif_images({"g1": ("123", "", "", "")})
    assert "123" in result


def test_build_gbif_images_keeps_better_score(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage",
         "identifier": "https://img.example.com/bysa.jpg", "license": "CC BY-SA"},
        {"gbifID": "g2", "type": "StillImage",
         "identifier": "https://img.example.com/cc0.jpg", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    gbif_to_taxon = {
        "g1": ("123", "alive", "organism", "flowers"),
        "g2": ("123", "alive", "organism", "flowers"),
    }
    result = build_tree._build_gbif_images(gbif_to_taxon)
    assert result["123"]["gbif_backup_image"] == "https://img.example.com/cc0.jpg"


def test_run_gbif_backup_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", tmp_path / "occurrence.txt")
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", tmp_path / "multimedia.txt")
    assert build_tree.run_gbif_backup({"123": {}}) == (0, 0)


def test_run_gbif_backup_updates_catalog(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    mm = tmp_path / "multimedia.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar", "taxonRank": "SPECIES"},
    ]))
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage",
         "identifier": "https://img.example.com/1.jpg", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    catalog = {"123": {"scientific_name": "Foo_bar"}}
    assert build_tree.run_gbif_backup(catalog) == (1, 0)
    assert "gbif_backup_image" in catalog["123"]


# ===========================================================================
# rebuild_index
# ===========================================================================

def test_rebuild_index(tmp_path, monkeypatch):
    catalog_path = tmp_path / "taxon_catalog.pkl"
    payload = {
        "catalog": {"123": {"common_name": "Oak"}},
        "combined_name_index": {},
    }
    catalog_path.write_bytes(pickle.dumps(payload))
    monkeypatch.setattr(build_tree, "CATALOG_PATH", catalog_path)
    build_tree.rebuild_index()
    updated = pickle.loads(catalog_path.read_bytes())
    assert "oak" in updated["combined_name_index"]


# ===========================================================================
# Edge-case coverage for remaining uncovered lines
# ===========================================================================

def test_build_mapping_skips_empty_scientific_name(tmp_path, monkeypatch):
    monkeypatch.setattr(build_tree, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(build_tree, "MAPPING_PATH", tmp_path / "mapping.csv")
    catalog = _catalog_entry()
    dwca = _make_dwca_bytes([
        {"id": "11", "taxonID": "", "taxonRank": "SPECIES", "scientificName": ""},
    ])
    build_tree.build_mapping(catalog, dwca)
    with open(tmp_path / "mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


def test_update_name_index_skips_empty_normalized_key():
    # "_" passes the `if raw` guard but normalizes to "" via _normalize_index_key
    payload = {
        "catalog": {"123": {"common_name": "_"}},
        "combined_name_index": {},
    }
    assert build_tree.update_name_index(payload) == 0


def test_build_gbif_to_taxon_invalid_dynamic_properties(tmp_path, monkeypatch):
    occ = tmp_path / "occurrence.txt"
    occ.write_text(_make_occurrence_tsv([
        {"gbifID": "g1", "scientificName": "Foo bar",
         "taxonRank": "SPECIES", "dynamicProperties": "not-json"},
    ]))
    monkeypatch.setattr(build_tree, "OCCURRENCE_PATH", occ)
    result = build_tree._build_gbif_to_taxon(SPECIES_CATALOG)
    assert result["g1"][2] == ""  # evidence remains empty on parse error


def test_build_gbif_images_skips_unmatched_gbifid(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "unmatched", "type": "StillImage",
         "identifier": "https://img.example.com/x.jpg", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    result = build_tree._build_gbif_images({})  # empty mapping → nothing matches
    assert result == {}


def test_build_gbif_images_skips_empty_identifier(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage", "identifier": "", "license": "CC0"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    assert "123" not in build_tree._build_gbif_images({"g1": ("123", "", "", "")})


def test_build_gbif_images_keeps_first_when_second_worse(tmp_path, monkeypatch):
    mm = tmp_path / "multimedia.txt"
    mm.write_text(_make_multimedia_tsv([
        {"gbifID": "g1", "type": "StillImage",
         "identifier": "https://img.example.com/cc0.jpg", "license": "CC0"},
        {"gbifID": "g2", "type": "StillImage",
         "identifier": "https://img.example.com/bysa.jpg", "license": "CC BY-SA"},
    ]))
    monkeypatch.setattr(build_tree, "MULTIMEDIA_PATH", mm)
    gbif_to_taxon = {
        "g1": ("123", "alive", "organism", "flowers"),
        "g2": ("123", "alive", "organism", "flowers"),
    }
    result = build_tree._build_gbif_images(gbif_to_taxon)
    assert result["123"]["gbif_backup_image"] == "https://img.example.com/cc0.jpg"


# ===========================================================================
# infer_species_inat_ids
# ===========================================================================

def _make_dwca_with_parents(taxa_rows: list[dict]) -> bytes:
    fieldnames = ["id", "taxonID", "taxonRank", "scientificName", "parentNameUsageID"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in taxa_rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})
        zf.writestr("taxa.csv", csv_buf.getvalue())
        zf.writestr("VernacularNames-0.csv", "id,vernacularName,language\n")
    return buf.getvalue()


def test_infer_species_inat_ids_basic():
    # Species 100 has no inat_id; varieties 200/201 both point to iNat parent 99
    catalog = {
        "100": {"taxon_key": "100", "path": "Plantae_1/Cactaceae_2/Foo_bar_100",
                "scientific_name": "Foo_bar", "rank": "SPECIES", "common_name": ""},
        "200": {"taxon_key": "200", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._a_200",
                "scientific_name": "Foo_bar_var._a", "rank": "VARIETY", "common_name": "",
                "inat_id": "1001"},
        "201": {"taxon_key": "201", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._b_201",
                "scientific_name": "Foo_bar_var._b", "rank": "VARIETY", "common_name": "",
                "inat_id": "1002"},
    }
    dwca = _make_dwca_with_parents([
        {"id": "99",   "taxonRank": "species", "scientificName": "OtherGenus bar",
         "parentNameUsageID": "https://www.inaturalist.org/taxa/50"},
        {"id": "1001", "taxonRank": "variety", "scientificName": "OtherGenus bar var. a",
         "parentNameUsageID": "https://www.inaturalist.org/taxa/99"},
        {"id": "1002", "taxonRank": "variety", "scientificName": "OtherGenus bar var. b",
         "parentNameUsageID": "https://www.inaturalist.org/taxa/99"},
    ])
    updated = build_tree.infer_species_inat_ids(catalog, dwca)
    assert updated == 1
    assert catalog["100"]["inat_id"] == "99"
    assert catalog["100"]["inat_taxon_url"] == "https://www.inaturalist.org/taxa/99"


def test_infer_species_inat_ids_skips_when_already_mapped():
    catalog = {
        "100": {"taxon_key": "100", "path": "Plantae_1/Cactaceae_2/Foo_bar_100",
                "scientific_name": "Foo_bar", "rank": "SPECIES", "common_name": "",
                "inat_id": "existing"},
        "200": {"taxon_key": "200", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._a_200",
                "scientific_name": "Foo_bar_var._a", "rank": "VARIETY", "common_name": "",
                "inat_id": "1001"},
    }
    dwca = _make_dwca_with_parents([
        {"id": "1001", "parentNameUsageID": "https://www.inaturalist.org/taxa/99"},
    ])
    updated = build_tree.infer_species_inat_ids(catalog, dwca)
    assert updated == 0
    assert catalog["100"]["inat_id"] == "existing"


def test_infer_species_inat_ids_no_infer_when_children_disagree():
    # Two children point to different iNat parents → ambiguous, skip
    catalog = {
        "100": {"taxon_key": "100", "path": "Plantae_1/Cactaceae_2/Foo_bar_100",
                "scientific_name": "Foo_bar", "rank": "SPECIES", "common_name": ""},
        "200": {"taxon_key": "200", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._a_200",
                "scientific_name": "Foo_bar_var._a", "rank": "VARIETY", "common_name": "",
                "inat_id": "1001"},
        "201": {"taxon_key": "201", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._b_201",
                "scientific_name": "Foo_bar_var._b", "rank": "VARIETY", "common_name": "",
                "inat_id": "1002"},
    }
    dwca = _make_dwca_with_parents([
        {"id": "1001", "parentNameUsageID": "https://www.inaturalist.org/taxa/99"},
        {"id": "1002", "parentNameUsageID": "https://www.inaturalist.org/taxa/98"},  # different!
    ])
    updated = build_tree.infer_species_inat_ids(catalog, dwca)
    assert updated == 0
    assert "inat_id" not in catalog["100"]


def test_infer_species_inat_ids_no_infer_when_no_matched_children():
    catalog = {
        "100": {"taxon_key": "100", "path": "Plantae_1/Cactaceae_2/Foo_bar_100",
                "scientific_name": "Foo_bar", "rank": "SPECIES", "common_name": ""},
        "200": {"taxon_key": "200", "path": "Plantae_1/Cactaceae_2/Foo_bar_100/Foo_bar_var._a_200",
                "scientific_name": "Foo_bar_var._a", "rank": "VARIETY", "common_name": ""},
        # child has no inat_id
    }
    dwca = _make_dwca_with_parents([])
    updated = build_tree.infer_species_inat_ids(catalog, dwca)
    assert updated == 0
