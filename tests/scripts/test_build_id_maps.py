import csv
import io
import json
import pickle
import zipfile
from unittest.mock import MagicMock

import scripts.build_id_maps as bim

MAPPING_FIELDNAMES = [
    "gbif_taxon_key", "inat_id", "inat_taxon_url", "rank", "scientific_name", "match_type",
]

CATALOG = {
    "2923970": {
        "taxon_key": "2923970",
        "scientific_name": "Opuntia_humifusa",
        "rank": "SPECIES",
        "common_name": "",
    },
    "2923968": {
        "taxon_key": "2923968",
        "scientific_name": "Opuntia",
        "rank": "GENUS",
        "common_name": "",
    },
    "3084112": {
        "taxon_key": "3084112",
        "scientific_name": "Echinocereus_triglochidiatus_var._mojavensis",
        "rank": "VARIETY",
        "common_name": "",
    },
}

PAYLOAD = {"catalog": CATALOG, "combined_name_index": {}}


def _make_dwca(rows: list[dict]) -> bytes:
    fieldnames = ["id", "taxonID", "scientificName", "taxonRank"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, "w") as zf:
        zf.writestr("taxa.csv", buf.getvalue())
    return zf_buf.getvalue()


# --- normalize_name ---

def test_normalize_name_basic():
    assert bim.normalize_name("Opuntia humifusa") == "opuntia humifusa"


def test_normalize_name_underscores():
    assert bim.normalize_name("Opuntia_humifusa") == "opuntia humifusa"


def test_normalize_name_hybrid():
    assert bim.normalize_name("Opuntia × humifusa") == "opuntia x humifusa"


def test_normalize_name_empty():
    assert bim.normalize_name("") == ""


# --- strip_infra_markers ---

def test_strip_infra_markers_var():
    assert bim.strip_infra_markers("echinocereus triglochidiatus var. mojavensis") == \
        "echinocereus triglochidiatus mojavensis"


def test_strip_infra_markers_subsp():
    assert bim.strip_infra_markers("opuntia humifusa subsp. austrina") == \
        "opuntia humifusa austrina"


def test_strip_infra_markers_no_marker():
    assert bim.strip_infra_markers("opuntia humifusa") == "opuntia humifusa"


# --- build_gbif_indexes ---

def test_build_gbif_indexes_exact():
    exact, stripped = bim.build_gbif_indexes(CATALOG)
    assert ("SPECIES", "opuntia humifusa") in exact
    assert exact[("SPECIES", "opuntia humifusa")] == ["2923970"]


def test_build_gbif_indexes_genus_excluded():
    exact, _ = bim.build_gbif_indexes(CATALOG)
    assert ("GENUS", "opuntia") in exact
    # genus is not in MAPPING_RANKS so will never be matched, but it IS indexed
    # (the rank filter happens during matching, not indexing)


def test_build_gbif_indexes_variety_stripped():
    _, stripped = bim.build_gbif_indexes(CATALOG)
    assert ("VARIETY", "echinocereus triglochidiatus mojavensis") in stripped


def test_build_gbif_indexes_skip_empty_name():
    catalog = {"999": {"scientific_name": "", "rank": "SPECIES", "common_name": ""}}
    exact, stripped = bim.build_gbif_indexes(catalog)
    assert len(exact) == 0


# --- download_dwca ---

def _make_urlopen_mock(head_etag: str, body: bytes):
    """urlopen mock: HEAD returns etag, GET returns body."""
    def mock_urlopen(req, timeout=30):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        if req.get_method() == "HEAD":
            resp.headers = {"ETag": head_etag}
            resp.read.return_value = b""
        else:
            resp.headers = {"ETag": head_etag}
            resp.read.return_value = body
        return resp
    return mock_urlopen


def test_download_dwca_fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(bim, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bim, "INAT_DWCA_CACHE", tmp_path / "inat_dwca.zip")
    monkeypatch.setattr(bim, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    fake_data = b"fake zip content"
    monkeypatch.setattr(bim, "urlopen", _make_urlopen_mock('"new-etag"', fake_data))
    result = bim.download_dwca()
    assert result == fake_data
    assert (tmp_path / "inat_dwca.zip").read_bytes() == fake_data
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert state["inat_taxonomy"]["etag"] == '"new-etag"'


def test_download_dwca_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(bim, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bim, "INAT_DWCA_CACHE", tmp_path / "inat_dwca.zip")
    monkeypatch.setattr(bim, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    cached = b"cached zip"
    (tmp_path / "inat_dwca.zip").write_bytes(cached)
    state = {"inat_taxonomy": {"etag": '"same-etag"'}}
    (tmp_path / "sync_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(bim, "urlopen", _make_urlopen_mock('"same-etag"', b"new data"))
    result = bim.download_dwca()
    assert result == cached


# --- extract_taxa_csv ---

def test_extract_taxa_csv():
    dwca = _make_dwca([
        {"id": "123", "taxonID": "https://www.inaturalist.org/taxa/123",
         "scientificName": "Opuntia humifusa", "taxonRank": "SPECIES"},
    ])
    reader = csv.DictReader(bim.extract_taxa_csv(dwca))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["id"] == "123"
    assert rows[0]["scientificName"] == "Opuntia humifusa"


# --- build_mapping ---

def test_build_mapping_exact_match(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    dwca = _make_dwca([
        {"id": "55555", "taxonID": "https://www.inaturalist.org/taxa/55555",
         "scientificName": "Opuntia humifusa", "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["gbif_taxon_key"] == "2923970"
    assert rows[0]["inat_id"] == "55555"
    assert rows[0]["match_type"] == "exact"


def test_build_mapping_stripped_match(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    # Catalog has no infra-marker in the name; iNat row has "var." → stripped fallback needed
    catalog_no_marker = {
        "3084112": {
            "taxon_key": "3084112",
            "scientific_name": "Echinocereus triglochidiatus mojavensis",
            "rank": "VARIETY",
            "common_name": "",
        },
    }
    dwca = _make_dwca([
        {"id": "77777", "taxonID": "https://www.inaturalist.org/taxa/77777",
         "scientificName": "Echinocereus triglochidiatus var. mojavensis", "taxonRank": "VARIETY"},
    ])
    bim.build_mapping(catalog_no_marker, dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["gbif_taxon_key"] == "3084112"
    assert rows[0]["match_type"] == "stripped"


def test_build_mapping_skips_empty_scientific_name(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    dwca = _make_dwca([
        {"id": "99999", "taxonID": "", "scientificName": "", "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_build_mapping_skips_non_leaf_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    dwca = _make_dwca([
        {"id": "99999", "taxonID": "", "scientificName": "Opuntia", "taxonRank": "GENUS"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_build_mapping_skips_no_inat_id(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    dwca = _make_dwca([
        {"id": "", "taxonID": "", "scientificName": "Opuntia humifusa", "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_build_mapping_conflict_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    # Two different iNat IDs claiming the same GBIF key
    dwca = _make_dwca([
        {"id": "11111", "taxonID": "", "scientificName": "Opuntia humifusa",
         "taxonRank": "SPECIES"},
        {"id": "22222", "taxonID": "", "scientificName": "Opuntia humifusa",
         "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    # First match written; second is a conflict and skipped
    assert len(rows) == 1
    assert rows[0]["inat_id"] == "11111"


def test_build_mapping_ambiguous_gbif(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    # Two GBIF entries with the same scientific name -> ambiguous, skip
    dup_catalog = {
        **CATALOG,
        "9999999": {
            "taxon_key": "9999999",
            "scientific_name": "Opuntia_humifusa",
            "rank": "SPECIES",
            "common_name": "",
        },
    }
    dwca = _make_dwca([
        {"id": "55555", "taxonID": "", "scientificName": "Opuntia humifusa",
         "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dup_catalog, dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_build_mapping_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    dwca = _make_dwca([
        {"id": "88888", "taxonID": "", "scientificName": "Completely unknown species",
         "taxonRank": "SPECIES"},
    ])
    bim.build_mapping(dict(CATALOG), dwca)
    with open(tmp_path / "inat_gbif_mapping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


# --- apply_mapping ---

def test_apply_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    with open(tmp_path / "inat_gbif_mapping.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "gbif_taxon_key": "2923970", "inat_id": "55555",
            "inat_taxon_url": "https://www.inaturalist.org/taxa/55555",
            "rank": "SPECIES", "scientific_name": "opuntia humifusa", "match_type": "exact",
        })
    catalog = {k: dict(v) for k, v in CATALOG.items()}
    updated = bim.apply_mapping(catalog)
    assert updated == 1
    assert catalog["2923970"]["inat_id"] == "55555"
    assert catalog["2923970"]["inat_taxon_url"] == "https://www.inaturalist.org/taxa/55555"


def test_apply_mapping_skips_missing_gbif_key(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    with open(tmp_path / "inat_gbif_mapping.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "gbif_taxon_key": "", "inat_id": "55555", "inat_taxon_url": "",
            "rank": "SPECIES", "scientific_name": "opuntia humifusa", "match_type": "exact",
        })
    catalog = {k: dict(v) for k, v in CATALOG.items()}
    updated = bim.apply_mapping(catalog)
    assert updated == 0


def test_apply_mapping_skips_unknown_taxon(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")
    with open(tmp_path / "inat_gbif_mapping.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "gbif_taxon_key": "9999999", "inat_id": "55555", "inat_taxon_url": "",
            "rank": "SPECIES", "scientific_name": "opuntia humifusa", "match_type": "exact",
        })
    catalog = {k: dict(v) for k, v in CATALOG.items()}
    updated = bim.apply_mapping(catalog)
    assert updated == 0


# --- load_catalog / save_catalog ---

def test_load_save_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_PATH", tmp_path / "taxon_catalog.pkl")
    with open(tmp_path / "taxon_catalog.pkl", "wb") as f:
        pickle.dump(PAYLOAD, f)
    loaded = bim.load_catalog()
    assert loaded["catalog"] == CATALOG
    loaded["catalog"]["new_key"] = {}
    bim.save_catalog(loaded)
    with open(tmp_path / "taxon_catalog.pkl", "rb") as f:
        reloaded = pickle.load(f)
    assert "new_key" in reloaded["catalog"]


# --- main ---

def test_main(tmp_path, monkeypatch):
    monkeypatch.setattr(bim, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(bim, "CATALOG_PATH", tmp_path / "taxon_catalog.pkl")
    monkeypatch.setattr(bim, "MAPPING_PATH", tmp_path / "inat_gbif_mapping.csv")

    with open(tmp_path / "taxon_catalog.pkl", "wb") as f:
        pickle.dump(PAYLOAD, f)

    dwca = _make_dwca([
        {"id": "55555", "taxonID": "https://www.inaturalist.org/taxa/55555",
         "scientificName": "Opuntia humifusa", "taxonRank": "SPECIES"},
    ])
    monkeypatch.setattr(bim, "download_dwca", lambda: dwca)

    bim.main()

    with open(tmp_path / "taxon_catalog.pkl", "rb") as f:
        result = pickle.load(f)
    assert result["catalog"]["2923970"]["inat_id"] == "55555"
