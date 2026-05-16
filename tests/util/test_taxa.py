from unittest.mock import patch

import pytest

import util.taxa as taxa

CATALOG = {
    "2923970": {
        "taxon_key": "2923970",
        "path": "Plantae_6/.../Opuntia_2923968/Opuntia_humifusa_2923970",
        "scientific_name": "Opuntia_humifusa",
        "common_name": "devil's tongue",
        "rank": "SPECIES",
    },
    "2923968": {
        "taxon_key": "2923968",
        "path": "Plantae_6/.../Opuntia_2923968",
        "scientific_name": "Opuntia",
        "common_name": "",
        "rank": "GENUS",
    },
}

NAME_INDEX = {
    "opuntia humifusa": ["2923970"],
    "devil's tongue": ["2923970"],
    "opuntia": ["2923968", "2923970"],
}

PAYLOAD = {"catalog": CATALOG, "combined_name_index": NAME_INDEX}


@pytest.fixture(autouse=True)
def clear_lru_caches():
    taxa._load_payload.cache_clear()
    taxa.load_catalog.cache_clear()
    taxa.load_name_index.cache_clear()
    taxa._slug_index.cache_clear()
    yield
    taxa._load_payload.cache_clear()
    taxa.load_catalog.cache_clear()
    taxa.load_name_index.cache_clear()
    taxa._slug_index.cache_clear()


@pytest.fixture(autouse=True)
def mock_payload():
    with patch.object(taxa, "_load_payload", return_value=PAYLOAD):
        yield


# --- normalize_name ---

def test_normalize_name_basic():
    assert taxa.normalize_name("Opuntia humifusa") == "opuntia humifusa"


def test_normalize_name_underscores():
    assert taxa.normalize_name("Opuntia_humifusa") == "opuntia humifusa"


def test_normalize_name_empty():
    assert taxa.normalize_name("") == ""


# --- taxon_slug ---

def test_taxon_slug_basic():
    assert taxa.taxon_slug("Opuntia humifusa") == "opuntia-humifusa"


def test_taxon_slug_underscores():
    assert taxa.taxon_slug("Opuntia_humifusa") == "opuntia-humifusa"


def test_taxon_slug_empty():
    assert taxa.taxon_slug("") == ""


def test_taxon_slug_none():
    assert taxa.taxon_slug(None) == ""


# --- get_taxon_by_id ---

def test_get_taxon_by_id_found():
    result = taxa.get_taxon_by_id("2923970")
    assert result is not None
    assert result["scientific_name"] == "Opuntia_humifusa"


def test_get_taxon_by_id_not_found():
    assert taxa.get_taxon_by_id("9999999") is None


def test_get_taxon_by_id_empty():
    assert taxa.get_taxon_by_id("") is None


def test_get_taxon_by_id_none():
    assert taxa.get_taxon_by_id(None) is None


# --- get_taxon_by_slug ---

def test_get_taxon_by_slug_found():
    result = taxa.get_taxon_by_slug("opuntia-humifusa")
    assert result is not None
    assert result["taxon_key"] == "2923970"


def test_get_taxon_by_slug_not_found():
    assert taxa.get_taxon_by_slug("no-such-taxon") is None


def test_get_taxon_by_slug_empty():
    assert taxa.get_taxon_by_slug("") is None


def test_get_taxon_by_slug_ambiguous():
    # Two catalog entries sharing the same scientific_name slug → neither returned
    dup_catalog = {
        **CATALOG,
        "9999999": {
            "taxon_key": "9999999",
            "path": "Plantae_6/Opuntia_9999999",
            "scientific_name": "Opuntia_humifusa",
            "common_name": "",
            "rank": "SPECIES",
        },
    }
    dup_payload = {"catalog": dup_catalog, "combined_name_index": NAME_INDEX}
    with patch.object(taxa, "_load_payload", return_value=dup_payload):
        taxa.load_catalog.cache_clear()
        taxa._slug_index.cache_clear()
        assert taxa.get_taxon_by_slug("opuntia-humifusa") is None


# --- search_taxa_by_name ---

def test_search_taxa_exact_match():
    results = taxa.search_taxa_by_name("opuntia humifusa")
    keys = [t["taxon_key"] for t, _ in results]
    assert "2923970" in keys


def test_search_taxa_common_name():
    results = taxa.search_taxa_by_name("devil's tongue")
    keys = [t["taxon_key"] for t, _ in results]
    assert "2923970" in keys


def test_search_taxa_empty_query():
    assert taxa.search_taxa_by_name("") == []


def test_search_taxa_limit():
    results = taxa.search_taxa_by_name("opuntia", limit=1)
    assert len(results) <= 1


def test_search_taxa_scores_descending():
    results = taxa.search_taxa_by_name("opuntia humifusa")
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_taxa_no_match():
    results = taxa.search_taxa_by_name("zzznomatch")
    assert results == []


# --- _adjust_score ---

def test_adjust_score_exact_match():
    score = taxa._adjust_score(
        "opuntia humifusa", "opuntia humifusa", ["opuntia", "humifusa"], 100.0
    )
    assert score is not None
    assert score > 100.0


def test_adjust_score_no_prefix_match():
    score = taxa._adjust_score("rosa canina", "opuntia", ["opuntia"], 80.0)
    assert score is None


def test_adjust_score_multi_token_partial_miss():
    score = taxa._adjust_score("opuntia canina", "opuntia humifusa", ["opuntia", "humifusa"], 70.0)
    assert score is None


def test_adjust_score_below_min():
    # raw=40 + exact boost=20 = 60, still below single-token minimum of 70
    score = taxa._adjust_score("opuntia", "opuntia", ["opuntia"], 40.0)
    assert score is None
