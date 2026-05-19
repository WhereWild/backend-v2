from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import util.taxa as taxa
from main import app

client = TestClient(app)

TAXON = {
    "taxon_key": "2923970",
    "path": "Plantae_6/Opuntia_2923968/Opuntia_humifusa_2923970",
    "scientific_name": "Opuntia_humifusa",
    "common_name": "devil's tongue",
    "rank": "SPECIES",
}


def test_data_sources():
    from unittest.mock import patch as _patch

    import util.citations as cit
    fake = {"gbif_backbone": {"name": "GBIF Backbone Taxonomy"}}
    with _patch.object(cit, "load_data_sources", return_value=fake):
        response = client.get("/data-sources")
    assert response.status_code == 200
    assert "gbif_backbone" in response.json()


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_taxon_by_id():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        response = client.get("/api/taxon/2923970")
    assert response.status_code == 200
    assert response.json()["taxon_key"] == "2923970"


def test_get_taxon_by_slug():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=TAXON):
        response = client.get("/api/taxon/opuntia-humifusa")
    assert response.status_code == 200
    assert response.json()["scientific_name"] == "Opuntia_humifusa"


def test_get_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        response = client.get("/api/taxon/nope")
    assert response.status_code == 404


def test_query_taxa():
    with patch.object(taxa, "search_taxa_by_name", return_value=[(TAXON, 95.0, "opuntia humifusa")]):
        response = client.get("/api/taxa/query?q=opuntia")
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "opuntia"
    assert len(body["results"]) == 1
    assert body["results"][0]["taxon_id"] == "2923970"
    assert body["results"][0]["scientific_name"] == "Opuntia humifusa"
    assert body["results"][0]["match_score"] == pytest.approx(95.0)


def test_query_taxa_no_query():
    response = client.get("/api/taxa/query")
    assert response.status_code == 200
    assert response.json()["empty_reason"] == "no_query"


def test_query_taxa_empty_query():
    response = client.get("/api/taxa/query?q=")
    assert response.status_code == 422


def test_query_taxa_limit():
    with patch.object(taxa, "search_taxa_by_name", return_value=[]) as mock_search:
        client.get("/api/taxa/query?q=opuntia&limit=5")
        mock_search.assert_called_once_with("opuntia", limit=5)
