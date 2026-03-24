"""Tests for GET /api/species and GET /api/species/{taxon_id}"""


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

def test_search_returns_200(client):
    r = client.get("/api/species?q=quercus")
    assert r.status_code == 200


def test_search_returns_list(client):
    body = client.get("/api/species?q=quercus").json()
    assert isinstance(body, list)


def test_search_returns_results_for_known_genus(client):
    body = client.get("/api/species?q=quercus").json()
    assert len(body) > 0, "Expected results for 'quercus'"


def test_search_result_has_required_fields(client):
    body = client.get("/api/species?q=quercus").json()
    required = {"taxon_id", "scientific_name", "common_name", "rank", "slug"}
    for item in body:
        missing = required - item.keys()
        assert not missing, f"Result missing fields {missing}: {item}"


def test_search_taxon_id_is_int(client):
    body = client.get("/api/species?q=quercus").json()
    for item in body:
        assert isinstance(item["taxon_id"], int), f"taxon_id not int: {item['taxon_id']}"


def test_search_rank_is_uppercase(client):
    body = client.get("/api/species?q=quercus").json()
    for item in body:
        rank = item.get("rank", "")
        assert rank == rank.upper(), f"rank not uppercase: '{rank}'"


def test_search_respects_limit(client):
    body = client.get("/api/species?q=quercus&limit=3").json()
    assert len(body) <= 3


def test_search_missing_q_returns_422(client):
    r = client.get("/api/species")
    assert r.status_code == 422


def test_search_empty_q_returns_422(client):
    r = client.get("/api/species?q=")
    assert r.status_code == 422


def test_search_common_name_query(client):
    """Search by common name (oak) should return results."""
    body = client.get("/api/species?q=oak").json()
    assert isinstance(body, list)
    assert len(body) > 0, "Expected results for common name 'oak'"


def test_search_matched_common_name_present(client):
    body = client.get("/api/species?q=quercus").json()
    for item in body:
        assert "matched_common_name" in item


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------

def test_get_species_detail_200(client, known_taxon_id):
    r = client.get(f"/api/species/{known_taxon_id}")
    assert r.status_code == 200


def test_get_species_detail_has_required_fields(client, known_taxon_id):
    body = client.get(f"/api/species/{known_taxon_id}").json()
    required = {"taxon_id", "scientific_name", "common_name", "rank", "slug"}
    missing = required - body.keys()
    assert not missing, f"Detail response missing fields: {missing}"


def test_get_species_detail_taxon_id_matches(client, known_taxon_id):
    body = client.get(f"/api/species/{known_taxon_id}").json()
    assert body["taxon_id"] == known_taxon_id


def test_get_species_detail_rank_is_uppercase(client, known_taxon_id):
    body = client.get(f"/api/species/{known_taxon_id}").json()
    rank = body.get("rank", "")
    assert rank == rank.upper()


def test_get_species_detail_404_for_unknown_id(client):
    r = client.get("/api/species/999999999")
    assert r.status_code == 404
