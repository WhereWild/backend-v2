"""Tests for GET /api/species/{taxon_id}"""


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
