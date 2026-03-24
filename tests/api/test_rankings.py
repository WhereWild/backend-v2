"""Tests for GET /relative-rankings/{taxon_id} and /relative-rankings/{taxon_id}/options"""


# ---------------------------------------------------------------------------
# /relative-rankings/{taxon_id}/options
# ---------------------------------------------------------------------------

def test_ranking_options_returns_200(client, known_genus_taxon_id):
    r = client.get(f"/relative-rankings/{known_genus_taxon_id}/options?rank=SPECIES")
    assert r.status_code == 200


def test_ranking_options_shape(client, known_genus_taxon_id):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}/options?rank=SPECIES"
    ).json()
    assert "ancestor_taxon_id" in body
    assert "rank" in body
    assert "options" in body


def test_ranking_options_rank_is_uppercase(client, known_genus_taxon_id):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}/options?rank=SPECIES"
    ).json()
    assert body["rank"] == "SPECIES"


def test_ranking_options_missing_rank_returns_422(client, known_genus_taxon_id):
    r = client.get(f"/relative-rankings/{known_genus_taxon_id}/options")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /relative-rankings/{taxon_id}
# ---------------------------------------------------------------------------

def test_rankings_returns_200(client, known_genus_taxon_id, known_numeric_var):
    r = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean"
    )
    assert r.status_code == 200


def test_rankings_top_level_fields(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean"
    ).json()
    required = {
        "ancestor_taxon_id", "rank", "variable", "metric",
        "total", "limit", "order", "entries",
    }
    missing = required - body.keys()
    assert not missing, f"Rankings response missing fields: {missing}"


def test_rankings_rank_is_uppercase(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean"
    ).json()
    assert body["rank"] == "SPECIES"


def test_rankings_entries_is_list(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean"
    ).json()
    assert isinstance(body["entries"], list)


def test_rankings_respects_limit(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean&limit=5"
    ).json()
    assert len(body["entries"]) <= 5


def test_rankings_order_asc(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean&order=asc&limit=20"
    ).json()
    values = [e["value"] for e in body["entries"] if e.get("value") is not None]
    assert values == sorted(values), "Entries should be in ascending order"


def test_rankings_order_desc(client, known_genus_taxon_id, known_numeric_var):
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean&order=desc&limit=20"
    ).json()
    values = [e["value"] for e in body["entries"] if e.get("value") is not None]
    assert values == sorted(values, reverse=True), "Entries should be in descending order"


def test_rankings_invalid_metric_returns_400(client, known_genus_taxon_id, known_numeric_var):
    r = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=not_a_real_metric"
    )
    assert r.status_code == 400


def test_rankings_missing_required_params_returns_422(client, known_genus_taxon_id):
    r = client.get(f"/relative-rankings/{known_genus_taxon_id}")
    assert r.status_code == 422


def test_rankings_with_include_distribution(client, known_genus_taxon_id, known_numeric_var):
    """include_distribution=true builds density curve from distribution values (line 981)."""
    body = client.get(
        f"/relative-rankings/{known_genus_taxon_id}"
        f"?rank=SPECIES&variable={known_numeric_var}&metric=mean&include_distribution=true"
    ).json()
    assert "distribution" in body
    # distribution may be null if no values, but key must be present
    assert "entries" in body


def test_ranking_options_invalid_taxon_returns_400(client):
    """Unknown taxon raises ValueError in list_rank_metric_options → 400 (lines 1022-1023)."""
    r = client.get("/relative-rankings/999999999/options?rank=SPECIES")
    assert r.status_code == 400
