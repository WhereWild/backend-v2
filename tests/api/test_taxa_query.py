"""Tests for GET /api/taxa/query."""

from __future__ import annotations

import main
from util.request_cancellation import RequestCancelledError


def test_taxa_query_returns_499_on_disconnect(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "_build_disconnect_checker",
        lambda _request, poll_every=32: lambda: (_ for _ in ()).throw(RequestCancelledError("Client disconnected")),
    )

    response = client.get("/api/taxa/query?q=oak")

    assert response.status_code == 499
    assert response.json()["detail"] == "Client disconnected"


def test_taxa_query_text_only_returns_search_payload(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 1,
            "matched_total": 1,
            "eligible_total": 1,
            "empty_reason": None,
            "results": [
                {
                    "taxon_id": 11,
                    "taxon": {"taxon_key": "11"},
                    "match_score": 91.0,
                    "sample_count": 7,
                    "sort_value": None,
                    "sort_variable": None,
                    "sort_metric": None,
                    "position": None,
                    "percentile": None,
                }
            ],
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "serialize_taxon",
        lambda _taxon: {
            "taxon_id": 11,
            "scientific_name": "Quercus robur",
            "common_name": "Oak",
            "common_names": ["Oak"],
            "rank": "SPECIES",
            "slug": "quercus-robur",
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"taxon_key": "11"},
    )

    body = client.get("/api/taxa/query?q=oak").json()

    assert body["total"] == 1
    assert len(body["results"]) == 1
    assert body["results"][0]["match_score"] == 91.0
    assert body["results"][0]["sample_count"] == 7
    assert body["results"][0]["sort_value"] is None
    assert body["results"][0]["sort_variable"] is None
    assert body["results"][0]["sort_metric"] is None
    assert body["results"][0]["position"] is None
    assert body["results"][0]["percentile"] is None


def test_taxa_query_normalizes_string_image_references(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 1,
            "matched_total": 1,
            "eligible_total": 1,
            "empty_reason": None,
            "results": [
                {
                    "taxon_id": 11,
                    "taxon": {"taxon_key": "11"},
                    "match_score": 91.0,
                    "sample_count": 7,
                    "sort_value": None,
                    "sort_variable": None,
                    "sort_metric": None,
                    "position": None,
                    "percentile": None,
                }
            ],
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "serialize_taxon",
        lambda _taxon: {
            "taxon_id": 11,
            "scientific_name": "Quercus robur",
            "common_name": "Oak",
            "common_names": ["Oak"],
            "rank": "SPECIES",
            "slug": "quercus-robur",
            "image_references": "https://www.inaturalist.org/photos/169945779",
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"taxon_key": "11"},
    )

    body = client.get("/api/taxa/query?q=oak").json()

    assert body["results"][0]["image_references"] == "https://www.inaturalist.org/photos/169945779"


def test_taxa_query_ranked_results_merge_search_and_sort_metadata(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 1,
            "matched_total": 1,
            "eligible_total": 1,
            "empty_reason": None,
            "results": [
                {
                    "taxon_id": 22,
                    "match_score": 97.0,
                    "sample_count": 18,
                    "sort_variable": "bio_1",
                    "sort_metric": "mean",
                    "sort_value": 4.2,
                    "position": 1,
                    "percentile": 0.0,
                    "taxon": {"taxon_key": "22"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"taxon_key": "22"},
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "serialize_taxon",
        lambda _taxon: {
            "taxon_id": 22,
            "scientific_name": "Pediocactus simpsonii",
            "common_name": "Mountain ball cactus",
            "common_names": ["Mountain ball cactus"],
            "rank": "SPECIES",
            "slug": "pediocactus-simpsonii",
        },
    )
    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"bio_1": {"units": "C"}}),
    )

    body = client.get(
        "/api/taxa/query?q=mountain%20ball%20cactus&within_taxon=10"
        "&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean"
    ).json()

    assert body["total"] == 1
    assert body["sort"]["variable"] == "bio_1"
    assert body["sort"]["units"] == "C"
    assert body["results"][0]["match_score"] == 97.0
    assert body["results"][0]["sort_value"] == 4.2
    assert body["results"][0]["sample_count"] == 18


def test_taxa_query_ranked_results_apply_unit_system(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 1,
            "matched_total": 1,
            "eligible_total": 1,
            "empty_reason": None,
            "results": [
                {
                    "taxon_id": 22,
                    "match_score": 97.0,
                    "sample_count": 18,
                    "sort_variable": "bio_1",
                    "sort_metric": "mean",
                    "sort_value": 0.0,
                    "position": 1,
                    "percentile": 0.0,
                    "taxon": {"taxon_key": "22"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"taxon_key": "22"},
    )
    monkeypatch.setattr(
        main.taxa_navigation,
        "serialize_taxon",
        lambda _taxon: {
            "taxon_id": 22,
            "scientific_name": "Pediocactus simpsonii",
            "common_name": "Mountain ball cactus",
            "common_names": ["Mountain ball cactus"],
            "rank": "SPECIES",
            "slug": "pediocactus-simpsonii",
        },
    )
    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"bio_1": {"units": "C"}}),
    )

    body = client.get(
        "/api/taxa/query?q=mountain%20ball%20cactus&within_taxon=10"
        "&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean&unit_system=imperial"
    ).json()

    assert body["sort"]["units"] == "°F"
    assert body["results"][0]["sort_value"] == 32.0


def test_taxa_query_normalizes_sort_order_in_response(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 0,
            "matched_total": 0,
            "eligible_total": 0,
            "empty_reason": "ranking_ineligible",
            "results": [],
        },
    )
    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"bio_1": {"units": "C"}}),
    )
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _taxon_id: {"taxon_key": "10"})

    body = client.get(
        "/api/taxa/query?within_taxon=10&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean&sort_order=%20DESC%20"
    ).json()

    assert body["sort"]["order"] == "desc"


def test_taxa_query_ranked_empty_does_not_fallback_to_text_matches(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 0,
            "matched_total": 1,
            "eligible_total": 1,
            "empty_reason": "ranking_ineligible",
            "results": [],
        },
    )
    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"bio_1": {"units": "C"}}),
    )
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _taxon_id: {"taxon_key": "10"})

    body = client.get(
        "/api/taxa/query?q=mountain%20ball%20cactus&within_taxon=10"
        "&descendant_rank=SPECIES&sort_variable=bio_1&sort_metric=mean"
    ).json()

    assert body["total"] == 0
    assert body["matched_total"] == 1
    assert body["eligible_total"] == 1
    assert body["empty_reason"] == "ranking_ineligible"
    assert body["results"] == []


def test_taxa_query_returns_structured_outcome_metadata(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 0,
            "matched_total": 4,
            "eligible_total": 0,
            "empty_reason": "filtered_out",
            "results": [],
        },
    )

    body = client.get("/api/taxa/query?q=oak").json()

    assert body["matched_total"] == 4
    assert body["eligible_total"] == 0
    assert body["empty_reason"] == "filtered_out"


def test_taxa_query_resolves_polymorphic_within_taxon_input(client, monkeypatch):
    captured: dict[str, str | None] = {}

    def fake_query_taxa(**kwargs):
        captured["within_taxon_id"] = kwargs.get("within_taxon_id")
        return {
            "total": 0,
            "matched_total": 0,
            "eligible_total": 0,
            "empty_reason": "no_text_matches",
            "results": [],
        }

    monkeypatch.setattr(main.indexing, "query_taxa", fake_query_taxa)
    monkeypatch.setattr(
        main.taxa_navigation,
        "resolve_taxon_reference",
        lambda value: {"taxon_key": "10"} if value == "alpha-genus" else None,
    )
    monkeypatch.setattr(main.taxa_navigation, "taxon_id_as_int", lambda value: int(str(value)))

    body = client.get("/api/taxa/query?q=oak&within_taxon=alpha-genus").json()

    assert captured["within_taxon_id"] == "10"
    assert body["scope"]["within_taxon"] == 10


def test_taxa_query_rejects_ambiguous_within_taxon_slug(client, monkeypatch):
    monkeypatch.setattr(
        main.taxa_navigation,
        "resolve_taxon_reference",
        lambda _value: (_ for _ in ()).throw(ValueError("Ambiguous taxon slug: iris")),
    )

    response = client.get("/api/taxa/query?q=iris&within_taxon=iris")

    assert response.status_code == 400
    assert response.json()["detail"] == "Ambiguous taxon slug: iris"


def test_taxa_query_rejects_unknown_numeric_within_taxon(client, monkeypatch):
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _taxon_id: None)

    response = client.get("/api/taxa/query?q=oak&within_taxon=999999")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown within_taxon value: 999999"


def test_taxa_query_rejects_unknown_descendant_rank(client):
    response = client.get("/api/taxa/query?q=oak&descendant_rank=spcies")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown descendant_rank: spcies"


def test_taxa_query_without_query_returns_no_query_reason(client, monkeypatch):
    monkeypatch.setattr(
        main.indexing,
        "query_taxa",
        lambda **_kwargs: {
            "total": 0,
            "matched_total": 0,
            "eligible_total": 0,
            "empty_reason": "no_query",
            "results": [],
        },
    )

    body = client.get("/api/taxa/query").json()

    assert body["total"] == 0
    assert body["empty_reason"] == "no_query"


def test_taxa_ranking_options_returns_scoped_options(client, monkeypatch):
    monkeypatch.setattr(main, "_resolve_within_taxon_id", lambda **_kwargs: "77")
    monkeypatch.setattr(
        main.indexing,
        "list_rank_metric_options",
        lambda ancestor_taxon_id, descendant_rank: (
            [
                {"variable": "bio_12", "metric": "max", "count": 14, "column": "bio_12::max"},
                {"variable": "bio_12", "metric": "min", "count": 14, "column": "bio_12::min"},
            ]
            if ancestor_taxon_id == "77" and descendant_rank == "SPECIES"
            else []
        ),
    )
    monkeypatch.setattr(main.taxa_navigation, "taxon_id_as_int", lambda value: int(str(value)))
    monkeypatch.setattr(main.taxa_navigation, "canonical_rank", lambda value: str(value).upper())

    body = client.get("/api/taxa/ranking-options?within_taxon=77&descendant_rank=SPECIES").json()

    assert body == {
        "ancestor_taxon_id": 77,
        "rank": "SPECIES",
        "options": [
            {"variable": "bio_12", "metric": "max", "label": None, "count": 14, "column": "bio_12::max"},
            {"variable": "bio_12", "metric": "min", "label": None, "count": 14, "column": "bio_12::min"},
        ],
    }


def test_taxa_ranking_options_rejects_invalid_rank(client, monkeypatch):
    monkeypatch.setattr(main, "_resolve_within_taxon_id", lambda **_kwargs: "77")
    monkeypatch.setattr(
        main.indexing,
        "list_rank_metric_options",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Unknown descendant_rank: spcies")),
    )

    response = client.get("/api/taxa/ranking-options?within_taxon=77&descendant_rank=spcies")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown descendant_rank: spcies"
