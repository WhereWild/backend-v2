"""Tests for GET /species/{taxon_id}/occurrences and /species/{taxon_id}/locations"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import main


# ---------------------------------------------------------------------------
# /species/{taxon_id}/occurrences
# ---------------------------------------------------------------------------

def test_occurrences_returns_200(client, known_taxon_id):
    r = client.get(f"/species/{known_taxon_id}/occurrences")
    assert r.status_code == 200


def test_occurrences_shape(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    assert "speciesId" in body
    assert "count" in body
    assert "occurrences" in body


def test_occurrences_species_id_matches(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    assert body["speciesId"] == known_taxon_id


def test_occurrences_count_matches_list_length(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    assert body["count"] == len(body["occurrences"])


def test_occurrences_count_positive(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    assert body["count"] > 0, "Expected at least one occurrence for Quercus robur"


def test_occurrences_point_fields(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    required = {"catalogNumber", "latitude", "longitude"}
    for pt in body["occurrences"][:20]:  # spot-check first 20
        missing = required - pt.keys()
        assert not missing, f"Occurrence point missing fields {missing}"


def test_occurrences_lat_lon_valid_ranges(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/occurrences").json()
    for pt in body["occurrences"][:100]:
        lat = pt["latitude"]
        lon = pt["longitude"]
        assert -90 <= lat <= 90, f"latitude out of range: {lat}"
        assert -180 <= lon <= 180, f"longitude out of range: {lon}"


def test_occurrences_invalid_taxon_returns_404(client):
    r = client.get("/species/999999999/occurrences")
    assert r.status_code == 404


def test_occurrences_invalid_location_returns_empty(client, known_taxon_id):
    body = client.get(
        f"/species/{known_taxon_id}/occurrences?location=NOT_A_REAL_GID"
    ).json()
    assert body["count"] == 0
    assert body["occurrences"] == []


def test_occurrences_sentinel_location_returns_empty(client, known_taxon_id):
    """Sentinel GID strings ('null', 'none', 'nan', etc.) fail is_valid_location_gid (line 183)."""
    for sentinel in ("null", "none", "nan", "na", "undefined"):
        body = client.get(f"/species/{known_taxon_id}/occurrences?location={sentinel}").json()
        assert body["count"] == 0, f"Expected 0 count for sentinel location '{sentinel}'"
        assert body["occurrences"] == []


def test_occurrences_location_filter_reduces_count(client, known_taxon_id, known_location_gid):
    global_count = client.get(f"/species/{known_taxon_id}/occurrences").json()["count"]
    filtered = client.get(
        f"/species/{known_taxon_id}/occurrences?location={known_location_gid}"
    ).json()["count"]
    assert filtered <= global_count, "Filtered count should not exceed global count"


# ---------------------------------------------------------------------------
# /species/{taxon_id}/locations
# ---------------------------------------------------------------------------

def test_species_locations_returns_200(client, known_taxon_id):
    r = client.get(f"/species/{known_taxon_id}/locations")
    assert r.status_code == 200


def test_species_locations_returns_list(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations").json()
    assert isinstance(body, list)


def test_species_locations_not_empty(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations").json()
    assert len(body) > 0, "Expected Quercus robur to appear in at least one location"


def test_species_locations_entry_fields(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations").json()
    required = {"gid", "name", "level", "hierarchy", "count"}
    for loc in body[:10]:
        missing = required - loc.keys()
        assert not missing, f"Location entry missing fields {missing}: {loc}"


def test_species_locations_counts_positive(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations").json()
    for loc in body:
        assert loc["count"] > 0, f"Location {loc['gid']} has count <= 0"


def test_species_locations_sorted_by_count_desc(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations").json()
    counts = [loc["count"] for loc in body]
    assert counts == sorted(counts, reverse=True), "Locations should be sorted by count descending"


def test_species_locations_filter_by_level_country(client, known_taxon_id):
    body = client.get(f"/species/{known_taxon_id}/locations?level=country").json()
    for loc in body:
        assert loc["level"] == 0


def test_species_locations_invalid_taxon_returns_404(client):
    r = client.get("/species/999999999/locations")
    assert r.status_code == 404


def test_species_locations_filter_by_parent(client, known_taxon_id, known_species_location_gid):
    """?parent builds parent_matchers and filters results (lines 239-292)."""
    body = client.get(f"/species/{known_taxon_id}/locations?parent={known_species_location_gid}").json()
    assert isinstance(body, list)
    # Q. robur is European — should match at least some locations under GBR


def test_occurrences_returns_404_when_taxon_path_missing(monkeypatch):
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _taxon_id: {"path": "/tmp/nope"})
    monkeypatch.setattr(main, "_path_exists", lambda _p: False)
    with pytest.raises(HTTPException) as exc:
        main.species_occurrences(2878688)
    assert exc.value.status_code == 404


def test_species_locations_returns_404_when_taxon_path_missing(monkeypatch):
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"path": "/tmp/nope", "taxon_key": "2878688"},
    )
    monkeypatch.setattr(main, "_path_exists", lambda _p: False)
    with pytest.raises(HTTPException) as exc:
        main.species_locations(2878688, level=None, parent=None, limit=500)
    assert exc.value.status_code == 404


def test_species_locations_returns_empty_when_taxon_key_not_numeric(monkeypatch):
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"path": "/tmp/ok", "taxon_key": "abc"},
    )
    monkeypatch.setattr(main, "_path_exists", lambda _p: True)
    monkeypatch.setattr(main.taxa_navigation, "taxon_id_as_int", lambda _v: None)
    assert main.species_locations(2878688, level=None, parent=None, limit=500) == []


def test_species_locations_handles_empty_catalog_and_counts(monkeypatch):
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"path": "/tmp/ok", "taxon_key": "2878688"},
    )
    monkeypatch.setattr(main, "_path_exists", lambda _p: True)
    monkeypatch.setattr(main.taxa_navigation, "taxon_id_as_int", lambda _v: 2878688)
    monkeypatch.setattr(main.gis_lookup, "load_location_catalog", lambda: ([], {}))
    assert main.species_locations(2878688, level=None, parent=None, limit=500) == []

    records = [
        main.gis_lookup.LocationRecord(gid="A", name="Alpha", level=0, parent_gid="B"),
        main.gis_lookup.LocationRecord(gid="B", name="Beta", level=0, parent_gid="A"),
    ]
    by_gid = {r.gid: r for r in records}
    by_gid["A_ALIAS"] = records[0]
    monkeypatch.setattr(main.gis_lookup, "resolve_location_context", lambda *_a, **_k: [])
    monkeypatch.setattr(main.gis_lookup, "load_location_catalog", lambda: (records, by_gid))

    class FakeCounts:
        def __bool__(self):
            return True

        @staticmethod
        def items():
            return [
                (("unknown_scope", "X"), 1),
                (("gadm_level0", "BAD"), 2),
                (("gadm_level0", "A"), 3),
                (("gadm_level0", "A_ALIAS"), 4),
                (("gadm_level0", "A"), 5),
            ]

    monkeypatch.setattr(main.gis_lookup, "location_counts_for_taxon", lambda _tid: FakeCounts())
    monkeypatch.setattr(main.gis_lookup, "is_valid_location_gid", lambda gid: gid != "BAD")
    out = main.species_locations(2878688, level=None, parent="Alpha", limit=500)
    gids = [item["gid"] for item in out]
    assert gids.count("A") == 1
    assert "A_ALIAS" in gids


def test_species_locations_returns_empty_when_no_counts(monkeypatch):
    monkeypatch.setattr(
        main.taxa_navigation,
        "get_taxon_by_id",
        lambda _taxon_id: {"path": "/tmp/ok", "taxon_key": "2878688"},
    )
    monkeypatch.setattr(main, "_path_exists", lambda _p: True)
    monkeypatch.setattr(main.taxa_navigation, "taxon_id_as_int", lambda _v: 2878688)
    records = [main.gis_lookup.LocationRecord(gid="A", name="Alpha", level=0, parent_gid=None)]
    monkeypatch.setattr(main.gis_lookup, "load_location_catalog", lambda: (records, {"A": records[0]}))
    monkeypatch.setattr(main.gis_lookup, "location_counts_for_taxon", lambda _tid: {})
    assert main.species_locations(2878688, level=None, parent=None, limit=500) == []
