"""Tests for GET /locations/search and GET /locations/search_hierarchy"""
from __future__ import annotations

import main


# ---------------------------------------------------------------------------
# /locations/search
# ---------------------------------------------------------------------------

def test_search_locations_returns_200(client):
    r = client.get("/locations/search?q=united+states")
    assert r.status_code == 200


def test_search_locations_returns_results_wrapper(client):
    body = client.get("/locations/search?q=united+states").json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_locations_returns_results(client):
    body = client.get("/locations/search?q=united+states").json()
    assert len(body["results"]) > 0


def test_search_locations_result_fields(client):
    body = client.get("/locations/search?q=united+states").json()
    required = {"gid", "name", "level", "hierarchy"}
    for loc in body["results"]:
        missing = required - loc.keys()
        assert not missing, f"Location missing fields {missing}: {loc}"


def test_search_locations_hierarchy_is_list(client):
    body = client.get("/locations/search?q=utah").json()
    for loc in body["results"]:
        assert isinstance(loc["hierarchy"], list)


def test_search_locations_level_is_int(client):
    body = client.get("/locations/search?q=united+states").json()
    for loc in body["results"]:
        assert isinstance(loc["level"], int)


def test_search_locations_missing_q_returns_422(client):
    r = client.get("/locations/search")
    assert r.status_code == 422


def test_search_locations_respects_limit(client):
    body = client.get("/locations/search?q=a&limit=3").json()
    assert len(body["results"]) <= 3


# ---------------------------------------------------------------------------
# /locations/search_hierarchy
# ---------------------------------------------------------------------------

def test_search_hierarchy_returns_200(client):
    r = client.get("/locations/search_hierarchy?q=utah")
    assert r.status_code == 200


def test_search_hierarchy_returns_results_wrapper(client):
    body = client.get("/locations/search_hierarchy?q=utah").json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_hierarchy_result_fields(client):
    body = client.get("/locations/search_hierarchy?q=utah").json()
    required = {"gid", "name", "level", "hierarchy"}
    for loc in body["results"]:
        missing = required - loc.keys()
        assert not missing, f"Hierarchy result missing fields {missing}: {loc}"


def test_search_hierarchy_empty_with_no_params_returns_empty(client):
    body = client.get("/locations/search_hierarchy").json()
    assert body == {"results": []}


def test_search_hierarchy_filter_by_level_country(client):
    body = client.get("/locations/search_hierarchy?level=country&limit=10").json()
    for loc in body["results"]:
        assert loc["level"] == 0, f"Expected country level (0), got {loc['level']}"


def test_search_hierarchy_filter_by_level_state(client):
    body = client.get("/locations/search_hierarchy?q=utah&level=state").json()
    for loc in body["results"]:
        assert loc["level"] == 1, f"Expected state level (1), got {loc['level']}"


def test_search_hierarchy_with_parent_filter(client):
    """?q + ?parent triggers parent resolution and matches_parent (lines 372-402)."""
    body = client.get("/locations/search_hierarchy?q=Utah&parent=United+States").json()
    assert "results" in body
    assert isinstance(body["results"], list)
    # Utah (state) or Utah County should appear; parent filter should narrow results
    assert len(body["results"]) > 0, "Expected at least one Utah result under United States"


def test_search_hierarchy_state_level_builds_parent_hierarchy(client):
    """?level=state (no q) hits catalog enumeration + while-parent_gid loop (lines 427, 433-437)."""
    body = client.get("/locations/search_hierarchy?level=state&limit=5").json()
    assert "results" in body
    for loc in body["results"]:
        assert loc["level"] == 1, f"Expected state level (1), got {loc['level']}"
        # States should have at least one parent name in hierarchy
        assert isinstance(loc["hierarchy"], list)


def test_search_hierarchy_parent_only_hits_list_children(client):
    """?parent alone (no q, no level) skips catalog enum and calls list_children (lines 452-466)."""
    body = client.get("/locations/search_hierarchy?parent=United+States&limit=10").json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_hierarchy_letter_scan_fallback(client):
    """Non-matching parent triggers letter-scan fallback (lines 469-482)."""
    body = client.get("/locations/search_hierarchy?parent=ZZZNOMATCH&limit=5").json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_hierarchy_parent_resolution_and_duplicate_drop(monkeypatch):
    monkeypatch.setattr(
        main.gis_lookup,
        "get_location_by_gid",
        lambda tok: {"gid": "USA", "name": "United States"} if tok == "US" else None,
        raising=False,
    )
    monkeypatch.setattr(
        main.gis_lookup,
        "search_locations",
        lambda _q, _limit: [
            {"gid": "X", "name": "Utah", "level": 1, "hierarchy": ["United States"]},
            {"gid": "X", "name": "Utah", "level": 1, "hierarchy": ["United States"]},
        ],
    )
    body = main.search_locations_by_hierarchy(q="utah", level=None, parent="US", limit=10)
    assert len(body["results"]) == 1


def test_search_hierarchy_handles_gid_resolution_exception(monkeypatch):
    def raise_on_gid(_tok):
        raise RuntimeError("lookup failure")

    monkeypatch.setattr(main.gis_lookup, "get_location_by_gid", raise_on_gid, raising=False)
    monkeypatch.setattr(main.gis_lookup, "search_locations", lambda _q, _limit: [])
    assert main.search_locations_by_hierarchy(q="x", level=None, parent="Y", limit=50) == {"results": []}


def test_search_hierarchy_catalog_enum_and_list_children_fallback_errors(monkeypatch):
    class FakeRec:
        gid = "UT"
        name = "Utah"
        level = 1
        parent_gid = "MISSING_PARENT"

    monkeypatch.setattr(main.gis_lookup, "search_locations", lambda _q, _limit: [])
    monkeypatch.setattr(main.gis_lookup, "load_location_catalog", lambda: ([FakeRec()], {}))
    monkeypatch.setattr(
        main.gis_lookup,
        "get_location_by_gid",
        lambda _tok: {"gid": "USA", "name": "US"},
        raising=False,
    )

    def bad_list_children(*_args, **_kwargs):
        raise RuntimeError("children failed")

    monkeypatch.setattr(main.gis_lookup, "list_children", bad_list_children)
    out = main.search_locations_by_hierarchy(q="", level="state", parent="US", limit=5)
    assert "results" in out


def test_search_hierarchy_catalog_enumeration_exception_is_ignored(monkeypatch):
    monkeypatch.setattr(main.gis_lookup, "search_locations", lambda _q, _limit: [])
    monkeypatch.setattr(
        main.gis_lookup,
        "load_location_catalog",
        lambda: (_ for _ in ()).throw(RuntimeError("catalog failed")),
    )
    monkeypatch.setattr(main.gis_lookup, "list_children", lambda *_a, **_k: [], raising=False)
    out = main.search_locations_by_hierarchy(q="", level="state", parent=None, limit=5)
    assert out == {"results": []}


def test_search_hierarchy_letter_scan_break_and_error_handling(monkeypatch):
    def partial(_q, _limit):
        if _q == "a":
            return [{"gid": "A", "name": "A", "level": 0, "hierarchy": []}]
        raise RuntimeError("letter lookup fail")

    monkeypatch.setattr(main.gis_lookup, "search_locations", partial)
    monkeypatch.setattr(main.gis_lookup, "list_children", lambda *_a, **_k: [], raising=False)
    out = main.search_locations_by_hierarchy(q="", level=None, parent="A", limit=1)
    assert len(out["results"]) == 1

    monkeypatch.setattr(
        main.gis_lookup,
        "search_locations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert main.search_locations_by_hierarchy(q="abc", level=None, parent=None, limit=50) == {"results": []}


def test_search_hierarchy_letter_scan_swallow_partial_errors(monkeypatch):
    monkeypatch.setattr(main.gis_lookup, "list_children", lambda *_a, **_k: [], raising=False)
    monkeypatch.setattr(
        main.gis_lookup,
        "search_locations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("letter failed")),
    )
    out = main.search_locations_by_hierarchy(q="", level=None, parent="A", limit=5)
    assert out == {"results": []}
