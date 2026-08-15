# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

import main as main_module
import util.rankings as rankings_module
import util.stats as st_module
import util.taxa as taxa
import util.temporal as temporal_module
import util.tiles as tiles
import util.upload as upload_module
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


def test_version_reports_crawl_date_and_build_date(tmp_path):
    sync_state = tmp_path / "sync_state.json"
    sync_state.write_text(json.dumps({
        "gbif_occurrences": {"crawl_finished": "2026-08-01T00:00:00.000Z"},
    }))
    build_date = tmp_path / "build_date.txt"
    build_date.write_text("2026-08-10T12:00:00Z\n")

    with patch.object(main_module, "_SYNC_STATE_PATH", sync_state), \
         patch.object(main_module, "_BUILD_DATE_PATH", build_date):
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "version": "2026-08-01T00:00:00.000Z",
        "api_build_date": "2026-08-10T12:00:00Z",
    }


def test_version_omits_build_date_when_file_missing(tmp_path):
    sync_state = tmp_path / "sync_state.json"
    missing_build_date = tmp_path / "build_date.txt"

    with patch.object(main_module, "_SYNC_STATE_PATH", sync_state), \
         patch.object(main_module, "_BUILD_DATE_PATH", missing_build_date):
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"version": None, "api_build_date": None}


def test_get_taxon_by_id():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(main_module._storage, "read_table", return_value=pa.table({})):
        response = client.get("/api/taxon/2923970")
    assert response.status_code == 200
    assert response.json()["taxon_key"] == "2923970"


def test_get_taxon_by_slug():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=TAXON), \
         patch.object(main_module._storage, "read_table", return_value=pa.table({})):
        response = client.get("/api/taxon/opuntia-humifusa")
    assert response.status_code == 200
    assert response.json()["scientific_name"] == "Opuntia humifusa"


def test_get_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        response = client.get("/api/taxon/nope")
    assert response.status_code == 404


def test_query_taxa():
    with patch.object(rankings_module, "search_taxa_by_name", return_value=[(TAXON, 95.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 100}):
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
    with patch.object(rankings_module, "search_taxa_by_name", return_value=[]) as mock_search:
        client.get("/api/taxa/query?q=opuntia&limit=5")
        # _query_text fetches max((limit+offset)*5, 200) candidates
        mock_search.assert_called_once_with("opuntia", limit=200)


def test_query_taxa_within_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/taxa/query?q=opuntia&within_taxon=99999")
    assert r.status_code == 404


def test_query_taxa_invalid_sort_order():
    r = client.get("/api/taxa/query?sort_order=random")
    assert r.status_code == 422


def test_query_taxa_scope_no_sort_no_catalog():
    """Scope without sort → catalog mode; empty scope returns no_catalog."""
    genus = {**TAXON, "taxon_key": "10", "path": "Plantae_6/Opuntia_2923968", "rank": "GENUS"}
    def _resolve(k):
        return genus if k == "10" else None
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[]):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    assert r.json()["empty_reason"] == "no_catalog"


def test_query_taxa_scope_catalog_mode():
    """Scope without sort lists catalog entries."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}

    def _resolve(k):
        if k == "10":
            return genus
        if k == "2923970":
            return TAXON
        return None

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[TAXON]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 50}):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["empty_reason"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["taxon_id"] == "2923970"
    assert body["results"][0]["sample_count"] == 50


def test_query_taxa_text_in_scope():
    """Text search filtered to scope."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}

    def _resolve(k):
        if k == "10":
            return genus
        if k == "2923970":
            return TAXON
        return None

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "search_taxa_by_name",
                      return_value=[(TAXON, 90.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 50}), \
         patch.object(rankings_module, "iter_descendants", return_value=[TAXON]):
        r = client.get("/api/taxa/query?q=opuntia&within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["scope"]["within_taxon"] == "10"
    assert body["scope"]["descendant_rank"] == "SPECIES"
    assert len(body["results"]) == 1
    assert body["results"][0]["match_score"] == pytest.approx(90.0)


def test_query_taxa_ranked_scoped_no_column():
    """Ranked-scoped mode with no matching rows returns no_column."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    def _resolve(k):
        return genus if k == "10" else None
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_read_rank_positions", return_value=[]):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean")
    assert r.status_code == 200
    assert r.json()["empty_reason"] == "no_column"


def _fake_rank_positions(rows: list[dict]):
    """side_effect for patching rankings_module._read_rank_positions with
    canned rows for a single (context, rank, variable, metric) group; returns
    [] for any other combination."""
    def _read(context_id, rank, variable, metric):
        if context_id == "10" and rank == "SPECIES" and variable == "bio1" and metric == "mean":
            return rows
        return []
    return _read


def test_query_taxa_ranked_scoped_mode():
    """Ranked-scoped mode reads rankings rows and returns sorted results with position/percentile."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "path": "Plantae_6/Opuntia_2923968/Other_111",
              "scientific_name": "Opuntia_other", "rank": "SPECIES"}
    rows = [
        {"taxon_key": "2923970", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
        {"taxon_key": "111", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_read_rank_positions", side_effect=_fake_rank_positions(rows)):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&sort_order=asc")
    assert r.status_code == 200
    body = r.json()
    assert body["empty_reason"] is None
    assert body["eligible_total"] == 2
    results = body["results"]
    assert len(results) == 2
    # asc order: 10.0 first, then 20.0
    assert results[0]["taxon_id"] == "2923970"
    assert results[0]["sort_value"] == pytest.approx(10.0)
    assert results[0]["position"] == 1
    assert results[0]["percentile"] == pytest.approx(0.0)
    assert results[1]["sort_value"] == pytest.approx(20.0)
    assert results[1]["position"] == 2


def test_query_taxa_ranked_scoped_desc():
    """sort_order=desc reverses order."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    rows = [
        {"taxon_key": "2923970", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
        {"taxon_key": "111", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_read_rank_positions", side_effect=_fake_rank_positions(rows)):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&sort_order=desc")
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["sort_value"] == pytest.approx(20.0)
    assert results[1]["sort_value"] == pytest.approx(10.0)


def test_query_taxa_ranked_scoped_min_samples():
    """min_samples filter excludes entries below threshold."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    rows = [
        {"taxon_key": "2923970", "value": 10.0, "position": 0, "count": 2, "sampleCount": 5},
        {"taxon_key": "111", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_read_rank_positions", side_effect=_fake_rank_positions(rows)):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&min_samples=10")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "111"


def test_query_taxa_ranked_scoped_location_filter():
    """Location filter excludes taxa not in the location."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    rows = [
        {"taxon_key": "2923970", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
        {"taxon_key": "111", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    loc_keys = frozenset({"2923970"})
    loc_counts = {"2923970": 42}
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_location_taxon_keys", return_value=(loc_keys, loc_counts)), \
         patch.object(rankings_module, "_read_rank_positions", side_effect=_fake_rank_positions(rows)):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&location=USA")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "2923970"
    assert results[0]["location_count"] == 42


def test_query_taxa_ranked_scoped_text_filter():
    """Mode 3: scope+sort+q filters index to text matches."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    rows = [
        {"taxon_key": "2923970", "value": 10.0, "position": 0, "count": 2, "sampleCount": 100},
        {"taxon_key": "111", "value": 20.0, "position": 1, "count": 2, "sampleCount": 200},
    ]

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "search_taxa_by_name",
                      return_value=[(TAXON, 90.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "_read_rank_positions", side_effect=_fake_rank_positions(rows)):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&q=opuntia")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "2923970"
    assert results[0]["match_score"] == pytest.approx(90.0)


def test_query_taxa_ranked_text_no_scope():
    """Mode 4: q+sort without scope reads global stats in a batch."""
    with patch.object(rankings_module, "search_taxa_by_name",
                      return_value=[(TAXON, 85.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "_batch_metric_values", return_value=({"2923970": 15.5}, {})), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 100}), \
         patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/taxa/query?q=opuntia&sort_variable=bio1&sort_metric=mean")
    assert r.status_code == 200
    body = r.json()
    assert body["empty_reason"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["sort_value"] == pytest.approx(15.5)
    assert body["results"][0]["position"] is None


def test_query_taxa_ranked_text_no_matches():
    """Mode 4 with no text matches returns no_text_matches."""
    with patch.object(rankings_module, "search_taxa_by_name", return_value=[]):
        r = client.get("/api/taxa/query?q=zzz&sort_variable=bio1&sort_metric=mean")
    assert r.status_code == 200
    assert r.json()["empty_reason"] == "no_text_matches"


def test_query_taxa_scope_include_species_like():
    """include_species_like=true accepts subspecies-rank entries."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    subsp = {**TAXON, "taxon_key": "999", "path": "x/999",
             "scientific_name": "Opuntia_humifusa_humifusa", "rank": "SUBSPECIES"}

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "999": subsp}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[TAXON, subsp]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 50, "999": 10}):
        r_no = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
        r_yes = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                           "&include_species_like=true")
    # Without flag: only SPECIES rank
    assert len(r_no.json()["results"]) == 1
    assert r_no.json()["results"][0]["taxon_id"] == "2923970"
    # With flag: SPECIES + SUBSPECIES
    assert len(r_yes.json()["results"]) == 2


def test_query_taxa_offset_pagination():
    """offset/limit pagination works in catalog mode."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxa_list = [
        {"taxon_key": str(i), "path": f"x/{i}", "scientific_name": f"Sp_{i}",
         "common_name": "", "rank": "SPECIES", "sample_count": i * 10}
        for i in range(1, 6)
    ]

    def _resolve(k):
        if k == "10":
            return genus
        for row in taxa_list:
            if row["taxon_key"] == k:
                return {**TAXON, "taxon_key": k, "path": row["path"],
                        "scientific_name": row["scientific_name"], "rank": "SPECIES"}
        return None

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[
             {**TAXON, "taxon_key": str(i), "rank": "SPECIES"} for i in range(1, 6)
         ]), \
         patch.object(rankings_module, "_batch_sample_counts",
                      return_value={str(i): i * 10 for i in range(1, 6)}):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES&limit=2&offset=2")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert len(body["results"]) == 2
    assert body["results"][0]["taxon_id"] == "3"


def test_query_taxa_stat_filter_narrows_results():
    """?filter=variable:metric:op:value chains onto catalog mode."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    taxon2 = {**TAXON, "taxon_key": "111", "scientific_name": "Opuntia_other", "rank": "SPECIES"}

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[TAXON, taxon2]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 50, "111": 50}), \
         patch.object(rankings_module, "_apply_stat_filters", return_value=frozenset({"111"})), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES&filter=bio1:mean:gte:10")
    assert r.status_code == 200
    body = r.json()
    assert body["scope"]["filters"] == ["bio1:mean:gte:10"]
    ids = [row["taxon_id"] for row in body["results"]]
    assert ids == ["111"]


def test_query_taxa_stat_filter_malformed_returns_422():
    r = client.get("/api/taxa/query?q=opuntia&filter=bio1:mean:xyz:10")
    assert r.status_code == 422
    assert "unknown filter operator" in r.json()["detail"]


def test_query_taxa_stat_filter_chains_multiple():
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}

    def _resolve(k):
        return {"10": genus, "2923970": TAXON}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "iter_descendants", return_value=[TAXON]), \
         patch.object(rankings_module, "_batch_sample_counts", return_value={"2923970": 50}), \
         patch.object(rankings_module, "_apply_stat_filters", return_value=frozenset({"2923970"})) as mock_apply, \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]):
        r = client.get(
            "/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
            "&filter=bio1:mean:gte:10&filter=bio1:std:lt:5"
        )
    assert r.status_code == 200
    # both filters parsed and passed through as one list, in order
    assert mock_apply.called
    passed_filters = mock_apply.call_args.args[1]
    assert [f.op for f in passed_filters] == ["gte", "lt"]


FAKE_LAYER = {
    "id": "bio1",
    "display_name": "Annual Mean Temperature",
    "units": "°C",
    "value_type": "interval",
    "source": "chelsa_v2_1",
    "filename": "bio1.tif",
    "scale_factor": 0.1,
    "add_offset": -273.15,
    "render_min": -50.0,
    "render_max": 35.0,
}
FAKE_CATEGORY = {"id": "bioclimate", "display_name": "Bioclimatic"}
FAKE_LAYER_WITH_IMPERIAL = {**FAKE_LAYER, "imperial_unit": "°F"}


def test_list_variables():
    with patch.object(tiles, "load_layers_with_category", return_value=[(FAKE_LAYER, FAKE_CATEGORY)]):
        response = client.get("/variables")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "bio1"
    assert body[0]["category"] == "Bioclimatic"
    assert body[0]["value_type"] == "continuous"
    assert body[0]["raw_value_type"] == "interval"


def test_list_layers():
    with patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]):
        response = client.get("/api/layers")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "bio1"


def test_layer_tile():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png):
        response = client.get("/api/layers/bio1/tiles/4/8/5.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_layer_tile_not_found():
    with patch.object(tiles, "get_layer", side_effect=KeyError("nope")):
        response = client.get("/api/layers/nope/tiles/4/8/5.png")
    assert response.status_code == 404


def test_elevation_terrain_tile():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "render_elevation_terrain_rgb_tile_bytes", return_value=png) as mock_render:
        response = client.get("/api/layers/elevation/terrain-tiles/4/8/5.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    mock_render.assert_called_once_with(4, 8, 5, 256)


def test_satellite_tile_proxies_to_esri_with_key_and_referer():
    # Well above _ARCGIS_NO_DATA_TILE_MAX_BYTES, representing a real tile.
    jpg = b"\xff\xd8\xff" + b"\x00" * 5000
    mock_response = MagicMock()
    mock_response.content = jpg
    mock_response.raise_for_status = MagicMock()
    with patch.object(main_module, "_ARCGIS_API_KEY", "fake-key"), \
         patch("httpx.get", return_value=mock_response) as mock_get:
        response = client.get("/api/tiles/satellite/4/8/5.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == jpg
    call_args = mock_get.call_args
    assert call_args.args[0] == "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/4/5/8"
    assert call_args.kwargs["params"] == {"token": "fake-key"}
    assert call_args.kwargs["headers"] == {"Referer": "https://wherewild.net"}


def test_satellite_tile_without_configured_key_returns_500():
    with patch.object(main_module, "_ARCGIS_API_KEY", None):
        response = client.get("/api/tiles/satellite/4/8/5.jpg")
    assert response.status_code == 500


def test_satellite_tile_upstream_error_returns_502():
    with patch.object(main_module, "_ARCGIS_API_KEY", "fake-key"), \
         patch("httpx.get", side_effect=main_module.httpx.ConnectError("boom")):
        response = client.get("/api/tiles/satellite/4/8/5.jpg")
    assert response.status_code == 502


def test_satellite_tile_no_data_placeholder_returns_404():
    # Matches the real confirmed no-data placeholder size (2521 bytes).
    placeholder = b"\xff\xd8\xff" + b"\x00" * 2518
    mock_response = MagicMock()
    mock_response.content = placeholder
    mock_response.raise_for_status = MagicMock()
    with patch.object(main_module, "_ARCGIS_API_KEY", "fake-key"), \
         patch("httpx.get", return_value=mock_response):
        response = client.get("/api/tiles/satellite/20/191949/394497.jpg")
    assert response.status_code == 404


def test_satellite_tile_at_no_data_threshold_boundary_returns_404():
    exactly_at_max = b"\x00" * main_module._ARCGIS_NO_DATA_TILE_MAX_BYTES
    mock_response = MagicMock()
    mock_response.content = exactly_at_max
    mock_response.raise_for_status = MagicMock()
    with patch.object(main_module, "_ARCGIS_API_KEY", "fake-key"), \
         patch("httpx.get", return_value=mock_response):
        response = client.get("/api/tiles/satellite/4/8/5.jpg")
    assert response.status_code == 404


def test_satellite_tile_just_above_no_data_threshold_returns_200():
    just_above_max = b"\x00" * (main_module._ARCGIS_NO_DATA_TILE_MAX_BYTES + 1)
    mock_response = MagicMock()
    mock_response.content = just_above_max
    mock_response.raise_for_status = MagicMock()
    with patch.object(main_module, "_ARCGIS_API_KEY", "fake-key"), \
         patch("httpx.get", return_value=mock_response):
        response = client.get("/api/tiles/satellite/4/8/5.jpg")
    assert response.status_code == 200


def test_variable_tile_compat():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png):
        response = client.get("/api/variables/bio_1/tiles/4/8/5.png")
    assert response.status_code == 200


def test_layer_tile_with_chain_param():
    """The `chain` query param (JSON) is parsed and forwarded to render_layer_tile_bytes."""
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    chain_json = json.dumps([
        {"layer_id": "kg2", "class_filter": [3, 4]},
        {"layer_id": "bio2", "value_ranges": [[1.0, 2.0], [5.0, 6.0]]},
    ])
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get(f"/api/layers/bio1/tiles/4/8/5.png?chain={chain_json}")
    assert response.status_code == 200
    call_args = mock_render.call_args
    passed_chain = call_args.args[-2]
    assert passed_chain == [
        {"layer_id": "kg2", "class_filter": [3, 4], "value_ranges": None},
        {"layer_id": "bio2", "class_filter": None, "value_ranges": [(1.0, 2.0), (5.0, 6.0)]},
    ]


def test_layer_tile_no_chain_param_passes_none():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get("/api/layers/bio1/tiles/4/8/5.png")
    assert response.status_code == 200
    assert mock_render.call_args.args[-2] is None


def test_layer_tile_with_render_range_param_forwards_parsed_tuple():
    """The `render_range` query param (JSON [min,max]) is parsed, unit-
    converted, and forwarded to render_layer_tile_bytes as a raw tuple —
    same convention as value_ranges/chain."""
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    render_range_json = json.dumps([32.0, 50.0])
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER_WITH_IMPERIAL), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get(
            f"/api/layers/bio1/tiles/4/8/5.png?render_range={render_range_json}&unit_system=imperial",
        )
    assert response.status_code == 200
    passed_min, passed_max = mock_render.call_args.args[-1]
    assert passed_min == pytest.approx(0.0)
    assert passed_max == pytest.approx(10.0)


def test_layer_tile_no_render_range_param_passes_none():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get("/api/layers/bio1/tiles/4/8/5.png")
    assert response.status_code == 200
    assert mock_render.call_args.args[-1] is None


def test_layer_tile_malformed_render_range_param_passes_none():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get("/api/layers/bio1/tiles/4/8/5.png?render_range=not-json")
    assert response.status_code == 200
    assert mock_render.call_args.args[-1] is None


def test_layer_tile_range_stats_converts_to_display_units():
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER_WITH_IMPERIAL), \
         patch.object(tiles, "layer_tile_range_stats", return_value=(0.0, 10.0)) as mock_stats:
        response = client.get(
            "/api/layers/bio1/tile-range/stats?z=4&x0=8&y0=5&x1=9&y1=6&unit_system=imperial",
        )
    assert response.status_code == 200
    body = response.json()
    assert body["min"] == pytest.approx(32.0)
    assert body["max"] == pytest.approx(50.0)
    mock_stats.assert_called_once_with("bio1", 4, 8, 5, 9, 6, "")


def test_layer_tile_range_stats_all_nodata_returns_null_range():
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "layer_tile_range_stats", return_value=None):
        response = client.get(
            "/api/layers/bio1/tile-range/stats?z=4&x0=8&y0=5&x1=9&y1=6",
        )
    assert response.status_code == 200
    assert response.json() == {"min": None, "max": None}


def test_layer_tile_range_stats_not_found():
    with patch.object(tiles, "get_layer", side_effect=KeyError("nope")):
        response = client.get(
            "/api/layers/nope/tile-range/stats?z=4&x0=8&y0=5&x1=9&y1=6",
        )
    assert response.status_code == 404


def test_layer_tile_range_stats_forecast_h_forwarded():
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "layer_tile_range_stats", return_value=(0.0, 10.0)) as mock_stats:
        response = client.get(
            "/api/layers/bio1/tile-range/stats?z=4&x0=8&y0=5&x1=9&y1=6&forecast_h=24",
        )
    assert response.status_code == 200
    mock_stats.assert_called_once_with("bio1", 4, 8, 5, 9, 6, "__f024h")


def test_layer_tile_with_value_ranges_param():
    """The `value_ranges` query param (JSON list of [min,max] pairs) is parsed,
    unit-converted, and forwarded to render_layer_tile_bytes as a list of tuples."""
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    ranges_json = json.dumps([[1.0, 2.0], [5.0, 6.0]])
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get(
            f"/api/layers/bio1/tiles/4/8/5.png?value_ranges={ranges_json}&unit_system=metric",
        )
    assert response.status_code == 200
    call_args = mock_render.call_args
    passed_value_ranges = call_args.args[9]
    assert passed_value_ranges == [(1.0, 2.0), (5.0, 6.0)]


def test_layer_tile_no_value_ranges_param_passes_none():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png) as mock_render:
        response = client.get("/api/layers/bio1/tiles/4/8/5.png")
    assert response.status_code == 200
    assert mock_render.call_args.args[9] is None


# ---------------------------------------------------------------------------
# Shared fixtures for new tests
# ---------------------------------------------------------------------------

FAKE_NOM_LAYER = {
    "id": "kg2",
    "display_name": "Koppen-Geiger Climate",
    "units": None,
    "value_type": "nominal",
    "domain": None,
    "source": None,
}

NONLEAF_TAXON = {
    "taxon_key": "2923968",
    "path": "Plantae_6/Opuntia_2923968",
    "scientific_name": "Opuntia",
    "rank": "GENUS",
}

DESC_TAXON = {
    "taxon_key": "2923970",
    "path": "Plantae_6/Opuntia_2923968/Opuntia_humifusa_2923970",
    "scientific_name": "Opuntia_humifusa",
    "rank": "SPECIES",
}

_NUM_STATS_TABLE = pa.table({
    "variable": ["bio1"],
    "count": [100],
    "min": [5.0],
    "mean": [15.0],
    "max": [25.0],
    "std": [3.0],
    "10th_percentile": [8.0],
    "90th_percentile": [22.0],
})

_NOM_STATS_TABLE = pa.table({
    "variable": ["kg2", "kg2", "kg2"],
    "metric": ["total_samples", "class_1", "class_2"],
    "value": [100.0, 0.6, 0.4],
})

_DENSITY_TABLE = pa.table({
    "variable": ["bio1"],
    "points": [[1.0, 2.0, 3.0]],
    "density": [[0.25, 0.5, 0.25]],
    "bandwidth": [0.5],
    "count": [100],
    "sampleCount": [100],
    "pointCount": [3],
    "min": [1.0],
    "max": [3.0],
})

_OCC_TABLE = pa.table({
    "catalogNumber": ["OCC001", "OCC002"],
    "decimalLatitude": [40.5, 41.0],
    "decimalLongitude": [-75.0, -74.5],
    "obscured": ["No", "No"],
    "coordinateUncertaintyInMeters": [100.0, 200.0],
})


def _env_stats_read(path, **kw):
    return {
        st_module.NUMERICAL_STATS_FILE: _NUM_STATS_TABLE,
        st_module.NOMINAL_STATS_FILE: _NOM_STATS_TABLE,
        st_module.DENSITY_FILE: _DENSITY_TABLE,
    }.get(Path(str(path)).name, pa.table({}))


def test_load_legend_missing_returns_empty():
    main_module._load_legend.cache_clear()
    assert main_module._load_legend("no_such_layer_xyz") == []


def test_load_legend_temporal_fallback(tmp_path, monkeypatch):
    data = {"classes": [{"id": 0, "name": "Clear sky"}]}
    (tmp_path / "weather_code_simple_legend.json").write_text(json.dumps(data))
    monkeypatch.setattr(main_module, "_LEGEND_DIR", tmp_path)
    main_module._load_legend.cache_clear()
    assert main_module._load_legend("weather_code_simple_mode_24h") == [{"id": 0, "name": "Clear sky"}]
    assert main_module._load_legend("weather_code_simple_mode_168h") == [{"id": 0, "name": "Clear sky"}]


# ---------------------------------------------------------------------------
# _filter_occ_df (lines 32-36)
# ---------------------------------------------------------------------------

def test_filter_occ_df_removes_obscured():
    df = pd.DataFrame({"obscured": ["No", "Yes", "No"], "x": [1, 2, 3]})
    result = main_module._filter_occ_df(df)
    assert list(result["x"]) == [1, 3]


def test_filter_occ_df_removes_high_uncertainty():
    df = pd.DataFrame({"coordinateUncertaintyInMeters": [100.0, 501.0, 500.0]})
    assert len(main_module._filter_occ_df(df)) == 2


def test_filter_occ_df_passthrough():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(main_module._filter_occ_df(df)) == 3


# ---------------------------------------------------------------------------
# /api/species/{id}/obscured (lines 120-123)
# ---------------------------------------------------------------------------

def test_get_species_obscured_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON]):
        r = client.get("/api/species/2923970/obscured")
    assert r.status_code == 200
    assert not r.json()["allObscured"]


def test_get_species_obscured_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/species/nope/obscured")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/taxon/{id}/env-stats (lines 128-184)
# ---------------------------------------------------------------------------

def test_get_taxon_env_stats_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/taxon/nope/env-stats")
    assert r.status_code == 404


def test_get_taxon_env_stats_all_files():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_env_stats_read):
        r = client.get("/api/taxon/2923970/env-stats")
    assert r.status_code == 200
    body = r.json()
    bio1 = next(v for v in body["variables"] if v["id"] == "bio1")
    assert bio1["stats"]["count"] == 100
    assert bio1["density"] is not None
    kg2 = next(v for v in body["variables"] if v["id"] == "kg2")
    assert kg2["density"] is None
    assert kg2["classes"] is not None


def test_get_taxon_env_stats_no_files():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[]), \
         patch.object(pq, "read_table", return_value=pa.table({})):
        r = client.get("/api/taxon/2923970/env-stats")
    assert r.status_code == 200
    assert r.json()["variables"] == []


# ---------------------------------------------------------------------------
# /species/{id}/environment/{var} (lines 193-270)
# ---------------------------------------------------------------------------

def test_get_species_environment_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/environment/bio1")
    assert r.status_code == 404


def test_get_species_environment_nominal_no_file():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch.object(pq, "read_table", return_value=pa.table({})):
        r = client.get("/species/2923970/environment/kg2")
    assert r.status_code == 404


def test_get_species_environment_nominal_no_rows():
    empty = pa.table({
        "variable": pa.array([], pa.string()),
        "metric": pa.array([], pa.string()),
        "value": pa.array([], pa.float64()),
    })
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=empty), \
         patch("main._load_legend", return_value=[]):
        r = client.get("/species/2923970/environment/kg2")
    assert r.status_code == 404


def test_get_species_environment_nominal_success():
    legend = [
        {"id": 1, "name": "Tropical", "description": "Wet", "traits": {"color": "#0f0"}},
        {"id": 2, "name": "Arid", "description": "Dry", "traits": None},
    ]
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_env_stats_read), \
         patch("main._load_legend", return_value=legend):
        r = client.get("/species/2923970/environment/kg2")
    assert r.status_code == 200
    body = r.json()
    assert body["variable"] == "kg2"
    assert body["density_curve"] is None
    dist = body["categorical_distribution"]
    assert len(dist) == 2
    assert dist[0]["fraction"] == pytest.approx(0.6)
    assert dist[0]["color"] == "#0f0"
    assert dist[1]["color"] is None


def test_get_species_environment_numerical_no_file():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch.object(pq, "read_table", return_value=pa.table({})):
        r = client.get("/species/2923970/environment/bio1")
    assert r.status_code == 404


def test_get_species_environment_numerical_no_row():
    empty_num = pa.table({"variable": pa.array([], pa.string())})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=empty_num):
        r = client.get("/species/2923970/environment/bio1")
    assert r.status_code == 404


def test_get_species_environment_numerical_with_density():
    def _read(path, **kw):
        name = Path(str(path)).name
        if name == st_module.NUMERICAL_STATS_FILE:
            return _NUM_STATS_TABLE
        return pa.table({"variable": ["bio1"], "points": [[1.0, 2.0]], "density": [[0.5, 0.5]]})

    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_read):
        r = client.get("/species/2923970/environment/bio1")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["count"] == 100
    assert body["density_curve"]["points"] == [1.0, 2.0]


def test_get_species_environment_numerical_no_density_row():
    # density file exists but has no row for bio1 → density_curve=None
    def _read(path, **kw):
        name = Path(str(path)).name
        if name == st_module.NUMERICAL_STATS_FILE:
            return _NUM_STATS_TABLE
        return pa.table({})

    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_read):
        r = client.get("/species/2923970/environment/bio1")
    assert r.status_code == 200
    assert r.json()["density_curve"] is None


def test_get_species_environment_underscore_variable():
    # bio_1 must be normalized to bio1
    def _read(path, **kw):
        name = Path(str(path)).name
        if name == st_module.NUMERICAL_STATS_FILE:
            return _NUM_STATS_TABLE
        return pa.table({})

    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_read):
        r = client.get("/species/2923970/environment/bio_1")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /species/{id}/occurrences (lines 283-310)
# ---------------------------------------------------------------------------

def test_get_species_occurrences_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/occurrences")
    assert r.status_code == 404


def test_get_species_occurrences_leaf():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON]), \
         patch.object(pq, "read_schema", return_value=_OCC_TABLE.schema), \
         patch.object(pq, "read_table", return_value=_OCC_TABLE):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    occs = r.json()["occurrences"]
    assert len(occs) == 2
    assert occs[0]["catalogNumber"] == "OCC001"
    assert occs[0]["latitude"] == pytest.approx(40.5)


def test_get_species_occurrences_leaf_no_file():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON]), \
         patch("pathlib.Path.exists", return_value=False):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    assert r.json()["occurrences"] == []


def test_get_species_occurrences_subspecies():
    subspecies_taxon = {**TAXON, "rank": "SUBSPECIES"}
    with patch.object(taxa, "get_taxon_by_id", return_value=subspecies_taxon), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(pq, "read_schema", return_value=_OCC_TABLE.schema), \
         patch.object(pq, "read_table", return_value=_OCC_TABLE):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    assert len(r.json()["occurrences"]) == 2


def test_get_species_occurrences_nonleaf():
    with patch.object(taxa, "get_taxon_by_id", return_value=NONLEAF_TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[DESC_TAXON]), \
         patch.object(pq, "read_schema", return_value=_OCC_TABLE.schema), \
         patch.object(pq, "read_table", return_value=_OCC_TABLE):
        r = client.get("/species/2923968/occurrences")
    assert r.status_code == 200
    assert len(r.json()["occurrences"]) == 2


def test_get_species_occurrences_species_includes_subspecies():
    """SPECIES occurrences endpoint scopes to self + descendants to include subspecies."""
    subspecies = {**DESC_TAXON, "taxon_key": "9999", "rank": "SUBSPECIES",
                  "path": DESC_TAXON["path"] + "/Sub_9999"}
    # One consolidated-file read now covers the whole scope (species + subspecies rows).
    combined_table = pa.table({
        "catalogNumber": ["OCC001", "OCC002", "SUB001"],
        "decimalLatitude": [40.5, 41.0, 41.0],
        "decimalLongitude": [-75.0, -74.5, -76.0],
        "obscured": ["No", "No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0, 100.0],
    })

    def _read_table_side_effect(path, **kwargs):
        if "numerical_stats" in str(path):
            return pa.table({"taxon_key": pa.array([], type=pa.string()), "count": pa.array([], type=pa.int64())})
        return combined_table

    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON, subspecies]), \
         patch.object(pq, "read_schema", return_value=combined_table.schema), \
         patch.object(pq, "read_table", side_effect=_read_table_side_effect):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    occs = r.json()["occurrences"]
    catalog_numbers = {o["catalogNumber"] for o in occs}
    assert "OCC001" in catalog_numbers
    assert "SUB001" in catalog_numbers


def test_get_species_occurrences_deduplication():
    dup_table = pa.table({
        "catalogNumber": ["DUP001", "DUP001"],
        "decimalLatitude": [40.5, 40.5],
        "decimalLongitude": [-75.0, -75.0],
        "obscured": ["No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0],
    })
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON]), \
         patch.object(pq, "read_schema", return_value=dup_table.schema), \
         patch.object(pq, "read_table", return_value=dup_table):
        r = client.get("/species/2923970/occurrences")
    assert len(r.json()["occurrences"]) == 1


def test_get_species_occurrences_includes_media():
    media_table = pa.table({
        "catalogNumber": ["OCC001", "OCC002"],
        "decimalLatitude": [40.5, 41.0],
        "decimalLongitude": [-75.0, -74.5],
        "obscured": ["No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 200.0],
        "mediaUrl": ["https://example.com/1.jpg", None],
        "mediaAttribution": ["Jane Doe", None],
        "mediaLicense": ["https://creativecommons.org/licenses/by-nc/4.0/", None],
    })
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON]), \
         patch.object(pq, "read_schema", return_value=media_table.schema), \
         patch.object(pq, "read_table", return_value=media_table):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    occs = {o["catalogNumber"]: o for o in r.json()["occurrences"]}
    with_media = occs["OCC001"]
    assert with_media["media_url"] == "https://example.com/1.jpg"
    assert with_media["media_attribution"] == "Jane Doe"
    assert with_media["media_license_url"] == "https://creativecommons.org/licenses/by-nc/4.0/"
    assert with_media["media_license"] == "CC BY-NC 4.0"
    without_media = occs["OCC002"]
    assert "media_url" not in without_media
    assert "media_attribution" not in without_media
    assert "media_license" not in without_media


# ---------------------------------------------------------------------------
# /occurrence/{catalog_number}
# ---------------------------------------------------------------------------

_CATALOG_NUMBER_INDEX_TABLE = pa.table({
    "catalogNumber": ["143391331"],
    "taxon_key": ["2923970"],
    "decimalLatitude": [40.5],
    "decimalLongitude": [-75.0],
})


_EMPTY_CATALOG_NUMBER_INDEX_TABLE = pa.table({
    "catalogNumber": pa.array([], type=pa.string()),
    "taxon_key": pa.array([], type=pa.string()),
    "decimalLatitude": pa.array([], type=pa.float64()),
    "decimalLongitude": pa.array([], type=pa.float64()),
})


def test_get_occurrence_found():
    with patch.object(pq, "read_table", return_value=_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/occurrence/143391331")
    assert r.status_code == 200
    body = r.json()
    assert body["catalog_number"] == "143391331"
    assert body["taxon_id"] == "2923970"
    assert body["scientific_name"] == "Opuntia humifusa"
    assert body["common_name"] == "devil's tongue"
    assert body["slug"] == "opuntia-humifusa"
    assert body["latitude"] == pytest.approx(40.5)
    assert body["longitude"] == pytest.approx(-75.0)
    assert body["ingested"] is True
    # No media/timestamp fields on the ingested path — the frontend already
    # has those from the taxon's normal /occurrences fetch once it lands on
    # the species page; only the not-ingested fallback needs to carry them.
    assert "media_url" not in body
    assert "event_timestamp" not in body


def test_get_occurrence_taxon_not_found():
    # Found in our index, but the taxon it points to doesn't resolve — a
    # data-integrity gap distinct from "not ingested", so no iNat fallback.
    with patch.object(pq, "read_table", return_value=_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(taxa, "get_taxon_by_id", return_value=None) as mock_by_id, \
         patch.object(main_module, "_lookup_inat_observation") as mock_fallback:
        r = client.get("/occurrence/143391331")
    assert r.status_code == 404
    mock_by_id.assert_called_once()
    mock_fallback.assert_not_called()


def _inat_observation_response(
    taxon_id: int = 48815, lat: float = 41.0, lon: float = -76.0,
    time_observed_at: str | None = None, photos: list | None = None,
):
    return {
        "results": [{
            "taxon": {"id": taxon_id},
            "geojson": {"type": "Point", "coordinates": [lon, lat]},
            "time_observed_at": time_observed_at,
            "photos": photos or [],
        }],
    }


def test_get_occurrence_not_in_index_falls_back_to_inat():
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    assert r.status_code == 200
    body = r.json()
    assert body["taxon_id"] == "2923970"
    assert body["ingested"] is False
    assert body["latitude"] == pytest.approx(41.0)
    assert body["longitude"] == pytest.approx(-76.0)
    assert body["event_timestamp"] is None
    assert body["media_url"] is None
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"] == {"id": "999888777"}


def test_get_occurrence_fallback_parses_timestamp():
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(time_observed_at="2018-09-05T14:06:00+02:00"),
            raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    assert r.status_code == 200
    assert r.json()["event_timestamp"] == 1536149160


def test_get_occurrence_fallback_includes_usable_license_photo():
    photos = [{
        "url": "https://static.inaturalist.org/photos/61482854/square.jpg?1581761020",
        "attribution": "(c) Andrew Harvey, some rights reserved (CC BY)",
        "license_code": "cc-by",
    }]
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(photos=photos), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    body = r.json()
    assert body["media_url"] == "https://static.inaturalist.org/photos/61482854/original.jpg?1581761020"
    # Reduced to the bare name — matches the ingested path's mediaAttribution
    # shape (multimedia.txt's rightsHolder), not iNat's own boilerplate
    # wording, since the license is already shown separately.
    assert body["media_attribution"] == "Andrew Harvey"
    assert body["media_license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert body["media_license"] == "CC BY 4.0"


def test_get_occurrence_fallback_attribution_falls_back_to_raw_on_unrecognized_format():
    photos = [{
        "url": "https://static.inaturalist.org/photos/1/square.jpg",
        "attribution": "Photo by Lucas Pearce",
        "license_code": "cc-by-nc",
    }]
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(photos=photos), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    assert r.json()["media_attribution"] == "Photo by Lucas Pearce"


def test_get_occurrence_fallback_skips_unusable_license_photo_for_next():
    photos = [
        {
            "url": "https://static.inaturalist.org/photos/1/square.jpg",
            "attribution": "all rights reserved",
            "license_code": "",
        },
        {
            "url": "https://static.inaturalist.org/photos/2/square.jpg",
            "attribution": "(c) Jane Doe, some rights reserved (CC0)",
            "license_code": "cc0",
        },
    ]
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(photos=photos), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    body = r.json()
    assert body["media_url"] == "https://static.inaturalist.org/photos/2/original.jpg"


def test_get_occurrence_fallback_no_usable_photos():
    photos = [{
        "url": "https://static.inaturalist.org/photos/1/square.jpg",
        "attribution": "all rights reserved",
        "license_code": "",
    }]
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(photos=photos), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    body = r.json()
    assert body["media_url"] is None
    assert body["media_attribution"] is None
    assert body["media_license"] is None
    assert body["media_license_url"] is None


def test_get_occurrence_index_file_missing_falls_back_to_inat():
    with patch.object(pq, "read_table", side_effect=FileNotFoundError), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=TAXON):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/143391331")
    assert r.status_code == 200
    assert r.json()["ingested"] is False


def test_get_occurrence_inat_fallback_no_results():
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: {"results": []}, raise_for_status=lambda: None)
        r = client.get("/occurrence/nope")
    assert r.status_code == 404


def test_get_occurrence_inat_fallback_unmapped_taxon():
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get") as mock_get, \
         patch.object(taxa, "get_taxon_by_inat_id", return_value=None):
        mock_get.return_value = MagicMock(
            json=lambda: _inat_observation_response(), raise_for_status=lambda: None,
        )
        r = client.get("/occurrence/999888777")
    assert r.status_code == 404


def test_get_occurrence_inat_fallback_network_error():
    with patch.object(pq, "read_table", return_value=_EMPTY_CATALOG_NUMBER_INDEX_TABLE), \
         patch.object(main_module.httpx, "get", side_effect=Exception("boom")):
        r = client.get("/occurrence/999888777")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /species/{id}/locations
# ---------------------------------------------------------------------------

_LOC_TABLE = pa.table({
    "scope": ["gadm_level0", "gadm_level0", "gadm_level1", "gbif_region"],
    "gid": ["USA", "CAN", "USA.1_1", "NORTH_AMERICA"],
    "taxon_key": ["2923970", "2923970", "2923970", "2923970"],
    "count": [100, 20, 80, 500],
})

_HIERARCHY_CSV = (
    "level,gid,name,parent_gid\n"
    "0,USA,United States,\n"
    "0,CAN,Canada,\n"
    "1,USA.1_1,California,USA\n"
)


def _patch_locations(tmp_path: Path, monkeypatch):
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(_LOC_TABLE, loc_path)
    hier_path = tmp_path / "hierarchy.csv"
    hier_path.write_text(_HIERARCHY_CSV, encoding="utf-8")
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    return loc_path


def test_get_species_locations_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/locations")
    assert r.status_code == 404


def test_get_species_locations_no_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", tmp_path / "missing.parquet")
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    assert r.status_code == 200
    assert r.json() == []


def test_get_species_locations_returns_results(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    assert r.status_code == 200
    data = r.json()
    gids = {row["gid"] for row in data}
    assert "USA" in gids
    assert "CAN" in gids
    assert "USA.1_1" in gids
    assert "NORTH_AMERICA" in gids


def test_get_species_locations_response_shape(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    usa = next(row for row in r.json() if row["gid"] == "USA")
    assert usa["name"] == "United States"
    assert usa["level"] == 0
    assert usa["count"] == 100
    assert isinstance(usa["hierarchy"], list)


def test_get_species_locations_hierarchy(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    state = next(row for row in r.json() if row["gid"] == "USA.1_1")
    assert state["name"] == "California"
    assert state["level"] == 1
    assert "United States" in state["hierarchy"]


def test_get_species_locations_gbif_region_level(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    region = next(row for row in r.json() if row["gid"] == "NORTH_AMERICA")
    assert region["level"] == -1
    assert region["count"] == 500


def test_get_species_locations_sorted_by_count(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    counts = [row["count"] for row in r.json()]
    assert counts == sorted(counts, reverse=True)


def test_get_species_locations_level_filter(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations?level=0")
    gids = {row["gid"] for row in r.json()}
    assert "USA" in gids
    assert "CAN" in gids
    assert "USA.1_1" not in gids
    assert "NORTH_AMERICA" not in gids


def test_get_species_locations_limit(tmp_path, monkeypatch):
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations?limit=2")
    assert len(r.json()) == 2


def test_get_species_locations_no_data_for_taxon(tmp_path, monkeypatch):
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(pa.table({
        "scope": ["gadm_level0"],
        "gid": ["USA"],
        "taxon_key": ["9999999"],
        "count": [1],
    }), loc_path)
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    (tmp_path / "hierarchy.csv").write_text("level,gid,name,parent_gid\n")
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    assert r.json() == []


def test_get_species_locations_parquet_read_error(tmp_path, monkeypatch):
    loc_path = tmp_path / "bad.parquet"
    loc_path.write_bytes(b"not a parquet file")
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    assert r.status_code == 200
    assert r.json() == []


def test_get_species_locations_missing_hierarchy(tmp_path, monkeypatch):
    # hierarchy.csv absent → _load_hierarchy returns {} → gid used as name
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(pa.table({
        "scope": ["gadm_level0"],
        "gid": ["USA"],
        "taxon_key": ["2923970"],
        "count": [5],
    }), loc_path)
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    assert r.status_code == 200
    row = r.json()[0]
    assert row["gid"] == "USA"
    assert row["name"] == "USA"  # falls back to gid when no hierarchy
    assert row["hierarchy"] == []


def test_get_species_locations_unknown_scope_skipped(tmp_path, monkeypatch):
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(pa.table({
        "scope": ["unknown_scope", "gadm_level0"],
        "gid": ["X1", "USA"],
        "taxon_key": ["2923970", "2923970"],
        "count": [99, 10],
    }), loc_path)
    hier_path = tmp_path / "hierarchy.csv"
    hier_path.write_text("level,gid,name,parent_gid\n0,USA,United States,\n")
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    gids = {row["gid"] for row in r.json()}
    assert "X1" not in gids
    assert "USA" in gids


def test_get_species_locations_broken_parent_chain(tmp_path, monkeypatch):
    # parent_gid references a gid not in hierarchy → _resolve_hierarchy breaks cleanly
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(pa.table({
        "scope": ["gadm_level1"],
        "gid": ["USA.1_1"],
        "taxon_key": ["2923970"],
        "count": [7],
    }), loc_path)
    hier_path = tmp_path / "hierarchy.csv"
    hier_path.write_text("level,gid,name,parent_gid\n1,USA.1_1,California,MISSING\n")
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations")
    row = r.json()[0]
    assert row["gid"] == "USA.1_1"
    assert row["hierarchy"] == []  # parent lookup failed, chain stops


# ---------------------------------------------------------------------------
# _location_filter_col
# ---------------------------------------------------------------------------

def _patch_hierarchy(monkeypatch, by_gid: dict) -> None:
    """Patch _load_hierarchy to return by_gid without filesystem interaction."""
    main_module._load_hierarchy.cache_clear()
    monkeypatch.setattr(main_module, "_load_hierarchy", lambda: by_gid)


_USA = {"level": 0, "name": "United States", "parent_gid": None}
_CA  = {"level": 1, "name": "California", "parent_gid": "USA"}
_LA  = {"level": 2, "name": "Los Angeles", "parent_gid": "USA.1_1"}


def test_location_filter_col_level0(monkeypatch):
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    assert main_module._location_filter_col("USA") == "level0Gid"


def test_location_filter_col_level1(monkeypatch):
    _patch_hierarchy(monkeypatch, {"USA.1_1": _CA})
    assert main_module._location_filter_col("USA.1_1") == "level1Gid"


def test_location_filter_col_level2(monkeypatch):
    _patch_hierarchy(monkeypatch, {"USA.1.1_1": _LA})
    assert main_module._location_filter_col("USA.1.1_1") == "level2Gid"


def test_location_filter_col_unknown_returns_gbif_region(monkeypatch):
    _patch_hierarchy(monkeypatch, {})
    assert main_module._location_filter_col("NORTH_AMERICA") == "gbifRegion"


# ---------------------------------------------------------------------------
# /species/{id}/environment/{var} with location param
# ---------------------------------------------------------------------------

def _make_occ_with_loc(tmp_path: Path, taxon_key: str, loc_col: str, gid: str, var_col: str, values: list) -> Path:
    occ_dir = tmp_path / "taxonomy"
    occ_dir.mkdir(parents=True, exist_ok=True)
    n = len(values)
    data = {
        "catalogNumber": [f"obs{i}" for i in range(n)],
        "decimalLatitude": [40.0] * n,
        "decimalLongitude": [-75.0] * n,
        "obscured": ["No"] * n,
        "coordinateUncertaintyInMeters": [100.0] * n,
        "taxon_key": [taxon_key] * n,
        loc_col: [gid] * n,
        var_col: values,
    }
    occ_path = occ_dir / "occurrences.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False), occ_path)
    return occ_path


def _patch_stats_storage(monkeypatch, tmp_path: Path) -> None:
    """Point util.stats at a tmp consolidated occurrences file + a catalog
    containing just TAXON (subtree scoping resolves through the catalog now,
    not a stored path column)."""
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "OCCURRENCES_FILE", tmp_path / "taxonomy" / "occurrences.parquet")
    monkeypatch.setattr(st_module, "load_catalog", lambda: {TAXON["taxon_key"]: TAXON})


def test_get_species_environment_with_location_continuous(tmp_path, monkeypatch):
    import numpy as np

    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "USA", "bio1",
                       list(np.linspace(5.0, 25.0, 20)))
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]):
        r = client.get("/species/2923970/environment/bio1?location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["observation_count"] == 20
    assert body["density_curve"] is not None
    assert body["relative_ranks"] == []
    assert body["categorical_distribution"] is None


def test_get_species_environment_with_location_nominal(tmp_path, monkeypatch):
    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "USA", "kg2",
                       [1.0] * 15 + [2.0] * 5)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    legend = [{"id": 1, "name": "Tropical", "description": None, "traits": None},
              {"id": 2, "name": "Arid", "description": None, "traits": {"color": "#f00"}}]
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("main._load_legend", return_value=legend):
        r = client.get("/species/2923970/environment/kg2?location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["observation_count"] == 20
    dist = body["categorical_distribution"]
    assert len(dist) == 2
    assert dist[0]["fraction"] == pytest.approx(0.75)
    assert body["relative_ranks"] == []


def test_get_species_environment_with_location_no_data_falls_through(monkeypatch):
    """compute_location_filtered_stats returns None → 404 (no fallback to precomputed)."""
    monkeypatch.setattr(st_module, "collect_taxon_df", lambda t, **kwargs: None)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_env_stats_read), \
         patch("main.iter_descendants", return_value=[TAXON]):
        r = client.get("/species/2923970/environment/bio1?location=USA")
    assert r.status_code == 404


def test_get_species_environment_with_location_no_layer_falls_through(monkeypatch):
    """layer=None skips location block and falls through to precomputed path."""
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[]), \
         patch.object(pq, "read_table", return_value=pa.table({})):
        r = client.get("/species/2923970/environment/bio1?location=USA")
    assert r.status_code == 404


def test_get_species_environment_with_location_filter_col_none(monkeypatch):
    """filter_col None (level not in 0-2) falls through to precomputed."""
    _patch_hierarchy(monkeypatch, {"WEIRD": {"level": 99, "name": "Weird", "parent_gid": None}})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_env_stats_read):
        r = client.get("/species/2923970/environment/bio1?location=WEIRD")
    assert r.status_code == 200
    assert r.json()["observation_count"] == 100  # from precomputed


def test_get_species_environment_with_extra_filter_reflects_chained_slice(tmp_path, monkeypatch):
    """The density curve/summary for bio1 should be computed over just the
    extra-filtered subset (kg2==1), the same on-the-fly recompute path
    location/phenology filters already use — not the full occurrence set."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_var_occ(occurrences_file)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "kg2", "classValue": 1}])
        r = client.get(f"/species/2923970/environment/bio1?extra={extra}")
    assert r.status_code == 200
    body = r.json()
    # Only A (bio1=10, kg2=1) and B (bio1=20, kg2=1) match; C (kg2=2) is excluded.
    assert body["observation_count"] == 2


def test_get_species_environment_with_extra_filter_reflects_chained_slice_categorical(tmp_path, monkeypatch):
    """Same recompute path, but for a categorical primary variable (kg2)
    chained with a numeric extra filter on bio1 — the class distribution
    should be computed over just the bio1-filtered subset."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_var_occ(occurrences_file)
    legend = [{"id": 1, "name": "ClassA", "description": None, "traits": None},
              {"id": 2, "name": "ClassB", "description": None, "traits": None}]
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER, FAKE_NOM_LAYER]), \
         patch("main._load_legend", return_value=legend):
        extra = json.dumps([{"variable": "bio1", "min": 15, "max": 100}])
        r = client.get(f"/species/2923970/environment/kg2?extra={extra}")
    assert r.status_code == 200
    body = r.json()
    # bio1 15-100 keeps B (bio1=20, kg2=1) and C (bio1=30, kg2=2); A
    # (bio1=10) is excluded — so the distribution is now 1 of each class,
    # not the unfiltered 2-vs-1 split across all three rows.
    assert body["observation_count"] == 2
    fractions = {d["value"]: d["fraction"] for d in body["categorical_distribution"]}
    assert fractions == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# Slice / class-samples shared index table
# ---------------------------------------------------------------------------

_INDEX_TABLE = pa.table({
    "catalogNumber": ["OCC001", "OCC002", "OCC003"],
    "decimalLatitude": [40.5, 41.0, 42.0],
    "decimalLongitude": [-75.0, -74.5, -73.0],
    "bio1": [10.0, 20.0, 30.0],
    "kg2": [1.0, 2.0, 1.0],
})

_INDEX_SCHEMA = MagicMock()
_INDEX_SCHEMA.names = list(_INDEX_TABLE.schema.names)

FAKE_DISC_LAYER = {
    "id": "bio1",
    "display_name": "Annual Mean Temperature",
    "units": "°C",
    "value_type": "interval",
    "domain": None,
}



def test_slice_layer_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30")
    assert r.status_code == 404


def test_slice_nominal_rejected():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/slice?min=0&max=30")
    assert r.status_code == 400


def test_class_samples_layer_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples")
    assert r.status_code == 404


def test_class_samples_not_nominal():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/class/10/samples")
    assert r.status_code == 400


def test_class_samples_invalid_class():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/notanumber/samples")
    assert r.status_code == 400


def test_slice_with_location_no_data(tmp_path, monkeypatch):
    """No occurrences.parquet → collect_taxon_df returns None → empty results."""
    _patch_stats_storage(monkeypatch, tmp_path)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_slice_with_location_empty_after_gid_filter(tmp_path, monkeypatch):
    """Data exists but no rows match the requested GID → empty results."""
    import numpy as np

    _patch_stats_storage(monkeypatch, tmp_path)
    # Occurrence file has CAN rows, not USA
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "CAN", "bio1",
                       list(np.linspace(5.0, 25.0, 20)))
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0
def test_slice_from_raw_occ_circular_wrap(tmp_path, monkeypatch):
    """_slice_from_raw_occ handles circular_wrap=True correctly."""
    _patch_stats_storage(monkeypatch, tmp_path)
    occ_dir = tmp_path / "taxonomy"
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["A", "B", "C"],
        "decimalLatitude": [40.0, 41.0, 42.0],
        "decimalLongitude": [-75.0, -74.0, -73.0],
        "obscured": ["No", "No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0, 100.0],
        "taxon_key": [TAXON["taxon_key"]] * 3,
        "level0Gid": ["USA", "USA", "USA"],
        "aspectdeg": [350.0, 10.0, 180.0],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrences.parquet")
    result = main_module._slice_from_raw_occ(
        TAXON, "aspectdeg", "level0Gid", "USA", 315.0, 45.0, True, None,
    )
    catalogs = {r["catalogNumber"] for r in result}
    assert "A" in catalogs
    assert "B" in catalogs
    assert "C" not in catalogs


def test_slice_with_location_limit(tmp_path, monkeypatch):
    import numpy as np

    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "USA", "bio1",
                       list(np.linspace(1.0, 20.0, 20)))
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=100&location=USA&limit=5")
    assert r.status_code == 200
    assert r.json()["count"] == 5


# ---------------------------------------------------------------------------
# Class-samples with location param
# ---------------------------------------------------------------------------

def test_class_samples_with_location_success(tmp_path, monkeypatch):
    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "USA", "kg2",
                       [1.0] * 10 + [2.0] * 10)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples?location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["class_value"] == 1
    assert body["count"] == 10
    assert all(obs["value"] == pytest.approx(1.0) for obs in body["observations"])


def test_class_samples_with_location_no_data(tmp_path, monkeypatch):
    """No occurrences.parquet → collect_taxon_df returns None → empty results."""
    _patch_stats_storage(monkeypatch, tmp_path)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples?location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_class_samples_with_location_empty_after_gid_filter(tmp_path, monkeypatch):
    """Data exists but no rows match the requested GID → empty results."""
    _patch_stats_storage(monkeypatch, tmp_path)
    # Occurrence file has CAN rows, not USA
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "CAN", "kg2", [1.0] * 10)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples?location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0
def test_class_samples_with_location_limit(tmp_path, monkeypatch):
    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_loc(tmp_path, TAXON["taxon_key"], "level0Gid", "USA", "kg2", [1.0] * 20)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples?location=USA&limit=3")
    assert r.status_code == 200
    assert r.json()["count"] == 3


# ---------------------------------------------------------------------------
# Chained multi-variable filters (`extra` query param)
# ---------------------------------------------------------------------------

def _patch_occ_source(monkeypatch, occurrences_file):
    monkeypatch.setattr(st_module, "OCCURRENCES_FILE", occurrences_file)
    monkeypatch.setattr(st_module, "load_catalog", lambda: {TAXON["taxon_key"]: TAXON})


def _write_multi_var_occ(occurrences_file):
    """3 rows spanning a numeric (bio1) and categorical (kg2) variable, so
    tests can chain a filter on one while slicing/sampling the other:
    A: bio1=10, kg2=1   B: bio1=20, kg2=1   C: bio1=30, kg2=2
    """
    data = {
        "catalogNumber": ["A", "B", "C"],
        "decimalLatitude": [40.0, 41.0, 42.0],
        "decimalLongitude": [-75.0, -74.0, -73.0],
        "bio1": [10.0, 20.0, 30.0],
        "kg2": [1.0, 1.0, 2.0],
        "taxon_key": [TAXON["taxon_key"]] * 3,
    }
    occurrences_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
        occurrences_file,
    )


def test_slice_with_extra_class_filter_chains_categorical_onto_numeric(tmp_path, monkeypatch):
    """Slicing bio1 (0-100, matches all 3) with an extra kg2 classValue=1
    filter chained on should exclude C (kg2=2), leaving only A and B."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_var_occ(occurrences_file)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "kg2", "classValue": 1}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=100&extra={extra}")
    assert r.status_code == 200
    body = r.json()
    catalogs = {obs["catalogNumber"] for obs in body["observations"]}
    assert catalogs == {"A", "B"}


def test_class_samples_with_extra_range_filter_chains_numeric_onto_categorical(tmp_path, monkeypatch):
    """Sampling kg2==1 (matches A and B) with an extra bio1 15-100 range
    chained on should exclude A (bio1=10), leaving only B."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_var_occ(occurrences_file)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "bio1", "min": 15, "max": 100}])
        r = client.get(f"/species/2923970/environment/kg2/class/1/samples?extra={extra}")
    assert r.status_code == 200
    body = r.json()
    catalogs = {obs["catalogNumber"] for obs in body["observations"]}
    assert catalogs == {"B"}


def _write_multi_class_occ(occurrences_file):
    """4 rows spanning bio1 (numeric) and kg2 (3 distinct classes), for
    testing OR-matching against multiple classValues of one variable:
    A: bio1=10, kg2=1   B: bio1=20, kg2=1   C: bio1=30, kg2=2   D: bio1=40, kg2=3
    """
    data = {
        "catalogNumber": ["A", "B", "C", "D"],
        "decimalLatitude": [40.0, 41.0, 42.0, 43.0],
        "decimalLongitude": [-75.0, -74.0, -73.0, -72.0],
        "bio1": [10.0, 20.0, 30.0, 40.0],
        "kg2": [1.0, 1.0, 2.0, 3.0],
        "taxon_key": [TAXON["taxon_key"]] * 4,
    }
    occurrences_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
        occurrences_file,
    )


def test_slice_with_extra_class_values_filter_ors_within_one_variable(tmp_path, monkeypatch):
    """Slicing bio1 (0-100, matches all 4) with an extra kg2 classValues=[1,3]
    filter chained on should keep A, B (kg2=1) and D (kg2=3), excluding C (kg2=2)."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_class_occ(occurrences_file)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "kg2", "classValues": [1, 3]}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=100&extra={extra}")
    assert r.status_code == 200
    body = r.json()
    catalogs = {obs["catalogNumber"] for obs in body["observations"]}
    assert catalogs == {"A", "B", "D"}


def test_extra_filter_class_values_must_be_a_non_empty_list():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "kg2", "classValues": []}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 400


def test_extra_filter_class_values_rejected_for_non_categorical_variable():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "bio1", "classValues": [1, 2]}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 400


def test_get_species_environment_with_extra_ranges_filter_ors_within_one_variable(tmp_path, monkeypatch):
    """kg2 (categorical primary) chained with an extra bio1 `ranges` filter
    OR-matching [5,15] and [35,45] should keep A (bio1=10) and D (bio1=40),
    excluding B (bio1=20) and C (bio1=30) — same OR-within-one-variable
    shape as classValues, but for a numeric variable's multi-select."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_class_occ(occurrences_file)
    legend = [{"id": 1, "name": "ClassA", "description": None, "traits": None},
              {"id": 2, "name": "ClassB", "description": None, "traits": None},
              {"id": 3, "name": "ClassC", "description": None, "traits": None}]
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]), \
         patch("main._load_legend", return_value=legend):
        extra = json.dumps([{
            "variable": "bio1",
            "ranges": [{"min": 5, "max": 15}, {"min": 35, "max": 45}],
        }])
        r = client.get(f"/species/2923970/environment/kg2?extra={extra}")
    assert r.status_code == 200
    body = r.json()
    assert body["observation_count"] == 2


def test_slice_with_extra_ranges_filter_ors_within_one_variable(tmp_path, monkeypatch):
    """Slicing kg2 class 1 samples (A, B) with an extra bio1 ranges filter
    OR-matching [5,15] and [35,45] should keep only A (bio1=10); B (bio1=20)
    falls outside both ranges."""
    occurrences_file = tmp_path / "occurrences.parquet"
    _patch_occ_source(monkeypatch, occurrences_file)
    _write_multi_class_occ(occurrences_file)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{
            "variable": "bio1",
            "ranges": [{"min": 5, "max": 15}, {"min": 35, "max": 45}],
        }])
        r = client.get(f"/species/2923970/environment/kg2/class/1/samples?extra={extra}")
    assert r.status_code == 200
    body = r.json()
    catalogs = {obs["catalogNumber"] for obs in body["observations"]}
    assert catalogs == {"A"}


def test_extra_filter_ranges_must_be_a_non_empty_list():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "bio1", "ranges": []}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 400


def test_extra_filter_ranges_rejected_for_categorical_variable():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER, FAKE_NOM_LAYER]):
        extra = json.dumps([{"variable": "kg2", "ranges": [{"min": 0, "max": 1}]}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 400


def test_slice_extra_filter_malformed_json_rejected():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&extra=not-json")
    assert r.status_code == 400


def test_slice_extra_filter_unknown_variable_rejected():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        extra = json.dumps([{"variable": "doesnotexist", "min": 0, "max": 10}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 404


def test_slice_extra_filter_type_mismatch_rejected():
    """A numeric-only variable can't be chained with a classValue filter."""
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        extra = json.dumps([{"variable": "bio1", "classValue": 1}])
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&extra={extra}")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /species/{id}/occurrences with location param
# ---------------------------------------------------------------------------

def test_get_species_occurrences_with_location(tmp_path, monkeypatch):
    """location filter restricts returned pins to matching rows only."""
    occ_dir = tmp_path / "taxonomy"
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["USA001", "USA002", "CAN001"],
        "decimalLatitude": [40.0, 41.0, 50.0],
        "decimalLongitude": [-75.0, -74.0, -80.0],
        "obscured": ["No", "No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0, 100.0],
        "taxon_key": [TAXON["taxon_key"]] * 3,
        "level0Gid": ["USA", "USA", "CAN"],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrences.parquet")
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.OCCURRENCES_FILE", occ_dir / "occurrences.parquet"), \
         patch("main.iter_descendants", return_value=[TAXON]):
        r = client.get("/species/2923970/occurrences?location=USA")
    assert r.status_code == 200
    occs = r.json()["occurrences"]
    catalog_numbers = {o["catalogNumber"] for o in occs}
    assert catalog_numbers == {"USA001", "USA002"}
    assert "CAN001" not in catalog_numbers


def test_get_species_occurrences_with_location_no_match(tmp_path, monkeypatch):
    """location filter with no matching rows returns empty list."""
    occ_dir = tmp_path / "taxonomy"
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["CAN001"],
        "decimalLatitude": [50.0],
        "decimalLongitude": [-80.0],
        "obscured": ["No"],
        "coordinateUncertaintyInMeters": [100.0],
        "taxon_key": [TAXON["taxon_key"]],
        "level0Gid": ["CAN"],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrences.parquet")
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.OCCURRENCES_FILE", occ_dir / "occurrences.parquet"), \
         patch("main.iter_descendants", return_value=[TAXON]):
        r = client.get("/species/2923970/occurrences?location=USA")
    assert r.status_code == 200
    assert r.json()["occurrences"] == []


# ---------------------------------------------------------------------------
# /species/{id}/locations — parent filter (_ancestor_gids coverage)
# ---------------------------------------------------------------------------

def test_get_species_locations_parent_filter_matches(tmp_path, monkeypatch):
    """parent=USA returns only locations whose ancestor chain includes USA."""
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations?level=1&parent=USA")
    assert r.status_code == 200
    gids = {row["gid"] for row in r.json()}
    assert "USA.1_1" in gids
    assert "CAN" not in gids
    assert "USA" not in gids


def test_get_species_locations_parent_filter_excludes_all(tmp_path, monkeypatch):
    """parent=CAN at level=1 returns nothing (no level-1 children of Canada in fixture)."""
    _patch_locations(tmp_path, monkeypatch)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations?level=1&parent=CAN")
    assert r.status_code == 200
    assert r.json() == []


def test_get_species_locations_cycle_safe(tmp_path, monkeypatch):
    """Cyclic parent_gid entries in hierarchy do not cause infinite loops."""
    loc_path = tmp_path / "location_taxa.parquet"
    pq.write_table(pa.table({
        "scope": ["gadm_level1"],
        "gid": ["X.1_1"],
        "taxon_key": ["2923970"],
        "count": [5],
    }), loc_path)
    hier = "level,gid,name,parent_gid\n1,X.1_1,Child,ROOT\n0,ROOT,Root,ROOT\n"
    (tmp_path / "hierarchy.csv").write_text(hier, encoding="utf-8")
    monkeypatch.setattr(main_module, "_LOC_TAXA_PATH", loc_path)
    monkeypatch.setattr(main_module, "_LOCATIONS_DIR", tmp_path)
    main_module._load_hierarchy.cache_clear()
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/species/2923970/locations?parent=ROOT")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/taxa/ranking-options
# ---------------------------------------------------------------------------

def test_ranking_options_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/taxa/ranking-options?within_taxon=999&descendant_rank=SPECIES")
    assert r.status_code == 404


def test_ranking_options_no_rankings_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "GLOBAL_STATS_DIR", tmp_path)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["options"] == []
    assert body["rank"] == "SPECIES"


def test_ranking_options_corrupt_rankings_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "GLOBAL_STATS_DIR", tmp_path)
    (tmp_path / main_module.RANKINGS_FILE).write_bytes(b"garbage")
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    assert r.json()["options"] == []


def test_ranking_options_returns_options(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "GLOBAL_STATS_DIR", tmp_path)
    pq.write_table(pa.table({
        "contextTaxonId": ["2923970", "2923970", "2923970", "2923970", "111"],
        "rank":           ["SPECIES", "SPECIES", "SPECIES", "SPECIES", "SPECIES"],
        "variable":       ["bio1", "bio1", "bio12", "bio12", "bio1"],
        "metric":         ["mean", "class_0", "median", "p10", "mean"],
        "count":          [100, 100, 100, 0, 999],
    }), tmp_path / main_module.RANKINGS_FILE)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[
             {"id": "bio1", "display_name": "Temperature"},
             {"id": "bio12", "display_name": "Precipitation"},
         ]), \
         patch.object(main_module, "_load_legend", return_value=[{"id": 0, "name": "Dry"}]):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["ancestor_taxon_id"] == TAXON["taxon_key"]
    assert body["rank"] == "SPECIES"
    options = body["options"]
    variables = [o["variable"] for o in options]
    assert "bio1" in variables
    assert "bio12" in variables
    # a different contextTaxonId's row (111) is excluded by the WHERE filter
    assert not any(o["count"] == 999 for o in options)
    # class_ metrics are included as sort options
    class_options = [o for o in options if o["metric"].startswith("class_")]
    assert len(class_options) == 1
    assert class_options[0]["variable"] == "bio1"
    # p10 with count==0 is skipped
    assert not any(o["metric"] == "p10" for o in options)
    # label populated for all options
    assert all(isinstance(o["label"], str) and o["label"] for o in options)
    assert all(o["count"] > 0 for o in options)


# ---------------------------------------------------------------------------
# _lookup_index_value
# ---------------------------------------------------------------------------

def test_lookup_index_value_missing_file(tmp_path):
    from main import _lookup_index_value
    with patch.object(main_module, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_column_absent(tmp_path):
    from main import _lookup_index_value
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "catalogNumber": pa.array(["12345"]),
        "taxon_key": pa.array([TAXON["taxon_key"]]),
    }), occ_path)
    with patch.object(main_module, "OCCURRENCES_FILE", occ_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_catalog_number_not_found(tmp_path):
    from main import _lookup_index_value
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "catalogNumber": pa.array(["99999"]),
        "taxon_key": pa.array([TAXON["taxon_key"]]),
        "bio1": pa.array([14.35]),
    }), occ_path)
    with patch.object(main_module, "OCCURRENCES_FILE", occ_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_null_value_returns_none(tmp_path):
    from main import _lookup_index_value
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "catalogNumber": pa.array(["12345"]),
        "taxon_key": pa.array([TAXON["taxon_key"]]),
        "bio1": pa.array([None], type=pa.float64()),
    }), occ_path)
    with patch.object(main_module, "OCCURRENCES_FILE", occ_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_found(tmp_path):
    from main import _lookup_index_value
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "catalogNumber": pa.array(["12345"]),
        "taxon_key": pa.array([TAXON["taxon_key"]]),
        "bio1": pa.array([14.35]),
    }), occ_path)
    with patch.object(main_module, "OCCURRENCES_FILE", occ_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result == pytest.approx(14.35)


def test_lookup_index_value_wrong_taxon_key_not_matched(tmp_path):
    """A row with a matching catalogNumber but a different taxon_key isn't returned."""
    from main import _lookup_index_value
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "catalogNumber": pa.array(["12345"]),
        "taxon_key": pa.array(["999999"]),
        "bio1": pa.array([14.35]),
    }), occ_path)
    with patch.object(main_module, "OCCURRENCES_FILE", occ_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


# ---------------------------------------------------------------------------
# _load_relative_ranks
# ---------------------------------------------------------------------------

def test_load_relative_ranks_reads_consolidated_positions_file(tmp_path):
    """One row per ancestor context for this taxon+variable, read straight
    from the single global positions file (no per-ancestor directory walk)."""
    from main import _load_relative_ranks
    positions_path = tmp_path / main_module.POSITION_FILE
    pq.write_table(pa.table({
        "taxon_key": pa.array([TAXON["taxon_key"], TAXON["taxon_key"], "other"]),
        "variable": pa.array(["bio1", "bio1", "bio1"]),
        "metric": pa.array(["mean", "mean", "mean"]),
        "position": pa.array([4, 19, 0], type=pa.int32()),
        "count": pa.array([5, 40, 1], type=pa.int32()),
        "sampleCount": pa.array([30, 30, 1], type=pa.int32()),
        "contextTaxonId": pa.array(["genusX", "familyY", "genusX"]),
        "contextLabel": pa.array(["Genus X", "Family Y", "Genus X"]),
    }), positions_path)
    with patch.object(main_module, "GLOBAL_STATS_DIR", tmp_path):
        result = _load_relative_ranks(TAXON["taxon_key"], "bio1")
    assert len(result) == 2
    by_label = {r["context_label"]: r for r in result}
    assert by_label["Genus X"]["position"] == 5
    assert by_label["Genus X"]["percentile"] == pytest.approx(1.0)
    assert by_label["Family Y"]["position"] == 20
    assert by_label["Family Y"]["percentile"] == pytest.approx(0.5)


def test_load_relative_ranks_missing_file_returns_empty(tmp_path):
    from main import _load_relative_ranks
    with patch.object(main_module, "GLOBAL_STATS_DIR", tmp_path):
        assert _load_relative_ranks(TAXON["taxon_key"], "bio1") == []


# ---------------------------------------------------------------------------
# /gis/point endpoint
# ---------------------------------------------------------------------------

_STATIC_LAYER = {
    "id": "bio1",
    "filename": "bio1.tif",
    "scale_factor": 0.1,
    "add_offset": -273.15,
    "units": "°C",
    "value_type": "interval",
    "window_hours": None,
}

_NOMINAL_LAYER = {
    "id": "kg2",
    "filename": "kg2.tif",
    "scale_factor": None,
    "add_offset": None,
    "units": "",
    "value_type": "nominal",
    "window_hours": None,
}

_TEMPORAL_LAYER = {
    "id": "temperature_2m_avg_1h",
    "var_id": "temperature_2m",
    "window_hours": 1,
    "window_label": "1h",
    "model": "copernicus_era5",
    "units": "°C",
    "value_type": "interval",
}


def test_gis_point_nonfinite_lat():
    r = client.get("/gis/point?lat=inf&lon=0&variable=bio1")
    assert r.status_code == 400


def test_gis_point_unknown_variable():
    with patch.object(tiles, "get_layer", side_effect=KeyError("nope")):
        r = client.get("/gis/point?lat=40&lon=-105&variable=nope")
    assert r.status_code == 404


def test_gis_point_raster_lookup():
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_STATIC_LAYER), \
         patch.object(gis_module, "sample_point", return_value=9.5):
        r = client.get("/gis/point?lat=40&lon=-105&variable=bio1")
    assert r.status_code == 200
    body = r.json()
    assert body["value"] == pytest.approx(9.5)
    assert body["variable"] == "bio1"
    assert body["units"] == "°C"
    assert body["class_name"] is None


def test_gis_point_index_hit_used_over_raster():
    """When taxon_id + catalog_number are supplied and the index has the value,
    _lookup_index_value result is returned without calling sample_point."""
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_STATIC_LAYER), \
         patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(main_module, "_lookup_index_value", return_value=14.35) as mock_lookup, \
         patch.object(gis_module, "sample_point", return_value=0.0) as mock_sample:
        r = client.get("/gis/point?lat=40&lon=-105&variable=bio1&taxon_id=2923970&catalog_number=12345")
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(14.35)
    mock_lookup.assert_called_once()
    mock_sample.assert_not_called()


def test_gis_point_index_miss_falls_back_to_raster():
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_STATIC_LAYER), \
         patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(main_module, "_lookup_index_value", return_value=None), \
         patch.object(gis_module, "sample_point", return_value=8.1):
        r = client.get("/gis/point?lat=40&lon=-105&variable=bio1&taxon_id=2923970&catalog_number=12345")
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(8.1)


def test_gis_point_nominal_resolves_class_name():
    import util.gis as gis_module
    fake_legend = [{"id": 9, "name": "Cold semi-arid", "traits": {"color": "#F00"}}]
    with patch.object(tiles, "get_layer", return_value=_NOMINAL_LAYER), \
         patch.object(gis_module, "sample_point", return_value=9.0), \
         patch.object(main_module, "_load_legend", return_value=fake_legend):
        r = client.get("/gis/point?lat=40&lon=-105&variable=kg2")
    assert r.status_code == 200
    body = r.json()
    assert body["value"] == pytest.approx(9.0)
    assert body["class_name"] == "Cold semi-arid"



def test_gis_point_nodata_returns_null_value():
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_STATIC_LAYER), \
         patch.object(gis_module, "sample_point", return_value=None):
        r = client.get("/gis/point?lat=40&lon=-105&variable=bio1")
    assert r.status_code == 200
    assert r.json()["value"] is None
    assert r.json()["class_name"] is None


def test_gis_point_event_ts_used_when_no_index_lookup():
    """No taxon_id/catalog_number given (e.g. a not-yet-ingested observation
    pin) but event_ts is — the live historical-at-timestamp lookup should be
    tried before falling to the current/live raster."""
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_TEMPORAL_LAYER), \
         patch.object(main_module, "_lookup_temporal_value_at_timestamp", return_value=12.5) as mock_temporal, \
         patch.object(gis_module, "sample_point", return_value=999.0) as mock_sample:
        r = client.get(
            "/gis/point?lat=40&lon=-105&variable=temperature_2m_avg_1h&event_ts=1700000000",
        )
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(12.5)
    mock_temporal.assert_called_once_with("temperature_2m_avg_1h", 40.0, -105.0, 1700000000)
    mock_sample.assert_not_called()


def test_gis_point_event_ts_ignored_when_index_hits():
    """A precomputed index hit (real ingested observation) wins even if
    event_ts is also supplied — no need for a live lookup."""
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_TEMPORAL_LAYER), \
         patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(main_module, "_lookup_index_value", return_value=14.35), \
         patch.object(main_module, "_lookup_temporal_value_at_timestamp") as mock_temporal, \
         patch.object(gis_module, "sample_point") as mock_sample:
        r = client.get(
            "/gis/point?lat=40&lon=-105&variable=temperature_2m_avg_1h"
            "&taxon_id=2923970&catalog_number=12345&event_ts=1700000000",
        )
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(14.35)
    mock_temporal.assert_not_called()
    mock_sample.assert_not_called()


def test_gis_point_no_event_ts_skips_temporal_lookup():
    """Existing behavior preserved: without event_ts, go straight to raster."""
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_TEMPORAL_LAYER), \
         patch.object(main_module, "_lookup_temporal_value_at_timestamp") as mock_temporal, \
         patch.object(gis_module, "sample_point", return_value=8.1):
        r = client.get("/gis/point?lat=40&lon=-105&variable=temperature_2m_avg_1h")
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(8.1)
    mock_temporal.assert_not_called()


def test_gis_point_event_ts_miss_falls_back_to_raster():
    import util.gis as gis_module
    with patch.object(tiles, "get_layer", return_value=_TEMPORAL_LAYER), \
         patch.object(main_module, "_lookup_temporal_value_at_timestamp", return_value=None), \
         patch.object(gis_module, "sample_point", return_value=8.1):
        r = client.get(
            "/gis/point?lat=40&lon=-105&variable=temperature_2m_avg_1h&event_ts=1700000000",
        )
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(8.1)


# ---------------------------------------------------------------------------
# _lookup_temporal_value_at_timestamp
# ---------------------------------------------------------------------------

def _fake_temporal_layer(**overrides):
    defaults = dict(
        id="temperature_2m", model="copernicus_era5", grid_mode="lat_asc_lon_pm180",
        agg="avg", windows=[1, 24], derived=False, sources=[],
    )
    defaults.update(overrides)
    return temporal_module.TemporalLayer(**defaults)


def test_lookup_temporal_value_no_matching_layer():
    with patch.object(main_module, "load_temporal_layers", return_value=[_fake_temporal_layer()]):
        result = main_module._lookup_temporal_value_at_timestamp(
            "precipitation_sum_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


def test_lookup_temporal_value_skips_derived_layers():
    layer = _fake_temporal_layer(derived=True)
    with patch.object(main_module, "load_temporal_layers", return_value=[layer]):
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


def test_lookup_temporal_value_returns_matching_column():
    layer = _fake_temporal_layer()
    fake_occ_table = object()
    updates = {"__upload__": {"temperature_2m_avg_24h": [(pd.array([0]), pd.array([21.5]))]}}
    with patch.object(main_module, "load_temporal_layers", return_value=[layer]), \
         patch.object(upload_module, "_df_to_occ_table", return_value=fake_occ_table) as mock_df, \
         patch.object(upload_module, "_process_one_layer", return_value=updates) as mock_process:
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result == pytest.approx(21.5)
    mock_df.assert_called_once()
    mock_process.assert_called_once_with(layer, fake_occ_table)


def test_lookup_temporal_value_no_column_produced():
    layer = _fake_temporal_layer()
    with patch.object(main_module, "load_temporal_layers", return_value=[layer]), \
         patch.object(upload_module, "_df_to_occ_table", return_value=object()), \
         patch.object(upload_module, "_process_one_layer", return_value={"__upload__": {}}):
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


def test_lookup_temporal_value_nan_returns_none():
    layer = _fake_temporal_layer()
    updates = {"__upload__": {"temperature_2m_avg_24h": [(pd.array([0]), [float("nan")])]}}
    with patch.object(main_module, "load_temporal_layers", return_value=[layer]), \
         patch.object(upload_module, "_df_to_occ_table", return_value=object()), \
         patch.object(upload_module, "_process_one_layer", return_value=updates):
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


def test_lookup_temporal_value_layer_exception_is_swallowed():
    layer = _fake_temporal_layer()
    with patch.object(main_module, "load_temporal_layers", return_value=[layer]), \
         patch.object(upload_module, "_df_to_occ_table", return_value=object()), \
         patch.object(upload_module, "_process_one_layer", side_effect=Exception("boom")):
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


def test_lookup_temporal_value_catalog_load_failure_returns_none():
    with patch.object(main_module, "load_temporal_layers", side_effect=Exception("boom")):
        result = main_module._lookup_temporal_value_at_timestamp(
            "temperature_2m_avg_24h", 40.0, -105.0, 1700000000,
        )
    assert result is None


# ---------------------------------------------------------------------------
# POST /upload/raw-observations
# ---------------------------------------------------------------------------

def _csv_file(content: str = "latitude,longitude\n45.0,-120.0\n46.0,-121.0\n"):
    return ("file", ("obs.csv", content.encode(), "text/csv"))


def test_upload_unsupported_extension():
    r = client.post("/upload/raw-observations",
                    files=[("file", ("obs.json", b"{}", "application/json"))])
    assert r.status_code == 400


def test_upload_invalid_csv_raises_422():
    r = client.post("/upload/raw-observations",
                    files=[("file", ("obs.csv", b"\x00\x01\x02\x03\xff", "text/csv"))])
    assert r.status_code == 422


def test_upload_missing_coordinates_raises_422():
    csv = b"foo,bar\n1,2\n"
    r = client.post("/upload/raw-observations",
                    files=[("file", ("obs.csv", csv, "text/csv"))])
    assert r.status_code == 422


def test_upload_csv_success():
    csv = b"latitude,longitude\n45.0,-120.0\n46.0,-121.0\n"
    with patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.csv", csv, "text/csv"))])
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "queued"


def test_upload_tsv_parsed_correctly():
    tsv = b"latitude\tlongitude\n45.0\t-120.0\n"
    with patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.tsv", tsv, "text/tab-separated-values"))])
    assert r.status_code == 202
    assert r.json()["status"] == "queued"


def test_upload_parquet_parsed_correctly():
    import io as _io

    import pyarrow as pa
    import pyarrow.parquet as pq_local
    df_in = pd.DataFrame({"latitude": [45.0], "longitude": [-120.0]})
    buf = _io.BytesIO()
    pq_local.write_table(pa.Table.from_pandas(df_in), buf)
    parquet_bytes = buf.getvalue()
    with patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.parquet", parquet_bytes, "application/octet-stream"))])
    assert r.status_code == 202
    assert r.json()["status"] == "queued"


# ---------------------------------------------------------------------------
# `polygon` param on /environment/{var}, /slice, /class/{value}/samples
# ---------------------------------------------------------------------------

def _encode_polyline(points: list[tuple[float, float]], precision: int = 5) -> str:
    """Test-only mirror of the frontend's encoder (speciesOccurrenceMapHelpers.ts
    encodePolyline) — production code only ever needs to decode."""
    factor = 10**precision
    out: list[str] = []
    prev_lat = prev_lon = 0
    for lat, lon in points:
        lat_i = round(lat * factor)
        lon_i = round(lon * factor)
        for delta in (lat_i - prev_lat, lon_i - prev_lon):
            v = ~(delta << 1) if delta < 0 else (delta << 1)
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        prev_lat, prev_lon = lat_i, lon_i
    return "".join(out)


# A 10x10 degree box straddling nothing in particular — used across the
# polygon tests below alongside points explicitly inside/outside it.
_POLYGON_BOX = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
_POLYGON_PARAM = _encode_polyline(_POLYGON_BOX)


def _make_occ_with_coords(
    tmp_path: Path, taxon_key: str, var_col: str, values: list, lats: list[float], lons: list[float],
) -> Path:
    occ_dir = tmp_path / "taxonomy"
    occ_dir.mkdir(parents=True, exist_ok=True)
    n = len(values)
    data = {
        "catalogNumber": [f"obs{i}" for i in range(n)],
        "decimalLatitude": lats,
        "decimalLongitude": lons,
        "obscured": ["No"] * n,
        "coordinateUncertaintyInMeters": [100.0] * n,
        "taxon_key": [taxon_key] * n,
        var_col: values,
    }
    occ_path = occ_dir / "occurrences.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False), occ_path)
    return occ_path


def test_slice_with_polygon_filters_points(tmp_path, monkeypatch):
    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_coords(
        tmp_path, TAXON["taxon_key"], "bio1",
        values=[10.0, 20.0],
        lats=[5.0, 50.0],
        lons=[5.0, 50.0],
    )
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get(f"/species/2923970/environment/bio1/slice?min=0&max=30&polygon={_POLYGON_PARAM}")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["observations"][0]["catalogNumber"] == "obs0"


def test_class_samples_with_polygon_filters_points(tmp_path, monkeypatch):
    _patch_stats_storage(monkeypatch, tmp_path)
    _make_occ_with_coords(
        tmp_path, TAXON["taxon_key"], "kg2",
        values=[1.0, 1.0],
        lats=[5.0, 50.0],
        lons=[5.0, 50.0],
    )
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get(f"/species/2923970/environment/kg2/class/1/samples?polygon={_POLYGON_PARAM}")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["observations"][0]["catalogNumber"] == "obs0"


def test_environment_with_polygon_filters_stats(tmp_path, monkeypatch):
    import numpy as np

    _patch_stats_storage(monkeypatch, tmp_path)
    # 20 points inside the box, 5 well outside — only the 20 should count.
    inside_lats = list(np.linspace(1.0, 9.0, 20))
    inside_lons = list(np.linspace(1.0, 9.0, 20))
    outside_lats = [50.0] * 5
    outside_lons = [50.0] * 5
    _make_occ_with_coords(
        tmp_path, TAXON["taxon_key"], "bio1",
        values=list(np.linspace(5.0, 25.0, 20)) + [999.0] * 5,
        lats=inside_lats + outside_lats,
        lons=inside_lons + outside_lons,
    )
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get(f"/species/2923970/environment/bio1?polygon={_POLYGON_PARAM}")
    assert r.status_code == 200
    assert r.json()["observation_count"] == 20


def test_slice_with_invalid_polygon_returns_400():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&polygon=!!!not-valid")
    assert r.status_code == 400


def test_class_samples_with_invalid_polygon_returns_400():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg2/class/1/samples?polygon=!!!not-valid")
    assert r.status_code == 400


def test_environment_with_invalid_polygon_returns_400():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1?polygon=!!!not-valid")
    assert r.status_code == 400
