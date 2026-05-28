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
import util.tiles as tiles
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
    with patch.object(rankings_module, "search_taxa_by_name", return_value=[(TAXON, 95.0, "opuntia humifusa")]):
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


def test_query_taxa_scope_no_sort_no_catalog(tmp_path):
    """Scope without sort → catalog mode; missing catalog returns no_catalog."""
    genus = {**TAXON, "taxon_key": "10", "path": "Plantae_6/Opuntia_2923968", "rank": "GENUS"}
    def _resolve(k):
        return genus if k == "10" else None
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    assert r.json()["empty_reason"] == "no_catalog"


def test_query_taxa_scope_catalog_mode(tmp_path):
    """Scope without sort lists catalog entries."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    catalog_dir = tmp_path / "Opuntia"
    catalog_dir.mkdir(parents=True)
    catalog_rows = [
        {"taxon_key": "2923970", "path": TAXON["path"], "scientific_name": "Opuntia_humifusa",
         "common_name": "devil's tongue", "rank": "SPECIES", "sample_count": 50},
    ]
    pq.write_table(pa.Table.from_pylist(catalog_rows), catalog_dir / "species.parquet")

    def _resolve(k):
        if k == "10":
            return genus
        if k == "2923970":
            return TAXON
        return None

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["empty_reason"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["taxon_id"] == "2923970"
    assert body["results"][0]["sample_count"] == 50


def test_query_taxa_text_in_scope(tmp_path):
    """Text search filtered to scope."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    catalog_dir = tmp_path / "Opuntia"
    catalog_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"taxon_key": "2923970", "path": TAXON["path"], "scientific_name": "Opuntia_humifusa",
             "common_name": "devil's tongue", "rank": "SPECIES", "sample_count": 50},
        ]),
        catalog_dir / "species.parquet",
    )

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
         patch.object(rankings_module, "_infer_sample_count", return_value=50), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?q=opuntia&within_taxon=10&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["scope"]["within_taxon"] == "10"
    assert body["scope"]["descendant_rank"] == "SPECIES"
    assert len(body["results"]) == 1
    assert body["results"][0]["match_score"] == pytest.approx(90.0)


def test_query_taxa_ranked_scoped_no_index(tmp_path):
    """Ranked-scoped mode with missing index returns no_index."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    (tmp_path / "Opuntia").mkdir(parents=True)
    def _resolve(k):
        return genus if k == "10" else None
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean")
    assert r.status_code == 200
    assert r.json()["empty_reason"] == "no_index"


def _build_index_parquet(ancestor_dir: Path, col_name: str, entries: list[dict]) -> None:
    import json
    struct_fields = [
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int64()),
    ]
    arr = pa.StructArray.from_arrays(
        [pa.array([e["taxonKey"] for e in entries], type=pa.string()),
         pa.array([e["value"] for e in entries], type=pa.float64()),
         pa.array([e["sampleCount"] for e in entries], type=pa.int64())],
        fields=struct_fields,
    )
    table = pa.table({col_name: arr}).replace_schema_metadata(
        {b"column_lengths": json.dumps({col_name: len(entries)}).encode()}
    )
    pq.write_table(table, ancestor_dir / "species_index.parquet")


def test_query_taxa_ranked_scoped_mode(tmp_path):
    """Ranked-scoped mode reads index and returns sorted results with position/percentile."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    ancestor_dir = tmp_path / "Opuntia"
    ancestor_dir.mkdir(parents=True)
    taxon2 = {**TAXON, "taxon_key": "111", "path": "Plantae_6/Opuntia_2923968/Other_111",
              "scientific_name": "Opuntia_other", "rank": "SPECIES"}
    _build_index_parquet(ancestor_dir, "bio1::mean", [
        {"taxonKey": "2923970", "value": 10.0, "sampleCount": 100},
        {"taxonKey": "111", "value": 20.0, "sampleCount": 200},
    ])

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
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


def test_query_taxa_ranked_scoped_desc(tmp_path):
    """sort_order=desc reverses order."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    ancestor_dir = tmp_path / "Opuntia"
    ancestor_dir.mkdir(parents=True)
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    _build_index_parquet(ancestor_dir, "bio1::mean", [
        {"taxonKey": "2923970", "value": 10.0, "sampleCount": 100},
        {"taxonKey": "111", "value": 20.0, "sampleCount": 200},
    ])

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&sort_order=desc")
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["sort_value"] == pytest.approx(20.0)
    assert results[1]["sort_value"] == pytest.approx(10.0)


def test_query_taxa_ranked_scoped_min_samples(tmp_path):
    """min_samples filter excludes entries below threshold."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    ancestor_dir = tmp_path / "Opuntia"
    ancestor_dir.mkdir(parents=True)
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    _build_index_parquet(ancestor_dir, "bio1::mean", [
        {"taxonKey": "2923970", "value": 10.0, "sampleCount": 5},
        {"taxonKey": "111", "value": 20.0, "sampleCount": 200},
    ])

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&min_samples=10")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "111"


def test_query_taxa_ranked_scoped_location_filter(tmp_path):
    """Location filter excludes taxa not in the location."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    ancestor_dir = tmp_path / "Opuntia"
    ancestor_dir.mkdir(parents=True)
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    _build_index_parquet(ancestor_dir, "bio1::mean", [
        {"taxonKey": "2923970", "value": 10.0, "sampleCount": 100},
        {"taxonKey": "111", "value": 20.0, "sampleCount": 200},
    ])

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    loc_keys = frozenset({"2923970"})
    loc_counts = {"2923970": 42}
    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "_location_taxon_keys", return_value=(loc_keys, loc_counts)), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&location=USA")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "2923970"
    assert results[0]["location_count"] == 42


def test_query_taxa_ranked_scoped_text_filter(tmp_path):
    """Mode 3: scope+sort+q filters index to text matches."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    ancestor_dir = tmp_path / "Opuntia"
    ancestor_dir.mkdir(parents=True)
    taxon2 = {**TAXON, "taxon_key": "111", "path": "x/111", "scientific_name": "Other", "rank": "SPECIES"}
    _build_index_parquet(ancestor_dir, "bio1::mean", [
        {"taxonKey": "2923970", "value": 10.0, "sampleCount": 100},
        {"taxonKey": "111", "value": 20.0, "sampleCount": 200},
    ])

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "111": taxon2}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "search_taxa_by_name",
                      return_value=[(TAXON, 90.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                       "&sort_variable=bio1&sort_metric=mean&q=opuntia")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["taxon_id"] == "2923970"
    assert results[0]["match_score"] == pytest.approx(90.0)


def test_query_taxa_ranked_text_no_scope(tmp_path):
    """Mode 4: q+sort without scope reads per-taxon stats."""
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    with patch.object(rankings_module, "search_taxa_by_name",
                      return_value=[(TAXON, 85.0, "opuntia humifusa")]), \
         patch.object(rankings_module, "_taxon_metric_value", return_value=15.5), \
         patch.object(rankings_module, "_infer_sample_count", return_value=100), \
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


def test_query_taxa_scope_include_species_like(tmp_path):
    """include_species_like=true accepts subspecies-rank entries."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    catalog_dir = tmp_path / "Opuntia"
    catalog_dir.mkdir(parents=True)
    subsp = {**TAXON, "taxon_key": "999", "path": "x/999",
             "scientific_name": "Opuntia_humifusa_humifusa", "rank": "SUBSPECIES"}
    pq.write_table(
        pa.Table.from_pylist([
            {"taxon_key": "2923970", "path": TAXON["path"], "scientific_name": "Opuntia_humifusa",
             "common_name": "", "rank": "SPECIES", "sample_count": 50},
            {"taxon_key": "999", "path": "x/999", "scientific_name": "Opuntia_humifusa_humifusa",
             "common_name": "", "rank": "SUBSPECIES", "sample_count": 10},
        ]),
        catalog_dir / "species.parquet",
    )

    def _resolve(k):
        return {"10": genus, "2923970": TAXON, "999": subsp}.get(k)

    with patch.object(taxa, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(rankings_module, "get_taxon_by_id", side_effect=_resolve), \
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r_no = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES")
        r_yes = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES"
                           "&include_species_like=true")
    # Without flag: only SPECIES rank
    assert len(r_no.json()["results"]) == 1
    assert r_no.json()["results"][0]["taxon_id"] == "2923970"
    # With flag: SPECIES + SUBSPECIES
    assert len(r_yes.json()["results"]) == 2


def test_query_taxa_offset_pagination(tmp_path):
    """offset/limit pagination works in catalog mode."""
    genus = {**TAXON, "taxon_key": "10", "path": "Opuntia", "rank": "GENUS"}
    catalog_dir = tmp_path / "Opuntia"
    catalog_dir.mkdir(parents=True)
    taxa_list = [
        {"taxon_key": str(i), "path": f"x/{i}", "scientific_name": f"Sp_{i}",
         "common_name": "", "rank": "SPECIES", "sample_count": i * 10}
        for i in range(1, 6)
    ]
    pq.write_table(pa.Table.from_pylist(taxa_list), catalog_dir / "species.parquet")

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
         patch.object(rankings_module, "TREE_ROOT", tmp_path):
        r = client.get("/api/taxa/query?within_taxon=10&descendant_rank=SPECIES&limit=2&offset=2")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert len(body["results"]) == 2
    assert body["results"][0]["taxon_id"] == "3"


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


def test_list_variables():
    with patch.object(tiles, "load_layers_with_category", return_value=[(FAKE_LAYER, FAKE_CATEGORY)]):
        response = client.get("/variables")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "bio1"
    assert body[0]["category"] == "Bioclimatic"
    assert body[0]["value_type"] == "continuous"


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


def test_variable_tile_compat():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch.object(tiles, "get_layer", return_value=FAKE_LAYER), \
         patch.object(tiles, "render_layer_tile_bytes", return_value=png):
        response = client.get("/api/variables/bio_1/tiles/4/8/5.png")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Shared fixtures for new tests
# ---------------------------------------------------------------------------

FAKE_NOM_LAYER = {
    "id": "kg0",
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
    "variable": ["kg0", "kg0", "kg0"],
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


# ---------------------------------------------------------------------------
# _load_relative_ranks
# ---------------------------------------------------------------------------

def test_load_relative_ranks_no_file(tmp_path):
    assert main_module._load_relative_ranks(tmp_path, "bio1") == []


def test_load_relative_ranks_corrupt_file(tmp_path):
    (tmp_path / rankings_module.POSITION_FILE).write_bytes(b"garbage")
    assert main_module._load_relative_ranks(tmp_path, "bio1") == []


def test_load_relative_ranks_filters_by_variable(tmp_path):
    pq.write_table(pa.table({
        "variable":       ["bio1", "kg0"],
        "metric":         ["mean", "entropy"],
        "position":       [4, 1],
        "count":          [10, 5],
        "sampleCount":    [50, 30],
        "contextTaxonId": ["100", "100"],
        "contextLabel":   ["Cactaceae", "Cactaceae"],
    }), tmp_path / rankings_module.POSITION_FILE)
    rows = main_module._load_relative_ranks(tmp_path, "bio1")
    assert len(rows) == 1
    assert rows[0]["metric"] == "mean"
    assert rows[0]["position"] == 5          # 0-based 4 → 1-based 5
    assert rows[0]["count"] == 10
    assert rows[0]["percentile"] == pytest.approx(0.4)     # 4/10
    assert rows[0]["label"] == "Cactaceae"
    assert rows[0]["context_label"] == "Cactaceae"


def test_load_relative_ranks_zero_count(tmp_path):
    """count=0 edge case must not divide by zero."""
    pq.write_table(pa.table({
        "variable":       ["bio1"],
        "metric":         ["mean"],
        "position":       [0],
        "count":          [0],
        "sampleCount":    [0],
        "contextTaxonId": ["1"],
        "contextLabel":   ["Plantae"],
    }), tmp_path / rankings_module.POSITION_FILE)
    rows = main_module._load_relative_ranks(tmp_path, "bio1")
    assert rows[0]["percentile"] == 0.0


# ---------------------------------------------------------------------------
# _load_legend (lines 25-28)
# ---------------------------------------------------------------------------

def test_load_legend_file_present(tmp_path, monkeypatch):
    data = {"classes": [{"id": 1, "name": "Forest"}]}
    (tmp_path / "kg0_legend.json").write_text(json.dumps(data))
    monkeypatch.setattr(main_module, "_LEGEND_DIR", tmp_path)
    main_module._load_legend.cache_clear()
    assert main_module._load_legend("kg0") == [{"id": 1, "name": "Forest"}]


def test_load_legend_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "_LEGEND_DIR", tmp_path)
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
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/species/2923970/obscured")
    assert r.status_code == 200
    assert r.json() == {"allObscured": False}


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
    kg0 = next(v for v in body["variables"] if v["id"] == "kg0")
    assert kg0["density"] is None
    assert kg0["classes"] is not None


def test_get_taxon_env_stats_no_files():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[]), \
         patch("pathlib.Path.exists", return_value=False):
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
         patch("pathlib.Path.exists", return_value=False):
        r = client.get("/species/2923970/environment/kg0")
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
        r = client.get("/species/2923970/environment/kg0")
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
         patch.object(pq, "read_table", return_value=_NOM_STATS_TABLE), \
         patch("main._load_legend", return_value=legend):
        r = client.get("/species/2923970/environment/kg0")
    assert r.status_code == 200
    body = r.json()
    assert body["variable"] == "kg0"
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
         patch("pathlib.Path.exists", return_value=False):
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
        return pa.table({"variable": ["other"]})

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
        return pa.table({"variable": ["other"]})

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
         patch("pathlib.Path.exists", return_value=True), \
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
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=_OCC_TABLE):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    assert len(r.json()["occurrences"]) == 2


def test_get_species_occurrences_nonleaf():
    with patch.object(taxa, "get_taxon_by_id", return_value=NONLEAF_TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[DESC_TAXON]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=_OCC_TABLE):
        r = client.get("/species/2923968/occurrences")
    assert r.status_code == 200
    assert len(r.json()["occurrences"]) == 2


def test_get_species_occurrences_species_includes_subspecies():
    """SPECIES occurrences endpoint iterates self + descendants to include subspecies."""
    subspecies = {**DESC_TAXON, "taxon_key": "9999", "rank": "SUBSPECIES",
                  "path": DESC_TAXON["path"] + "/Sub_9999"}
    sub_table = pa.table({
        "catalogNumber": ["SUB001"],
        "decimalLatitude": [41.0],
        "decimalLongitude": [-76.0],
        "obscured": ["No"],
        "coordinateUncertaintyInMeters": [100.0],
    })
    call_count = {"n": 0}
    def _read_table_side_effect(path, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _OCC_TABLE   # species own obs
        return sub_table         # subspecies obs

    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.iter_descendants", return_value=[TAXON, subspecies]), \
         patch("pathlib.Path.exists", return_value=True), \
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
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=dup_table):
        r = client.get("/species/2923970/occurrences")
    assert len(r.json()["occurrences"]) == 1


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

def _make_occ_with_loc(tmp_path: Path, taxon_path: str, loc_col: str, gid: str, var_col: str, values: list) -> Path:
    occ_dir = tmp_path / taxon_path
    occ_dir.mkdir(parents=True, exist_ok=True)
    n = len(values)
    data = {
        "catalogNumber": [f"obs{i}" for i in range(n)],
        "decimalLatitude": [40.0] * n,
        "decimalLongitude": [-75.0] * n,
        "obscured": ["No"] * n,
        "coordinateUncertaintyInMeters": [100.0] * n,
        loc_col: [gid] * n,
        var_col: values,
    }
    occ_path = occ_dir / "occurrence.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False), occ_path)
    return occ_path


def test_get_species_environment_with_location_continuous(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "bio1",
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
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "kg0",
                       [1.0] * 15 + [2.0] * 5)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    legend = [{"id": 1, "name": "Tropical", "description": None, "traits": None},
              {"id": 2, "name": "Arid", "description": None, "traits": {"color": "#f00"}}]
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("main._load_legend", return_value=legend):
        r = client.get("/species/2923970/environment/kg0?location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["observation_count"] == 20
    dist = body["categorical_distribution"]
    assert len(dist) == 2
    assert dist[0]["fraction"] == pytest.approx(0.75)
    assert body["relative_ranks"] == []


def test_get_species_environment_with_location_no_data_falls_through(monkeypatch):
    """compute_location_filtered_stats returns None → falls back to precomputed stats."""
    monkeypatch.setattr(st_module, "collect_taxon_df", lambda t: None)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", side_effect=_env_stats_read):
        r = client.get("/species/2923970/environment/bio1?location=USA")
    assert r.status_code == 200
    assert r.json()["observation_count"] == 100  # from precomputed table


def test_get_species_environment_with_location_no_layer_falls_through(monkeypatch):
    """layer=None skips location block and falls through to precomputed path."""
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[]), \
         patch("pathlib.Path.exists", return_value=False):
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


# ---------------------------------------------------------------------------
# Slice / class-samples shared index table
# ---------------------------------------------------------------------------

_INDEX_TABLE = pa.table({
    "catalogNumber": ["OCC001", "OCC002", "OCC003"],
    "decimalLatitude": [40.5, 41.0, 42.0],
    "decimalLongitude": [-75.0, -74.5, -73.0],
    "bio1": [10.0, 20.0, 30.0],
    "kg0": [1.0, 2.0, 1.0],
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


# ---------------------------------------------------------------------------
# _read_index_for_slice (lines 333-350)
# ---------------------------------------------------------------------------

def test_read_index_variable_not_in_schema(tmp_path):
    path = tmp_path / "idx.parquet"
    pq.write_table(_INDEX_TABLE, path)
    result = main_module._read_index_for_slice(path, "missing_var", value_min=0.0, value_max=100.0)
    assert result == []


def test_read_index_range_filter(tmp_path):
    path = tmp_path / "idx.parquet"
    pq.write_table(_INDEX_TABLE, path)
    result = main_module._read_index_for_slice(path, "bio1", value_min=5.0, value_max=15.0)
    assert len(result) == 1
    assert result[0]["catalogNumber"] == "OCC001"
    assert result[0]["value"] == pytest.approx(10.0)


def test_read_index_class_filter(tmp_path):
    path = tmp_path / "idx.parquet"
    pq.write_table(_INDEX_TABLE, path)
    result = main_module._read_index_for_slice(path, "kg0", class_value=1.0)
    assert len(result) == 2
    assert all(r["value"] == pytest.approx(1.0) for r in result)


def test_read_index_circular_wrap(tmp_path):
    wrap_table = pa.table({
        "catalogNumber": ["A", "B", "C"],
        "decimalLatitude": [40.0, 41.0, 42.0],
        "decimalLongitude": [-75.0, -74.0, -73.0],
        "aspect_deg": [350.0, 10.0, 180.0],
    })
    path = tmp_path / "idx.parquet"
    pq.write_table(wrap_table, path)
    # selection 315→45 wraps through north; 350 and 10 should match, 180 should not
    result = main_module._read_index_for_slice(
        path, "aspect_deg", value_min=315.0, value_max=45.0, circular_wrap=True,
    )
    catalogs = {r["catalogNumber"] for r in result}
    assert "A" in catalogs
    assert "B" in catalogs
    assert "C" not in catalogs


def test_read_index_limit(tmp_path):
    path = tmp_path / "idx.parquet"
    pq.write_table(_INDEX_TABLE, path)
    result = main_module._read_index_for_slice(path, "bio1", value_min=0.0, value_max=100.0, limit=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# /species/{id}/environment/{var}/slice (lines 369-397)
# ---------------------------------------------------------------------------

def test_slice_not_finite():
    r = client.get("/species/2923970/environment/bio1/slice?min=nan&max=20")
    assert r.status_code == 400


def test_slice_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/environment/bio1/slice?min=0&max=30")
    assert r.status_code == 404


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
        r = client.get("/species/2923970/environment/kg0/slice?min=0&max=30")
    assert r.status_code == 400


def test_slice_no_index_file():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]), \
         patch("pathlib.Path.exists", return_value=False):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30")
    assert r.status_code == 404


def test_slice_success(tmp_path):
    idx_path = tmp_path / "idx.parquet"
    pq.write_table(_INDEX_TABLE, idx_path)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("main.TREE_ROOT", tmp_path / TAXON["path"]):
        # Patch the index path directly via TREE_ROOT so the endpoint builds path correctly
        pass

    # Simpler: mock read_schema and read_table
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_schema", return_value=_INDEX_SCHEMA), \
         patch.object(pq, "read_table", return_value=_INDEX_TABLE):
        r = client.get("/species/2923970/environment/bio1/slice?min=5&max=25")
    assert r.status_code == 200
    body = r.json()
    assert body["variable"] == "bio1"
    assert body["range"] == {"min": 5.0, "max": 25.0}
    assert body["count"] == 2
    assert body["observations"][0]["catalogNumber"] == "OCC001"


def test_slice_min_greater_than_max_swapped():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_schema", return_value=_INDEX_SCHEMA), \
         patch.object(pq, "read_table", return_value=_INDEX_TABLE):
        r = client.get("/species/2923970/environment/bio1/slice?min=25&max=5")
    assert r.status_code == 200
    # After swap, min=5 max=25, same 2 results
    assert r.json()["count"] == 2


# ---------------------------------------------------------------------------
# /species/{id}/environment/{var}/class/{val}/samples (lines 407-434)
# ---------------------------------------------------------------------------

def test_class_samples_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/environment/kg0/class/1/samples")
    assert r.status_code == 404


def test_class_samples_layer_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[]):
        r = client.get("/species/2923970/environment/kg0/class/1/samples")
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
        r = client.get("/species/2923970/environment/kg0/class/notanumber/samples")
    assert r.status_code == 400


def test_class_samples_no_index():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("pathlib.Path.exists", return_value=False):
        r = client.get("/species/2923970/environment/kg0/class/1/samples")
    assert r.status_code == 404


def test_class_samples_success():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_schema", return_value=_INDEX_SCHEMA), \
         patch.object(pq, "read_table", return_value=_INDEX_TABLE):
        r = client.get("/species/2923970/environment/kg0/class/1/samples")
    assert r.status_code == 200
    body = r.json()
    assert body["variable"] == "kg0"
    assert body["class_value"] == 1
    assert body["count"] == 2
    assert all(obs["value"] == pytest.approx(1.0) for obs in body["observations"])


# ---------------------------------------------------------------------------
# Slice with location param
# ---------------------------------------------------------------------------

def test_slice_with_location_success(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "bio1",
                       list(np.linspace(5.0, 25.0, 20)))
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=10&max=20&location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    assert all(10.0 <= obs["value"] <= 20.0 for obs in body["observations"])


def test_slice_with_location_no_data(tmp_path, monkeypatch):
    """No occurrence.parquet → collect_taxon_df returns None → empty results."""
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    (tmp_path / TAXON["path"]).mkdir(parents=True, exist_ok=True)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_slice_with_location_empty_after_gid_filter(tmp_path, monkeypatch):
    """Data exists but no rows match the requested GID → empty results."""
    import numpy as np

    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    # Occurrence file has CAN rows, not USA
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "CAN", "bio1",
                       list(np.linspace(5.0, 25.0, 20)))
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=30&location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_slice_with_location_filter_col_none_falls_through(tmp_path, monkeypatch):
    """filter_col None → falls through to precomputed index path."""
    _patch_hierarchy(monkeypatch, {"WEIRD": {"level": 99, "name": "Weird", "parent_gid": None}})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_DISC_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_schema", return_value=_INDEX_SCHEMA), \
         patch.object(pq, "read_table", return_value=_INDEX_TABLE):
        r = client.get("/species/2923970/environment/bio1/slice?min=0&max=100&location=WEIRD")
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_slice_from_raw_occ_circular_wrap(tmp_path, monkeypatch):
    """_slice_from_raw_occ handles circular_wrap=True correctly."""
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    occ_dir = tmp_path / TAXON["path"]
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["A", "B", "C"],
        "decimalLatitude": [40.0, 41.0, 42.0],
        "decimalLongitude": [-75.0, -74.0, -73.0],
        "obscured": ["No", "No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0, 100.0],
        "level0Gid": ["USA", "USA", "USA"],
        "aspectdeg": [350.0, 10.0, 180.0],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrence.parquet")
    result = main_module._slice_from_raw_occ(
        TAXON, "aspectdeg", "level0Gid", "USA", 315.0, 45.0, True, None,
    )
    catalogs = {r["catalogNumber"] for r in result}
    assert "A" in catalogs
    assert "B" in catalogs
    assert "C" not in catalogs


def test_slice_with_location_limit(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "bio1",
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
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "kg0",
                       [1.0] * 10 + [2.0] * 10)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg0/class/1/samples?location=USA")
    assert r.status_code == 200
    body = r.json()
    assert body["class_value"] == 1
    assert body["count"] == 10
    assert all(obs["value"] == pytest.approx(1.0) for obs in body["observations"])


def test_class_samples_with_location_no_data(tmp_path, monkeypatch):
    """No occurrence.parquet → collect_taxon_df returns None → empty results."""
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    (tmp_path / TAXON["path"]).mkdir(parents=True, exist_ok=True)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg0/class/1/samples?location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_class_samples_with_location_empty_after_gid_filter(tmp_path, monkeypatch):
    """Data exists but no rows match the requested GID → empty results."""
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    # Occurrence file has CAN rows, not USA
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "CAN", "kg0", [1.0] * 10)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg0/class/1/samples?location=USA")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_class_samples_with_location_filter_col_none_falls_through(monkeypatch):
    """filter_col None → falls through to precomputed index path."""
    _patch_hierarchy(monkeypatch, {"WEIRD": {"level": 99, "name": "Weird", "parent_gid": None}})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_schema", return_value=_INDEX_SCHEMA), \
         patch.object(pq, "read_table", return_value=_INDEX_TABLE):
        r = client.get("/species/2923970/environment/kg0/class/1/samples?location=WEIRD")
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_class_samples_with_location_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(st_module, "TREE_ROOT", tmp_path)
    monkeypatch.setattr(st_module, "iter_descendants", lambda t, **kw: [t])
    _make_occ_with_loc(tmp_path, TAXON["path"], "level0Gid", "USA", "kg0", [1.0] * 20)
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[FAKE_NOM_LAYER]):
        r = client.get("/species/2923970/environment/kg0/class/1/samples?location=USA&limit=3")
    assert r.status_code == 200
    assert r.json()["count"] == 3


# ---------------------------------------------------------------------------
# /species/{id}/occurrences with location param
# ---------------------------------------------------------------------------

def test_get_species_occurrences_with_location(tmp_path, monkeypatch):
    """location filter restricts returned pins to matching rows only."""
    occ_dir = tmp_path / TAXON["path"]
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["USA001", "USA002", "CAN001"],
        "decimalLatitude": [40.0, 41.0, 50.0],
        "decimalLongitude": [-75.0, -74.0, -80.0],
        "obscured": ["No", "No", "No"],
        "coordinateUncertaintyInMeters": [100.0, 100.0, 100.0],
        "level0Gid": ["USA", "USA", "CAN"],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrence.parquet")
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.TREE_ROOT", tmp_path), \
         patch("main.iter_descendants", return_value=[TAXON]):
        r = client.get("/species/2923970/occurrences?location=USA")
    assert r.status_code == 200
    occs = r.json()["occurrences"]
    catalog_numbers = {o["catalogNumber"] for o in occs}
    assert catalog_numbers == {"USA001", "USA002"}
    assert "CAN001" not in catalog_numbers


def test_get_species_occurrences_with_location_no_match(tmp_path, monkeypatch):
    """location filter with no matching rows returns empty list."""
    occ_dir = tmp_path / TAXON["path"]
    occ_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "catalogNumber": ["CAN001"],
        "decimalLatitude": [50.0],
        "decimalLongitude": [-80.0],
        "obscured": ["No"],
        "coordinateUncertaintyInMeters": [100.0],
        "level0Gid": ["CAN"],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(data), preserve_index=False),
                   occ_dir / "occurrence.parquet")
    _patch_hierarchy(monkeypatch, {"USA": _USA})
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None), \
         patch("main.TREE_ROOT", tmp_path), \
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

def _write_rank_index_for_main(index_path: Path, entries: dict) -> None:
    """Write a minimal rank index parquet (mirrors the rankings test helper)."""
    struct_type = pa.struct([
        pa.field("taxonKey", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("sampleCount", pa.int64()),
    ])
    max_len = max(len(v) for v in entries.values()) if entries else 1
    arrays: dict = {}
    column_lengths: dict = {}
    for col_name, rows in entries.items():
        column_lengths[col_name] = len(rows)
        arr = pa.StructArray.from_arrays(
            [
                pa.array([r[0] for r in rows], type=pa.string()),
                pa.array([r[1] for r in rows], type=pa.float64()),
                pa.array([r[2] for r in rows], type=pa.int64()),
            ],
            fields=list(struct_type),
        )
        if len(arr) < max_len:
            arr = pa.concat_arrays([arr, pa.nulls(max_len - len(arr), type=struct_type)])
        arrays[col_name] = arr
    table = pa.table(arrays)
    metadata = {b"column_lengths": json.dumps(column_lengths).encode("utf-8")}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), index_path)


def test_ranking_options_taxon_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/api/taxa/ranking-options?within_taxon=999&descendant_rank=SPECIES")
    assert r.status_code == 404


def test_ranking_options_no_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rankings_module, "TREE_ROOT", tmp_path)
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["options"] == []
    assert body["rank"] == "SPECIES"


def test_ranking_options_corrupt_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rankings_module, "TREE_ROOT", tmp_path)
    index_path = tmp_path / TAXON["path"] / "species_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"garbage")
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    assert r.json()["options"] == []


def test_ranking_options_returns_options(tmp_path, monkeypatch):
    monkeypatch.setattr(rankings_module, "TREE_ROOT", tmp_path)
    index_path = tmp_path / TAXON["path"] / "species_index.parquet"
    _write_rank_index_for_main(index_path, {
        "bio1::mean": [("2923970", 10.0, 100)],
        "bio1::class_0": [("2923970", 1.0, 100)],
        "bio12::median": [("2923970", 5.0, 100)],
        "no_double_colon": [("2923970", 1.0, 100)],  # skipped: no ::
        "bio12::p10": [],  # skipped: count == 0
    })
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(tiles, "load_layers", return_value=[
             {"id": "bio1", "display_name": "Temperature"},
             {"id": "bio12", "display_name": "Precipitation"},
         ]):
        r = client.get("/api/taxa/ranking-options?within_taxon=2923970&descendant_rank=SPECIES")
    assert r.status_code == 200
    body = r.json()
    assert body["ancestor_taxon_id"] == TAXON["taxon_key"]
    assert body["rank"] == "SPECIES"
    options = body["options"]
    variables = [o["variable"] for o in options]
    assert "bio1" in variables
    assert "bio12" in variables
    # class_ metrics are now included as sort options
    class_options = [o for o in options if o["metric"].startswith("class_")]
    assert len(class_options) == 1
    assert class_options[0]["variable"] == "bio1"
    # label populated for all options
    assert all(isinstance(o["label"], str) and o["label"] for o in options)
    assert all(o["count"] > 0 for o in options)


# ---------------------------------------------------------------------------
# _lookup_index_value
# ---------------------------------------------------------------------------

def test_lookup_index_value_missing_file(tmp_path):
    from main import _lookup_index_value
    taxon = {**TAXON, "path": "no_such_path"}
    with patch.object(main_module, "TREE_ROOT", tmp_path):
        result = _lookup_index_value(taxon, "bio1", "12345")
    assert result is None


def test_lookup_index_value_column_absent(tmp_path):
    from main import _lookup_index_value
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    pq.write_table(pa.table({"catalogNumber": pa.array(["12345"])}), taxon_dir / "occurrence_index.parquet")
    with patch.object(main_module, "TREE_ROOT", tmp_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_catalog_number_not_found(tmp_path):
    from main import _lookup_index_value
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"catalogNumber": pa.array(["99999"]), "bio1": pa.array([14.35])}),
        taxon_dir / "occurrence_index.parquet",
    )
    with patch.object(main_module, "TREE_ROOT", tmp_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_returns_float(tmp_path):
    from main import _lookup_index_value
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"catalogNumber": pa.array(["12345"]), "bio1": pa.array([14.35])}),
        taxon_dir / "occurrence_index.parquet",
    )
    with patch.object(main_module, "TREE_ROOT", tmp_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result == pytest.approx(14.35)


def test_lookup_index_value_read_error_returns_none(tmp_path):
    from main import _lookup_index_value
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"catalogNumber": pa.array(["12345"]), "bio1": pa.array([14.35])}),
        taxon_dir / "occurrence_index.parquet",
    )
    with patch.object(main_module, "TREE_ROOT", tmp_path), \
         patch("main.pq.read_table", side_effect=Exception("corrupt")):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


def test_lookup_index_value_null_value_returns_none(tmp_path):
    from main import _lookup_index_value
    taxon_dir = tmp_path / TAXON["path"]
    taxon_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"catalogNumber": pa.array(["12345"]), "bio1": pa.array([None], type=pa.float64())}),
        taxon_dir / "occurrence_index.parquet",
    )
    with patch.object(main_module, "TREE_ROOT", tmp_path):
        result = _lookup_index_value(TAXON, "bio1", "12345")
    assert result is None


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
    "id": "kg0",
    "filename": "kg0.tif",
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
        r = client.get("/gis/point?lat=40&lon=-105&variable=kg0")
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
    csv = b"x,y\n1,2\n"
    r = client.post("/upload/raw-observations",
                    files=[("file", ("obs.csv", csv, "text/csv"))])
    assert r.status_code == 422


def test_upload_csv_success():
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    csv = b"latitude,longitude\n45.0,-120.0\n46.0,-121.0\n"
    fake_archive = Path(tempfile.mkdtemp()) / "out.zip"
    fake_work_dir = fake_archive.parent
    with zipfile.ZipFile(fake_archive, "w") as zf:
        zf.writestr("occurrence.parquet", b"")

    with patch("util.upload.enrich_with_gis", return_value=pd.DataFrame({
            "catalogNumber": ["Observation #1", "Observation #2"],
            "decimalLatitude": [45.0, 46.0],
            "decimalLongitude": [-120.0, -121.0],
        })), \
         patch("util.upload.build_archive", return_value=(fake_archive, "processed_observations.zip", fake_work_dir)), \
         patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.csv", csv, "text/csv"))])
    shutil.rmtree(fake_work_dir, ignore_errors=True)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"


def test_upload_tsv_parsed_correctly():
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    tsv = b"latitude\tlongitude\n45.0\t-120.0\n"
    fake_archive = Path(tempfile.mkdtemp()) / "out.zip"
    fake_work_dir = fake_archive.parent
    with zipfile.ZipFile(fake_archive, "w") as zf:
        zf.writestr("occurrence.parquet", b"")

    with patch("util.upload.enrich_with_gis", return_value=pd.DataFrame({
            "catalogNumber": ["Observation #1"],
            "decimalLatitude": [45.0],
            "decimalLongitude": [-120.0],
        })), \
         patch("util.upload.build_archive", return_value=(fake_archive, "processed_observations.zip", fake_work_dir)), \
         patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.tsv", tsv, "text/tab-separated-values"))])
    shutil.rmtree(fake_work_dir, ignore_errors=True)
    assert r.status_code == 200


def test_upload_parquet_parsed_correctly():
    import io
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq_local

    df_in = pd.DataFrame({"latitude": [45.0], "longitude": [-120.0]})
    buf = io.BytesIO()
    pq_local.write_table(pa.Table.from_pandas(df_in), buf)
    parquet_bytes = buf.getvalue()

    fake_archive = Path(tempfile.mkdtemp()) / "out.zip"
    fake_work_dir = fake_archive.parent
    with zipfile.ZipFile(fake_archive, "w") as zf:
        zf.writestr("occurrence.parquet", b"")

    with patch("util.upload.enrich_with_gis", return_value=pd.DataFrame({
            "catalogNumber": ["Observation #1"],
            "decimalLatitude": [45.0],
            "decimalLongitude": [-120.0],
        })), \
         patch("util.upload.build_archive", return_value=(fake_archive, "processed_observations.zip", fake_work_dir)), \
         patch("util.tiles.load_layers", return_value=[]):
        r = client.post("/upload/raw-observations",
                        files=[("file", ("obs.parquet", parquet_bytes, "application/octet-stream"))])
    shutil.rmtree(fake_work_dir, ignore_errors=True)
    assert r.status_code == 200
