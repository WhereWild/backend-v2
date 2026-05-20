import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

import main as main_module
import util.taxa as taxa
import util.tiles as tiles
from main import app
from util.rankings import POSITION_FILE
from util.stats import NOMINAL_STATS_FILE, NUMERICAL_DENSITY_FILE, NUMERICAL_STATS_FILE

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
        NUMERICAL_STATS_FILE: _NUM_STATS_TABLE,
        NOMINAL_STATS_FILE: _NOM_STATS_TABLE,
        NUMERICAL_DENSITY_FILE: _DENSITY_TABLE,
    }.get(Path(str(path)).name, pa.table({}))


# ---------------------------------------------------------------------------
# _load_relative_ranks
# ---------------------------------------------------------------------------

def test_load_relative_ranks_no_file(tmp_path):
    assert main_module._load_relative_ranks(tmp_path, "bio1") == []


def test_load_relative_ranks_corrupt_file(tmp_path):
    (tmp_path / POSITION_FILE).write_bytes(b"garbage")
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
    }), tmp_path / POSITION_FILE)
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
    }), tmp_path / POSITION_FILE)
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
        if name == NUMERICAL_STATS_FILE:
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
        if name == NUMERICAL_STATS_FILE:
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
        if name == NUMERICAL_STATS_FILE:
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
         patch("pathlib.Path.exists", return_value=False):
        r = client.get("/species/2923970/occurrences")
    assert r.status_code == 200
    assert r.json()["occurrences"] == []


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
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(pq, "read_table", return_value=dup_table):
        r = client.get("/species/2923970/occurrences")
    assert len(r.json()["occurrences"]) == 1


# ---------------------------------------------------------------------------
# /species/{id}/locations (lines 315-318)
# ---------------------------------------------------------------------------

def test_get_species_locations_not_found():
    with patch.object(taxa, "get_taxon_by_id", return_value=None), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/nope/locations")
    assert r.status_code == 404


def test_get_species_locations_returns_empty():
    with patch.object(taxa, "get_taxon_by_id", return_value=TAXON), \
         patch.object(taxa, "get_taxon_by_slug", return_value=None):
        r = client.get("/species/2923970/locations")
    assert r.status_code == 200
    assert r.json() == []


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
