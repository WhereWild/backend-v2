import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

import main as main_module
import util.stats as st_module
import util.taxa as taxa
import util.tiles as tiles
from main import app
from util.rankings import POSITION_FILE

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
        st_module.NUMERICAL_STATS_FILE: _NUM_STATS_TABLE,
        st_module.NOMINAL_STATS_FILE: _NOM_STATS_TABLE,
        st_module.NUMERICAL_DENSITY_FILE: _DENSITY_TABLE,
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
