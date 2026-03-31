"""Data integrity tests for the GIS catalog (data/gis/catalog.json)."""
from __future__ import annotations

import json
import pytest


@pytest.fixture(scope="module")
def catalog(data_root, parquet_storage):
    catalog_path = data_root / "gis" / "catalog.json"
    if not parquet_storage.exists(catalog_path):
        pytest.skip(f"GIS catalog not found: {catalog_path}")
    with parquet_storage.open_input_file(catalog_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_layers(catalog):
    """Flat list of every layer entry across all categories."""
    layers = []
    for category in catalog.get("categories", []):
        for layer in category.get("layers", []):
            layers.append(layer)
    return layers


@pytest.fixture(scope="module")
def variables_map(data_root):
    """Use the live load_variable_metadata so we test the same view the API sees."""
    from util import gis_lookup
    _, by_id = gis_lookup.load_variable_metadata()
    return by_id


# ---------------------------------------------------------------------------
# Catalog structure
# ---------------------------------------------------------------------------

def test_catalog_has_categories(catalog):
    assert "categories" in catalog
    assert len(catalog["categories"]) > 0


def test_catalog_each_category_has_name(catalog):
    for cat in catalog["categories"]:
        assert "name" in cat, f"Category missing 'name': {cat}"


def test_catalog_each_layer_has_id(all_layers):
    for layer in all_layers:
        assert "id" in layer, f"Layer missing 'id': {layer}"


def test_catalog_each_layer_has_value_type(all_layers):
    known_types = {"numeric", "categorical"}
    for layer in all_layers:
        vtype = (layer.get("value_type") or "").lower()
        assert vtype in known_types, (
            f"Layer '{layer.get('id')}' has unknown value_type '{vtype}'"
        )


def test_catalog_layer_ids_are_unique(all_layers):
    ids = [layer["id"] for layer in all_layers if "id" in layer]
    assert len(ids) == len(set(ids)), f"Duplicate layer IDs: {sorted(set(x for x in ids if ids.count(x) > 1))}"


# ---------------------------------------------------------------------------
# Variable metadata consistency
# ---------------------------------------------------------------------------

def test_variable_metadata_loads(variables_map):
    assert len(variables_map) > 0, "load_variable_metadata returned empty mapping"


def test_variable_metadata_count_reasonable(variables_map):
    assert len(variables_map) >= 10


def test_variable_metadata_each_has_name(variables_map):
    for var_id, entry in variables_map.items():
        assert entry.get("name"), f"Variable '{var_id}' has no name"


def test_variable_metadata_each_has_value_type(variables_map):
    known = {"numeric", "categorical"}
    for var_id, entry in variables_map.items():
        vtype = (entry.get("value_type") or "").lower()
        assert vtype in known, f"Variable '{var_id}' has unknown value_type '{vtype}'"


# ---------------------------------------------------------------------------
# Legends exist for categorical variables
# ---------------------------------------------------------------------------

def test_categorical_variables_have_legend_files(variables_map, parquet_storage):
    from util.config import load_config
    legends_dir = load_config("global").gis_legends_root
    if not parquet_storage.exists(legends_dir):
        pytest.skip(f"Legends directory not found: {legends_dir}")
    missing = []
    for var_id, entry in variables_map.items():
        vtype = (entry.get("value_type") or "").lower()
        if vtype != "categorical":
            continue
        legend_file = legends_dir / f"{var_id}_legend.json"
        if not parquet_storage.exists(legend_file):
            missing.append(var_id)
    assert not missing, f"Missing legend files for categorical variables: {missing}"
