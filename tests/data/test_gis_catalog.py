"""Data integrity tests for the GIS catalog (data/gis/catalog.json)."""
from __future__ import annotations

import json
import pytest


@pytest.fixture(scope="module")
def catalog(parquet_storage):
    from util.config import load_config
    catalog_path = load_config("global").gis_catalog_path
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
def static_layers(catalog):
    """Flat list of layers excluding temporal categories (whose IDs are composed at runtime)."""
    layers = []
    for category in catalog.get("categories", []):
        if category.get("windows"):
            continue
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
    known_types = {"numeric", "categorical", "circular"}
    for layer in all_layers:
        vtype = (layer.get("value_type") or "").lower()
        assert vtype in known_types, (
            f"Layer '{layer.get('id')}' has unknown value_type '{vtype}'"
        )


def test_catalog_layer_ids_are_unique(static_layers):
    # Temporal category layers have base IDs (e.g. "temperature_2m") that are
    # composed into full IDs at runtime ("temperature_2m_avg_24h"), so they are
    # excluded from this uniqueness check.
    ids = [layer["id"] for layer in static_layers if "id" in layer]
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
    known = {"numeric", "categorical", "circular"}
    for var_id, entry in variables_map.items():
        vtype = (entry.get("value_type") or "").lower()
        assert vtype in known, f"Variable '{var_id}' has unknown value_type '{vtype}'"


# ---------------------------------------------------------------------------
# Legends exist for categorical variables
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Citation completeness
# ---------------------------------------------------------------------------

def test_data_sources_present(catalog):
    assert "data_sources" in catalog
    assert len(catalog["data_sources"]) > 0


def test_each_source_has_required_fields(catalog):
    required = ["name", "url", "license", "references"]
    for sid, source in catalog["data_sources"].items():
        missing = [f for f in required if not source.get(f) and source.get(f) != []]
        assert not missing, f"Source '{sid}' missing fields: {missing}"


def test_each_reference_has_required_fields(catalog):
    for sid, source in catalog["data_sources"].items():
        for i, ref in enumerate(source.get("references", [])):
            assert ref.get("authors"), f"Source '{sid}' ref[{i}] missing 'authors'"
            assert ref.get("title"), f"Source '{sid}' ref[{i}] missing 'title'"
            assert ref.get("year"), f"Source '{sid}' ref[{i}] missing 'year'"
            has_link = ref.get("doi") or ref.get("url")
            assert has_link, f"Source '{sid}' ref[{i}] missing both 'doi' and 'url'"


def test_all_layers_have_source_ids(catalog):
    for cat in catalog["categories"]:
        cat_has = bool(cat.get("source_ids"))
        for layer in cat.get("layers", []):
            effective = bool(layer.get("source_ids")) or cat_has
            assert effective, (
                f"Layer '{layer.get('id')}' in category '{cat.get('name')}' has no effective source_ids"
            )


def test_all_source_ids_are_valid(catalog):
    valid_ids = set(catalog.get("data_sources", {}).keys())
    for cat in catalog["categories"]:
        for sid in cat.get("source_ids", []):
            assert sid in valid_ids, (
                f"Category '{cat.get('name')}' references unknown source_id '{sid}'"
            )
        for layer in cat.get("layers", []):
            for sid in layer.get("source_ids", []):
                assert sid in valid_ids, (
                    f"Layer '{layer.get('id')}' references unknown source_id '{sid}'"
                )


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
