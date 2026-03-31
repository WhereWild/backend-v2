"""Data integrity tests for the location index (data/gis/locations/)."""
from __future__ import annotations

import pyarrow.compute as pc
import pytest


@pytest.fixture(scope="module")
def locations_dir(data_root, parquet_storage):
    d = data_root / "gis" / "locations"
    if not parquet_storage.exists(d):
        pytest.skip(f"Locations directory not found: {d}")
    return d


@pytest.fixture(scope="module")
def location_taxa_path(locations_dir, parquet_storage):
    p = locations_dir / "location_taxa.parquet"
    if not parquet_storage.exists(p):
        pytest.skip(f"location_taxa.parquet not found: {p}")
    return p


@pytest.fixture(scope="module")
def location_taxa_schema(location_taxa_path, parquet_storage):
    return parquet_storage.read_schema(location_taxa_path)


@pytest.fixture(scope="module")
def location_taxa_meta(location_taxa_path, parquet_storage):
    return parquet_storage.read_metadata(location_taxa_path)


# ---------------------------------------------------------------------------
# Schema — reads schema only, no row data loaded
# ---------------------------------------------------------------------------

def test_location_taxa_exists(location_taxa_path):
    assert location_taxa_path is not None


def test_location_taxa_has_required_columns(location_taxa_schema):
    required = {"scope", "gid", "taxon_id", "count"}
    actual = set(location_taxa_schema.names)
    missing = required - actual
    assert not missing, f"location_taxa.parquet missing columns: {missing}"


def test_location_taxa_column_types(location_taxa_schema):
    schema = {f.name: str(f.type) for f in location_taxa_schema}
    assert "int" in schema["taxon_id"], f"taxon_id should be int, got {schema['taxon_id']}"
    assert "int" in schema["count"], f"count should be int, got {schema['count']}"


def test_location_taxa_has_rows(location_taxa_meta):
    assert location_taxa_meta.num_rows > 0


# ---------------------------------------------------------------------------
# Value sanity — reads single columns only, uses compute kernels not to_pylist
# ---------------------------------------------------------------------------

def test_location_taxa_no_null_gids(location_taxa_path, parquet_storage):
    col = parquet_storage.read_table(location_taxa_path, columns=["gid"])["gid"]
    assert col.null_count == 0, f"{col.null_count} null GIDs in location_taxa.parquet"


def test_location_taxa_no_null_taxon_ids(location_taxa_path, parquet_storage):
    col = parquet_storage.read_table(location_taxa_path, columns=["taxon_id"])["taxon_id"]
    assert col.null_count == 0, f"{col.null_count} null taxon_ids in location_taxa.parquet"


def test_location_taxa_counts_are_positive(location_taxa_path, parquet_storage):
    col = parquet_storage.read_table(location_taxa_path, columns=["count"])["count"]
    min_count = pc.min(col).as_py()
    assert min_count > 0, f"Minimum count is {min_count} — expected all counts > 0"


def test_location_taxa_scopes_are_known(location_taxa_path, parquet_storage):
    col = parquet_storage.read_table(location_taxa_path, columns=["scope"])["scope"]
    scopes = pc.unique(col).to_pylist()
    assert len(scopes) > 0, "No scopes found"
    for scope in scopes:
        assert isinstance(scope, str) and scope, f"Invalid scope value: {scope!r}"


# ---------------------------------------------------------------------------
# Hierarchy / location catalog
# ---------------------------------------------------------------------------

def test_hierarchy_csv_exists(locations_dir, parquet_storage):
    p = locations_dir / "hierarchy.csv"
    assert parquet_storage.exists(p), f"hierarchy.csv not found at {p}"
