"""Data integrity tests for per-species parquet files."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def known_species_dir(data_root, parquet_storage):
    """Quercus robur — confirmed to have all data files."""
    p = (
        data_root
        / "species" / "taxonomy"
        / "Plantae_6"
        / "Tracheophyta_7707728"
        / "Magnoliopsida_220"
        / "Fagales_1354"
        / "Fagaceae_4689"
        / "Quercus_2877951"
        / "Quercus_robur_2878688"
    )
    if not parquet_storage.exists(p):
        pytest.skip(f"Known species dir not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Known species — all required files present
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "occurrence.parquet",
    "occurrence_index.parquet",
    "summary_stats.parquet",
    "density_graph.parquet",
]

def test_known_species_has_all_required_files(known_species_dir, parquet_storage):
    missing = [f for f in REQUIRED_FILES if not parquet_storage.exists(known_species_dir / f)]
    assert not missing, f"Known species dir missing files: {missing}"


# ---------------------------------------------------------------------------
# occurrence.parquet schema & value sanity (known species)
# ---------------------------------------------------------------------------

OCCURRENCE_REQUIRED_COLS = {
    "decimalLatitude", "decimalLongitude", "catalogNumber",
    "obscured", "gbifRegion",
}

def test_occurrence_parquet_has_required_columns(known_species_dir, parquet_storage):
    schema = parquet_storage.read_schema(known_species_dir / "occurrence.parquet")
    actual = set(schema.names)
    missing = OCCURRENCE_REQUIRED_COLS - actual
    assert not missing, f"occurrence.parquet missing columns: {missing}"


def test_occurrence_parquet_has_rows(known_species_dir, parquet_storage):
    meta = parquet_storage.read_metadata(known_species_dir / "occurrence.parquet")
    assert meta.num_rows > 0


def test_occurrence_lat_lon_ranges(known_species_dir, parquet_storage):
    table = parquet_storage.read_table(
        known_species_dir / "occurrence.parquet",
        columns=["decimalLatitude", "decimalLongitude"],
    )
    lats = table["decimalLatitude"].to_pylist()
    lons = table["decimalLongitude"].to_pylist()
    bad_lat = [v for v in lats if v is not None and not (-90 <= v <= 90)]
    bad_lon = [v for v in lons if v is not None and not (-180 <= v <= 180)]
    assert not bad_lat, f"{len(bad_lat)} latitudes out of [-90, 90]"
    assert not bad_lon, f"{len(bad_lon)} longitudes out of [-180, 180]"


def test_occurrence_no_null_catalog_numbers(known_species_dir, parquet_storage):
    table = parquet_storage.read_table(
        known_species_dir / "occurrence.parquet",
        columns=["catalogNumber"],
    )
    null_count = table["catalogNumber"].null_count
    assert null_count == 0, f"{null_count} null catalogNumbers in occurrence.parquet"


# ---------------------------------------------------------------------------
# summary_stats.parquet schema & value sanity (known species)
# ---------------------------------------------------------------------------

SUMMARY_REQUIRED_COLS = {"variable", "count", "min", "max", "mean"}

def test_summary_stats_has_required_columns(known_species_dir, parquet_storage):
    schema = parquet_storage.read_schema(known_species_dir / "summary_stats.parquet")
    actual = set(schema.names)
    missing = SUMMARY_REQUIRED_COLS - actual
    assert not missing, f"summary_stats.parquet missing columns: {missing}"


def test_summary_stats_has_rows(known_species_dir, parquet_storage):
    meta = parquet_storage.read_metadata(known_species_dir / "summary_stats.parquet")
    assert meta.num_rows > 0


def test_summary_stats_min_lte_max(known_species_dir, parquet_storage):
    table = parquet_storage.read_table(
        known_species_dir / "summary_stats.parquet",
        columns=["variable", "min", "max"],
    )
    variables = table["variable"].to_pylist()
    mins = table["min"].to_pylist()
    maxs = table["max"].to_pylist()
    violations = [
        variables[i]
        for i in range(len(variables))
        if mins[i] is not None and maxs[i] is not None and mins[i] > maxs[i]
    ]
    assert not violations, f"min > max for variables: {violations}"


def test_summary_stats_counts_positive(known_species_dir, parquet_storage):
    table = parquet_storage.read_table(
        known_species_dir / "summary_stats.parquet",
        columns=["variable", "count"],
    )
    variables = table["variable"].to_pylist()
    counts = table["count"].to_pylist()
    bad = [variables[i] for i, c in enumerate(counts) if c is not None and c <= 0]
    assert not bad, f"Non-positive counts for variables: {bad}"


