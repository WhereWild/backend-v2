"""Data integrity tests for per-species parquet files.

Samples a small number of species rather than iterating all of them
so the test suite stays fast even with thousands of taxa.
"""
from __future__ import annotations

import random
from pathlib import Path

import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def taxonomy_root(data_root):
    p = data_root / "species" / "taxonomy"
    if not p.exists():
        pytest.skip(f"Taxonomy root not found: {p}")
    return p


@pytest.fixture(scope="module")
def sampled_species_dirs(taxonomy_root):
    """Walk the taxonomy tree and return up to 30 leaf species directories
    that contain occurrence.parquet."""
    candidates = []
    for path in taxonomy_root.rglob("occurrence.parquet"):
        candidates.append(path.parent)
        if len(candidates) >= 300:  # stop early to keep glob fast
            break
    if not candidates:
        pytest.skip("No species directories with occurrence.parquet found")
    rng = random.Random(42)  # deterministic sample
    return rng.sample(candidates, min(30, len(candidates)))


@pytest.fixture(scope="module")
def known_species_dir(taxonomy_root):
    """Quercus robur — confirmed to have all data files."""
    p = (
        taxonomy_root
        / "Plantae_6"
        / "Tracheophyta_7707728"
        / "Magnoliopsida_220"
        / "Fagales_1354"
        / "Fagaceae_4689"
        / "Quercus_2877951"
        / "Quercus_robur_2878688"
    )
    if not p.exists():
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

def test_known_species_has_all_required_files(known_species_dir):
    missing = [f for f in REQUIRED_FILES if not (known_species_dir / f).exists()]
    assert not missing, f"Known species dir missing files: {missing}"


# ---------------------------------------------------------------------------
# occurrence.parquet schema & value sanity (known species)
# ---------------------------------------------------------------------------

OCCURRENCE_REQUIRED_COLS = {
    "decimalLatitude", "decimalLongitude", "catalogNumber",
    "obscured", "gbifRegion",
}

def test_occurrence_parquet_has_required_columns(known_species_dir):
    schema = pq.read_schema(known_species_dir / "occurrence.parquet")
    actual = set(schema.names)
    missing = OCCURRENCE_REQUIRED_COLS - actual
    assert not missing, f"occurrence.parquet missing columns: {missing}"


def test_occurrence_parquet_has_rows(known_species_dir):
    meta = pq.read_metadata(known_species_dir / "occurrence.parquet")
    assert meta.num_rows > 0


def test_occurrence_lat_lon_ranges(known_species_dir):
    table = pq.read_table(
        known_species_dir / "occurrence.parquet",
        columns=["decimalLatitude", "decimalLongitude"],
    )
    lats = table["decimalLatitude"].to_pylist()
    lons = table["decimalLongitude"].to_pylist()
    bad_lat = [v for v in lats if v is not None and not (-90 <= v <= 90)]
    bad_lon = [v for v in lons if v is not None and not (-180 <= v <= 180)]
    assert not bad_lat, f"{len(bad_lat)} latitudes out of [-90, 90]"
    assert not bad_lon, f"{len(bad_lon)} longitudes out of [-180, 180]"


def test_occurrence_no_null_catalog_numbers(known_species_dir):
    table = pq.read_table(
        known_species_dir / "occurrence.parquet",
        columns=["catalogNumber"],
    )
    null_count = table["catalogNumber"].null_count
    assert null_count == 0, f"{null_count} null catalogNumbers in occurrence.parquet"


# ---------------------------------------------------------------------------
# summary_stats.parquet schema & value sanity (known species)
# ---------------------------------------------------------------------------

SUMMARY_REQUIRED_COLS = {"variable", "count", "min", "max", "mean"}

def test_summary_stats_has_required_columns(known_species_dir):
    schema = pq.read_schema(known_species_dir / "summary_stats.parquet")
    actual = set(schema.names)
    missing = SUMMARY_REQUIRED_COLS - actual
    assert not missing, f"summary_stats.parquet missing columns: {missing}"


def test_summary_stats_has_rows(known_species_dir):
    meta = pq.read_metadata(known_species_dir / "summary_stats.parquet")
    assert meta.num_rows > 0


def test_summary_stats_min_lte_max(known_species_dir):
    table = pq.read_table(
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


def test_summary_stats_counts_positive(known_species_dir):
    table = pq.read_table(
        known_species_dir / "summary_stats.parquet",
        columns=["variable", "count"],
    )
    variables = table["variable"].to_pylist()
    counts = table["count"].to_pylist()
    bad = [variables[i] for i, c in enumerate(counts) if c is not None and c <= 0]
    assert not bad, f"Non-positive counts for variables: {bad}"


# ---------------------------------------------------------------------------
# Sampled species — spot-check required files and coordinate ranges
# ---------------------------------------------------------------------------

def test_sampled_species_have_occurrence_parquet(sampled_species_dirs):
    missing = [str(d) for d in sampled_species_dirs if not (d / "occurrence.parquet").exists()]
    assert not missing, f"{len(missing)} sampled species missing occurrence.parquet"


def test_sampled_species_have_summary_stats(sampled_species_dirs):
    missing = [str(d) for d in sampled_species_dirs if not (d / "summary_stats.parquet").exists()]
    assert not missing, f"{len(missing)} sampled species missing summary_stats.parquet"


def test_sampled_species_lat_lon_valid(sampled_species_dirs):
    violations = []
    for d in sampled_species_dirs:
        table = pq.read_table(
            d / "occurrence.parquet",
            columns=["decimalLatitude", "decimalLongitude"],
        )
        lats = table["decimalLatitude"].to_pylist()
        lons = table["decimalLongitude"].to_pylist()
        bad = [
            (lat, lon) for lat, lon in zip(lats, lons)
            if lat is not None and lon is not None
            and (not (-90 <= lat <= 90) or not (-180 <= lon <= 180))
        ]
        if bad:
            violations.append(f"{d.name}: {len(bad)} invalid points")
    assert not violations, "Coordinate range violations:\n" + "\n".join(violations)


def test_sampled_species_summary_min_lte_max(sampled_species_dirs):
    violations = []
    for d in sampled_species_dirs:
        p = d / "summary_stats.parquet"
        if not p.exists():
            continue
        table = pq.read_table(p, columns=["variable", "min", "max"])
        for var, mn, mx in zip(
            table["variable"].to_pylist(),
            table["min"].to_pylist(),
            table["max"].to_pylist(),
        ):
            if mn is not None and mx is not None and mn > mx:
                violations.append(f"{d.name}/{var}: min={mn} > max={mx}")
    assert not violations, "summary_stats min > max:\n" + "\n".join(violations)
