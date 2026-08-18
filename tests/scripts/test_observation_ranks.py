# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for scripts/observation_ranks.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import scripts.observation_ranks as obs_ranks

# Genus G1
#   Species S1 "Echinocereus triglochidiatus"
#     Subspecies T1 "ssp. triglochidiatus"
#     Subspecies T2 "ssp. mojavensis"
#   Species S2 "Echinocereus other" (pure species rank, no subspecies)
_CATALOG = {
    "G1": {"taxon_key": "G1", "rank": "GENUS"},
    "S1": {"taxon_key": "S1", "rank": "SPECIES"},
    "T1": {"taxon_key": "T1", "rank": "SUBSPECIES"},
    "T2": {"taxon_key": "T2", "rank": "SUBSPECIES"},
    "S2": {"taxon_key": "S2", "rank": "SPECIES"},
}
_ANCESTORS = {
    "G1": {"GENUS": "G1"},
    "S1": {"GENUS": "G1", "SPECIES": "S1"},
    "T1": {"GENUS": "G1", "SPECIES": "S1"},
    "T2": {"GENUS": "G1", "SPECIES": "S1"},
    "S2": {"GENUS": "G1", "SPECIES": "S2"},
}

# Wasatch-like dense cluster center vs. an Oquirrh-like isolated point ~3
# degrees of longitude away (~334km at any latitude in Web Mercator, since X
# is linear in longitude — comfortably past the 200km coarsest band cell).
_CLUSTER_LAT, _CLUSTER_LON = 40.5, -111.6
_ISOLATED_LAT, _ISOLATED_LON = 40.5, -114.6


def _jitter_rows(prefix: str, taxon_key: str, n: int, lat: float, lon: float) -> list[dict]:
    return [
        {
            "catalogNumber": f"{prefix}{i}",
            "decimalLatitude": lat + (i % 5) * 0.001,
            "decimalLongitude": lon + (i // 5) * 0.001,
            "taxon_key": taxon_key,
        }
        for i in range(n)
    ]


def _make_parquet(path: Path, rows: list[dict]) -> None:
    """Writes physically sorted by taxon_key — matching the real
    occurrences.parquet's maintained invariant (see main()'s reliance on it
    to skip an explicit sort), which _rows() below doesn't uphold on its own."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["taxon_key"])
    cols: dict[str, list] = {}
    for row in rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}), path)


def _rows() -> list[dict]:
    return [
        *_jitter_rows("t1-cluster-", "T1", 10, _CLUSTER_LAT, _CLUSTER_LON),
        {"catalogNumber": "t1-isolated-0", "decimalLatitude": _ISOLATED_LAT,
         "decimalLongitude": _ISOLATED_LON, "taxon_key": "T1"},
        *_jitter_rows("t2-cluster-", "T2", 5, _CLUSTER_LAT, _CLUSTER_LON),
        *_jitter_rows("s2-", "S2", 3, _CLUSTER_LAT + 5.0, _CLUSTER_LON + 5.0),
    ]


def _run_main(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    occ_path = tmp_path / "occurrences.parquet"
    _make_parquet(occ_path, rows)
    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS):
        obs_ranks.main()
    return pq.read_table(occ_path).to_pandas()


# ---------------------------------------------------------------------------
# Coverage guarantee
# ---------------------------------------------------------------------------

def test_isolated_point_gets_earliest_band(tmp_path):
    result = _run_main(tmp_path, _rows())
    isolated = result.loc[result["catalogNumber"] == "t1-isolated-0"].iloc[0]
    assert isolated["minZoomInfra"] == obs_ranks._BANDS[0][0]


def test_dense_cluster_spreads_across_bands(tmp_path):
    """Not every point in a 10-point same-cell cluster should get the same
    minZoom — some must be deferred to finer bands, or thinning isn't
    happening at all."""
    result = _run_main(tmp_path, _rows())
    cluster = result.loc[result["catalogNumber"].str.startswith("t1-cluster-")]
    assert cluster["minZoomInfra"].nunique() > 1


def test_exactly_one_cluster_representative_at_coarsest_band(tmp_path):
    result = _run_main(tmp_path, _rows())
    cluster = result.loc[result["catalogNumber"].str.startswith("t1-cluster-")]
    at_coarsest = cluster.loc[cluster["minZoomInfra"] == obs_ranks._BANDS[0][0]]
    assert len(at_coarsest) == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_across_reruns(tmp_path):
    rows = _rows()
    first = _run_main(tmp_path / "run1", rows)
    second = _run_main(tmp_path / "run2", rows)
    first_sorted = first.sort_values("catalogNumber").reset_index(drop=True)
    second_sorted = second.sort_values("catalogNumber").reset_index(drop=True)
    pd.testing.assert_series_equal(first_sorted["minZoomInfra"], second_sorted["minZoomInfra"])
    pd.testing.assert_series_equal(first_sorted["minZoomSpecies"], second_sorted["minZoomSpecies"])


# ---------------------------------------------------------------------------
# INFRA vs. SPECIES grouping
# ---------------------------------------------------------------------------

def test_sibling_subspecies_are_separate_infra_groups(tmp_path):
    """T1's dense cluster and T2's smaller cluster share the same real
    location, but different taxon_key — T2 must get its own coarsest-band
    representative independent of T1's, not be crowded out by it."""
    result = _run_main(tmp_path, _rows())
    t2 = result.loc[result["catalogNumber"].str.startswith("t2-cluster-")]
    assert (t2["minZoomInfra"] == obs_ranks._BANDS[0][0]).sum() == 1


def test_species_column_pools_sibling_subspecies(tmp_path):
    """T1's and T2's clusters share a location and the same SPECIES ancestor
    (S1) — minZoomSpecies should treat their combined cluster points as one
    pool, i.e. only one representative across both at the coarsest band.
    (The isolated T1 point is excluded here: it's a separate cell even
    within the pooled SPECIES group, so it legitimately earns its own
    representative too — that's covered by test_isolated_point_gets_earliest_band.)
    """
    result = _run_main(tmp_path, _rows())
    combined = result.loc[result["catalogNumber"].str.startswith(("t1-cluster-", "t2-cluster-"))]
    at_coarsest = combined.loc[combined["minZoomSpecies"] == obs_ranks._BANDS[0][0]]
    assert len(at_coarsest) == 1


def test_species_rank_row_has_null_infra_populated_species(tmp_path):
    result = _run_main(tmp_path, _rows())
    s2 = result.loc[result["catalogNumber"].str.startswith("s2-")]
    assert s2["minZoomInfra"].isna().all()
    assert s2["minZoomSpecies"].notna().all()


def test_original_columns_preserved(tmp_path):
    result = _run_main(tmp_path, _rows())
    for col in ("catalogNumber", "decimalLatitude", "decimalLongitude", "taxon_key"):
        assert col in result.columns

def test_output_is_taxon_key_sorted_even_from_unsorted_source(tmp_path):
    """The final write does an explicit ORDER BY taxon_key (see main()), so
    the output is correctly sorted regardless of source order — unlike an
    earlier version of this script, which relied on the source already being
    sorted and skipped the sort for performance. That version's safety
    check (still present, now effectively a regression guard rather than
    something normal input can trigger) is exercised implicitly here: it
    must pass even when fed deliberately unsorted input."""
    occ_path = tmp_path / "occurrences.parquet"
    unsorted_rows = [
        {"catalogNumber": "a", "decimalLatitude": 40.5, "decimalLongitude": -111.6, "taxon_key": "T2"},
        {"catalogNumber": "b", "decimalLatitude": 40.5, "decimalLongitude": -111.6, "taxon_key": "S2"},
    ]
    occ_path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {}
    for row in unsorted_rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}), occ_path)

    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS):
        obs_ranks.main()

    result = pq.read_table(occ_path).to_pandas()
    assert list(result["taxon_key"]) == sorted(result["taxon_key"])
