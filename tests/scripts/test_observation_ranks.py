# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for scripts/observation_ranks.py."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import scripts.observation_ranks as obs_ranks
from util.gis import hilbert_index

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


def _row(catalog_number: str, taxon_key: str, lat: float, lon: float) -> dict:
    """Includes mediaUrl/mediaAttribution/mediaLicense and hilbertIdx —
    always-present columns in the real occurrences.parquet schema (see
    scripts/populate_tree.py's SCHEMA) that the spatial-index phase in
    main() selects unconditionally, so fixtures need them too even though
    these particular tests don't exercise media/hilbert content directly."""
    return {
        "catalogNumber": catalog_number,
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "taxon_key": taxon_key,
        "mediaUrl": None,
        "mediaAttribution": None,
        "mediaLicense": None,
        "hilbertIdx": hilbert_index(lat, lon),
    }


def _jitter_rows(prefix: str, taxon_key: str, n: int, lat: float, lon: float) -> list[dict]:
    return [
        _row(f"{prefix}{i}", taxon_key, lat + (i % 5) * 0.001, lon + (i // 5) * 0.001)
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
        _row("t1-isolated-0", "T1", _ISOLATED_LAT, _ISOLATED_LON),
        *_jitter_rows("t2-cluster-", "T2", 5, _CLUSTER_LAT, _CLUSTER_LON),
        *_jitter_rows("s2-", "S2", 3, _CLUSTER_LAT + 5.0, _CLUSTER_LON + 5.0),
    ]


def _run_main(tmp_path: Path, rows: list[dict], max_representatives: int = 1) -> pd.DataFrame:
    """max_representatives defaults to 1 (applied uniformly across every
    band), not the real production ramp (_MAX_REPRESENTATIVES_BY_BAND, 1 to
    50 coarsest-to-finest) — most of these tests are about the underlying
    coverage/determinism/grouping guarantees the grid+hash mechanism
    provides, which read most clearly at "exactly one representative wins"
    the same way they did before that became configurable. See
    test_multiple_representatives_promoted_when_cluster_exceeds_cap and
    test_representatives_vary_independently_per_band for dedicated coverage
    of the K>1 and per-band-K behavior itself."""
    occ_path = tmp_path / "occurrences.parquet"
    _make_parquet(occ_path, rows)
    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "OCCURRENCES_BY_HILBERT_FILE", tmp_path / "occurrences_by_hilbert.parquet"), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS), \
         patch.object(
             obs_ranks, "_MAX_REPRESENTATIVES_BY_BAND",
             (max_representatives,) * len(obs_ranks._BANDS),
         ):
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


def test_multiple_representatives_promoted_when_cluster_exceeds_cap(tmp_path):
    """With the per-band cap raised above 1, a same-cell cluster larger than
    the cap should still get exactly that many representatives at the
    coarsest band (not just one, not the whole cluster) — the point of
    raising it above 1 is denser-looking coverage in genuinely dense areas,
    capped so it doesn't just show everyone."""
    result = _run_main(tmp_path, _rows(), max_representatives=3)
    cluster = result.loc[result["catalogNumber"].str.startswith("t1-cluster-")]
    at_coarsest = cluster.loc[cluster["minZoomInfra"] == obs_ranks._BANDS[0][0]]
    assert len(at_coarsest) == 3
    # The rest of the 10-point cluster still exists and is still deferred
    # to a finer band — raising the cap isn't the same as disabling
    # thinning entirely.
    assert cluster["minZoomInfra"].nunique() > 1


def test_representatives_vary_independently_per_band(tmp_path):
    """_MAX_REPRESENTATIVES_BY_BAND is one value per band, not one global
    constant — a coarse band (small cap, since one map tile spans many of
    its large cells) and a fine band (large cap, since a tile only
    overlaps a handful of its small cells) must each respect their OWN
    value independently. Regression coverage for the real production bug:
    a single constant (50) was confirmed live to blow up to 500k+ rows in
    one coarse-zoom tile, because it applied the SAME generous cap to
    bands where a tile can contain thousands of cells, not just the fine
    bands where that's actually safe.

    Uses a dedicated zero-jitter fixture (every point at the identical
    coordinate) rather than the shared _rows() cluster — with real jitter,
    points can resolve into different cells as cell size shrinks band to
    band, making per-band counts depend on the exact jitter pattern rather
    than cleanly isolating the cap itself. Zero jitter means every point
    stays in the SAME cell at every band regardless of cell size, so the
    cap is the only thing determining how many get promoted at each one."""
    rows = [_row(f"t1-same-{i}", "T1", _CLUSTER_LAT, _CLUSTER_LON) for i in range(10)]
    occ_path = tmp_path / "occurrences.parquet"
    _make_parquet(occ_path, rows)
    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "OCCURRENCES_BY_HILBERT_FILE", tmp_path / "occurrences_by_hilbert.parquet"), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS), \
         patch.object(obs_ranks, "_MAX_REPRESENTATIVES_BY_BAND", (1, 1, 1, 1, 1, 8)):
        obs_ranks.main()
    result = pq.read_table(occ_path).to_pandas()
    cluster = result.loc[result["catalogNumber"].str.startswith("t1-same-")]
    # Every point stays in the same cell at every band (zero jitter), so
    # with caps (1,1,1,1,1,8): band 1 promotes 1 (9 left), bands 2-5 each
    # promote 1 more (5 left), and band 6's cap of 8 comfortably covers
    # what's left — 5, not 8, since only 5 actually remain by then. That
    # last part is exactly the point: the fine band's generous cap doesn't
    # force it to promote MORE than what's actually left just because it
    # can.
    at_coarsest = cluster.loc[cluster["minZoomInfra"] == obs_ranks._BANDS[0][0]]
    assert len(at_coarsest) == 1
    at_finest = cluster.loc[cluster["minZoomInfra"] == obs_ranks._BANDS[-1][0]]
    assert len(at_finest) == 5
    assert len(cluster) == 10


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


def test_rerunning_against_already_ranked_data_overwrites_not_shadows(tmp_path):
    """Regression test: main() reads OCCURRENCES_FILE as both input (for
    coordinates) and, after a first run, as a file that ALREADY carries
    minZoom* columns from that run. The join-back step used to do
    `SELECT o.*, r.minZoomKingdom` — since `o` (the file being re-read) now
    has a column with that exact name too, DuckDB silently renamed the new
    one to `minZoomKingdom_1` instead of erroring, so a second run with
    DIFFERENT _BANDS left the live-served minZoomKingdom column completely
    unchanged (still the FIRST run's value) while the real second-run
    output sat unused under a suffixed column name. Confirmed live in
    production: 5 stacked generations (minZoomKingdom through _4) had
    accumulated with the originally-published value still being served the
    entire time. A second run must actually overwrite the real column and
    must not leave any _N-suffixed duplicate behind."""
    rows = _rows()
    occ_path = tmp_path / "occurrences.parquet"
    _make_parquet(occ_path, rows)

    def _run_with_bands(bands):
        with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
             patch.object(obs_ranks, "OCCURRENCES_BY_HILBERT_FILE", tmp_path / "occurrences_by_hilbert.parquet"), \
             patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
             patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
             patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
             patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS), \
             patch.object(obs_ranks, "_MAX_REPRESENTATIVES_BY_BAND", (1,) * len(bands)), \
             patch.object(obs_ranks, "_BANDS", bands):
            obs_ranks.main()
        return pq.read_table(occ_path).to_pandas()

    first_bands = ((0, 20_000.0), (2, 5_000.0), (4, 1_000.0), (6, 250.0), (8, 25.0), (10, 1.0))
    second_bands = ((0, 20_000.0), (3, 5_000.0), (6, 1_000.0), (9, 250.0), (12, 25.0), (15, 1.0))
    _run_with_bands(first_bands)
    second_result = _run_with_bands(second_bands)

    # No shadow columns left over from either run.
    shadow_cols = [c for c in second_result.columns if re.match(r"^minZoom\w+_\d+$", c)]
    assert shadow_cols == []

    # The real column reflects the SECOND run's bands, not the first's —
    # every value must be one of second_bands' zoom values.
    second_zoom_values = {zoom for zoom, _ in second_bands}
    actual_values = set(second_result["minZoomInfra"].dropna().unique().tolist())
    assert actual_values <= second_zoom_values
    # And genuinely not the first run's values (distinguishable since the
    # two band sets barely overlap beyond zoom=0).
    assert not actual_values <= {zoom for zoom, _ in first_bands}

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
        _row("a", "T2", 40.5, -111.6),
        _row("b", "S2", 40.5, -111.6),
    ]
    occ_path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {}
    for row in unsorted_rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}), occ_path)

    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "OCCURRENCES_BY_HILBERT_FILE", tmp_path / "occurrences_by_hilbert.parquet"), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS):
        obs_ranks.main()

    result = pq.read_table(occ_path).to_pandas()
    assert list(result["taxon_key"]) == sorted(result["taxon_key"])

def test_spatial_index_is_slim_and_hilbert_sorted(tmp_path):
    """occurrences_by_hilbert.parquet should be a narrow projection (not a
    full-width copy — see main()'s comment on why), physically sorted by
    hilbertIdx, carrying the minZoom* columns the main file also has."""
    occ_path = tmp_path / "occurrences.parquet"
    spatial_path = tmp_path / "occurrences_by_hilbert.parquet"
    _make_parquet(occ_path, _rows())

    with patch.object(obs_ranks, "OCCURRENCES_FILE", occ_path), \
         patch.object(obs_ranks, "OCCURRENCES_BY_HILBERT_FILE", spatial_path), \
         patch.object(obs_ranks, "_DUCKDB_SPILL_DIR", tmp_path / "duckdb_spill"), \
         patch.object(obs_ranks, "_DUCKDB_SCRATCH_DB", tmp_path / "scratch.duckdb"), \
         patch.object(obs_ranks, "load_catalog", return_value=_CATALOG), \
         patch.object(obs_ranks, "ancestor_keys_by_rank", return_value=_ANCESTORS):
        obs_ranks.main()

    result = pq.read_table(spatial_path).to_pandas()
    expected_cols = {
        "catalogNumber", "taxon_key", "decimalLatitude", "decimalLongitude",
        "mediaUrl", "mediaAttribution", "mediaLicense", "hilbertIdx",
        "minZoomKingdom", "minZoomPhylum", "minZoomClass", "minZoomOrder",
        "minZoomFamily", "minZoomGenus", "minZoomSpecies", "minZoomInfra",
    }
    assert set(result.columns) == expected_cols
    assert len(result) == len(_rows())
    assert list(result["hilbertIdx"]) == sorted(result["hilbertIdx"])
