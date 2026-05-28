"""Tests for scripts/carry_forward.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.carry_forward as cf

_TEMPORAL_IDS = frozenset(["temperature_2m", "precipitation"])

_BASE_ROW = {
    "decimalLatitude": 35.0,
    "decimalLongitude": -112.0,
    "catalogNumber": "cat001",
    "hilbertIdx": 12345,
    "eventTimestamp": 1700000000,
    "coordinateUncertaintyInMeters": 50.0,
    "obscured": "No",
    "gbifRegion": "NORTH_AMERICA",
    "level0Gid": "USA",
    "level1Gid": "USA.4_1",
    "level2Gid": None,
    "dp": "",
    "vitality": "",
    "rcs": "",
}


def _make_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {}
    for row in rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    arrays = {k: pa.array(v) for k, v in cols.items()}
    pq.write_table(pa.table(arrays), path)


# ---------------------------------------------------------------------------
# _carry_one
# ---------------------------------------------------------------------------

def test_carry_one_unchanged_copies_all(tmp_path):
    """Unchanged observation → all enrichment cols copied."""
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {k: v for k, v in _BASE_ROW.items()}  # base cols only

    old_path = tmp_path / "old" / "occ.parquet"
    new_path = tmp_path / "new" / "occ.parquet"
    _make_parquet(old_path, [old_row])
    _make_parquet(new_path, [new_row])

    n_carried, n_changed, n_new_obs, n_total = cf._carry_one(new_path, old_path, _TEMPORAL_IDS)

    assert n_carried == 1
    assert n_changed == 0
    assert n_new_obs == 0
    assert n_total == 1
    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)
    assert result.at[0, "temperature_2m_avg_24h"] == pytest.approx(20.5)


def test_carry_one_coords_changed_copies_nothing(tmp_path):
    """Coords changed → no enrichment cols copied, counted as changed."""
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {**_BASE_ROW, "decimalLatitude": 36.0, "decimalLongitude": -113.0}

    old_path = tmp_path / "old" / "occ.parquet"
    new_path = tmp_path / "new" / "occ.parquet"
    _make_parquet(old_path, [old_row])
    _make_parquet(new_path, [new_row])

    n_carried, n_changed, n_new_obs, _ = cf._carry_one(new_path, old_path, _TEMPORAL_IDS)

    assert n_carried == 0
    assert n_changed == 1
    assert n_new_obs == 0
    result = pq.read_table(new_path).to_pandas()
    assert "elevation" not in result.columns
    assert "temperature_2m_avg_24h" not in result.columns


def test_carry_one_timestamp_changed_copies_tree_only(tmp_path):
    """Timestamp changed only → tree (GIS) cols copied, temporal cols not."""
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {**_BASE_ROW, "eventTimestamp": 1800000000}  # different timestamp

    old_path = tmp_path / "old" / "occ.parquet"
    new_path = tmp_path / "new" / "occ.parquet"
    _make_parquet(old_path, [old_row])
    _make_parquet(new_path, [new_row])

    n_carried, n_changed, _, _ = cf._carry_one(new_path, old_path, _TEMPORAL_IDS)

    assert n_carried == 1
    assert n_changed == 0
    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)
    assert np.isnan(result.at[0, "temperature_2m_avg_24h"])


def test_carry_one_new_observation_not_copied(tmp_path):
    """New catalogNumber → not in old → counted as new_obs."""
    old_row = {**_BASE_ROW, "elevation": 1500.0}
    new_row = {**_BASE_ROW, "catalogNumber": "cat_new"}

    old_path = tmp_path / "old" / "occ.parquet"
    new_path = tmp_path / "new" / "occ.parquet"
    _make_parquet(old_path, [old_row])
    _make_parquet(new_path, [new_row])

    n_carried, n_changed, n_new_obs, n_total = cf._carry_one(new_path, old_path, _TEMPORAL_IDS)

    assert n_carried == 0
    assert n_changed == 0
    assert n_new_obs == 1
    assert n_total == 1


def test_carry_one_no_enrichment_in_old(tmp_path):
    """Old parquet has no enrichment cols → nothing to copy → no-op."""
    _make_parquet(tmp_path / "old.parquet", [_BASE_ROW])
    _make_parquet(tmp_path / "new.parquet", [_BASE_ROW])

    n_carried, n_changed, n_new_obs, _ = cf._carry_one(
        tmp_path / "new.parquet", tmp_path / "old.parquet", _TEMPORAL_IDS
    )
    assert n_carried == 0
    assert n_new_obs == 1  # old has no enrich cols → early return treats all as new


def test_carry_one_empty_parquets(tmp_path):
    """Empty parquets → returns (0, 0, 0, 0) without error."""
    for fname in ("old.parquet", "new.parquet"):
        pq.write_table(pa.table({"catalogNumber": pa.array([], pa.string()),
                                  "decimalLatitude": pa.array([], pa.float64())}), tmp_path / fname)

    n_carried, n_changed, n_new_obs, n_total = cf._carry_one(
        tmp_path / "new.parquet", tmp_path / "old.parquet", _TEMPORAL_IDS
    )
    assert n_carried == 0
    assert n_total == 0


def test_carry_one_mixed_rows(tmp_path):
    """Multiple rows: some unchanged, one with changed coords, one new."""
    base = _BASE_ROW
    old_rows = [
        {**base, "catalogNumber": "cat001", "elevation": 100.0, "temperature_2m_avg_24h": 10.0},
        {**base, "catalogNumber": "cat002", "decimalLatitude": 40.0, "elevation": 200.0, "temperature_2m_avg_24h": 20.0},
    ]
    new_rows = [
        {**base, "catalogNumber": "cat001"},                                    # unchanged
        {**base, "catalogNumber": "cat002", "decimalLatitude": 41.0},          # coords changed
        {**base, "catalogNumber": "cat003"},                                    # new
    ]
    old_path = tmp_path / "old.parquet"
    new_path = tmp_path / "new.parquet"
    _make_parquet(old_path, old_rows)
    _make_parquet(new_path, new_rows)

    n_carried, n_changed, n_new_obs, n_total = cf._carry_one(new_path, old_path, _TEMPORAL_IDS)

    assert n_total == 3
    assert n_carried == 1   # cat001: unchanged
    assert n_changed == 1   # cat002: coords differ
    assert n_new_obs == 1   # cat003: not in old

    result = pq.read_table(new_path).to_pandas().set_index("catalogNumber")
    assert result.at["cat001", "elevation"] == pytest.approx(100.0)
    assert result.at["cat001", "temperature_2m_avg_24h"] == pytest.approx(10.0)
    assert np.isnan(result.at["cat002", "elevation"])
    assert np.isnan(result.at["cat003", "elevation"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_no_old_tree(tmp_path, capsys):
    """No old tree → no-op."""
    with patch.object(cf, "OLD_TREE_PATH", tmp_path / "nonexistent"):
        cf.main()
    out = capsys.readouterr().out
    assert "first run" in out


def test_main_carries_forward(tmp_path):
    """main() matches old parquets at same path and copies enrichment."""
    old_tree = tmp_path / "old_tree"
    new_tree = tmp_path / "new_tree"

    rel = Path("cactaceae") / "opuntia" / "occurrence.parquet"

    old_row = {**_BASE_ROW, "elevation": 999.0}
    new_row = {k: v for k, v in _BASE_ROW.items()}

    _make_parquet(old_tree / rel, [old_row])
    _make_parquet(new_tree / rel, [new_row])

    with (
        patch.object(cf, "OLD_TREE_PATH", old_tree),
        patch.object(cf, "TREE_ROOT", new_tree),
        patch.object(cf, "_load_temporal_ids", return_value=_TEMPORAL_IDS),
    ):
        cf.main()

    result = pq.read_table(new_tree / rel).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(999.0)
    assert not old_tree.exists()  # cleaned up


def test_main_skips_taxa_not_in_old_tree(tmp_path):
    """Taxa with no old parquet are left untouched (new taxon)."""
    old_tree = tmp_path / "old_tree"
    new_tree = tmp_path / "new_tree"

    rel = Path("cactaceae") / "opuntia_new_species" / "occurrence.parquet"
    new_row = {**_BASE_ROW}
    _make_parquet(new_tree / rel, [new_row])
    old_tree.mkdir(parents=True)  # exists but has no parquet at this path

    with (
        patch.object(cf, "OLD_TREE_PATH", old_tree),
        patch.object(cf, "TREE_ROOT", new_tree),
        patch.object(cf, "_load_temporal_ids", return_value=_TEMPORAL_IDS),
    ):
        cf.main()

    result = pq.read_table(new_tree / rel).to_pandas()
    assert "elevation" not in result.columns  # nothing added
