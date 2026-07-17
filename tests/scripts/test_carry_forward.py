# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for scripts/carry_forward.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.carry_forward as cf

_STATIC_IDS   = frozenset(["elevation", "bio1", "kg2"])
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
    "taxon_key": "2923970",
}


def _make_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {}
    for row in rows:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    arrays = {k: pa.array(v) for k, v in cols.items()}
    pq.write_table(pa.table(arrays), path)


def _run_main(tmp_path: Path, old_rows: list[dict] | None, new_rows: list[dict]) -> Path:
    old_path = tmp_path / "old_occurrences.parquet"
    new_path = tmp_path / "occurrences.parquet"
    if old_rows is not None:
        _make_parquet(old_path, old_rows)
    _make_parquet(new_path, new_rows)
    with patch.object(cf, "OLD_OCCURRENCES_PATH", old_path), \
         patch.object(cf, "OCCURRENCES_FILE", new_path), \
         patch.object(cf, "SYNC_STATE_PATH", tmp_path / "sync_state.json"), \
         patch.object(cf, "_load_catalog_ids", return_value=(_STATIC_IDS, _TEMPORAL_IDS)):
        cf.main()
    return new_path


# ---------------------------------------------------------------------------
# main — copy-rule scenarios
# ---------------------------------------------------------------------------

def test_main_unchanged_copies_all(tmp_path, capsys):
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {k: v for k, v in _BASE_ROW.items()}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)
    assert result.at[0, "temperature_2m_avg_24h"] == pytest.approx(20.5)
    out = capsys.readouterr().out
    assert "1/1 rows carried forward" in out


def test_main_coords_changed_copies_nothing(tmp_path):
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {**_BASE_ROW, "decimalLatitude": 36.0, "decimalLongitude": -113.0}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert "elevation" not in result.columns
    assert "temperature_2m_avg_24h" not in result.columns


def test_main_timestamp_changed_copies_tree_only(tmp_path):
    old_row = {**_BASE_ROW, "elevation": 1500.0, "temperature_2m_avg_24h": 20.5}
    new_row = {**_BASE_ROW, "eventTimestamp": 1800000000}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)
    assert np.isnan(result.at[0, "temperature_2m_avg_24h"])


def test_main_new_observation_not_copied(tmp_path, capsys):
    old_row = {**_BASE_ROW, "elevation": 1500.0}
    new_row = {**_BASE_ROW, "catalogNumber": "cat_new"}

    _run_main(tmp_path, [old_row], [new_row])

    out = capsys.readouterr().out
    assert "1 new" in out
    assert "0.0%" in out  # nothing carried


def test_main_no_enrichment_in_old(tmp_path, capsys):
    _run_main(tmp_path, [_BASE_ROW], [_BASE_ROW])
    out = capsys.readouterr().out
    assert "0/1 rows carried forward" in out
    assert "1 new" in out  # no enrich cols → early return treats all as new


def test_main_empty_parquets(tmp_path, capsys):
    empty = {"catalogNumber": pa.array([], pa.string()), "decimalLatitude": pa.array([], pa.float64())}
    old_path = tmp_path / "old_occurrences.parquet"
    new_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table(empty), old_path)
    pq.write_table(pa.table(empty), new_path)
    with patch.object(cf, "OLD_OCCURRENCES_PATH", old_path), \
         patch.object(cf, "OCCURRENCES_FILE", new_path), \
         patch.object(cf, "SYNC_STATE_PATH", tmp_path / "sync_state.json"), \
         patch.object(cf, "_load_catalog_ids", return_value=(_STATIC_IDS, _TEMPORAL_IDS)):
        cf.main()
    out = capsys.readouterr().out
    assert "0/0 rows carried forward" in out


def test_main_mixed_rows(tmp_path):
    old_rows = [
        {**_BASE_ROW, "catalogNumber": "cat001", "elevation": 100.0, "temperature_2m_avg_24h": 10.0},
        {**_BASE_ROW, "catalogNumber": "cat002", "decimalLatitude": 40.0, "elevation": 200.0, "temperature_2m_avg_24h": 20.0},
    ]
    new_rows = [
        {**_BASE_ROW, "catalogNumber": "cat001"},                              # unchanged
        {**_BASE_ROW, "catalogNumber": "cat002", "decimalLatitude": 41.0},     # coords changed
        {**_BASE_ROW, "catalogNumber": "cat003"},                              # new
    ]
    new_path = _run_main(tmp_path, old_rows, new_rows)

    result = pq.read_table(new_path).to_pandas().set_index("catalogNumber")
    assert result.at["cat001", "elevation"] == pytest.approx(100.0)
    assert result.at["cat001", "temperature_2m_avg_24h"] == pytest.approx(10.0)
    assert np.isnan(result.at["cat002", "elevation"])
    assert np.isnan(result.at["cat003", "elevation"])


def test_main_reidentified_observation_still_matches(tmp_path):
    """A catalogNumber whose taxon_key changed between runs still matches
    (global catalogNumber matching, unlike the old per-path-only matching)."""
    old_row = {**_BASE_ROW, "elevation": 1500.0, "taxon_key": "111"}
    new_row = {**_BASE_ROW, "taxon_key": "222"}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)
    assert result.at[0, "taxon_key"] == "222"  # new taxon assignment kept


# ---------------------------------------------------------------------------
# main — no-op / cleanup
# ---------------------------------------------------------------------------

def test_main_no_old_occurrences(tmp_path, capsys):
    with patch.object(cf, "OLD_OCCURRENCES_PATH", tmp_path / "nonexistent.parquet"):
        cf.main()
    out = capsys.readouterr().out
    assert "first run" in out


def test_main_no_new_occurrences(tmp_path, capsys):
    old_path = tmp_path / "old_occurrences.parquet"
    _make_parquet(old_path, [_BASE_ROW])
    with patch.object(cf, "OLD_OCCURRENCES_PATH", old_path), \
         patch.object(cf, "OCCURRENCES_FILE", tmp_path / "nonexistent.parquet"):
        cf.main()
    out = capsys.readouterr().out
    assert "nothing to carry into" in out


def test_main_cleans_up_old_file(tmp_path):
    old_row = {**_BASE_ROW, "elevation": 1500.0}
    new_row = {k: v for k, v in _BASE_ROW.items()}
    old_path = tmp_path / "old_occurrences.parquet"
    _run_main(tmp_path, [old_row], [new_row])
    assert not old_path.exists()


def test_main_writes_sync_state_stats(tmp_path):
    old_row = {**_BASE_ROW, "elevation": 1500.0}
    new_row = {k: v for k, v in _BASE_ROW.items()}
    _run_main(tmp_path, [old_row], [new_row])
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert state["carry_forward"]["carried"] == 1
    assert state["carry_forward"]["total_rows"] == 1


# ---------------------------------------------------------------------------
# Catalog-awareness: removed variables are dropped, still-current ones kept
# ---------------------------------------------------------------------------

def test_removed_static_not_carried(tmp_path):
    old_row = {**_BASE_ROW, "elevation": 1500.0, "old_removed_var": 42.0}
    new_row = {k: v for k, v in _BASE_ROW.items()}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "elevation"] == pytest.approx(1500.0)  # in catalog → carried
    assert "old_removed_var" not in result.columns              # removed → dropped


def test_temporal_cols_all_carried(tmp_path):
    old_row = {
        **_BASE_ROW,
        "temperature_2m_avg_24h": 10.0,
        "temperature_2m_max_168h": 25.0,
        "precipitation_sum_24h": 3.5,
    }
    new_row = {k: v for k, v in _BASE_ROW.items()}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "temperature_2m_avg_24h"]  == pytest.approx(10.0)
    assert result.at[0, "temperature_2m_max_168h"] == pytest.approx(25.0)
    assert result.at[0, "precipitation_sum_24h"]   == pytest.approx(3.5)


def test_removed_temporal_not_carried(tmp_path):
    old_row = {
        **_BASE_ROW,
        "temperature_2m_avg_24h": 10.0,   # still in catalog
        "wind_speed_avg_24h": 5.0,         # removed temporal variable
    }
    new_row = {k: v for k, v in _BASE_ROW.items()}

    new_path = _run_main(tmp_path, [old_row], [new_row])

    result = pq.read_table(new_path).to_pandas()
    assert result.at[0, "temperature_2m_avg_24h"] == pytest.approx(10.0)
    assert "wind_speed_avg_24h" not in result.columns
