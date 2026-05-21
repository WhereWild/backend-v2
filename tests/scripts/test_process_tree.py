from unittest.mock import patch

import pytest

import scripts.process_tree as pt

# Capture before autouse fixture patches it.
_real_load_layers = pt._load_layers

_FAKE_LAYERS = [
    {"id": "bio1", "value_type": "ratio", "scale_factor": 0.1, "add_offset": -273.15},
]

_FAKE_TAXON = {
    "taxon_key": "6",
    "path": "Plantae_6",
    "scientific_name": "Plantae",
    "common_name": "",
    "rank": "KINGDOM",
}


@pytest.fixture(autouse=True)
def patch_load_layers(monkeypatch):
    monkeypatch.setattr(pt, "_load_layers", lambda: _FAKE_LAYERS)


def test_load_layers(monkeypatch):
    monkeypatch.setattr(pt, "_load_layers", _real_load_layers)
    monkeypatch.setattr("scripts.process_tree.load_layers", lambda: _FAKE_LAYERS)
    assert pt._load_layers() == _FAKE_LAYERS


def test_main_root_not_found(capsys, monkeypatch):
    with patch("scripts.process_tree.get_taxon_by_id", return_value=None):
        pt.main()
    assert "not found" in capsys.readouterr().out


def test_main_runs_compute_for_all(capsys, monkeypatch):
    with patch("scripts.process_tree.get_taxon_by_id", return_value=_FAKE_TAXON), \
         patch("scripts.process_tree.iter_descendants", return_value=[_FAKE_TAXON]), \
         patch("scripts.process_tree.compute_taxon_stats") as mock_stats, \
         patch("scripts.process_tree.compute_relative_ranks") as mock_ranks:
        pt.main()
    mock_stats.assert_called_once_with(_FAKE_TAXON, _FAKE_LAYERS)
    mock_ranks.assert_called_once_with(_FAKE_TAXON, _FAKE_LAYERS)
    out = capsys.readouterr().out
    assert "taxa" in out
    assert "done" in out


def test_main_logs_failed_stats_node(capsys, monkeypatch):
    with patch("scripts.process_tree.get_taxon_by_id", return_value=_FAKE_TAXON), \
         patch("scripts.process_tree.iter_descendants", return_value=[_FAKE_TAXON]), \
         patch("scripts.process_tree.compute_taxon_stats", side_effect=RuntimeError("boom")), \
         patch("scripts.process_tree.compute_relative_ranks"):
        pt.main()
    out = capsys.readouterr().out
    assert "failed" in out
    assert "boom" in out


def test_main_logs_failed_rankings_node(capsys, monkeypatch):
    with patch("scripts.process_tree.get_taxon_by_id", return_value=_FAKE_TAXON), \
         patch("scripts.process_tree.iter_descendants", return_value=[_FAKE_TAXON]), \
         patch("scripts.process_tree.compute_taxon_stats"), \
         patch("scripts.process_tree.compute_relative_ranks", side_effect=RuntimeError("rank-fail")):
        pt.main()
    out = capsys.readouterr().out
    assert "failed" in out
    assert "rank-fail" in out


def test_main_prints_progress_at_1000(capsys, monkeypatch):
    taxa = [dict(_FAKE_TAXON, taxon_key=str(i), path=f"Plantae_6/T_{i}") for i in range(1001)]
    with patch("scripts.process_tree.get_taxon_by_id", return_value=_FAKE_TAXON), \
         patch("scripts.process_tree.iter_descendants", return_value=taxa), \
         patch("scripts.process_tree.compute_taxon_stats"), \
         patch("scripts.process_tree.compute_relative_ranks"):
        pt.main()
    out = capsys.readouterr().out
    assert "1000/" in out
