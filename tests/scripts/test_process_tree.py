# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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