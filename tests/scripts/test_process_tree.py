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


# ---------------------------------------------------------------------------
# _setup — multi-root independence
#
# Two configured roots must be treated as fully separate trees: there's no
# "Life" node above kingdom in the catalog, so _setup() just unions each
# root's own subtree. These tests lock in that no taxon from one root's
# subtree ever leaks into the other's, and that a missing root still raises
# (generalizing the single-root test_main_root_not_found behavior above).
# ---------------------------------------------------------------------------

_FAKE_PLANTAE_ROOT = {
    "taxon_key": "P", "path": "Plantae_P", "scientific_name": "Plantae",
    "common_name": "", "rank": "KINGDOM",
}
_FAKE_PLANTAE_CHILD = {
    "taxon_key": "G1", "path": "Plantae_P/GenusA_G1", "scientific_name": "GenusA",
    "common_name": "", "rank": "GENUS",
}
_FAKE_FUNGI_ROOT = {
    "taxon_key": "F", "path": "Fungi_F", "scientific_name": "Fungi",
    "common_name": "", "rank": "KINGDOM",
}
_FAKE_FUNGI_CHILD = {
    "taxon_key": "G2", "path": "Fungi_F/GenusB_G2", "scientific_name": "GenusB",
    "common_name": "", "rank": "GENUS",
}

_FAKE_TWO_ROOT_CATALOG = {
    "P": _FAKE_PLANTAE_ROOT,
    "G1": _FAKE_PLANTAE_CHILD,
    "F": _FAKE_FUNGI_ROOT,
    "G2": _FAKE_FUNGI_CHILD,
}


def _fake_iter_descendants(root, *, include_self=True):
    subtree = {
        "P": [_FAKE_PLANTAE_ROOT, _FAKE_PLANTAE_CHILD],
        "F": [_FAKE_FUNGI_ROOT, _FAKE_FUNGI_CHILD],
    }[root["taxon_key"]]
    return subtree if include_self else subtree[1:]


def test_setup_unions_two_roots(monkeypatch):
    monkeypatch.setattr(pt.CONFIG, "taxonomy_roots", ("P", "F"))
    monkeypatch.setattr(
        "scripts.process_tree.get_taxon_by_id", lambda tid: _FAKE_TWO_ROOT_CATALOG.get(tid)
    )
    monkeypatch.setattr("scripts.process_tree.iter_descendants", _fake_iter_descendants)
    _, _, by_depth, _, _, total = pt._setup()

    assert total == 4
    taxon_keys_by_depth = {
        depth: {t["taxon_key"] for t in taxa} for depth, taxa in by_depth.items()
    }
    assert taxon_keys_by_depth[0] == {"P", "F"}
    assert taxon_keys_by_depth[1] == {"G1", "G2"}
    # No taxon from one root's subtree ever ends up misattributed to the
    # other — every taxon_key present is exactly one of the four fixture nodes.
    all_keys = {t["taxon_key"] for taxa in by_depth.values() for t in taxa}
    assert all_keys == {"P", "G1", "F", "G2"}


def test_setup_raises_if_any_configured_root_missing(monkeypatch):
    monkeypatch.setattr(pt.CONFIG, "taxonomy_roots", ("MISSING", "P"))
    monkeypatch.setattr(
        "scripts.process_tree.get_taxon_by_id", lambda tid: _FAKE_TWO_ROOT_CATALOG.get(tid)
    )
    with pytest.raises(RuntimeError, match="MISSING"):
        pt._setup()