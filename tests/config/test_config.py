# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass

import pytest

from config.config import GlobalConfig, clear_config_cache, load_config, register_config


def test_load_config_global(monkeypatch):
    monkeypatch.delenv("TAXONOMY_ROOTS", raising=False)
    clear_config_cache()
    cfg = load_config("global")
    assert isinstance(cfg, GlobalConfig)
    assert cfg.taxonomy_roots == ("P","F")
    assert cfg.species_rank == "SPECIES"
    assert "SUBSPECIES" in cfg.leaf_rank_set
    assert "SPECIES" in cfg.leaf_ranks


def test_env_override(monkeypatch):
    monkeypatch.setenv("TAXONOMY_ROOTS", "7HS")
    clear_config_cache()
    assert load_config("global").taxonomy_roots == ("7HS",)


def test_env_override_multiple_roots(monkeypatch):
    monkeypatch.setenv("TAXONOMY_ROOTS", "7HS,CXQ")
    clear_config_cache()
    assert load_config("global").taxonomy_roots == ("7HS", "CXQ")


def test_env_override_roots_strips_whitespace_and_drops_empty(monkeypatch):
    monkeypatch.setenv("TAXONOMY_ROOTS", " 7HS ,, CXQ ,")
    clear_config_cache()
    assert load_config("global").taxonomy_roots == ("7HS", "CXQ")


def test_load_config_cached():
    assert load_config("global") is load_config("global")


def test_load_config_unknown():
    with pytest.raises(KeyError, match="unknown"):
        load_config("unknown")


def test_register_config():
    @dataclass
    @register_config("_test")
    class _TestConfig:
        x: int = 99

    assert load_config("_test").x == 99
