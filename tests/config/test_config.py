from dataclasses import dataclass

import pytest

from config.config import GlobalConfig, clear_config_cache, load_config, register_config


def test_load_config_global(monkeypatch):
    monkeypatch.delenv("PLANTAE_KEY", raising=False)
    clear_config_cache()
    cfg = load_config("global")
    assert isinstance(cfg, GlobalConfig)
    assert cfg.plantae_key == 6
    assert cfg.species_rank == "SPECIES"
    assert "SUBSPECIES" in cfg.leaf_rank_set
    assert "SPECIES" in cfg.leaf_ranks


def test_env_override(monkeypatch):
    monkeypatch.setenv("PLANTAE_KEY", "2519")
    clear_config_cache()
    assert load_config("global").plantae_key == 2519


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
