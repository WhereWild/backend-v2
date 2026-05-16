from dataclasses import dataclass

import pytest

from config.config import GlobalConfig, load_config, register_config


def test_load_config_global():
    cfg = load_config("global")
    assert isinstance(cfg, GlobalConfig)
    assert cfg.plantae_key == 6
    assert cfg.species_rank == "SPECIES"
    assert "SUBSPECIES" in cfg.leaf_rank_set
    assert cfg.do_write_dirs is True


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
