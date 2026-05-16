from dataclasses import dataclass
from typing import Any

_REGISTRY: dict[str, type] = {}
_CACHE: dict[str, Any] = {}


def register_config(name: str):
    def decorator(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls
    return decorator


def load_config(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown config '{name}'")
    if name not in _CACHE:
        _CACHE[name] = _REGISTRY[name]()
    return _CACHE[name]


@dataclass
@register_config("global")
class GlobalConfig:
    plantae_key: int = 6
    leaf_ranks: tuple[str, ...] = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")
    subspecies_equivalents: tuple[str, ...] = ("SUBSPECIES", "VARIETY", "FORM")
    species_rank: str = "SPECIES"
    do_write_dirs: bool = False  # writes taxonomy folder tree to data/taxonomy/tree/

    @property
    def leaf_rank_set(self) -> frozenset[str]:
        return frozenset(self.leaf_ranks)
