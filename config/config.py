import os
from dataclasses import dataclass, fields
from typing import Any, get_type_hints

_REGISTRY: dict[str, type] = {}
_CACHE: dict[str, Any] = {}

_SCALAR_TYPES = (int, str, float)


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


def clear_config_cache() -> None:
    _CACHE.clear()


@dataclass
@register_config("global")
class GlobalConfig:
    plantae_key: int = 6
    leaf_ranks: tuple[str, ...] = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")
    subspecies_equivalents: tuple[str, ...] = ("SUBSPECIES", "VARIETY", "FORM")
    species_rank: str = "SPECIES"

    def __post_init__(self):
        hints = get_type_hints(self.__class__)
        for f in fields(self):
            if hints.get(f.name) not in _SCALAR_TYPES:
                continue
            val = os.environ.get(f.name.upper())
            if val is not None:
                setattr(self, f.name, hints[f.name](val))

    @property
    def leaf_rank_set(self) -> frozenset[str]:
        return frozenset(self.leaf_ranks)
