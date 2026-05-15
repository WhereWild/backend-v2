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
