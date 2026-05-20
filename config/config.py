import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any, get_type_hints


class ValueType(StrEnum):
    RATIO = "ratio"
    INTERVAL = "interval"
    ORDINAL = "ordinal"
    CIRCULAR = "circular"
    AGGREGATE = "aggregate"
    NOMINAL = "nominal"


_CONTINUOUS_METRICS: tuple[str, ...] = (
    "count", "unique_samples", "min",
    "10th_percentile", "25th_percentile", "median",
    "75th_percentile", "90th_percentile", "max",
    "mean", "std", "iqr", "10_90_range", "range", "mode",
)

_NOMINAL_METRICS: tuple[str, ...] = (
    "unique_samples", "total_samples", "unique_classes", "entropy", "mode",
)

METRICS_BY_TYPE: dict[ValueType, tuple[str, ...]] = {
    ValueType.RATIO:     _CONTINUOUS_METRICS,
    ValueType.INTERVAL:  _CONTINUOUS_METRICS,
    ValueType.NOMINAL:   _NOMINAL_METRICS,
    ValueType.ORDINAL:   (),
    ValueType.CIRCULAR:  (),
    ValueType.AGGREGATE: (),
}

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

    # Location / GADM
    gbif_regions: tuple[str, ...] = (
        "AFRICA", "ANTARCTICA", "ASIA", "EUROPE",
        "LATIN_AMERICA", "NORTH_AMERICA", "OCEANIA",
    )
    location_levels: tuple[int, ...] = (0, 1, 2)
    location_level_columns: dict[int, str] = field(
        default_factory=lambda: {0: "level0Gid", 1: "level1Gid", 2: "level2Gid"}
    )
    location_scope_by_level: dict[int, str] = field(
        default_factory=lambda: {0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}
    )

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

    @property
    def location_columns(self) -> tuple[tuple[str, str], ...]:
        """Return ((column_name, scope_name), ...) pairs for each location level."""
        return tuple(
            (self.location_level_columns[lvl], self.location_scope_by_level[lvl])
            for lvl in self.location_levels
        )
