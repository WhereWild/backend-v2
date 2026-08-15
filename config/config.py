# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, get_type_hints


class ValueType(StrEnum):
    RATIO = "ratio"
    INTERVAL = "interval"
    ORDINAL = "ordinal"
    CIRCULAR = "circular"
    NOMINAL = "nominal"


_CONTINUOUS_METRICS: tuple[str, ...] = (
    "count", "unique_samples", "min",
    "10th_percentile", "25th_percentile", "median",
    "75th_percentile", "90th_percentile", "max",
    "mean", "std", "variance", "iqr", "10_90_range", "range", "entropy", "mode",
)

_NOMINAL_METRICS: tuple[str, ...] = (
    "unique_samples", "total_samples", "unique_classes", "entropy", "mode",
)

_ORDINAL_METRICS: tuple[str, ...] = (
    "count", "unique_samples", "total_samples", "unique_classes", "entropy", "mode",
    "10th_percentile", "25th_percentile", "median", "75th_percentile", "90th_percentile",
)

METRICS_BY_TYPE: dict[ValueType, tuple[str, ...]] = {
    ValueType.RATIO:     _CONTINUOUS_METRICS,
    ValueType.INTERVAL:  _CONTINUOUS_METRICS,
    ValueType.NOMINAL:   _NOMINAL_METRICS,
    ValueType.ORDINAL:   _ORDINAL_METRICS,
    ValueType.CIRCULAR:  ("count", "unique_samples", "circular_mean", "rbar", "circular_var", "circular_std", "entropy", "mode"),
}

# GIS layers where the raster nodata value means the property is absent (= 0),
# not that the data is missing. E.g. scd=nodata at the equator → 0 snow cover days.
#
# The single source of truth for which layers get this treatment — the
# actual fill (burn 0 into nodata pixels, clear the nodata flag) happens
# once, in scripts/gis/build_overviews.py, before overviews are built from
# the corrected data. That covers every consumer (map tiles, the /gis/point
# background-point endpoint, and enrich_tree.py's per-observation sampling)
# from one place, rather than each caller needing its own nodata-aware
# special-casing (enrich_tree.py's is a harmless no-op once a layer's
# nodata has actually been cleared, since ds.nodata reads back as None).
ZERO_NODATA_LAYERS: frozenset[str] = frozenset({
    "swe", "scd", "fcf",
    "gdd0", "gdd5", "gdd10",
    "scdur", "scsl", "sfsl", "sper", "ssper", "sreg",
})

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
    # Root taxa the pipeline treats as independent trees — no combining "Life"
    # node exists in the catalog above these, so each root's stats/rankings
    # (see scripts/process_tree.py) stop at its own node and never aggregate
    # into any other root's. Override via TAXONOMY_ROOTS (comma-separated
    # COL XR ids, e.g. "7HS,CXQ" for a small dev subset) — see __post_init__,
    # this can't go through the generic scalar-field env loader below since
    # it's a tuple. Adding a new root (e.g. Animalia) is purely a config/env
    # change; every call site iterates this instead of naming a kingdom.
    taxonomy_roots: tuple[str, ...] = ("P","F")
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

    # Phenology filter values (must match rcs column in occurrence.parquet, pipe-separated)
    phenology_values: tuple[str, ...] = (
        "flower buds", "flowers", "fruits or seeds", "no flowers or fruits"
    )

    # Taxonomy / occurrence
    occurrence_parquet_filename: str = "occurrences.parquet"
    data_root: str = "data"

    # Temporal enrichment
    # ERA5 starts 1940-01-01; earliest obs needs a full 90-day (2160h) lookback,
    # so the first enrichable date is 1940-01-01 + 90d = 1940-04-01.
    temporal_min_date: str = "1940-04-01"
    temporal_worklist_batch_rows: int = 100_000
    temporal_cache_dir: str = "data/cache/temporal"
    temporal_overwrite_all: bool = False
    temporal_elevation_correctable_vars: tuple[str, ...] = (
        "temperature_2m", "dew_point_2m", "soil_temperature_0_to_7cm"
    )

    # Temporal raster builder
    temporal_raster_out_dir: str = "data/gis/temporal/rasters"
    temporal_raster_vars: str = ""        # CSV, empty = all
    temporal_raster_windows: str = ""     # CSV window labels (e.g. "24h,7d"), empty = all
    temporal_raster_force_rebuild: bool = False

    @property
    def gis_root(self) -> Path:
        return Path(self.data_root) / "gis"

    def __post_init__(self):
        raw_roots = os.environ.get("TAXONOMY_ROOTS")
        if raw_roots is not None:
            self.taxonomy_roots = tuple(
                part.strip() for part in raw_roots.split(",") if part.strip()
            )

        hints = get_type_hints(self.__class__)
        for f in fields(self):
            if hints.get(f.name) not in _SCALAR_TYPES:
                continue
            val = os.environ.get(f.name.upper())
            if val is None and f.name == "data_root":
                val = os.environ.get("WHEREWILD_DATA_ROOT")
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
