"""Central configuration for pipeline scripts.

Edit the constants below to tune behavior locally without modifying each script.

ONLY constants that are present in more than one file should be present here to reduce clutter.
This file aims to provide an area to define constants used across files to ensure no inconsistencies arise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Type


_REGISTRY: Dict[str, Type] = {}
_CONFIG_CACHE: Dict[str, Any] = {}


def register_config(name: str):
    """Decorator that records dataclasses for auto-loading."""

    def decorator(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return decorator


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_env_path(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


@dataclass
@register_config("global")
class GlobalConfig:
    """Shared knobs consumed by multiple scripts."""

    # Filenames and templates
    taxa_csv_filename: str = "taxa.csv"
    vernacular_filename: str = "VernacularName.tsv"
    occurrence_filename: str = "occurrence.txt"
    occurrence_parquet_filename: str = "occurrence.parquet"
    taxon_catalog_filename: str = "taxon_catalog.pkl"
    catalog_json_filename: str = "catalog.json"
    gadm_gpkg_filename: str = "gadm.gpkg"
    location_hierarchy_filename: str = "hierarchy.csv"
    gbif_regions_filename: str = "gbif_regions.csv"
    location_catalog_filename: str = "location_taxa.parquet"

    # GBIF occurrence download files (for media extraction)
    gbif_occurrence_txt: str = "occurrence.txt"
    gbif_multimedia_txt: str = "multimedia.txt"
    gbif_taxon_lookup_filename: str = "gbif_taxon_lookup.txt"
    taxon_media_filename: str = "taxon_media.pkl"
    # Set to None for full dataset, or a number to limit rows (for testing/sampling)

    # Base root
    project_root: Path = field(default_factory=_project_root)

    # Pipeline tuning
    root_taxon_id: str = "1"
    process_tree_ranks_only: bool = False
    do_write_dirs: bool = False

    # Taxonomy
    leaf_ranks: tuple[str, ...] = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")
    species_rank: str = "SPECIES"
    subspecies_equivalents: tuple[str, ...] = ("SUBSPECIES", "VARIETY", "FORM")
    _rank_synonyms: dict[str, frozenset[str]] = field(
        default_factory=lambda: {
            "SPECIES": frozenset(("SPECIES", "SP", "SPP", "SPECIESGROUP")),
            "SUBSPECIES": frozenset(
                ("SUBSPECIES", "SUBSPECIE", "SUBSP", "SSP", "SUBSP.", "SSP.")
            ),
            "VARIETY": frozenset(("VARIETY", "VAR", "VAR.", "VARIETAS")),
            "FORM": frozenset(("FORM", "FORMA", "F.", "FOR.")),
        }
    )

    # Occurrence ingestion + schema
    annotation_columns: tuple[str, ...] = (
        "dp",
        "sex",
        "lifeStage",
        "rcs",
        "vitality",
        "gall",
    )
    occurrence_base_columns: tuple[str, ...] = (
        "decimalLatitude",
        "decimalLongitude",
        "catalogNumber",
        "tileId",
        "eventTimestamp",
        "coordinateUncertaintyInMeters",
        "obscured",
        "gbifRegion",
        "level0Gid",
        "level1Gid",
        "level2Gid",
    )
    occurrence_list_columns: tuple[str, ...] = ("dp", "rcs", "gall")

    # API + aggregation behavior

    # Enrichment

    # GIS + locations
    gbif_regions: tuple[str, ...] = (
        "NORTH_AMERICA",
        "ASIA",
        "LATIN_AMERICA",
        "OCEANIA",
        "EUROPE",
        "AFRICA",
        "ANTARCTICA",
    )
    location_levels: tuple[int, ...] = (0, 1, 2)
    location_level_columns: dict[int, str] = field(
        default_factory=lambda: {
            0: "level0Gid",
            1: "level1Gid",
            2: "level2Gid",
        }
    )

    # Temporal weather enrichment
    temporal_cache_dirname: str = "temporal_cache"
    inat_mapping_offline_filename: str = "inat_gbif_mapping.csv"
    inat_mapping_api_filename: str = "inat_gbif_mapping_api.csv"
    inat_mapping_obs_filename: str = "inat_gbif_mapping_obs.csv"
    inat_preferred_common_name_locale: str = "en"
    inat_preferred_common_name_batch_size: int = 200
    inat_preferred_common_name_request_limit: int = 0
    inat_preferred_common_name_progress_every: int = 50
    inat_preferred_common_name_overwrite: bool = False
    inat_preferred_common_name_rate_limit_per_second: float = 1.0
    inat_preferred_common_name_max_requests: int = 10_000
    common_name_language: str = "en"
    location_scope_by_level: dict[int, str] = field(
        default_factory=lambda: {
            0: "gadm_level0",
            1: "gadm_level1",
            2: "gadm_level2",
        }
    )

    # API server

    # Docs

    # Derived values (properties)
    @property
    def data_root(self) -> Path:
        return _resolve_env_path("WHEREWILD_DATA_ROOT", self.project_root / "data")

    @property
    def species_dir(self) -> Path:
        return _resolve_env_path("SPECIES_DIR", self.data_root / "species")

    @property
    def taxonomy_root(self) -> Path:
        return self.species_dir / "taxonomy"

    @property
    def gis_root(self) -> Path:
        return self.data_root / "gis"

    @property
    def gis_catalog_path(self) -> Path:
        return _resolve_env_path(
            "GIS_CATALOG_PATH",
            self.gis_root / self.catalog_json_filename,
        )

    @property
    def gis_legends_root(self) -> Path:
        return self.gis_root / "legends"

    @property
    def gis_locations_root(self) -> Path:
        return self.gis_root / "locations"

    @property
    def gis_regions_root(self) -> Path:
        return self.gis_root / "regions"

    @property
    def gis_landcover_root(self) -> Path:
        return self.gis_root / "landcover"

    @property
    def parquet_root(self) -> Path:
        return self.data_root / "parquet"

    @property
    def taxon_catalog_path(self) -> Path:
        return _resolve_env_path(
            "TAXON_CATALOG_PATH",
            self.taxonomy_root / self.taxon_catalog_filename,
        )

    @property
    def taxa_csv_path(self) -> Path:
        return self.species_dir / self.taxa_csv_filename
    @property
    def vernacular_tsv_path(self) -> Path:
        return self.data_root / self.vernacular_filename

    @property
    def occurrence_path(self) -> Path:
        return self.species_dir / self.occurrence_filename

    @property
    def gbif_occurrence_path(self) -> Path:
        return self.species_dir / self.gbif_occurrence_txt

    @property
    def gbif_multimedia_path(self) -> Path:
        return self.species_dir / self.gbif_multimedia_txt

    @property
    def gbif_taxon_lookup_path(self) -> Path:
        return self.taxonomy_root / self.gbif_taxon_lookup_filename

    @property
    def taxon_media_path(self) -> Path:
        return self.taxonomy_root / self.taxon_media_filename

    @property
    def inat_mapping_offline_path(self) -> Path:
        return self.taxonomy_root / self.inat_mapping_offline_filename

    @property
    def inat_mapping_api_path(self) -> Path:
        return self.taxonomy_root / self.inat_mapping_api_filename

    @property
    def inat_mapping_obs_path(self) -> Path:
        return self.taxonomy_root / self.inat_mapping_obs_filename

    @property
    def location_hierarchy_path(self) -> Path:
        return self.gis_locations_root / self.location_hierarchy_filename

    @property
    def gbif_regions_path(self) -> Path:
        return self.gis_locations_root / self.gbif_regions_filename

    @property
    def location_catalog_path(self) -> Path:
        return self.gis_locations_root / self.location_catalog_filename

    @property
    def gadm_gpkg_path(self) -> Path:
        return self.gis_root / self.gadm_gpkg_filename

    @property
    def bioclim_root(self) -> Path:
        return self.data_root / "bioclim"

    @property
    def temporal_cache_root(self) -> Path:
        return _resolve_env_path(
            "TEMPORAL_CACHE_ROOT",
            self.data_root / self.temporal_cache_dirname,
        )

    @property
    def leaf_rank_set(self) -> frozenset[str]:
        return frozenset(self.leaf_ranks)

    @property
    def gbif_region_set(self) -> frozenset[str]:
        return frozenset(self.gbif_regions)

    @property
    def rank_synonyms(self) -> dict[str, frozenset[str]]:
        return {
            key: frozenset(value.upper() for value in values)
            for key, values in self._rank_synonyms.items()
        }

    @property
    def location_columns(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.location_level_columns[level], self.location_scope_by_level[level])
            for level in self.location_levels
        )

    @property
    def occurrence_all_columns(self) -> tuple[str, ...]:
        return self.occurrence_base_columns + self.annotation_columns

    @property
    def occurrence_list_column_indices(self) -> dict[str, int]:
        return {
            col: self.occurrence_all_columns.index(col)
            for col in self.occurrence_list_columns
        }


def load_config(name: str) -> Any:
    """Return a cached config object by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown config '{name}'")
    if name not in _CONFIG_CACHE:
        _CONFIG_CACHE[name] = _REGISTRY[name]()
    return _CONFIG_CACHE[name]
