from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "processed"
SPECIES_STATS_DIR = PROCESSED_DIR / "species" / "stats"
LEADERBOARD_DIR = SPECIES_STATS_DIR / "leaderboard"
GIS_CATALOG_PATH = REPO_ROOT / "gis_catalog.json"


class VariableNotFoundError(KeyError):
    """Raised when a requested GIS variable is missing from the catalog."""


class SpeciesStatsNotFoundError(FileNotFoundError):
    """Raised when the precomputed stats file is missing for a species/variable."""


@lru_cache()
def _load_gis_catalog() -> Dict[str, Dict[str, Any]]:
    if not GIS_CATALOG_PATH.exists():
        msg = f"GIS catalog not found at {GIS_CATALOG_PATH}"
        raise FileNotFoundError(msg)
    with GIS_CATALOG_PATH.open() as fp:
        entries = json.load(fp)
    return {entry["id"]: entry for entry in entries}


def get_variable_definition(variable_id: str) -> dict[str, Any]:
    catalog = _load_gis_catalog()
    try:
        return catalog[variable_id]
    except KeyError as exc:
        raise VariableNotFoundError(variable_id) from exc


def list_variables() -> list[dict[str, Any]]:
    return list(_load_gis_catalog().values())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return json.load(fp)


def get_species_stats(species_id: int, variable_id: str) -> dict[str, Any]:
    _ = get_variable_definition(variable_id)
    stats_path = SPECIES_STATS_DIR / variable_id / f"{species_id}.json"
    if not stats_path.exists():
        raise SpeciesStatsNotFoundError(
            f"Stats not found for species {species_id} and variable '{variable_id}'"
        )
    return _read_json(stats_path)


def get_variable_leaderboard(variable_id: str) -> dict[str, Any]:
    _ = get_variable_definition(variable_id)
    leaderboard_path = LEADERBOARD_DIR / f"{variable_id}_leaderboard.json"
    if not leaderboard_path.exists():
        raise SpeciesStatsNotFoundError(
            f"Leaderboard not found for variable '{variable_id}'"
        )
    return _read_json(leaderboard_path)
