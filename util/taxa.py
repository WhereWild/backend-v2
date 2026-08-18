# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pickle
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from rapidfuzz import fuzz, process

from config.config import load_config
from util.storage import ParquetStorageProxy

CONFIG = load_config("global")

CATALOG_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "catalog"

_storage = ParquetStorageProxy(
    data_root=Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")),
    project_root=Path(__file__).parent.parent,
)


class TaxonRecord(TypedDict):
    taxon_key: str
    path: str
    scientific_name: str
    common_name: str
    rank: str


def normalize_name(value: str) -> str:
    if not value:
        return ""
    return " ".join(value.replace("_", " ").lower().split())


def format_common_name(value: str) -> str:
    """Title-case a common name, preserving short all-caps acronyms (e.g. 'NW', 'USA')."""
    if not value:
        return ""
    words = []
    for word in value.split(" "):
        if len(word) <= 4 and word.isupper():
            words.append(word)
        elif "'" in word:
            parts = word.lower().split("'", 1)
            first = (parts[0][0].upper() + parts[0][1:]) if parts[0] else ""
            words.append(f"{first}'{parts[1]}" if parts[1] else first)
        else:
            parts = word.lower().split("-")
            words.append("-".join((p[0].upper() + p[1:]) if p else p for p in parts))
    return " ".join(words).strip()


def taxon_slug(value: str | None) -> str:
    normalized = normalize_name(value or "")
    if not normalized:
        return ""
    return "-".join(normalized.split())


@lru_cache(maxsize=1)
def _load_payload() -> dict:
    with _storage.open_input_file(CATALOG_DIR / "taxon_catalog.pkl") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, TaxonRecord]:
    return {str(k): v for k, v in _load_payload()["catalog"].items()}


@lru_cache(maxsize=1)
def load_name_index() -> dict[str, list[str]]:
    return {
        str(k): [str(key) for key in v]
        for k, v in _load_payload()["combined_name_index"].items()
    }


@lru_cache(maxsize=1)
def _slug_index() -> dict[str, str]:
    index: dict[str, list[str]] = {}
    for taxon_key, taxon in load_catalog().items():
        slug = taxon_slug(taxon.get("scientific_name", ""))
        if slug:
            index.setdefault(slug, []).append(taxon_key)
    # Discard ambiguous slugs (multiple taxa share a scientific name)
    return {slug: keys[0] for slug, keys in index.items() if len(keys) == 1}


@lru_cache(maxsize=1)
def _path_index() -> dict[str, str]:
    """Map taxon path → taxon_key (built once from catalog)."""
    return {taxon["path"]: key for key, taxon in load_catalog().items()}


@lru_cache(maxsize=1)
def _children_index() -> dict[str, list[str]]:
    """Map taxon_key → list of direct-child taxon_keys."""
    path_to_key = _path_index()
    index: dict[str, list[str]] = {}
    for key, taxon in load_catalog().items():
        path = taxon["path"]
        if "/" not in path:
            continue
        parent_path = path.rsplit("/", 1)[0]
        parent_key = path_to_key.get(parent_path)
        if parent_key:
            index.setdefault(parent_key, []).append(key)
    return index


# Ranks that can own their own display/map view — matches build_tree.py's
# TAXONOMY_LEVELS, uppercased to match TaxonRecord["rank"]'s casing.
ANCESTOR_RANK_LEVELS: tuple[str, ...] = (
    "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES",
)

# Every level a minZoom* column exists for: the 7 ancestor ranks above, plus
# INFRA (subspecies/variety/form — see display_level_for_rank). Single
# source of truth for the "minZoom" + label column-naming convention, shared
# by scripts/observation_ranks.py (which writes these columns) and main.py
# (which reads them) so the two don't duplicate the naming scheme.
DISPLAY_LEVELS: tuple[str, ...] = (*ANCESTOR_RANK_LEVELS, "INFRA")

LEVEL_LABELS: dict[str, str] = {rank: rank.capitalize() for rank in ANCESTOR_RANK_LEVELS}
LEVEL_LABELS["INFRA"] = "Infra"


def zoom_column(level: str) -> str:
    """minZoom column name for a display level, e.g. "GENUS" -> "minZoomGenus"."""
    return f"minZoom{LEVEL_LABELS[level]}"


def display_level_for_rank(rank: str) -> str:
    """Which DISPLAY_LEVELS entry (and therefore which minZoom column) a
    taxon of this rank uses for its own map view. Infraspecific ranks
    (config.subspecies_equivalents) get the separate INFRA column rather
    than being treated as SPECIES — see ancestor_keys_by_rank's docstring
    and scripts/observation_ranks.py for why."""
    if rank in CONFIG.subspecies_equivalents:
        return "INFRA"
    return rank


@lru_cache(maxsize=1)
def ancestor_keys_by_rank() -> dict[str, dict[str, str]]:
    """taxon_key -> {rank: ancestor_taxon_key} for every rank in
    ANCESTOR_RANK_LEVELS present in that taxon's own lineage.

    Walks each taxon's own `path` (a "/"-joined chain of
    "{name}_{taxon_key}" segments, one per ancestor level — see
    scripts/build_tree.py's TAXONOMY_LEVELS loop) from itself upward,
    looking up each prefix's taxon via _path_index() and recording it under
    its own `rank`. Driven by each node's actual `rank` field rather than
    assuming positional correspondence between path segments and
    TAXONOMY_LEVELS, since some lineages skip a rank (a GBIF record can lack
    e.g. an "order" designation) — a taxon simply has no entry for a rank
    missing from its lineage, which callers should treat as legitimately
    absent, not an error.

    A rank-N node's own entry includes itself as its own rank-N ancestor
    (e.g. a GENUS node's dict has "GENUS" -> its own key) — the natural
    result of walking prefixes starting at the full path.

    Catalog-sized (not occurrence-sized), cached: safe to call per-observation.
    """
    catalog = load_catalog()
    path_to_key = _path_index()
    result: dict[str, dict[str, str]] = {}
    for key, taxon in catalog.items():
        segments = taxon["path"].split("/")
        ancestors: dict[str, str] = {}
        for i in range(len(segments), 0, -1):
            prefix = "/".join(segments[:i])
            anc_key = path_to_key.get(prefix)
            if anc_key is None:
                continue
            anc_rank = catalog[anc_key]["rank"]
            if anc_rank in ANCESTOR_RANK_LEVELS and anc_rank not in ancestors:
                ancestors[anc_rank] = anc_key
        result[key] = ancestors
    return result


def get_children(taxon_key: Any) -> list[TaxonRecord]:
    """Return the direct children of a taxon in catalog order."""
    catalog = load_catalog()
    return [catalog[k] for k in _children_index().get(str(taxon_key), []) if k in catalog]


def iter_descendants(taxon: TaxonRecord, *, include_self: bool = True) -> Iterable[TaxonRecord]:
    """DFS over a taxon and all its descendants."""
    if include_self:
        yield taxon
    stack = get_children(taxon["taxon_key"])
    while stack:
        child = stack.pop()
        yield child
        stack.extend(get_children(child["taxon_key"]))


def get_taxon_by_id(taxon_id: Any) -> TaxonRecord | None:
    key = str(taxon_id).strip() if taxon_id is not None else ""
    if not key:
        return None
    return load_catalog().get(key)


def get_taxon_by_slug(slug: str) -> TaxonRecord | None:
    normalized = taxon_slug(slug)
    if not normalized:
        return None
    key = _slug_index().get(normalized)
    return get_taxon_by_id(key) if key else None


@lru_cache(maxsize=1)
def _inat_id_index() -> dict[str, str]:
    """Map iNaturalist taxon id -> our taxon_key (first-seen wins on collision)."""
    index: dict[str, str] = {}
    for taxon_key, taxon in load_catalog().items():
        inat_id = str(taxon.get("inat_id") or "").strip()
        if inat_id and inat_id not in index:
            index[inat_id] = taxon_key
    return index


def get_taxon_by_inat_id(inat_id: Any) -> TaxonRecord | None:
    """Resolve our own taxon from an iNaturalist taxon id (build_tree.py's inat_id field)."""
    key = str(inat_id).strip() if inat_id is not None else ""
    if not key:
        return None
    mapped = _inat_id_index().get(key)
    return get_taxon_by_id(mapped) if mapped else None


def reload_catalog() -> None:
    """Clear all catalog caches so the next request re-reads from disk."""
    _load_payload.cache_clear()
    load_catalog.cache_clear()
    load_name_index.cache_clear()
    _slug_index.cache_clear()
    _path_index.cache_clear()
    _children_index.cache_clear()
    _inat_id_index.cache_clear()


def search_taxa_by_name(
    name_query: str,
    limit: int = 10,
) -> list[tuple[TaxonRecord, float, str]]:
    normalized_query = normalize_name(name_query)
    tokens = normalized_query.split()
    if not tokens:
        return []

    name_index = load_name_index()
    matches = process.extract(
        normalized_query,
        name_index.keys(),
        scorer=fuzz.token_set_ratio,
        limit=max(limit * 25, 100),
    )

    best_by_taxon: dict[str, tuple[TaxonRecord, float, str]] = {}
    for name, score, _ in matches:
        adjusted = _adjust_score(name, normalized_query, tokens, float(score))
        if adjusted is None:
            continue
        for key in name_index.get(name, []):
            taxon = get_taxon_by_id(key)
            if taxon is None:
                continue
            existing = best_by_taxon.get(key)
            if existing is None or adjusted > existing[1]:
                best_by_taxon[key] = (taxon, adjusted, _display_name_for_match(taxon, name))

    results = sorted(best_by_taxon.values(), key=lambda x: x[1], reverse=True)
    return results[:limit]


def _display_name_for_match(taxon: TaxonRecord, matched_key: str) -> str:
    """Return the display name for a search match.

    If the matched index key is a vernacular name, return it (the frontend
    formats casing). If it's a scientific or synonym name, return the taxon's
    preferred common name instead so synonym searches don't show the old name.
    """
    vernacular_keys: set[str] = set()
    for field in ("common_name", "inat_preferred_common_name"):
        v = normalize_name(str(taxon.get(field) or ""))
        if v:
            vernacular_keys.add(v)
    for vn in (taxon.get("vernacular_names") or []):
        k = normalize_name(str(vn))
        if k:
            vernacular_keys.add(k)
    if matched_key in vernacular_keys:
        return matched_key
    # Scientific or synonym name — substitute preferred common name
    preferred = (
        normalize_name(str(taxon.get("inat_preferred_common_name") or ""))
        or normalize_name(str(taxon.get("common_name") or ""))
    )
    return preferred if preferred else matched_key


def _adjust_score(
    normalized_name: str,
    normalized_query: str,
    query_tokens: list[str],
    raw_score: float,
) -> float | None:
    name_tokens = normalized_name.split()
    if len(query_tokens) > 1:
        if not all(
            any(nt.startswith(qt) for nt in name_tokens) for qt in query_tokens
        ):
            return None
    else:
        if not any(nt.startswith(query_tokens[0]) for nt in name_tokens):
            return None

    score = raw_score
    if normalized_name == normalized_query:
        score += 20.0
    score -= float(max(0, len(name_tokens) - len(query_tokens)) * 2)
    min_score = 60.0 if len(query_tokens) > 1 else 70.0
    return score if score >= min_score else None
