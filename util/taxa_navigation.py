"""
This file functions as a library, providing functions that perform standard operations one might use when traversing the taxonomy tree.
It aims to accomplish more generic things on the tree and its parquets that other areas of the code can use.
"""

import pickle
from functools import lru_cache
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process

from util.config import load_config
from util.request_cancellation import CancelCheck
from util.storage import ParquetStorageProxy
from util import gis_lookup

CONFIG = load_config("global")
PARQUET = ParquetStorageProxy(CONFIG.data_root, CONFIG.project_root)
LOGGER = logging.getLogger("uvicorn.error")
PC = cast(Any, pc)

STANDARD_DESCENDANT_RANKS: tuple[str, ...] = (
    "KINGDOM",
    "PHYLUM",
    "CLASS",
    "ORDER",
    "FAMILY",
    "GENUS",
    "SPECIES",
    "SUBSPECIES",
)

base_occurrence_columns = frozenset(
    {
        "catalogNumber",
        "decimalLatitude",
        "decimalLongitude",
        "obscured",
        "coordinateUncertaintyInMeters",
        "level0Gid",
        "level1Gid",
        "level2Gid",
        "gbifRegion",
    }
)

combined_parquet_filename = "combined.parquet"


class TaxonRecord(TypedDict):
    taxon_key: str
    path: Path | str
    scientific_name: str
    common_name: object
    rank: str


def _normalize_taxon_path(value: Any) -> Path:
    raw = value if isinstance(value, Path) else Path(str(value))
    if raw.is_absolute():
        parts = raw.parts
        if "taxonomy" in parts:
            idx = parts.index("taxonomy")
            rel = Path(*parts[idx + 1 :])
            return CONFIG.taxonomy_root / rel
        return raw
    return CONFIG.taxonomy_root / raw


def normalize_taxon_path(value: Any) -> Path:
    """Normalize a serialized taxon path to an absolute taxonomy path."""
    return _normalize_taxon_path(value)


def normalize_name(value: str) -> str:
    """Normalizes a taxon name to match name index keys."""
    if not value:
        return ""
    return " ".join(value.replace("_", " ").lower().split())


def taxon_slug(value: str | None) -> str:
    """Build the canonical API slug for a taxon scientific name."""
    normalized = normalize_name(value or "")
    if not normalized:
        return ""
    return "-".join(part for part in normalized.split(" ") if part)


def get_parent_taxon(taxon: TaxonRecord) -> TaxonRecord | None:
    """Returns the parent taxon record when available."""
    raw_path = taxon.get("path")
    if not raw_path:
        return None
    path = _normalize_taxon_path(raw_path)
    if path == CONFIG.taxonomy_root:
        return None
    parent = path.parent
    if parent == CONFIG.taxonomy_root:
        return None
    parent_key = taxon_key_from_path(parent)
    return get_taxon_by_id(parent_key)


def _common_name_score(language: str, lexicon: str, source: str) -> int:
    lang = (language or "").strip().lower()
    lex = (lexicon or "").strip().lower()
    src = (source or "").strip().lower()
    score = 0
    if lang == "en":
        score += 100
    if "english" in lex:
        score += 50
    if "american" in lex or "united states" in lex or "u.s." in lex or "usa" in lex:
        score += 10
    if src == "inat":
        score += 25
    return score


def _matches_language(language: str, lexicon: str, target_language: str) -> bool:
    lang = (language or "").strip().lower()
    lex = (lexicon or "").strip().lower()
    target = (target_language or "").strip().lower()
    if not target:
        return True
    if lang == target:
        return True
    if target == "en" and "english" in lex:
        return True
    return False


def _format_common_name(value: str) -> str:
    """Title-case common names while preserving short acronyms."""
    if not value:
        return ""
    words = []
    for word in value.split(" "):
        if len(word) <= 4 and word.isupper():
            words.append(word)
        else:
            lower = word.lower()
            if "'" in lower:
                parts = lower.split("'", 1)
                if parts[0]:
                    first = parts[0][0].upper() + parts[0][1:]
                else:
                    first = ""
                second = parts[1]
                words.append(f"{first}'{second}" if second else first)
            else:
                words.append(lower[:1].upper() + lower[1:])
    return " ".join(words).strip()


def _extract_common_names(taxon: TaxonRecord, language: str | None) -> list[str]:
    raw_common_name = taxon.get("common_name")
    preferred_name = ""
    if language and str(language).lower().startswith("en"):
        preferred_name = str(taxon.get("inat_preferred_common_name") or "").strip()
    if isinstance(raw_common_name, list):
        if raw_common_name and isinstance(raw_common_name[0], dict):
            scored: dict[str, int] = {}
            for entry in raw_common_name:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                entry_language = str(entry.get("language") or "")
                lexicon = str(entry.get("lexicon") or "")
                source = str(entry.get("source") or "")
                if language and not _matches_language(entry_language, lexicon, language):
                    continue
                score = _common_name_score(entry_language, lexicon, source)
                if name not in scored or score > scored[name]:
                    scored[name] = score
            ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0].lower()))
            names = [_format_common_name(name) for name, _score in ordered if name]
            if preferred_name:
                preferred_formatted = _format_common_name(preferred_name)
                if preferred_formatted not in names:
                    names.insert(0, preferred_formatted)
                else:
                    names = [preferred_formatted] + [n for n in names if n != preferred_formatted]
            return names
        names = [name.strip() for name in raw_common_name if isinstance(name, str)]
        names = [_format_common_name(name) for name in names if name]
        if preferred_name:
            preferred_formatted = _format_common_name(preferred_name)
            if preferred_formatted not in names:
                names.insert(0, preferred_formatted)
            else:
                names = [preferred_formatted] + [n for n in names if n != preferred_formatted]
        return names
    if isinstance(raw_common_name, str) and raw_common_name.strip():
        names = [name.strip() for name in raw_common_name.split(",")]
        names = [_format_common_name(name) for name in names if name]
        if preferred_name:
            preferred_formatted = _format_common_name(preferred_name)
            if preferred_formatted not in names:
                names.insert(0, preferred_formatted)
            else:
                names = [preferred_formatted] + [n for n in names if n != preferred_formatted]
        return names
    return []


def extract_common_names(taxon: TaxonRecord) -> list[str]:
    """Returns all common names from a taxon record (strings or {name, language, lexicon})."""
    return _extract_common_names(taxon, language=None)


def extract_common_names_for_language(taxon: TaxonRecord, language: str = "en") -> list[str]:
    """Returns common names for a specific language, ordered by source/lexicon preference."""
    names = _extract_common_names(taxon, language=language)
    if names:
        return names
    rank = (taxon.get("rank") or "").strip().upper()
    if rank in CONFIG.subspecies_equivalents:
        parent = get_parent_taxon(taxon)
        if parent:
            return _extract_common_names(parent, language=language)
    return []


def iter_descendants_dfs(taxon: TaxonRecord) -> Iterable[TaxonRecord]:
    """Iterates descendants in depth-first order."""
    stack: list[TaxonRecord] = list(get_children(taxon["taxon_key"]))
    while stack:
        current = stack.pop()
        yield current
        children = get_children(current["taxon_key"])
        if children:
            stack.extend(children)


@lru_cache(maxsize=4096)
def resolve_taxon_media(taxon_key: str) -> dict | None:
    """Resolve media for a taxon, preferring descendant species for higher ranks."""
    taxon = get_taxon_by_id(taxon_key)
    if not taxon:
        return None
    media_index = load_taxon_media()
    rank = canonical_rank(taxon.get("rank"))
    is_species_or_lower = rank == "SPECIES" or rank in CONFIG.subspecies_equivalents
    direct = media_index.get(taxon_key)

    first_descendant_media = None
    for descendant in iter_descendants_dfs(taxon):
        record = media_index.get(descendant["taxon_key"])
        if not record:
            continue
        if first_descendant_media is None:
            first_descendant_media = record
        if canonical_rank(descendant.get("rank")) == "SPECIES":
            return record

    if is_species_or_lower and direct:
        return direct
    if first_descendant_media:
        return first_descendant_media
    if direct:
        return direct

    parent = get_parent_taxon(taxon)
    if parent:
        siblings = [sib for sib in get_children(parent["taxon_key"]) if sib["taxon_key"] != taxon_key]
        for sibling in siblings:
            record = media_index.get(sibling["taxon_key"])
            if record:
                return record
            for descendant in iter_descendants_dfs(sibling):
                record = media_index.get(descendant["taxon_key"])
                if record:
                    return record

    return None


@lru_cache(maxsize=1)
def _load_payload() -> dict:
    """Loads the taxon catalog payload pickle.

    Args:
        None.

    Returns:
        A dict containing the catalog and lookup indices.
    """
    with PARQUET.open_input_file(CONFIG.taxon_catalog_path) as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, TaxonRecord]:
    """Loads the raw catalog keyed by taxon id.

    Paths are stored as serialized in the payload and normalized lazily on
    per-record access, which keeps cold-start load time low.
    """
    payload = _load_payload()
    return {str(key): value for key, value in payload["catalog"].items()}


def _iter_search_index_names(taxon: TaxonRecord) -> Iterable[str]:
    preferred_name = str(taxon.get("inat_preferred_common_name") or "").strip()
    if preferred_name:
        yield preferred_name

    raw_common_name = taxon.get("common_name")
    if isinstance(raw_common_name, list):
        if raw_common_name and isinstance(raw_common_name[0], dict):
            for entry in raw_common_name:
                if not isinstance(entry, dict):
                    continue
                value = str(entry.get("name") or "").strip()
                if value:
                    yield value
        else:
            for value in raw_common_name:
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        yield cleaned
    elif isinstance(raw_common_name, str):
        for value in raw_common_name.split(","):
            cleaned = value.strip()
            if cleaned:
                yield cleaned

    scientific_name = str(taxon.get("scientific_name") or "").strip()
    if scientific_name:
        yield scientific_name


@lru_cache(maxsize=1)
def load_name_index() -> dict:
    """Load name index and expand it to include all comma-separated common names."""
    start = time.perf_counter()
    payload = _load_payload()
    name_index = {
        normalized_name: list(dict.fromkeys(str(key) for key in keys))
        for raw_name, keys in payload["combined_name_index"].items()
        if (normalized_name := normalize_name(str(raw_name or "")))
    }
    keys_by_name = {name: set(keys) for name, keys in name_index.items()}
    catalog = payload["catalog"]

    # Expand the precomputed index with raw catalog names without paying the
    # full common-name formatting cost on first typeahead request.
    for taxon_key, taxon in catalog.items():
        taxon_key = str(taxon_key)
        for raw_name in _iter_search_index_names(taxon):
            normalized_name = normalize_name(raw_name)
            if not normalized_name:
                continue
            if normalized_name not in name_index:
                name_index[normalized_name] = []
                keys_by_name[normalized_name] = set()
            known_keys = keys_by_name[normalized_name]
            if taxon_key in known_keys:
                continue
            known_keys.add(taxon_key)
            name_index[normalized_name].append(taxon_key)

    LOGGER.info(
        "[taxa.load-name-index] elapsed=%.3fs names=%s taxa=%s",
        time.perf_counter() - start,
        len(name_index),
        len(catalog),
    )

    return name_index


@lru_cache(maxsize=1)
def load_search_names_by_taxon() -> dict[str, tuple[str, ...]]:
    """Invert the expanded name index to normalized search names per taxon."""
    names_by_taxon: dict[str, list[str]] = {}
    for normalized_name, taxon_keys in load_name_index().items():
        if not normalized_name:
            continue
        for taxon_key in taxon_keys:
            bucket = names_by_taxon.setdefault(str(taxon_key), [])
            if normalized_name not in bucket:
                bucket.append(normalized_name)
    return {taxon_key: tuple(names) for taxon_key, names in names_by_taxon.items()}


@lru_cache(maxsize=1)
def load_slug_index() -> dict[str, tuple[str, ...]]:
    """Load a scientific-name slug to taxon id index."""
    slug_index: dict[str, list[str]] = {}
    for taxon_key, taxon in load_catalog().items():
        slug = taxon_slug(str(taxon.get("scientific_name") or ""))
        if not slug:
            continue
        slug_index.setdefault(slug, []).append(str(taxon_key))
    return {slug: tuple(taxon_keys) for slug, taxon_keys in slug_index.items()}


@lru_cache(maxsize=65536)
def _normalized_taxon_record(lookup_key: Any) -> TaxonRecord | None:
    """Returns a single normalized taxon record for the requested catalog key."""
    catalog = load_catalog()
    record = catalog.get(lookup_key)
    if record is None:
        return None
    updated = dict(record)
    updated["path"] = _normalize_taxon_path(updated["path"])
    return cast(TaxonRecord, updated)


@lru_cache(maxsize=1)
def load_taxon_media() -> dict[str, dict]:
    """Load taxon_key -> media mapping from taxon_media.pkl.

    Returns:
        Dictionary mapping taxon_key to media record with url, license, creator, rightsHolder.
    """
    if not PARQUET.exists(CONFIG.taxon_media_path):
        return {}
    with PARQUET.open_input_file(CONFIG.taxon_media_path) as handle:
        return pickle.load(handle)


# ---- Public API ----
def get_taxon_by_id(taxon_id: Any) -> TaxonRecord | None:
    """Simply returns the taxon record for a given taxon id.

    Args:
        taxon_id: The taxon id in question.

    Returns:
        The taxon record for the corresponding id.
    """
    normalized_key = str(taxon_id).strip() if taxon_id is not None else ""
    if not normalized_key:
        return None
    return _normalized_taxon_record(normalized_key)


def get_taxon_by_slug(slug: str) -> TaxonRecord | None:
    """Return a taxon record for a canonical scientific-name slug."""
    normalized_slug = taxon_slug(slug)
    if not normalized_slug:
        return None
    taxon_keys = load_slug_index().get(normalized_slug)
    if not taxon_keys:
        return None
    if len(taxon_keys) > 1:
        raise ValueError(f"Ambiguous taxon slug: {normalized_slug}")
    taxon_key = cast(tuple[str], taxon_keys)[0]
    return get_taxon_by_id(taxon_key)


def resolve_taxon_reference(value: str | None) -> TaxonRecord | None:
    """Resolve a taxon from a numeric id or canonical scientific-name slug."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return None

    exact = get_taxon_by_id(raw_value)
    if exact is not None:
        return exact

    if any(ch.isspace() for ch in raw_value):
        return None

    by_slug = get_taxon_by_slug(raw_value)
    if by_slug is not None:
        return by_slug
    return None


def is_valid_descendant_rank(value: str | None) -> bool:
    """Return whether a descendant rank is recognized by the API."""
    canonical = canonical_rank(value)
    if not canonical:
        return False
    valid_ranks = set(STANDARD_DESCENDANT_RANKS) | set(CONFIG.rank_synonyms) | set(CONFIG.subspecies_equivalents)
    return canonical in valid_ranks


def get_children(taxon_id: str) -> List[TaxonRecord]:
    """Returns a list of taxon records for all direct children of the taxon defined by the passed taxon id.

    Args:
        taxon_id: The taxon id in question.

    Returns:
        A list of taxon records for all direct children of the argument taxon.
    """
    parent = get_taxon_by_id(taxon_id)
    if parent is None:
        return []

    cached_children = _child_index().get(str(taxon_id))
    if cached_children is not None:
        resolved: list[TaxonRecord] = []
        for child_key in cached_children:
            child_taxon = get_taxon_by_id(child_key)
            if child_taxon is not None:
                resolved.append(child_taxon)
        return resolved

    if PARQUET.is_remote:
        return []

    parent_path = _normalize_taxon_path(parent["path"])
    if not parent_path.exists():
        return []

    children: list[TaxonRecord] = []
    for child_dir in parent_path.iterdir():
        if child_dir.is_dir():
            # match child by taxonKey suffix in folder name
            key = taxon_key_from_path(child_dir)
            child_taxon = get_taxon_by_id(key)
            if child_taxon:
                children.append(child_taxon)
    return children


def taxon_key_from_path(path: Path) -> str:
    """Implicitly gets the taxon key from the filepath of a taxon. Used in areas where it's convenient.

    Args:
        path: The path to the taxon in the filesystem.

    Returns:
        The taxon key which is implicitly stored in the path.
    """
    name = path.name
    if "_" in name:
        return name.split("_")[-1]
    return name


def search_taxa_by_name(
    name_query: str,
    limit: int | None = 10,
    cancel_check: CancelCheck | None = None,
) -> list[Tuple[TaxonRecord, float]]:
    """Performs fuzzy search to get a list of taxa with a name that matches the query.

    Args:
        name_query: The name of the taxon being searched for. Can be common or scientific.
        limit: How many taxa to return. When None, return all matches.

    Returns:
        A list of tuples of (TaxonRecord, score).
    """
    start = time.perf_counter()
    normalized_query = normalize_name(name_query)
    tokens = normalized_query.split()
    try:
        name_index = load_name_index()
        if not name_index:
            return []
        if not tokens:
            return []

        result_limit = max(int(limit), 1) if limit is not None else None
        if limit is None:
            extract_iter = getattr(process, "extract_iter", None)
            if extract_iter is not None:
                matches = extract_iter(
                    normalized_query,
                    name_index.keys(),
                    scorer=fuzz.token_set_ratio,
                )
            else:
                matches = process.extract(
                    normalized_query,
                    name_index.keys(),
                    scorer=fuzz.token_set_ratio,
                    limit=max(len(name_index), 1),
                )
        else:
            bounded_limit = max(int(limit), 1)
            # RapidFuzz needs a wider candidate pool than the final API page
            # because score adjustment and taxon deduplication happen after the
            # fuzzy match step. Keep this bounded to avoid scanning the entire
            # shared name index for every broad text query.
            extract_limit = max(bounded_limit * 25, 100)
            matches = process.extract(
                normalized_query,
                name_index.keys(),
                scorer=fuzz.token_set_ratio,
                limit=extract_limit,
            )

        best_by_taxon: dict[str, Tuple[TaxonRecord, float]] = {}
        for name, score, _ in matches:
            if cancel_check is not None:
                cancel_check()
            adjusted_score = _adjust_search_name_score(name, normalized_query, tokens, float(score))
            if adjusted_score is None:
                continue

            keys = name_index.get(name, [])
            for key in keys:
                if cancel_check is not None:
                    cancel_check()
                taxon_key = str(key)
                taxon = get_taxon_by_id(taxon_key)
                if not taxon:
                    continue
                existing = best_by_taxon.get(taxon_key)
                if existing is None or adjusted_score > existing[1]:
                    best_by_taxon[taxon_key] = (taxon, adjusted_score)

        results = list(best_by_taxon.values())
        results.sort(key=lambda entry: entry[1], reverse=True)
        return results if result_limit is None else results[:result_limit]
    finally:
        LOGGER.info(
            "[taxa.search-by-name] elapsed=%.3fs query=%r normalized=%r token_count=%s limit=%r",
            time.perf_counter() - start,
            name_query,
            normalized_query,
            len(tokens),
            limit,
        )


def _adjust_search_name_score(
    normalized_name: str,
    normalized_query: str,
    query_tokens: list[str],
    raw_score: float,
) -> float | None:
    if not normalized_query or not query_tokens:
        return None

    name_tokens = normalized_name.split()
    if len(query_tokens) > 1:
        token_matches = all(any(name_token.startswith(token) for name_token in name_tokens) for token in query_tokens)
        if not token_matches:
            return None
    else:
        token = query_tokens[0]
        if not any(name_token.startswith(token) for name_token in name_tokens):
            return None

    adjusted_score = float(raw_score)
    if normalized_name == normalized_query:
        adjusted_score += 20.0
    token_penalty = max(0, len(name_tokens) - len(query_tokens)) * 2
    adjusted_score -= float(token_penalty)
    min_score = 60 if len(query_tokens) > 1 else 70
    if adjusted_score < min_score:
        return None
    return adjusted_score


def taxon_name_match_score(taxon: TaxonRecord, name_query: str) -> float | None:
    """Return the best search-match score for a single taxon against a query.

    This mirrors the score adjustments used by search_taxa_by_name but evaluates
    only the names attached to a single taxon. It is intended for scoped ranked
    queries where the leaderboard already constrains the candidate universe.
    """
    normalized_query = normalize_name(name_query)
    query_tokens = normalized_query.split()
    if not query_tokens:
        return None

    best_score: float | None = None
    seen_names: set[str] = set()
    taxon_key = str(taxon.get("taxon_key") or "").strip()
    search_names = list(load_search_names_by_taxon().get(taxon_key, ()))
    if not search_names:
        search_names = [normalize_name(raw_name) for raw_name in _iter_search_index_names(taxon)]
    for normalized_name in search_names:
        if not normalized_name or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        adjusted_score = _adjust_search_name_score(
            normalized_name,
            normalized_query,
            query_tokens,
            float(fuzz.token_set_ratio(normalized_query, normalized_name)),
        )
        if adjusted_score is None:
            continue
        if best_score is None or adjusted_score > best_score:
            best_score = adjusted_score
    return best_score


def canonical_rank(value: str | None) -> str:
    """Normalizes a rank string using configured synonyms, e.g. sp. maps to SPECIES.

    Args:
        value: The rank string to normalize.

    Returns:
        The canonical rank string.
    """
    if not value:
        return ""
    cleaned = value.strip().upper()
    for canonical, synonyms in CONFIG.rank_synonyms.items():
        if cleaned in synonyms:
            return canonical
    return cleaned


def iter_descendants(taxon: TaxonRecord, include_self: bool = False) -> List[TaxonRecord]:
    """Iterates all descendants of a taxon to essentially get an iterable of the subtree at its node.

    Args:
        taxon: The taxon record in question.
        include_self: Whether or not to include the taxon in question in the iterable.

    Returns:
        An iterable of all descendants of the taxon, e.g. its subtree.
    """
    stack: list[TaxonRecord] = [taxon] if include_self else list(get_children(taxon["taxon_key"]))
    result: List[TaxonRecord] = []
    while stack:
        current = stack.pop()
        if current is taxon:
            if include_self:
                result.append(current)
        else:
            result.append(current)
        children = get_children(current["taxon_key"])
        if children:
            stack.extend(children)
    return result


def iter_descendants_by_rank(
    taxon: TaxonRecord,
    target_rank: str,
    include_self: bool = False,
) -> List[TaxonRecord]:
    """Iterates all descendants of a taxon, filtering results to a given rank.

    Args:
        taxon: The taxon record in question.
        target_rank: The target rank to only include in the iterable.
        include_self: Whether to include the starting taxon if it matches.

    Returns:
        An iterable including all descendants of the given taxon, filtered to the target rank.
    """
    canonical_target = canonical_rank(target_rank)
    if not canonical_target:
        return []
    matches: list[TaxonRecord] = []
    stack = list(get_children(taxon["taxon_key"]))
    if include_self:
        stack.append(taxon)
    while stack:
        current = stack.pop()
        normalized = canonical_rank(current["rank"])
        if normalized == canonical_target:
            matches.append(current)
            continue
        children = get_children(current["taxon_key"])
        if children:
            stack.extend(children)
    return matches


def taxon_id_as_int(taxon_key: str | None) -> int | None:
    """Converts a taxon key to an integer when possible.

    Args:
        taxon_key: The taxon key string to parse.

    Returns:
        The integer taxon id, or None when parsing fails.
    """
    if not taxon_key:
        return None
    try:
        return int(taxon_key)
    except (TypeError, ValueError):
        return None


def preferred_image_url(taxon: TaxonRecord | None) -> str | None:
    """Returns catalog-stored preferred image URL when available."""
    if not taxon:
        return None
    value = str(taxon.get("inat_preferred_image") or "").strip()
    return value or None


def normalize_image_reference(value: Any) -> str | None:
    """Normalize image references to a single primary API string.

    Upstream catalog and media payloads may store references as a single
    string, a list of strings, or richer dict records. The public API only
    uses one primary attribution link today, so collapse these variants to the
    first usable string value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    return cleaned
            elif isinstance(entry, dict):
                normalized = normalize_image_reference(entry)
                if normalized:
                    return normalized
        return None
    if isinstance(value, dict):
        for key in ("url", "href", "reference", "references"):
            normalized = normalize_image_reference(value.get(key))
            if normalized:
                return normalized
    return None


@lru_cache(maxsize=4096)
def resolve_preferred_image_taxon_key(taxon_key: str) -> str | None:
    """Resolve taxon key whose preferred image metadata should be used."""
    taxon = get_taxon_by_id(taxon_key)
    if not taxon:
        return None

    # If this taxon already has a preferred image, never traverse descendants.
    if preferred_image_url(taxon):
        return taxon_key

    for descendant in iter_descendants_dfs(taxon):
        image = preferred_image_url(descendant)
        if not image:
            continue
        return descendant["taxon_key"]
    return None


def preferred_image_payload(taxon: TaxonRecord | None) -> dict[str, Any]:
    """Returns API image fields derived from catalog-stored preferred image metadata."""
    if not taxon:
        return {}
    source_taxon_key = resolve_preferred_image_taxon_key(str(taxon.get("taxon_key") or ""))
    source_taxon = get_taxon_by_id(source_taxon_key) if source_taxon_key else None
    image_url = preferred_image_url(source_taxon)
    if not image_url:
        return {}
    payload: dict[str, Any] = {"image_url": image_url}
    image_license = str((source_taxon or {}).get("inat_preferred_image_license") or "").strip()
    image_creator = str((source_taxon or {}).get("inat_preferred_image_creator") or "").strip()
    image_attribution = str((source_taxon or {}).get("inat_preferred_image_attribution") or "").strip()
    image_references = normalize_image_reference((source_taxon or {}).get("inat_preferred_image_references"))
    if image_license:
        payload["image_license"] = image_license
    if image_creator:
        payload["image_rights_holder"] = image_creator
    elif image_attribution:
        payload["image_rights_holder"] = image_attribution
    if image_creator:
        payload["image_creator"] = image_creator
    elif image_attribution:
        payload["image_creator"] = image_attribution
    if image_attribution:
        payload["image_attribution"] = image_attribution
    if image_references:
        payload["image_references"] = image_references
    return payload


def count_taxon_rows(taxon: TaxonRecord) -> int | None:
    """Returns the numnber of rows within the occurrence parquet of a taxon.

    Args:
        taxon: The taxon record in question.

    Returns:
        The number of rows within the taxon's occurrence parquet.
    """
    data_path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    if not PARQUET.exists(data_path):
        return None
    return PARQUET.read_metadata(data_path).num_rows


def serialize_taxon(taxon: TaxonRecord) -> dict[str, Any] | None:
    """Returns a serialized version of taxon metadata for the API.

    Args:
        taxon: The taxon record in question.

    Returns:
        A serialized version of the taxon containing its id, scientific and common names, description, image, and rank metadata.
    """
    taxon_id = taxon_id_as_int(taxon.get("taxon_key"))
    if taxon_id is None:
        return None
    scientific_name = (taxon.get("scientific_name") or "").replace("_", " ").strip()

    common_names = extract_common_names_for_language(taxon, language=CONFIG.common_name_language)
    common_name = common_names[0] if common_names else scientific_name

    rank = (taxon.get("rank") or "").upper()
    slug = taxon_slug(scientific_name)
    # Avoid per-result metadata reads during search serialization.
    occurrences: int | None = None
    description = f"{scientific_name}. Rank: {rank.title()}."

    # Prefer catalog-stored preferred image metadata when available.
    taxon_key = taxon.get("taxon_key")
    preferred_image = preferred_image_payload(taxon)
    media_record = None
    if not preferred_image and taxon_key:
        media_record = resolve_taxon_media(taxon_key)

    path_str = str(taxon.get("path", ""))
    if "Arthropoda_54" in path_str:
        taxon_group = "arthropods"
    elif "Aves_212" in path_str:
        taxon_group = "birds"
    elif "Animalia_1" in path_str:
        taxon_group = "animals"
    elif "Fungi_5" in path_str:
        taxon_group = "fungi"
    elif "Plantae_6" in path_str:
        taxon_group = "plants"
    else:
        taxon_group = "other"

    result = {
        "taxon_id": taxon_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "common_names": common_names,
        "description": description,
        "image_url": preferred_image.get("image_url"),
        "slug": slug,
        "rank": rank,
        "occurrences": occurrences,
        "taxon_group": taxon_group,
    }

    # Add image and attribution if available
    if preferred_image:
        result.update(preferred_image)
    elif media_record:
        result["image_url"] = media_record.get("url")
        result["image_license"] = media_record.get("license")
        result["image_creator"] = media_record.get("creator")
        result["image_rights_holder"] = media_record.get("rightsHolder")
        result["image_references"] = normalize_image_reference(media_record.get("references"))

    return result


def base_observation_mask(table: pa.Table) -> pa.Array:
    """Adds a mask over a parquet to only return rows that are not obscured and are positionally accurate.

    Args:
        table: The input table.

    Returns:
        A pyarrow boolean array mask that can be used to filter the table.
    """
    mask = PC.equal(table["obscured"], "No")
    precision_mask = PC.less_equal(table["coordinateUncertaintyInMeters"], 500)
    mask = PC.and_(mask, precision_mask)
    lat_col = table["decimalLatitude"]
    lon_col = table["decimalLongitude"]
    mask = PC.and_(mask, PC.invert(PC.is_null(lat_col)))
    mask = PC.and_(mask, PC.invert(PC.is_null(lon_col)))
    return mask


def iter_filtered_occurrence_tables(
    taxon_id: int,
    extra_columns: Iterable[str] = (),
    location_gid: Optional[str] = None,
) -> Iterable[pa.Table]:
    """Yields filtered occurrence tables for a taxon and its descendants.

    Args:
        taxon_id: Taxon id whose subtree should be scanned for occurrence tables.
        extra_columns: Additional column names to include in the yielded tables.
        location_gid: Optional location GID to filter observations by location membership.

    Returns:
        An iterator of pyarrow tables filtered to non-obscured, precise observations.
    """
    taxon = get_taxon_by_id(str(taxon_id))
    if taxon is None:
        return
    columns = set(base_occurrence_columns)
    columns.update(extra_columns)
    column_list = list(columns)
    normalized_location = location_gid.strip() if location_gid else None
    taxa = iter_descendants(taxon, include_self=True)
    for taxon_record in taxa:
        taxon_dir = Path(taxon_record["path"])
        if not PARQUET.is_remote and not taxon_dir.exists():
            continue
        for candidate in (
            CONFIG.occurrence_parquet_filename,
            combined_parquet_filename,
        ):
            path = taxon_dir / candidate
            if not PARQUET.exists(path):
                continue
            try:
                table = PARQUET.read_table(path, columns=column_list).combine_chunks()
            except Exception:
                continue
            mask = base_observation_mask(table)
            if normalized_location:
                loc_mask = gis_lookup.build_location_mask(table, normalized_location)
                if loc_mask is None:
                    continue
                mask = PC.and_(mask, loc_mask)
            filtered = table.filter(mask).combine_chunks()
            if filtered.num_rows:
                yield filtered


def load_occurrence_points(
    taxon_id: int,
    location_gid: Optional[str] = None,
) -> list[dict[str, float | str]]:
    """Loads occurrence coordinates for a taxon and its descendants.

    Args:
        taxon_id: Taxon id whose occurrence points should be loaded.
        location_gid: Optional location GID to filter occurrences by location membership.

    Returns:
        A list of unique occurrence points with catalog number and coordinates. As a dict so it can easily be serialized for the API.
    """
    points: list[dict[str, float | str]] = []
    seen_catalogs: set[str] = set()
    for table in iter_filtered_occurrence_tables(taxon_id, location_gid=location_gid):
        catalogs = table["catalogNumber"].to_pylist()
        latitudes = table["decimalLatitude"].to_pylist()
        longitudes = table["decimalLongitude"].to_pylist()
        for catalog, lat, lon in zip(catalogs, latitudes, longitudes):
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            catalog_id = str(catalog)
            if catalog_id in seen_catalogs:
                continue
            seen_catalogs.add(catalog_id)
            points.append(
                {
                    "catalogNumber": catalog_id,
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )
    return points


@lru_cache(maxsize=1)
def _child_index() -> dict[str, list[str]]:
    """Builds a parent taxon key -> child taxon keys index."""
    catalog = load_catalog()
    mapping: dict[str, list[str]] = {}
    for record in catalog.values():
        raw_path = record.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        parent_path = path.parent
        parent_key = taxon_key_from_path(parent_path)
        child_key = str(record.get("taxon_key") or "")
        if child_key:
            mapping.setdefault(parent_key, []).append(child_key)
    for children in mapping.values():
        children.sort()
    return mapping
