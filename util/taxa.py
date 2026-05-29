import os
import pickle
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from rapidfuzz import fuzz, process

from util.storage import ParquetStorageProxy

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
            w = word.lower()
            words.append(w[:1].upper() + w[1:])
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
            existing = best_by_taxon.get(key)
            if existing is None or adjusted > existing[1]:
                best_by_taxon[key] = (taxon, adjusted, name)

    results = sorted(best_by_taxon.values(), key=lambda x: x[1], reverse=True)
    return results[:limit]


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
