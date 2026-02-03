'''
This file functions as a library, providing functions that perform standard operations one might use when traversing the taxonomy tree.
It aims to accomplish more generic things on the tree and its parquets that other areas of the code can use.
'''

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from util.config import load_config
from util import gis_lookup

CONFIG = load_config("global")

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
    path: Path
    scientific_name: str
    common_name: str
    rank: str


@lru_cache(maxsize=1)
def _load_payload() -> dict:
    """Loads the taxon catalog payload pickle.
    
    Args:
        None.
    
    Returns:
        A dict containing the catalog and lookup indices.
    """
    with open(CONFIG.taxon_catalog_path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, TaxonRecord]:
    """Loads the catalog of taxon records keyed by taxon id.
    
    Args:
        None.
    
    Returns:
        A dict mapping taxon ids to taxon records.
    """
    payload = _load_payload()
    return payload["catalog"]


@lru_cache(maxsize=1)
def load_name_index() -> dict:
    """Loads the combined name index from the payload.
    
    Args:
        None.
    
    Returns:
        A dict mapping normalized names to lists of taxon ids.
    """
    payload = _load_payload()
    return payload["combined_name_index"]

# ---- Public API ----
def get_taxon_by_id(taxon_id: str) -> TaxonRecord | None:
    """Simply returns the taxon record for a given taxon id.
    
    Args:
        taxon_id: The taxon id in question.
    
    Returns:
        The taxon record for the corresponding id.
    """
    catalog = load_catalog()
    return catalog.get(taxon_id)

def get_children(taxon_id: str) -> List[TaxonRecord]:
    """Returns a list of taxon records for all direct children of the taxon defined by the passed taxon id.
    
    Args:
        taxon_id: The taxon id in question.
    
    Returns:
        A list of taxon records for all direct children of the argument taxon.
    """
    parent = get_taxon_by_id(taxon_id)
    if parent is None or not parent["path"].exists():
        return []

    children = []
    for child_dir in parent["path"].iterdir():
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

def search_taxa_by_name(name_query: str, limit: int = 10) -> list[Tuple[TaxonRecord, float]]:
    """Performs fuzzy search to get a list of taxa with a name that matches the query.
    
    Args:
        name_query: The name of the taxon being searched for. Can be common or scientific.
        limit: How many taxa to return. Defaults to 10.
    
    Returns:
        A list of tuples of (TaxonRecord, float) where the float denotes how strong the match was.
    """

    catalog = load_catalog()
    name_index = load_name_index()
    if not name_index:
        return []
    matches = process.extract(
        name_query.lower(),
        name_index.keys(),
        scorer=fuzz.WRatio,
        limit=limit
    )

    results = []
    for name, score, _ in matches:
        keys = name_index.get(name, [])
        for key in keys:
            taxon = catalog.get(key)
            if taxon:
                results.append((taxon, score))
    return results


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


def count_taxon_rows(taxon: TaxonRecord) -> int | None:
    """Returns the numnber of rows within the occurrence parquet of a taxon.
    
    Args:
        taxon: The taxon record in question.
    
    Returns:
        The number of rows within the taxon's occurrence parquet.
    """
    data_path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    if not data_path.exists():
        return None
    return pq.read_metadata(data_path).num_rows


def serialize_taxon(taxon: TaxonRecord) -> dict[str, Any] | None:
    """Returns a serialized version of taxon metadata for the API.
    
    Args:
        taxon: The taxon record in question.
    
    Returns:
        A serialized version of the taxon containing its id, scientific and common names, description, image, slug, rank, and number of occurrences.
    """
    taxon_id = taxon_id_as_int(taxon.get("taxon_key"))
    if taxon_id is None:
        return None
    scientific_name = (taxon.get("scientific_name") or "").replace("_", " ").strip()
    common_name = taxon.get("common_name") or scientific_name
    rank = (taxon.get("rank") or "").upper()
    slug = "-".join(part for part in scientific_name.lower().split() if part)
    data_path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    occurrences = count_taxon_rows(taxon) if data_path.exists() else 0
    description = (
        f"{scientific_name} has {occurrences:,} research-grade observations in this dataset."
        f" Rank: {rank.title()}."
    )
    return {
        "taxon_id": taxon_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "description": description,
        "image_url": None,
        "slug": slug,
        "rank": rank,
        "occurrences": occurrences,
    }


def base_observation_mask(table: pa.Table) -> pa.Array:
    """Adds a mask over a parquet to only return rows that are not obscured and are positionally accurate.
    
    Args:
        table: The input table.
    
    Returns:
        A pyarrow boolean array mask that can be used to filter the table.
    """
    mask = pc.equal(table["obscured"], "No")
    precision_mask = pc.less_equal(table["coordinateUncertaintyInMeters"], 500)
    mask = pc.and_(mask, precision_mask)
    lat_col = table["decimalLatitude"]
    lon_col = table["decimalLongitude"]
    mask = pc.and_(mask, pc.invert(pc.is_null(lat_col)))
    mask = pc.and_(mask, pc.invert(pc.is_null(lon_col)))
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
        if not taxon_dir.exists():
            continue
        for candidate in (
            CONFIG.occurrence_parquet_filename,
            combined_parquet_filename,
        ):
            path = taxon_dir / candidate
            if not path.exists():
                continue
            try:
                table = pq.read_table(path, columns=column_list).combine_chunks()
            except Exception:
                continue
            mask = base_observation_mask(table)
            if normalized_location:
                loc_mask = gis_lookup.build_location_mask(table, normalized_location)
                if loc_mask is None:
                    continue
                mask = pc.and_(mask, loc_mask)
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
