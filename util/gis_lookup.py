'''
This file functions as a library, providing functions that perform standard operations one might use when querying GIS data from taxa or coordinates.
'''

from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import defaultdict
import csv
import json
import math
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from util.config import load_config

CONFIG = load_config("global")

_LAYER_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class LocationRecord:
    gid: str
    name: str
    level: int
    parent_gid: Optional[str]

@lru_cache(maxsize=1)
def _load_gis_catalog() -> Dict[str, Any]:
    """Loads the GIS catalog JSON into memory.
    
    Args:
        None.
    
    Returns:
        The parsed GIS catalog dictionary with categories and layers.
    """
    with open(CONFIG.gis_catalog_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=256)
def _get_layer(layer_id: str) -> Dict[str, Any] | None:
    """Returns the layer "object" for a given layer id.
    
    Args:
        layer_id: The layer id for the requested layer, e.g. `bio_1`.
    
    Returns:
        A dictionary representation of the layer's info, accessed such as _get_layer("bio_1")["units"].
    """
    catalog = _load_gis_catalog()
    for cat in catalog["categories"]:
        for layer in cat["layers"]:
            if layer["id"] == layer_id:
                return layer
    return None


@lru_cache(maxsize=1)
def load_layer_metadata() -> Dict[str, Dict[str, Any]]:
    """Builds a mapping of layer id to raw layer metadata.
    
    Args:
        None.
    
    Returns:
        A dict of layer id to the layer entry in the GIS catalog.
    """
    catalog = _load_gis_catalog()
    layers: Dict[str, Dict[str, Any]] = {}
    for category in catalog.get("categories", []):
        for layer in category.get("layers", []):
            layer_id = layer.get("id")
            if not layer_id:
                continue
            layers[str(layer_id)] = layer
    return layers


@lru_cache(maxsize=1)
def load_variable_metadata() -> tuple[List[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Builds a list of variable metadata entries and a lookup by id.
    
    Args:
        None.
    
    Returns:
        A tuple of (entries, mapping) where entries are sorted variable metadata
        and mapping maps layer id to the same metadata entry.
    """
    catalog = _load_gis_catalog()
    entries: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for category in catalog.get("categories", []):
        category_name = category.get("display_name") or category.get("name")
        for layer in category.get("layers", []):
            layer_id = layer.get("id")
            if not layer_id:
                continue
            entry = {
                "id": layer_id,
                "name": layer.get("display_name") or layer.get("name") or layer_id,
                "units": layer.get("units"),
                "description": layer.get("description"),
                "value_type": layer.get("value_type"),
                "category": category_name,
            }
            entries.append(entry)
            mapping[layer_id] = entry
    entries.sort(key=lambda item: item["id"])
    return entries, mapping


@lru_cache(maxsize=64)
def load_layer_legend(layer_id: str) -> dict[str, dict[str, Any]]:
    """Loads a categorical legend mapping for a GIS layer.
    
    Args:
        layer_id: The layer id whose legend should be loaded.
    
    Returns:
        A mapping of class id or slugified class name to legend metadata.
    """
    path = CONFIG.gis_legends_root / f"{layer_id}_legend.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for entry in payload.get("classes", []):
        class_id = entry.get("id")
        name = entry.get("name")
        if class_id is None or name is None:
            continue
        slug = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
        data = {
            "id": class_id,
            "name": name,
            "description": entry.get("description"),
        }
        mapping[str(class_id)] = data
        if slug:
            mapping[slug] = data
    return mapping


@lru_cache(maxsize=1)
def load_location_catalog() -> tuple[List[LocationRecord], dict[str, LocationRecord]]:
    """Loads the location catalog into a tuple of a list of all locations (for search) and a dict of location ids to location records.
    
    Args:
        None.
    
    Returns:
        A list of all locations and a dict of location ids to location records.
    """
    entries: List[LocationRecord] = []
    by_gid: dict[str, LocationRecord] = {}
    if CONFIG.location_hierarchy_path.exists():
        with CONFIG.location_hierarchy_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    level = int(row.get("level", 0))
                except (TypeError, ValueError):
                    continue
                gid = row.get("gid") or ""
                name = row.get("name") or ""
                parent_gid = row.get("parent_gid") or None
                if not gid or not name:
                    continue
                record = LocationRecord(gid=gid, name=name, level=level, parent_gid=parent_gid)
                entries.append(record)
                by_gid[record.gid] = record
    for region in sorted(CONFIG.gbif_region_set):
        name = region.replace("_", " ").title()
        record = LocationRecord(gid=region, name=name, level=-1, parent_gid=None)
        entries.append(record)
        by_gid[record.gid] = record
    return entries, by_gid


def location_lookup_for_gid(gid: str) -> tuple[str, str, str]:
    """Returns the location column, scope label, and normalized gid for filtering.
    
    Args:
        gid: The gid of the region being filtered on.
    
    Returns:
        A tuple of (column_name, scope_label, normalized_gid).
    """
    normalized = gid.strip()
    upper = normalized.upper()
    if upper in CONFIG.gbif_region_set:
        return "gbifRegion", "gbif_region", upper
    dot_count = normalized.count(".")
    if dot_count == 0:
        return CONFIG.location_level_columns[0], CONFIG.location_scope_by_level[0], normalized
    if dot_count == 1:
        return CONFIG.location_level_columns[1], CONFIG.location_scope_by_level[1], normalized
    return CONFIG.location_level_columns[2], CONFIG.location_scope_by_level[2], normalized


def build_location_mask(table: pa.Table, location_gid: str) -> Optional[pa.Array]:
    """Builds a mask for a parquet so only rows that match the location gid are present after the filter.
    
    Args:
        table: The input parquet or table.
        location_gid: The gid for the region being filtered on.
    
    Returns:
        A mask for the parquet that filters for only rows within the gid in question.
    """
    column_name, _scope, target = location_lookup_for_gid(location_gid)
    if column_name not in table.column_names:
        return None
    column = table[column_name]
    return pc.equal(column, target)




@lru_cache(maxsize=1)
def location_taxa_membership() -> Dict[tuple[str, str], frozenset[int]]:
    """Returns a dict mapping gids at certain scopes to a set of taxon ids that occur within the region.
    
    Args:
        None.
    
    Returns:
        A dict mapping gids at certain scopes to a set of taxon ids that occur within the region.
    """
    mapping: dict[tuple[str, str], set[int]] = defaultdict(set)
    if not CONFIG.location_catalog_path.exists():
        return {}
    try:
        table = pq.read_table(
            CONFIG.location_catalog_path,
            columns=["scope", "gid", "taxon_id"],
        ).combine_chunks()
    except Exception:
        return {}
    scopes = table.column("scope").to_pylist()
    gids = table.column("gid").to_pylist()
    taxon_ids = table.column("taxon_id").to_pylist()
    for scope, gid, taxon_id in zip(scopes, gids, taxon_ids):
        if not scope or not gid:
            continue
        try:
            numeric_id = int(taxon_id)
        except (TypeError, ValueError):
            continue
        mapping[(str(scope), str(gid))].add(numeric_id)
    return {key: frozenset(value) for key, value in mapping.items()}


def resolve_location_context(
    record: LocationRecord,
    mapping: dict[str, LocationRecord],
) -> List[str]:
    """Builds the ancestor name path for a location record.
    
    Args:
        record: Location record whose parent chain should be resolved.
        mapping: Mapping of gid to location record for lookups.
    
    Returns:
        A list of ancestor names from top-level to immediate parent.
    """
    context: List[str] = []
    parent = record.parent_gid
    while parent:
        parent_record = mapping.get(parent)
        if not parent_record:
            break
        context.append(parent_record.name)
        parent = parent_record.parent_gid
    context.reverse()
    return context


def search_locations(query: str, limit: int) -> List[dict[str, Any]]:
    """Searches location records by name substring.
    
    Args:
        query: Search term matched against location names.
        limit: Maximum number of results to return.
    
    Returns:
        A list of match dictionaries with gid, name, level, and hierarchy path.
    """
    search_term = query.lower().strip()
    if not search_term:
        return []
    entries, mapping = load_location_catalog()
    results: List[dict[str, Any]] = []
    for record in entries:
        if search_term not in record.name.lower():
            continue
        context = resolve_location_context(record, mapping)
        results.append(
            {
                "gid": record.gid,
                "name": record.name,
                "level": record.level,
                "hierarchy": context,
            }
        )
        if len(results) >= limit:
            break
    return results

def _region_origin(value: float) -> float:
    """Simply converts a lat/lon to its 10° region origin. Origins are at the southwest corner of the region.
    
    Args:
        value: The lat/lon in question.
    
    Returns:
        The lat/lon of the origin of the region the point is in.
    """
    return math.floor(value / 10) * 10


def get_region_name(latitude: float, longitude: float) -> str:
    """Returns the "region name" of a point. The "region name" is the folder name of the region, e.g. `lat10lon-20`.
    
    Args:
        latitude: The latitude of the point.
        longitude: The longitude of the point.
    
    Returns:
        The region name of the region the point is contained within.
    """
    lat0 = _region_origin(latitude)
    lon0 = _region_origin(longitude)
    return f"lat{int(lat0)}_lon{int(lon0)}"


@lru_cache(maxsize=64)
def get_layer_tile_info(layer_id: str) -> dict:
    """Returns useful metadata about the COGs a layer stores. Obtained by reading values from any COG of the layer.
    
    Args:
        layer_id: The id of the layer in question.
    
    Returns:
        A dict containing the span of the region, the pixel size, and the block size and shape in terms of lat/lon. Values are currently always be the same for lat/lon.
    """
    import rasterio

    layer = _get_layer(layer_id) # this is the JSON object for the layer
    if layer is None:
        raise ValueError(f"Layer {layer_id} not found in catalog")

    # Get the region folder and sample a region from that folder
    region_root = CONFIG.gis_root / layer["region_root"]
    sample_dir = next((p for p in sorted(region_root.iterdir()) if p.is_dir()), None)
    if sample_dir is None:
        raise FileNotFoundError(f"No regions found for layer {layer_id} in {region_root}")

    # Look for a tif file for the layer inside the sampled folder
    filename = layer["filename_template"].format(id=layer_id)
    sample_path = sample_dir / filename
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample region {sample_path} missing for layer {layer_id}")

    # Get the bounds and relevant data of COGs of that layer
    with rasterio.open(sample_path) as ds:
        bounds = ds.bounds
        span_lat = abs(bounds.top - bounds.bottom)
        span_lon = abs(bounds.right - bounds.left)
        pixel_lat = abs(ds.transform.e)
        pixel_lon = abs(ds.transform.a)
        block_h, block_w = ds.block_shapes[0]
        block_span_lat = block_h * pixel_lat
        block_span_lon = block_w * pixel_lon

    return {
        "region_span_lat": span_lat,
        "region_span_lon": span_lon,
        "pixel_size_lat": pixel_lat,
        "pixel_size_lon": pixel_lon,
        "block_span_lat": block_span_lat,
        "block_span_lon": block_span_lon,
        "block_shape": (block_h, block_w),
    }

def list_layer_ids() -> List[str]:
    """Returns all GIS layer ids from the catalog.
    
    Args:
        None.
    
    Returns:
        A list of layer ids in catalog order.
    """
    catalog = _load_gis_catalog()
    return [
        layer["id"]
        for category in catalog["categories"]
        for layer in category["layers"]
    ]

def get_cog_path(layer_id: str, latitude: float, longitude: float) -> Optional[Path]:
    """Gets the path to the COG of a given layer at a given coordinate.
    
    Args:
        layer_id: The layer id in question.
        latitude: The latitude of the point in question.
        longitude: The longitude of the point in question.
    
    Returns:
        The filepath to the COG based on the parameters so values can be read.
    """
    layer = _get_layer(layer_id)
    if layer is None:
        return None

    region_root = layer["region_root"]
    filename = layer["filename_template"].format(id=layer_id)

    tile_id = get_region_name(latitude, longitude)

    return CONFIG.gis_root / region_root / tile_id / filename
