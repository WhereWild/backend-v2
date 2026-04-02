'''
This file functions as a library, providing functions that perform standard operations one might use when querying GIS data from taxa or coordinates.
'''

from contextlib import contextmanager
from pathlib import Path
import io
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterator
from collections import defaultdict, OrderedDict
import csv
import json
import math
import os
import re
import threading

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
from util.config import load_config
from util.storage import ParquetStorageProxy, get_parquet_storage_with_mode
import unicodedata

CONFIG = load_config("global")
PARQUET = ParquetStorageProxy(CONFIG.data_root, CONFIG.project_root)

_LAYER_CACHE: dict[str, dict[str, Any]] = {}
_INVALID_LOCATION_GID_TOKENS = frozenset({"nan", "none", "null", "na", "n/a", "undefined"})
_MAX_CACHED_RASTERS = 128
_raster_cache: OrderedDict[str, tuple] = OrderedDict()  # uri -> (ds, gdal_env)
_raster_cache_lock = threading.Lock()
_RASTER_STORAGE_ENV = "WHEREWILD_RASTER_STORAGE"


@dataclass(frozen=True)
class LocationRecord:
    gid: str
    name: str
    level: int
    parent_gid: Optional[str]



@dataclass(frozen=True)
class RasterSource:
    uri: str
    gdal_env: dict[str, str]
    is_remote: bool


def raster_exists(path: Path) -> bool:
    """Return True when a raster can be read from local disk or remote B2."""
    return resolve_raster_source(path) is not None


def _raster_storage_mode() -> str:
    mode = str(os.environ.get(_RASTER_STORAGE_ENV, "auto") or "").strip().lower()
    if mode in {"local", "b2"}:
        return mode
    return "auto"


def _raster_storage():
    mode = _raster_storage_mode()
    if mode == "auto":
        return PARQUET.current()
    try:
        return get_parquet_storage_with_mode(CONFIG.data_root, CONFIG.project_root, mode)
    except Exception:
        return PARQUET.current()


def resolve_raster_source(path: Path) -> Optional[RasterSource]:
    """Resolve a raster to either local path or /vsis3 remote URI."""
    mode = _raster_storage_mode()
    storage = _raster_storage()

    def _remote_source() -> Optional[RasterSource]:
        if not storage.is_remote:
            return None
        try:
            if not storage.exists(path):
                return None
            return RasterSource(
                uri=storage.vsis3_path(path),
                gdal_env=storage.gdal_env(),
                is_remote=True,
            )
        except Exception:
            return None

    # In explicit b2 mode, prefer /vsis3 over mounted/local files.
    if mode == "b2":
        remote = _remote_source()
        if remote is not None:
            return remote
        if path.exists():
            return RasterSource(uri=str(path), gdal_env={}, is_remote=False)
        return None

    # local or auto: prefer filesystem path first.
    if path.exists():
        return RasterSource(uri=str(path), gdal_env={}, is_remote=False)
    return _remote_source()


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def open_raster(source: RasterSource) -> Iterator[Any]:
    """Open a raster source, reusing a cached handle when available."""
    import rasterio

    uri = source.uri

    with _raster_cache_lock:
        if uri in _raster_cache:
            _raster_cache.move_to_end(uri)
            ds = _raster_cache[uri][0]
        else:
            ds = None

    if ds is None:
        if source.gdal_env:
            with _temporary_env(source.gdal_env):
                with rasterio.Env():
                    ds = rasterio.open(uri)
        else:
            ds = rasterio.open(uri)

        with _raster_cache_lock:
            if uri not in _raster_cache:
                _raster_cache[uri] = (ds, source.gdal_env or {})
                _raster_cache.move_to_end(uri)
                while len(_raster_cache) > _MAX_CACHED_RASTERS:
                    _, (evicted_ds, _) = _raster_cache.popitem(last=False)
                    try:
                        evicted_ds.close()
                    except Exception:
                        pass
            else:
                try:
                    ds.close()
                except Exception:
                    pass
                _raster_cache.move_to_end(uri)
                ds = _raster_cache[uri][0]

    if source.gdal_env:
        with _temporary_env(source.gdal_env):
            with rasterio.Env():
                yield ds
    else:
        yield ds


@contextmanager
def open_raster_path(path: Path) -> Iterator[Any]:
    """Open a raster from local storage or B2-backed /vsis3 fallback."""
    source = resolve_raster_source(path)
    if source is None:
        raise FileNotFoundError(path)
    with open_raster(source) as ds:
        yield ds

@lru_cache(maxsize=1)
def _load_gis_catalog() -> Dict[str, Any]:
    """Loads the GIS catalog JSON into memory.
    
    Args:
        None.
    
    Returns:
        The parsed GIS catalog dictionary with categories and layers.
    """
    with PARQUET.open_input_file(CONFIG.gis_catalog_path) as handle:
        return json.loads(handle.read())


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
        if category.get("name") == "temporal":
            for layer in _expand_temporal_layers(category):
                layer_id = layer.get("id")
                if not layer_id:
                    continue
                layers[str(layer_id)] = layer
            continue
        for layer in category.get("layers", []):
            layer_id = layer.get("id")
            if not layer_id:
                continue
            layers[str(layer_id)] = layer
    return layers


@lru_cache(maxsize=1)
def load_temporal_registry() -> dict[str, Any]:
    """Load temporal registry from the GIS catalog.

    Returns a dict with keys:
      - windows: default windows list
      - layers: list of base temporal layer dicts (id, agg, windows override, etc.)
    """
    catalog = _load_gis_catalog()
    for category in catalog.get("categories", []):
        if category.get("name") == "temporal":
            return {
                "windows": category.get("windows", []),
                "layers": category.get("layers", []),
            }
    return {"windows": [], "layers": []}


def _expand_temporal_layers(category: dict[str, Any]) -> list[dict[str, Any]]:
    windows = category.get("windows", []) or []
    base_layers = category.get("layers", []) or []
    expanded: list[dict[str, Any]] = []
    for layer in base_layers:
        base_id = layer.get("id")
        if not base_id:
            continue
        agg = layer.get("agg") or "avg"
        layer_windows = layer.get("windows") or windows
        display_name = layer.get("display_name") or base_id
        units = layer.get("units")
        value_type = layer.get("value_type") or "numeric"
        code = layer.get("code") or str(base_id).upper()
        if agg == "snapshot":
            expanded.append(
                {
                    "id": base_id,
                    "code": code,
                    "display_name": display_name,
                    "units": units,
                    "value_type": value_type,
                    "region_root": "regions",
                    "region_size": 10,
                    "filename_template": "{id}.tif",
                }
            )
            continue
        for hours in layer_windows:
            expanded.append(
                {
                    "id": f"{base_id}_{agg}_{hours}h",
                    "code": f"{code}_{agg.upper()}_{hours}H",
                    "display_name": f"{display_name} ({agg.capitalize()}, {hours}h)",
                    "units": units,
                    "value_type": value_type,
                    "region_root": "regions",
                    "region_size": 10,
                    "filename_template": "{id}.tif",
                }
            )
    return expanded


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
        if category.get("name") == "temporal":
            for layer in _expand_temporal_layers(category):
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
            continue
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
        with PARQUET.open_input_file(path) as handle:
            payload = json.loads(handle.read())
    except (OSError, json.JSONDecodeError):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for entry in payload.get("classes", []):
        class_id = entry.get("id")
        name = entry.get("name")
        if class_id is None or name is None:
            continue
        slug = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
        code_match = re.search(r"\(([A-Za-z0-9]{2,4})\)", str(name))
        data = {
            "id": class_id,
            "name": name,
            "description": entry.get("description"),
            "group": entry.get("group"),
            "group_label": entry.get("group_label"),
            "traits": entry.get("traits"),
        }
        mapping[str(class_id)] = data
        if slug:
            mapping[slug] = data
        if code_match:
            code = code_match.group(1)
            mapping[code] = data
            mapping[code.lower()] = data
    return mapping


@lru_cache(maxsize=1)
def preload_layer_legends() -> int:
    """Loads all categorical layer legends into memory.
    
    Returns:
        Count of legends loaded.
    """
    layers = load_layer_metadata()
    loaded = 0
    for layer_id, meta in layers.items():
        value_type = str(meta.get("value_type") or "").lower()
        if value_type != "categorical":
            continue
        load_layer_legend(layer_id)
        loaded += 1
    return loaded


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
    if PARQUET.exists(CONFIG.location_hierarchy_path):
        with PARQUET.open_input_file(CONFIG.location_hierarchy_path) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
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

def sample_raster_value(source: RasterSource, lat: float, lon: float) -> Optional[float]:
    try:
        import rasterio
        import rasterio.windows

        # If remote, try to find the file via the B2 mount instead of /vsis3/
        uri = source.uri
        if source.is_remote and uri.startswith("/vsis3/"):
            mount_root = os.environ.get("WW_B2_MOUNT", "/workspace/.b2-mount")
            # Strip /vsis3/bucket/prefix/ to get the relative data path
            storage = _raster_storage()
            bucket = storage.bucket or ""
            prefix = (storage.prefix or "").strip("/")
            vsis3_prefix = f"/vsis3/{bucket}/{prefix}/"
            if uri.startswith(vsis3_prefix):
                rel = uri[len(vsis3_prefix):]
                mount_path = Path(mount_root) / rel
                if mount_path.exists():
                    uri = str(mount_path)

        with rasterio.open(uri) as ds:
            row, col = ds.index(lon, lat)
            window = rasterio.windows.Window(col, row, 1, 1)
            data = ds.read(1, window=window, masked=True)
            return None if data.mask.all() else float(data.data.flat[0])
    except Exception as e:
        print(f"[sample_raster] exception: {e}")
        return None

_DERIVED_DEM_VARIABLES = frozenset({"slope", "aspect", "aspect_deg"})


def sample_dem_derived_value(layer_id: str, lat: float, lon: float) -> Optional[float]:
    """Sample a DEM-derived variable (slope, aspect, aspect_deg) at a point.

    Reads a 3×3 elevation window centred on the coordinate and applies the
    same np.gradient derivation used in tiles._derive_slope_aspect.
    Returns None when the window falls outside the raster or contains nodata.
    """
    import numpy as np
    import rasterio.windows

    cog_path = get_cog_path(layer_id, lat, lon)
    if cog_path is None:
        return None
    source = resolve_raster_source(cog_path)
    if source is None:
        return None

    try:
        with open_raster(source) as ds:
            row, col = ds.index(lon, lat)
            radius = 1
            if (
                row - radius < 0
                or col - radius < 0
                or row + radius >= ds.height
                or col + radius >= ds.width
            ):
                return None
            win = rasterio.windows.Window(col - radius, row - radius, 3, 3)
            data = ds.read(1, window=win).astype(np.float32, copy=False)
            if data.shape != (3, 3):
                return None
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan

            # Resolution in metres — mirrors tiles._resolution_meters logic.
            win_transform = ds.window_transform(win)
            xres = abs(win_transform.a)
            yres = abs(win_transform.e)
            crs_text = str(ds.crs).upper() if ds.crs is not None else ""
            if crs_text in {"EPSG:4326", "OGC:CRS84"}:
                mpd = 111_320.0
                xres_m = xres * mpd * max(0.01, math.cos(math.radians(lat)))
                yres_m = yres * mpd
            else:
                xres_m, yres_m = xres, yres
            if xres_m == 0 or yres_m == 0:
                return None

            dz_dy, dz_dx = np.gradient(data, yres_m, xres_m)
            slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
            raw_aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
            aspect_deg_arr = (90.0 - raw_aspect) % 360.0

            c = 1  # centre of the 3×3 window
            if layer_id == "slope":
                val = float(slope[c, c])
                return None if not math.isfinite(val) else val

            asp = float(aspect_deg_arr[c, c])
            if not math.isfinite(asp):
                return None
            if layer_id == "aspect_deg":
                return asp

            # aspect binned 1–8 — mirrors tiles._aspect_degrees_to_bins
            bin_val = int(math.floor((asp % 360.0 + 22.5) / 45.0) % 8) + 1
            return float(bin_val)
    except Exception as e:
        print(f"[sample_derived] exception: {e}")
        return None


def is_valid_location_gid(gid: Any) -> bool:
    """Returns True when a location token is usable as a GID filter."""
    if gid is None:
        return False
    text = str(gid).strip()
    if not text:
        return False
    return text.lower() not in _INVALID_LOCATION_GID_TOKENS


def location_lookup_for_gid(gid: str) -> tuple[str, str, str]:
    """Returns the location column, scope label, and normalized gid for filtering.
    
    Args:
        gid: The gid of the region being filtered on.
    
    Returns:
        A tuple of (column_name, scope_label, normalized_gid).
    """
    normalized = str(gid).strip()
    if not is_valid_location_gid(normalized):
        raise ValueError(f"Invalid location gid '{gid}'")
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
    try:
        column_name, _scope, target = location_lookup_for_gid(location_gid)
    except ValueError:
        return None
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
    table = _load_location_taxa_table()
    if table is None:
        return {}
    scopes = table.column("scope").to_pylist()
    gids = table.column("gid").to_pylist()
    taxon_ids = table.column("taxon_id").to_pylist()
    for scope, gid, taxon_id in zip(scopes, gids, taxon_ids):
        if not scope or not is_valid_location_gid(gid):
            continue
        try:
            numeric_id = int(taxon_id)
        except (TypeError, ValueError):
            continue
        mapping[(str(scope), str(gid))].add(numeric_id)
    return {key: frozenset(value) for key, value in mapping.items()}


@lru_cache(maxsize=4096)
def location_taxa_for(scope: str, gid: str) -> frozenset[int]:
    """Returns taxon ids known to occur for a single (scope, gid) location key.

    This avoids loading the full location->taxa map on first location query.
    """
    normalized_scope = str(scope or "").strip()
    normalized_gid = str(gid or "").strip()
    if not normalized_scope or not is_valid_location_gid(normalized_gid):
        return frozenset()
    if not PARQUET.exists(CONFIG.location_catalog_path):
        return frozenset()

    try:
        filtered = PARQUET.read_table(
            CONFIG.location_catalog_path,
            columns=["taxon_id"],
            filters=[("scope", "=", normalized_scope), ("gid", "=", normalized_gid)],
        ).combine_chunks()
    except TypeError:
        # Fallback path for readers that don't support parquet filters.
        try:
            table = PARQUET.read_table(
                CONFIG.location_catalog_path,
                columns=["scope", "gid", "taxon_id"],
            ).combine_chunks()
            scope_col = pc.cast(table["scope"], pa.string())
            gid_col = pc.cast(table["gid"], pa.string())
            mask = pc.and_(pc.equal(scope_col, normalized_scope), pc.equal(gid_col, normalized_gid))
            filtered = table.filter(mask).combine_chunks()
        except Exception:
            return frozenset()
    except Exception:
        return frozenset()

    if not filtered.num_rows:
        return frozenset()

    taxon_ids = filtered.column("taxon_id").to_pylist()
    resolved: set[int] = set()
    for taxon_id in taxon_ids:
        try:
            resolved.add(int(taxon_id))
        except (TypeError, ValueError):
            continue
    return frozenset(resolved)


@lru_cache(maxsize=1)
def _load_location_taxa_table() -> pa.Table | None:
    if not PARQUET.exists(CONFIG.location_catalog_path):
        return None
    try:
        return PARQUET.read_table(
            CONFIG.location_catalog_path,
            columns=["scope", "gid", "taxon_id", "count"],
        ).combine_chunks()
    except Exception:
        try:
            return PARQUET.read_table(
                CONFIG.location_catalog_path,
                columns=["scope", "gid", "taxon_id"],
            ).combine_chunks()
        except Exception:
            return None


def location_counts_for_taxon(taxon_id: int) -> Dict[tuple[str, str], int]:
    """Returns per-location observation counts for a taxon.

    For species taxa, counts include infraspecific descendants
    (subspecies/variety/form) so species presence reflects child observations.
    """
    try:
        normalized_taxon_id = int(taxon_id)
    except (TypeError, ValueError):
        return {}
    table = _load_location_taxa_table()
    if table is None or not table.num_rows:
        return {}
    target_taxon_ids: set[int] = {normalized_taxon_id}
    try:
        from util import taxa_navigation

        target_taxon = taxa_navigation.get_taxon_by_id(str(normalized_taxon_id))
        if target_taxon is not None:
            target_rank = taxa_navigation.canonical_rank(target_taxon.get("rank"))
            if target_rank == "SPECIES":
                for descendant in taxa_navigation.iter_descendants(target_taxon):
                    descendant_rank = taxa_navigation.canonical_rank(descendant.get("rank"))
                    if descendant_rank not in CONFIG.subspecies_equivalents:
                        continue
                    descendant_id = taxa_navigation.taxon_id_as_int(descendant.get("taxon_key"))
                    if descendant_id is not None:
                        target_taxon_ids.add(int(descendant_id))
    except Exception:
        # Fall back to direct-only counts if taxonomy lookups fail.
        target_taxon_ids = {normalized_taxon_id}
    try:
        taxon_col = table["taxon_id"]
        if len(target_taxon_ids) == 1:
            scalar = pa.scalar(normalized_taxon_id, type=taxon_col.type)
            mask = pc.equal(taxon_col, scalar)
        else:
            value_set = pa.array(sorted(target_taxon_ids), type=taxon_col.type)
            mask = pc.is_in(taxon_col, value_set=value_set)
        filtered = table.filter(mask).combine_chunks()
    except Exception:
        return {}
    if not filtered.num_rows:
        return {}
    has_count_column = "count" in filtered.column_names
    scopes = filtered.column("scope").to_pylist()
    gids = filtered.column("gid").to_pylist()
    counts = filtered.column("count").to_pylist() if has_count_column else [1] * len(scopes)
    mapping: dict[tuple[str, str], int] = defaultdict(int)
    for scope, gid, count in zip(scopes, gids, counts):
        if not scope or not is_valid_location_gid(gid):
            continue
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            numeric_count = 1
        if numeric_count <= 0:
            continue
        mapping[(str(scope), str(gid))] += numeric_count
    return dict(mapping)


@lru_cache(maxsize=1024)
def location_taxon_counts(
    scope: str,
    gid: str,
    *,
    include_species_rollup: bool = False,
) -> Dict[int, int]:
    """Returns taxon->observation count mapping for a specific location key.

    When include_species_rollup=True, infraspecific counts are also added to
    their parent species taxon id.
    """
    normalized_scope = str(scope or "").strip()
    normalized_gid = str(gid or "").strip()
    if not normalized_scope or not is_valid_location_gid(normalized_gid):
        return {}
    table = _load_location_taxa_table()
    if table is None or not table.num_rows:
        return {}
    try:
        scope_mask = pc.equal(table["scope"], pa.scalar(normalized_scope, type=table["scope"].type))
        gid_mask = pc.equal(table["gid"], pa.scalar(normalized_gid, type=table["gid"].type))
        filtered = table.filter(pc.and_(scope_mask, gid_mask)).combine_chunks()
    except Exception:
        return {}
    if not filtered.num_rows:
        return {}
    has_count_column = "count" in filtered.column_names
    taxon_ids = filtered.column("taxon_id").to_pylist()
    counts = filtered.column("count").to_pylist() if has_count_column else [1] * len(taxon_ids)
    mapping: dict[int, int] = defaultdict(int)
    for taxon_id, count in zip(taxon_ids, counts):
        try:
            numeric_taxon_id = int(taxon_id)
        except (TypeError, ValueError):
            continue
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            numeric_count = 1
        if numeric_count <= 0:
            continue
        mapping[numeric_taxon_id] += numeric_count
    if include_species_rollup and mapping:
        try:
            from util import taxa_navigation

            rolled: dict[int, int] = defaultdict(int)
            for numeric_taxon_id, numeric_count in mapping.items():
                rolled[numeric_taxon_id] += numeric_count
                taxon = taxa_navigation.get_taxon_by_id(str(numeric_taxon_id))
                if taxon is None:
                    continue
                rank = taxa_navigation.canonical_rank(taxon.get("rank"))
                if rank not in CONFIG.subspecies_equivalents:
                    continue
                parent = taxa_navigation.get_parent_taxon(taxon)
                while parent is not None:
                    parent_rank = taxa_navigation.canonical_rank(parent.get("rank"))
                    if parent_rank == "SPECIES":
                        parent_id = taxa_navigation.taxon_id_as_int(parent.get("taxon_key"))
                        if parent_id is not None:
                            rolled[int(parent_id)] += numeric_count
                        break
                    parent = taxa_navigation.get_parent_taxon(parent)
            mapping = rolled
        except Exception:
            pass
    return dict(mapping)


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

"Helper method to strip diacritics"
def strip_diacritics(text: str) -> str:
    if not text:
        return ''
    normalized = unicodedata.normalize('NFD', str(text))
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch)!= 'Mn')
    return stripped.lower().strip()

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
    search_norm = strip_diacritics(search_term)
    entries, mapping = load_location_catalog()
    results: List[dict[str, Any]] = []
    for record in entries:
        name_norm = strip_diacritics(record.name or '')
        if search_norm not in name_norm:
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

def list_children(parent_token: str, level: Optional[int] = None, limit: int = 500) -> List[dict[str, Any]]:
    """
    Return child locations of `parent_token` (gid or name). If level is provided, filter by level.
    """
    entries, by_gid = load_location_catalog()
    parent_token = (parent_token or "").strip()
    if not parent_token:
        return []

    results: List[dict[str, Any]] = []

    # 1) try treat parent_token as gid and return records whose gid starts with parent + '.'
    if '.' in parent_token or parent_token.upper() in by_gid:
        # If the token is exactly a gid key in by_gid, use its gid as parent
        parent_gid = parent_token if parent_token in by_gid else None
        if parent_gid is None:
            # maybe uppercase region codes
            if parent_token.upper() in by_gid:
                parent_gid = parent_token.upper()
        if parent_gid:
            pre = f"{parent_gid}."
            for rec in entries:
                if level is not None and rec.level != level:
                    continue
                if str(rec.gid).startswith(pre):
                    results.append({"gid": rec.gid, "name": rec.name, "level": rec.level, "hierarchy": resolve_location_context(rec, by_gid)})
                    if len(results) >= limit:
                        return results
            # if found some, return
            if results:
                return results

    # 2) fallback: match by name in hierarchy / parent_gid field
    lower = parent_token.lower()
    for rec in entries:
        if level is not None and rec.level != level:
            continue
        # check if parent_gid equals a gid whose name matches
        parent_gid = rec.parent_gid
        if parent_gid and parent_gid in by_gid and by_gid[parent_gid].name.lower() == lower:
            results.append({"gid": rec.gid, "name": rec.name, "level": rec.level, "hierarchy": resolve_location_context(rec, by_gid)})
            if len(results) >= limit:
                return results

    # 3) super-fallback: check if parent_token appears in record.hierarchy names
    if not results:
        for rec in entries:
            if level is not None and rec.level != level:
                continue
            ctx = resolve_location_context(rec, by_gid)
            if any(p.lower() == lower for p in ctx):
                results.append({"gid": rec.gid, "name": rec.name, "level": rec.level, "hierarchy": ctx})
                if len(results) >= limit:
                    return results

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
    layer = _get_layer(layer_id) # this is the JSON object for the layer
    if layer is None:
        raise ValueError(f"Layer {layer_id} not found in catalog")

    # Get the region folder and sample a region from that folder.
    region_root = CONFIG.gis_root / layer["region_root"]
    filename = layer["filename_template"].format(id=layer_id)
    sample_source: Optional[RasterSource] = None

    if region_root.exists():
        sample_dir = next((p for p in sorted(region_root.iterdir()) if p.is_dir()), None)
        if sample_dir is not None:
            sample_path = sample_dir / filename
            sample_source = resolve_raster_source(sample_path)

    if sample_source is None:
        storage = _raster_storage()
        if storage.is_remote and storage.filesystem is not None:
            try:
                selector = pafs.FileSelector(storage.resolve(region_root), recursive=True)
                for info in storage.filesystem.get_file_info(selector):
                    if info.type != pafs.FileType.File:
                        continue
                    if not str(info.path).endswith(f"/{filename}"):
                        continue
                    sample_source = RasterSource(
                        uri=f"/vsis3/{info.path}",
                        gdal_env=storage.gdal_env(),
                        is_remote=True,
                    )
                    break
            except Exception:
                sample_source = None

    if sample_source is None:
        raise FileNotFoundError(f"No regions found for layer {layer_id} in {region_root}")

    # Get the bounds and relevant data of COGs of that layer
    with open_raster(sample_source) as ds:
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
    ids: list[str] = []
    for category in catalog.get("categories", []):
        if category.get("name") == "temporal":
            for layer in _expand_temporal_layers(category):
                layer_id = layer.get("id")
                if layer_id:
                    ids.append(str(layer_id))
            continue
        for layer in category.get("layers", []):
            layer_id = layer.get("id")
            if layer_id:
                ids.append(str(layer_id))
    return ids

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

    if "region_root" not in layer:
        raise KeyError(f"Layer {layer_id} missing region_root in catalog.")
    region_root = layer["region_root"]
    filename = layer["filename_template"].format(id=layer_id)

    tile_id = get_region_name(latitude, longitude)

    return CONFIG.gis_root / region_root / tile_id / filename


def get_cog_source(layer_id: str, latitude: float, longitude: float) -> Optional[RasterSource]:
    """Resolve a layer COG to local path or B2-backed /vsis3 source."""
    cog_path = get_cog_path(layer_id, latitude, longitude)
    if cog_path is None:
        return None
    return resolve_raster_source(cog_path)
