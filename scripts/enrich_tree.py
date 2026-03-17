'''
This script is a sort of an outlier script in the sense that it contains lots of logic rather than using functions defined in libaries.
The reason for this is because it needs to have some special logic and structure to improve its performance.
It basically tries to read COGs in a "cache-friendly" access pattern, where the cache is keeping a COG file open,
rather than opening and closing COGs for each lookup request which has lots of overhead.
'''

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import tempfile
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rasterio.windows import Window
import util.gis_lookup as gis_lookup
import util.taxa_navigation as taxa_navigation
from util.config import load_config

CONFIG = load_config("global")

enrich_tree_row_limit = 10_000_000

DEM_FILENAME = "dem.tif"
DEM_REGION_ROOT = CONFIG.gis_regions_root
DERIVED_DEM_LAYER_METRICS = {
    "slope": "slope",
    "aspect": "aspect",
    "aspect_deg": "aspect_deg",
}


# We require these columns when writing


def _load_layer_ids() -> List[str]:
    with open(CONFIG.gis_catalog_path, "r") as f:
        catalog = json.load(f)
    ids: List[str] = []
    for category in catalog.get("categories", []):
        if category.get("name") == "temporal":
            continue
        for layer in category.get("layers", []):
            layer_id = layer.get("id")
            if layer_id:
                ids.append(str(layer_id))
    if not ids:
        raise RuntimeError("No non-temporal GIS layers defined in the catalog.")
    return ids

# TODO: does this belong in a library? do we want a separate library for parquet manipulation?
def _atomic_write(parquet_path: Path, table: pa.Table) -> None:
    parquet_path = parquet_path.resolve()
    with tempfile.NamedTemporaryFile(
        dir=parquet_path.parent,
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def _missing_rows_for_taxon(taxon, layer_ids: List[str]) -> pa.Table | None:
    '''Calculates the missing GIS rows for a taxon, e.g. what rows have not been populated for a taxon.'''
    data_path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    if not data_path.exists():
        return None
    table = pq.read_table(data_path)
    df = table.to_pandas()
    if df.empty:
        return None
    _cleanup_stale_gis_columns(df, layer_ids, data_path)
    required_columns = list(CONFIG.occurrence_base_columns[:4])
    if any(column not in df.columns for column in required_columns):
        return None
    base = df[required_columns].copy()
    missing_layers = [layer_id for layer_id in layer_ids if layer_id not in df.columns]
    if not missing_layers:
        return None
    subset = base.copy()
    subset["catalogNumber"] = subset["catalogNumber"].astype(str)
    subset["missingLayers"] = [missing_layers] * len(subset)
    subset["taxonKey"] = taxon["taxon_key"]
    subset["dataPath"] = str(data_path)
    return pa.table(
        {
            "catalogNumber": pa.array(
                subset["catalogNumber"].to_numpy(), type=pa.large_string()
            ),
            "tileId": pa.array(subset["tileId"].to_numpy(), type=pa.large_string()),
            "decimalLatitude": pa.array(
                subset["decimalLatitude"].to_numpy(), type=pa.float64()
            ),
            "decimalLongitude": pa.array(
                subset["decimalLongitude"].to_numpy(), type=pa.float64()
            ),
            "missingLayers": pa.array(
                subset["missingLayers"].to_list(), type=pa.list_(pa.large_string())
            ),
            "taxonKey": pa.array(subset["taxonKey"].to_numpy(), type=pa.large_string()),
            "dataPath": pa.array(subset["dataPath"].to_numpy(), type=pa.large_string()),
        }
    )

def _cleanup_stale_gis_columns(df, layer_ids: List[str], data_path: Path) -> None:
    if "gall" not in df.columns:
        return
    try:
        gall_idx = list(df.columns).index("gall")
    except ValueError:
        return
    allowed = set(layer_ids)
    columns = list(df.columns)
    drop_cols: list[str] = []
    for col in columns[gall_idx + 1 :]:
        if col not in allowed:
            drop_cols.append(col)
    if not drop_cols:
        return
    df.drop(columns=drop_cols, inplace=True)
    updated = pa.Table.from_pandas(df, preserve_index=False)
    _atomic_write(Path(data_path), updated)


def _iter_worklist_batches(
    layer_ids: List[str],
    ancestor: str,
    *,
    row_limit: int,
) -> Iterable[pa.Table]:
    """Yields sorted worklist batches capped by row count."""
    ancestor_record = taxa_navigation.get_taxon_by_id(ancestor)
    if ancestor_record is None:
        return
    chunks: list[pa.Table] = []
    total_rows = 0
    batch_rows = 0
    for idx, taxon in enumerate(
        taxa_navigation.iter_descendants(ancestor_record, include_self=True), 1
    ):
        chunk = _missing_rows_for_taxon(taxon, layer_ids)
        if chunk is None or chunk.num_rows == 0:
            continue
        chunks.append(chunk)
        total_rows += chunk.num_rows
        batch_rows += chunk.num_rows
        if idx % 1000 == 0:
            print(f"[worklist] scanned {idx} taxa, captured {total_rows} rows")
        if batch_rows >= row_limit:
            print(f"[worklist] concatenating {len(chunks)} chunks ({batch_rows} rows)")
            worklist = pa.concat_tables(chunks).combine_chunks().sort_by([("tileId", "ascending")])
            print(f"[worklist] batch rows pending GIS lookup: {worklist.num_rows}")
            yield worklist
            chunks = []
            batch_rows = 0
    if not chunks:
        return
    print(f"[worklist] concatenating {len(chunks)} chunks ({batch_rows} rows)")
    worklist = pa.concat_tables(chunks).combine_chunks().sort_by([("tileId", "ascending")])
    print(f"[worklist] batch rows pending GIS lookup: {worklist.num_rows}")
    yield worklist

def _sample_layer_values(
    layer_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    tile_id: str,
) -> List[float | None]:
    '''Samples a list of lats and lons for a certain layer and returns the values in a list.'''
    if lats.size == 0:
        return []
    layer_meta = gis_lookup.load_layer_metadata().get(layer_id)
    if layer_meta is not None and "region_root" not in layer_meta:
        return [None] * len(lats)
    if layer_id == "elevation":
        return _sample_dem_values(lats=lats, lons=lons, tile_id=tile_id)
    derived_metric = DERIVED_DEM_LAYER_METRICS.get(layer_id)
    if derived_metric is not None:
        return _sample_dem_derived_values(
            lats=lats,
            lons=lons,
            tile_id=tile_id,
            metrics=(derived_metric,),
        )[derived_metric]
    ref_lat = float(lats[0])
    ref_lon = float(lons[0])
    cog_source = gis_lookup.get_cog_source(layer_id, ref_lat, ref_lon)
    if cog_source is None:
        return [None] * len(lats)
    coords = list(zip(lons.tolist(), lats.tolist()))
    results: list[float | None] = []
    with gis_lookup.open_raster(cog_source) as ds:
        sampler = ds.sample(coords)
        for point in sampler:
            value = point[0]
            if ds.nodata is not None and value == ds.nodata:
                results.append(None)
            else:
                results.append(float(value))
    return results


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat_rad = np.deg2rad(lat_deg)
    m_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )
    return float(m_per_deg_lat), float(m_per_deg_lon)


def _compute_slope_aspect(
    window: np.ndarray,
    dx_m: float,
    dy_m: float,
) -> Tuple[float, float]:
    z1, z2, z3 = window[0, 0], window[0, 1], window[0, 2]
    z4, _, z6 = window[1, 0], window[1, 1], window[1, 2]
    z7, z8, z9 = window[2, 0], window[2, 1], window[2, 2]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * dx_m)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * dy_m)

    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = float(np.degrees(slope_rad))

    if dzdx == 0 and dzdy == 0:
        aspect_deg = 0.0
    else:
        aspect = np.degrees(np.arctan2(dzdy, -dzdx))
        aspect_deg = float(90.0 - aspect)
        if aspect_deg < 0:
            aspect_deg += 360.0

    return slope_deg, aspect_deg


def _aspect_bin(aspect_deg: float) -> int:
    # 8-bin compass rose. N centered at 0/360.
    if aspect_deg < 0:
        aspect_deg = (aspect_deg % 360.0)
    aspect_deg = aspect_deg % 360.0
    if aspect_deg < 22.5 or aspect_deg >= 337.5:
        return 1  # N
    if aspect_deg < 67.5:
        return 2  # NE
    if aspect_deg < 112.5:
        return 3  # E
    if aspect_deg < 157.5:
        return 4  # SE
    if aspect_deg < 202.5:
        return 5  # S
    if aspect_deg < 247.5:
        return 6  # SW
    if aspect_deg < 292.5:
        return 7  # W
    return 8  # NW


def _sample_dem_derived_values(
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    tile_id: str,
    metrics: Tuple[str, ...],
) -> Dict[str, List[float | None]]:
    results: Dict[str, List[float | None]] = {
        metric: [None] * len(lats) for metric in metrics
    }
    dem_path = DEM_REGION_ROOT / tile_id / DEM_FILENAME
    dem_source = gis_lookup.resolve_raster_source(dem_path)
    if dem_source is None:
        return results

    with gis_lookup.open_raster(dem_source) as ds:
        nodata = ds.nodata
        pixel_width_deg = float(ds.transform.a)
        pixel_height_deg = abs(float(ds.transform.e))
        for idx, (lat, lon) in enumerate(zip(lats.tolist(), lons.tolist())):
            row, col = ds.index(lon, lat)
            window_size = 3
            radius = window_size // 2
            if (
                row - radius < 0
                or col - radius < 0
                or row + radius >= ds.height
                or col + radius >= ds.width
            ):
                continue
            window = ds.read(
                1,
                window=Window(col - radius, row - radius, window_size, window_size),
                boundless=False,
            )
            if window.shape != (window_size, window_size):
                continue
            if nodata is not None and np.any(window == nodata):
                continue
            if np.any(np.isnan(window)):
                continue
            m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat)
            dx_m = pixel_width_deg * m_per_deg_lon
            dy_m = pixel_height_deg * m_per_deg_lat
            if dx_m == 0 or dy_m == 0:
                continue
            center = window_size // 2
            local3 = window[center - 1 : center + 2, center - 1 : center + 2]
            slope_deg, aspect_deg = _compute_slope_aspect(local3, dx_m, dy_m)
            for metric in metrics:
                if metric == "slope":
                    results[metric][idx] = slope_deg
                elif metric == "aspect":
                    results[metric][idx] = float(_aspect_bin(aspect_deg))
                elif metric == "aspect_deg":
                    results[metric][idx] = aspect_deg
    return results


def _sample_dem_values(
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    tile_id: str,
) -> List[float | None]:
    results: List[float | None] = [None] * len(lats)
    dem_path = DEM_REGION_ROOT / tile_id / DEM_FILENAME
    dem_source = gis_lookup.resolve_raster_source(dem_path)
    if dem_source is None:
        return results
    coords = list(zip(lons.tolist(), lats.tolist()))
    with gis_lookup.open_raster(dem_source) as ds:
        nodata = ds.nodata
        sampler = ds.sample(coords)
        for idx, point in enumerate(sampler):
            value = point[0]
            if nodata is not None and value == nodata:
                continue
            results[idx] = float(value)
    return results

def _flush_taxon_updates(
    taxon_key: str,
    data_path: str,
    pending: Dict[str, Dict[str, List[tuple[str, float]]]],
) -> None:
    '''Writes taxon updates into their parquet files. The pending param is complicated nesting, but expanded is:
    pending = {
    "taxon_key_1": {
        "layer_a": [("catalog_001", 12.3), ("catalog_009", 8.1)],
        "layer_b": [("catalog_001", 0.4), ("catalog_009", 2.3)],
    },
    "taxon_key_2": {
        "layer_a": [("catalog_777", 3.7), ("catalog_002", 7.6)]
    }
    }
    '''
    updates = pending.get(taxon_key)
    if not updates:
        return
    data_file = Path(data_path)
    if not data_file.exists():
        del pending[taxon_key]
        return
    table = pq.read_table(data_file)
    df = table.to_pandas()
    if df.empty or "catalogNumber" not in df.columns:
        del pending[taxon_key]
        return
    catalog_series = df["catalogNumber"].astype(str)
    # Index of where in the table to write values
    catalog_index = {value: idx for idx, value in enumerate(catalog_series)}
    # Iter through and write
    for layer_id, entries in updates.items():
        if layer_id not in df.columns:
            df[layer_id] = np.nan
        for catalog, value in entries:
            if value is None:
                continue
            idx = catalog_index.get(str(catalog))
            if idx is None:
                continue
            df.at[idx, layer_id] = float(value)
    updated = pa.Table.from_pandas(df, preserve_index=False)
    _atomic_write(data_file, updated)
    del pending[taxon_key]
    #print(f"[flush] wrote GIS values for taxon {taxon_key}")

def _process_tiles(worklist: pa.Table) -> None:
    df = worklist.to_pandas()
    if df.empty:
        print("No worklist entries to process.")
        return
    if "tileId" not in df.columns:
        print("[process] worklist missing tileId column.")
        return
    missing_tile_mask = df["tileId"].isna() | (df["tileId"].astype(str).str.strip() == "")
    if missing_tile_mask.any():
        print(f"[process] tileId missing for {int(missing_tile_mask.sum())} rows.")
    df.sort_values("tileId", inplace=True)
    total_tiles = df["tileId"].nunique(dropna=True)
    print(f"[process] total tiles to process: {total_tiles}")
    remaining = Counter(df["taxonKey"])
    taxon_paths: dict[str, str] = {}
    # Populate the dict
    for taxon_key, data_path in zip(df["taxonKey"], df["dataPath"]):
        taxon_paths.setdefault(taxon_key, data_path)
    # I don't really know how this works but it does
    pending: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_tiles = df["tileId"].nunique(dropna=True)
    for idx, (tile_id, group) in enumerate(df.groupby("tileId", dropna=True), 1):
        tile_df = group.reset_index(drop=True)
        lats = tile_df["decimalLatitude"].to_numpy(dtype=float)
        lons = tile_df["decimalLongitude"].to_numpy(dtype=float)
        catalogs = tile_df["catalogNumber"].astype(str).to_numpy()
        taxa = tile_df["taxonKey"].astype(str).to_numpy()
        layer_rows: dict[str, list[int]] = defaultdict(list)
        for row_idx, missing_layers in enumerate(tile_df["missingLayers"]):
            if missing_layers is None:
                continue
            if isinstance(missing_layers, (list, tuple, set, np.ndarray)) and len(missing_layers) == 0:
                continue
            for layer_id in missing_layers:
                layer_rows[layer_id].append(row_idx)
        derived_layers = {
            layer_id: DERIVED_DEM_LAYER_METRICS.get(layer_id)
            for layer_id in layer_rows.keys()
            if DERIVED_DEM_LAYER_METRICS.get(layer_id) is not None
        }
        if derived_layers:
            union_indices = sorted(
                {idx for rows in layer_rows.values() for idx in rows}
            )
            union_lats = lats[np.array(union_indices, dtype=int)]
            union_lons = lons[np.array(union_indices, dtype=int)]
            metrics = tuple(sorted(set(derived_layers.values())))
            derived_values = _sample_dem_derived_values(
                lats=union_lats,
                lons=union_lons,
                tile_id=tile_id,
                metrics=metrics,
            )
            for layer_id, metric in derived_layers.items():
                row_indices = layer_rows[layer_id]
                if not row_indices:
                    continue
                index_map = {idx: pos for pos, idx in enumerate(union_indices)}
                for row_idx in row_indices:
                    pos = index_map.get(row_idx)
                    if pos is None:
                        continue
                    value = derived_values[metric][pos]
                    if value is None:
                        continue
                    taxon_key = taxa[row_idx]
                    catalog = catalogs[row_idx]
                    pending[taxon_key][layer_id].append((catalog, value))

        for layer_id, row_indices in layer_rows.items():
            if layer_id in derived_layers:
                continue
            if not row_indices:
                continue
            row_indices_arr = np.array(row_indices, dtype=int)
            layer_lats = lats[row_indices_arr]
            layer_lons = lons[row_indices_arr]
            values = _sample_layer_values(layer_id, layer_lats, layer_lons, tile_id)
            for offset, value in zip(row_indices_arr, values):
                if value is None:
                    continue
                taxon_key = taxa[offset]
                catalog = catalogs[offset]
                pending[taxon_key][layer_id].append((catalog, value))
        counts = tile_df["taxonKey"].value_counts()
        for taxon_key, count in counts.items():
            remaining[taxon_key] -= count
            if remaining[taxon_key] <= 0:
                _flush_taxon_updates(taxon_key, taxon_paths[taxon_key], pending)
                del remaining[taxon_key]
        print(f"processed {idx}/{total_tiles} tiles")
    # flush any remaining taxa (in case of zero rows)
    for taxon_key in list(pending.keys()):
        _flush_taxon_updates(taxon_key, taxon_paths[taxon_key], pending)


def main():
    layer_ids = _load_layer_ids()
    batch_count = 0
    for batch in _iter_worklist_batches(
        layer_ids,
        CONFIG.root_taxon_id,
        row_limit=enrich_tree_row_limit,
    ):
        if batch is None or batch.num_rows == 0:
            continue
        batch_count += 1
        print(f"[worklist] processing batch {batch_count}")
        _process_tiles(batch)
    if batch_count == 0:
        print("All taxa already populated with GIS data.")
        return
    print("Completed GIS lookups for pending taxa.")


if __name__ == "__main__":
    main()
