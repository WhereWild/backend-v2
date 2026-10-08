# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows

# Ensure GDAL block cache is generous for the terrain tile reads.
os.environ.setdefault("GDAL_CACHEMAX", "4096")

from util.temporal import (
    _LAPSE_RATE,
    ELEVATION_CORRECTABLE_VARS,
    _read_model_elevation,
    grid_indices,
)
from util.tiles import (
    _MODEL_GRID_PARAMS,
    LAYERS_DIR,
    TEMPORAL_RASTERS_DIR,
    _load_temporal_npy,
    _load_vector_layer,
    resolve_layer_path,
    wgs84_to_mercator,
)

# ---------------------------------------------------------------------------
# Cached dataset handles
#
# Point-sampling (occurrence enrichment, per-request temporal elevation
# correction) opens the same COGs over and over — for a well-formed COG this
# is normally cheap, but for a very large file like elevation.tif (hundreds
# of GB, millions of tiles) re-parsing the tile-offset tables on every single
# point is a real, avoidable cost. Cache the open handle instead.
#
# rasterio/GDAL datasets aren't safe for concurrent access from multiple
# threads, so reads against a cached handle are serialized per-file with a
# lock (FastAPI serves requests from a threadpool). This still beats
# reopening: the expensive part (parsing the directory) happens once, only
# the read itself is serialized.
# ---------------------------------------------------------------------------

_dataset_cache: dict[str, rasterio.io.DatasetReader] = {}
_dataset_cache_lock = threading.Lock()
_dataset_read_locks: dict[str, threading.Lock] = {}
_dataset_read_locks_mx = threading.Lock()


def _cached_open(path: Path) -> rasterio.io.DatasetReader:
    key = str(path)
    ds = _dataset_cache.get(key)
    if ds is not None:
        return ds
    with _dataset_cache_lock:
        ds = _dataset_cache.get(key)
        if ds is None:
            ds = rasterio.open(path)
            _dataset_cache[key] = ds
        return ds


def _read_lock(path: Path) -> threading.Lock:
    key = str(path)
    lock = _dataset_read_locks.get(key)
    if lock is not None:
        return lock
    with _dataset_read_locks_mx:
        lock = _dataset_read_locks.setdefault(key, threading.Lock())
        return lock


def clear_dataset_cache() -> None:
    """Close and drop all cached dataset handles.

    Call after a data rebuild/reload so stale handles from before the swap
    don't linger (mirrors the other .cache_clear() calls in /internal/reload).
    """
    with _dataset_cache_lock:
        for ds in _dataset_cache.values():
            try:
                ds.close()
            except Exception:
                pass
        _dataset_cache.clear()


def sample_point(layer: dict, lat: float, lon: float, forecast_suffix: str = "") -> float | None:
    """Return the raster value for a layer at a lat/lon coordinate.

    For static COG layers: opens the file and reads a single pixel, applying
    scale/offset from the catalog.  For temporal layers: samples the .npy grid
    for the given forecast offset (empty string = current window).
    Returns None for nodata or out-of-bounds.
    """
    if layer.get("window_hours") is not None:
        return _sample_temporal_point(layer, lat, lon, forecast_suffix)
    if layer["id"] == "slope":
        return compute_slope_at_point(lat, lon)
    if layer["id"] == "aspect":
        return compute_aspect_at_point(lat, lon)
    if layer["id"] == "heat_load":
        return compute_heat_load_at_point(lat, lon)
    if layer["id"] == "convergence":
        return compute_convergence_at_point(lat, lon)
    if layer["id"] == "plan_curvature":
        return compute_plan_curvature_at_point(lat, lon)
    if layer["id"] == "profile_curvature":
        return compute_profile_curvature_at_point(lat, lon)
    if layer["id"] == "soil_texture":
        return compute_soil_texture_at_point(lat, lon)
    if layer.get("vector_field"):
        return _sample_vector_point(layer, lat, lon)
    return _sample_cog_point(layer, lat, lon)


def _sample_vector_point(layer: dict, lat: float, lon: float) -> float | None:
    """Point-in-polygon lookup for a native vector layer source (e.g. ecoregions/biome).

    Uses the same cached, full-detail (non-simplified) GeoDataFrame + spatial
    index util.tiles builds for tile rendering's high-zoom fallback — full
    precision matters here since this backs both the live /gis/point API and
    (via scripts/enrich_tree.py's equivalent) the canonical enrichment data
    written to occurrences.parquet, unlike tile rendering where a
    zoom-appropriate simplified copy is the right tradeoff.
    """
    path = resolve_layer_path(LAYERS_DIR, layer["filename"])
    val = sample_vector_batch(path, layer["vector_field"], np.array([lat]), np.array([lon]))[0]
    return None if np.isnan(val) else float(val)


_VECTOR_BATCH_CHUNK_SIZE = 5000


def sample_vector_batch(path: Path, field: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Point-in-polygon sample a native vector layer source (e.g. ecoregions/
    biome) for a batch of coordinates. Returns float64 array (NaN = no match).

    Shared by scripts/enrich_tree.py (tree enrichment) and util/upload.py
    (custom observation upload enrichment) — one implementation instead of
    two independently-written point-in-polygon samplers.

    Callers pass points in hilbert-curve order (enrich_tree.py sorts the
    whole batch by hilbertIdx before sampling any layer — see
    _process_batch), so consecutive points here are almost always spatially
    tight clusters. Exploits that: instead of querying the spatial index
    once per point (each paying its own independent tree descent even
    though a whole run of nearby points is likely to land in the very same
    one or two polygons), points are processed in contiguous chunks — one
    cheap bbox query per chunk narrows the candidate-polygon set, then only
    those few candidates get the real exact intersects test against the
    whole chunk. Same exact GEOS test either way, just against a much
    smaller, locality-derived candidate set instead of re-deriving it from
    scratch per point. If the input isn't actually hilbert-ordered (e.g. a
    scattered batch), this degrades gracefully to roughly the same cost as
    the flat per-point query, never worse in correctness.

    A point can land on a shared edge and match >1 polygon; only the first
    matching candidate (in the chunk's query order) is kept (harmless —
    both polygons carry the same boundary, so either's attribute value is
    correct there).
    """
    import shapely

    n = len(lats)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    gdf = _load_vector_layer(path)
    if gdf is None or gdf.empty:
        return out

    x, y = wgs84_to_mercator(lons, lats)
    geoms = gdf.geometry.values
    field_vals = gdf[field].to_numpy(dtype=np.float64)

    for start in range(0, n, _VECTOR_BATCH_CHUNK_SIZE):
        end = min(start + _VECTOR_BATCH_CHUNK_SIZE, n)
        cx, cy = x[start:end], y[start:end]
        bbox = shapely.box(cx.min(), cy.min(), cx.max(), cy.max())
        candidates = gdf.sindex.query(bbox, predicate="intersects")
        if candidates.size == 0:
            continue

        chunk_pts = shapely.points(cx, cy)
        chunk_out = np.full(end - start, np.nan, dtype=np.float64)
        unresolved = np.ones(end - start, dtype=bool)
        for cand_idx in candidates:
            if not unresolved.any():
                break
            hit = shapely.intersects(geoms[cand_idx], chunk_pts) & unresolved
            if hit.any():
                chunk_out[hit] = field_vals[cand_idx]
                unresolved &= ~hit
        out[start:end] = chunk_out

    return out


def _sample_cog_point(layer: dict, lat: float, lon: float) -> float | None:
    path = resolve_layer_path(LAYERS_DIR, layer["filename"])
    scale = layer.get("scale_factor") or 1.0
    offset = layer.get("add_offset") or 0.0
    try:
        ds = _cached_open(path)
        with _read_lock(path):
            row, col = ds.index(lon, lat)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                return None
            window = rasterio.windows.Window(col, row, 1, 1)
            data = ds.read(1, window=window, masked=True)
        if data.mask.all():
            return None
        raw = float(data.data.flat[0])
        if np.issubdtype(data.dtype, np.integer) and ds.nodata is not None:
            dtype_max = np.iinfo(data.dtype).max
            nd_int = round(ds.nodata)
            if raw == nd_int or raw >= dtype_max - 3:
                return None
        return raw * scale + offset
    except Exception:
        return None


def _sample_temporal_point(layer: dict, lat: float, lon: float, forecast_suffix: str = "") -> float | None:
    """Sample a temporal .npy at a lat/lon for the given forecast offset."""
    var_id = layer["var_id"]
    window_label = layer["window_label"]
    model = layer.get("model", "copernicus_era5")

    arr = _load_temporal_npy(TEMPORAL_RASTERS_DIR / f"{var_id}_{window_label}{forecast_suffix}.npy")
    if arr is None:
        return None

    shape_to_model = {(721, 1440): "copernicus_era5", (1801, 3600): "copernicus_era5_land"}
    grid = _MODEL_GRID_PARAMS.get(shape_to_model.get(arr.shape, model), _MODEL_GRID_PARAMS["copernicus_era5"])

    row = round((lat - grid["lat_min"]) / (grid["lat_max"] - grid["lat_min"]) * (grid["ny"] - 1))
    col = round((lon - grid["lon_min"]) / (grid["lon_max"] - grid["lon_min"]) * (grid["nx"] - 1))
    if not (0 <= row < grid["ny"] and 0 <= col < grid["nx"]):
        return None

    val = float(arr[row, col])
    if not np.isfinite(val):
        return None

    if var_id in ELEVATION_CORRECTABLE_VARS:
        step = layer.get("grid_step", 0.25)
        mode = layer.get("grid_mode", "lat_asc_lon_pm180")
        val = _apply_point_elevation_correction(val, lat, lon, model, step, mode)

    return val


def _apply_point_elevation_correction(
    val: float, lat: float, lon: float, model: str, step: float, mode: str,
) -> float:
    """Apply lapse-rate correction to a temporal value at a single point.

    Samples the elevation COG for the true surface elevation, then looks up
    the model's smoothed grid elevation (HSURF) at the same grid cell.
    Correction = (model_elev - obs_elev) * LAPSE_RATE.
    Returns val unchanged if either elevation is unavailable.
    """
    elev_layer = resolve_layer_path(LAYERS_DIR, "elevation.tif")
    if not elev_layer.exists():
        return val

    try:
        ds = _cached_open(elev_layer)
        with _read_lock(elev_layer):
            r, c = ds.index(lon, lat)
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                return val
            window = rasterio.windows.Window(c, r, 1, 1)
            data = ds.read(1, window=window, masked=True)
        if data.mask.all():
            return val
        obs_elev = float(data.data.flat[0])
    except Exception:
        return val

    if not np.isfinite(obs_elev) or obs_elev <= -9000:
        return val

    ny = int(round(180.0 / step)) + 1
    nx = int(round(360.0 / step)) + 1
    li, lo = grid_indices(lat, lon, ny, nx, mode, step)
    lat_arr = np.array([li], dtype=np.int32)
    lon_arr = np.array([lo], dtype=np.int32)
    model_elev = float(_read_model_elevation(model, lat_arr, lon_arr)[0])
    if not np.isfinite(model_elev):
        return val

    return val + (model_elev - obs_elev) * _LAPSE_RATE


# ---------------------------------------------------------------------------
# Terrain derivation helpers
# ---------------------------------------------------------------------------

_EARTH_A = 6378137.0        # WGS84 semi-major axis (m)
_EARTH_B = 6356752.3142     # WGS84 semi-minor axis (m)

# Layers derived on-the-fly from elevation.tif (no separate COG stored).
DERIVED_FROM_ELEVATION: frozenset[str] = frozenset(
    {"slope", "aspect", "heat_load", "convergence", "plan_curvature", "profile_curvature"},
)

# Slope/aspect match gdaldem's (Horn 1981 3×3 kernel, aspect undefined where
# the gradient is exactly zero, nodata neighbours replaced by the centre cell
# as with -compute_edges) — the tool Geomorpho90m used for its slope/aspect
# layers — so the point sampler and the tile renderer agree with each other
# and with gdaldem over the same pixels.


def _meters_per_degree_np(lat):
    """Vectorised (m_per_deg_lat, m_per_deg_lon) on the WGS84 ellipsoid."""
    lat_rad = np.radians(lat)
    m_per_deg_lat = (
        111132.954
        - 559.822 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )
    return m_per_deg_lat, m_per_deg_lon


def _meters_per_degree(lat: float) -> tuple[float, float]:
    """Return (m_per_deg_lat, m_per_deg_lon) at the given latitude."""
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree_np(lat)
    return float(m_per_deg_lat), float(m_per_deg_lon)


def _horn_gradient(dem: np.ndarray, dx_m, dy_m) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) partial derivatives of a DEM with NaN as nodata.

    Returns (dz/dx toward east, dz/dy toward north), NaN wherever the centre
    cell is NaN. A NaN neighbour takes the centre cell's value, as gdaldem
    -compute_edges does — never a fill value from elsewhere in the array,
    which would put fake cliffs along coastlines and data edges. Off-array
    neighbours are treated the same way (gdaldem extrapolates there instead;
    callers pad their reads, so the outermost ring is never displayed).
    `dx_m`/`dy_m` may be scalars or per-row (rows, 1) arrays so geographic
    grids get each row's own cell width instead of one latitude for all.
    """
    dem = np.asarray(dem, dtype=np.float64)
    z1, z2, z3, z4, _, z6, z7, z8, z9 = _neighbours(dem)
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * dx_m)
    # Row 0 is north, so top-row minus bottom-row is the northward rise.
    dzdy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8.0 * dy_m)
    centre_nan = np.isnan(dem)
    return np.where(centre_nan, np.nan, dzdx), np.where(centre_nan, np.nan, dzdy)


def _neighbours(dem: np.ndarray) -> tuple[np.ndarray, ...]:
    """The 3×3 window around every cell as nine arrays z1..z9 (row-major,
    z1 = NW, z5 = centre, z9 = SE). A NaN or off-array neighbour takes the
    centre cell's value, as gdaldem -compute_edges does."""
    h, w = dem.shape
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)

    def nb(dr: int, dc: int) -> np.ndarray:
        a = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        return np.where(np.isnan(a), dem, a)

    return tuple(nb(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _curvatures(dem: np.ndarray, dx_m, dy_m, dzdx, dzdy) -> tuple[np.ndarray, np.ndarray]:
    """Plan (tangential) and profile curvature, m⁻¹, from a DEM and its Horn
    first derivatives.

    Second derivatives are Evans-Young's (Evans 1979) least-squares fit over
    the 3×3 window, with each cell's own east-west and north-south size in
    metres, so a geographic grid's non-square cells are handled like
    slope/aspect handle them. Profile curvature is along the slope line.
    Plan curvature is the tangential form (the normal section through the
    contour direction, as GRASS r.slope.aspect's tcurvature), not the pure
    contour curvature: contour curvature divides by the gradient cubed and
    blows up on near-flat ground (valley floors measured wider than the
    Wasatch on FABDEM), while tangential is contour curvature × sin(slope)
    and fades to zero there (Florinsky 2016). Sign: negative is
    concave — flow decelerating (profile) or converging (plan), where water
    collects — and positive is convex. Undefined (NaN) where the gradient is
    exactly zero, like aspect.
    """
    dem = np.asarray(dem, dtype=np.float64)
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _neighbours(dem)
    r = (z1 + z3 + z4 + z6 + z7 + z9 - 2 * (z2 + z5 + z8)) / (3.0 * dx_m * dx_m)
    t = (z1 + z2 + z3 + z7 + z8 + z9 - 2 * (z4 + z5 + z6)) / (3.0 * dy_m * dy_m)
    # y increases northward (row 0 is north), matching dzdy.
    s = (z3 - z1 - z9 + z7) / (4.0 * dx_m * dy_m)
    p, q = dzdx, dzdy
    g2 = p * p + q * q
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = -(p * p * r + 2 * p * q * s + q * q * t) / (g2 * (1 + g2) ** 1.5)
        plan = -(q * q * r - 2 * p * q * s + p * p * t) / (g2 * np.sqrt(1 + g2))
    undefined = (g2 == 0) | np.isnan(dem)
    return np.where(undefined, np.nan, plan), np.where(undefined, np.nan, profile)


def _convergence_index(dem: np.ndarray, dx_m, dy_m) -> np.ndarray:
    """Convergence index (Köthe & Lehmeier 1996, as SAGA's), scaled to ±100.

    For each of the 8 neighbours, the angle between its downslope aspect and
    the bearing from it back to the centre cell; the index is the mean angle
    minus 90°, scaled by 100/90. Bearings use ground-metre offsets (each
    cell's own dx/dy), so diagonals point where they really do on a
    geographic grid. Sign follows SAGA and our curvatures: negative where
    the neighbours drain toward the cell (converging, -100 in a pit),
    positive where they drain away (diverging, +100 on a peak) — the
    opposite of Geomorpho90m's r.convergence. Flat neighbours (aspect
    undefined) are left out; NaN if the centre is nodata or no neighbour has
    an aspect. Needs a 5×5 window around a cell (its neighbours' aspects).
    """
    dem = np.asarray(dem, dtype=np.float64)
    dzdx, dzdy = _horn_gradient(dem, dx_m, dy_m)
    aspect = _aspect_deg(dzdx, dzdy)
    h, w = dem.shape
    padded = np.pad(aspect, 1, mode="constant", constant_values=np.nan)
    angles = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb_aspect = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            # Neighbour (dr, dc) -> centre: east by -dc cells, north by +dr cells.
            bearing = np.degrees(np.arctan2(-dc * dx_m, dr * dy_m)) % 360.0
            angles.append(np.abs((nb_aspect - bearing + 180.0) % 360.0 - 180.0))
    stack = np.stack(np.broadcast_arrays(*angles))
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid, stack, 0.0).sum(axis=0) / count
    ci = (mean - 90.0) * (100.0 / 90.0)
    return np.where((count == 0) | np.isnan(dem), np.nan, ci)


def _slope_deg(dzdx, dzdy):
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def _aspect_deg(dzdx, dzdy):
    """Compass bearing (N=0, clockwise) of the downslope direction; NaN if flat."""
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    return np.where((dzdx == 0) & (dzdy == 0), np.nan, aspect)


def _horn_slope(patch: np.ndarray, dx_m: float, dy_m: float) -> float:
    dzdx, dzdy = _horn_gradient(patch, dx_m, dy_m)
    return float(_slope_deg(dzdx[1, 1], dzdy[1, 1]))


def _horn_aspect(patch: np.ndarray, dx_m: float, dy_m: float) -> float | None:
    dzdx, dzdy = _horn_gradient(patch, dx_m, dy_m)
    aspect = float(_aspect_deg(dzdx[1, 1], dzdy[1, 1]))
    return None if np.isnan(aspect) else aspect


def _heat_load(slope_deg, aspect_deg, lat_deg):
    """McCune & Keon (2002) heat load index: potential annual direct incident
    radiation (MJ cm⁻² yr⁻¹) from slope, aspect and latitude.

    Aspect is folded about the hemisphere's warmest direction (SW = 225° in
    the north, NW = 315° in the south) so 0 is the coolest face and 180° the
    warmest, and latitude enters as its absolute value, mirroring the
    northern-hemisphere fit. Flat ground (aspect NaN) is still defined: every
    aspect term is multiplied by sin(slope) = 0 there. The equation was fitted
    on northern mid-latitudes; FABDEM's 60°S–80°N coverage is extrapolation
    at its poleward ends.
    """
    slope = np.radians(slope_deg)
    lat = np.asarray(lat_deg, dtype=np.float64)
    warmest = np.where(lat < 0, 315.0, 225.0)
    aspect = np.where(np.isnan(aspect_deg), 0.0, aspect_deg)
    folded = np.radians(np.abs(180.0 - np.abs(aspect - warmest)))
    lat_r = np.radians(np.abs(lat))
    ln_rad = (
        -1.467
        + 1.582 * np.cos(lat_r) * np.cos(slope)
        - 1.5 * np.cos(folded) * np.sin(slope) * np.sin(lat_r)
        - 0.262 * np.sin(lat_r) * np.sin(slope)
        + 0.607 * np.sin(folded) * np.sin(slope)
    )
    return np.exp(ln_rad)


def sample_elevation_terrain_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    want_elevation: bool = False,
    want_slope: bool = False,
    want_aspect: bool = False,
    want_heat_load: bool = False,
    want_convergence: bool = False,
    want_plan_curvature: bool = False,
    want_profile_curvature: bool = False,
) -> dict[str, list[float | None]]:
    """Sample elevation and/or any elevation-derived terrain value in a single
    pass over elevation.tif.

    Opens the file once and reads one 3×3 window per point, computing all
    requested outputs from the same read.  Points should be hilbert-sorted
    for GDAL block-cache locality.

    Returns a dict with keys matching the requested outputs. Aspect is None
    on perfectly flat ground, where it is undefined.
    """
    n = len(lats)
    elev_path = resolve_layer_path(LAYERS_DIR, "elevation.tif")
    results: dict[str, list[float | None]] = {}
    if want_elevation:
        results["elevation"] = [None] * n
    if want_slope:
        results["slope"] = [None] * n
    if want_aspect:
        results["aspect"] = [None] * n
    if want_heat_load:
        results["heat_load"] = [None] * n
    if want_convergence:
        results["convergence"] = [None] * n
    if want_plan_curvature:
        results["plan_curvature"] = [None] * n
    if want_profile_curvature:
        results["profile_curvature"] = [None] * n
    if not results or not elev_path.exists() or n == 0:
        return results
    try:
        with rasterio.open(elev_path) as ds:
            nodata = ds.nodata
            pixel_deg = abs(ds.transform.a)
            h, w = ds.height, ds.width
            need_curvature = want_plan_curvature or want_profile_curvature
            need_terrain = want_slope or want_aspect or want_heat_load or need_curvature or want_convergence
            # Convergence needs the neighbours' aspects, so a 5×5 window;
            # everything else only uses the centre's 3×3.
            half = 2 if want_convergence else 1
            for i, (lat, lon) in enumerate(zip(lats.tolist(), lons.tolist())):
                try:
                    row, col = ds.index(lon, lat)
                    if need_terrain:
                        if row < half or col < half or row >= h - half or col >= w - half:
                            continue
                        size = 2 * half + 1
                        win = rasterio.windows.Window(col - half, row - half, size, size)
                        patch = ds.read(1, window=win).astype(np.float64)
                        if patch.shape != (size, size):
                            continue
                        if nodata is not None:
                            patch[patch == nodata] = np.nan
                        patch[~np.isfinite(patch)] = np.nan
                        c = half
                        if np.isnan(patch[c, c]):
                            continue
                        if want_elevation:
                            results["elevation"][i] = float(patch[c, c])
                        m_lat, m_lon = _meters_per_degree(lat)
                        dx_m = pixel_deg * m_lon
                        dy_m = pixel_deg * m_lat
                        if dx_m == 0 or dy_m == 0:
                            continue
                        dzdx, dzdy = _horn_gradient(patch, dx_m, dy_m)
                        slope = float(_slope_deg(dzdx[c, c], dzdy[c, c]))
                        aspect = float(_aspect_deg(dzdx[c, c], dzdy[c, c]))
                        if want_slope:
                            results["slope"][i] = slope
                        if want_aspect:
                            results["aspect"][i] = None if np.isnan(aspect) else aspect
                        if want_heat_load:
                            results["heat_load"][i] = float(_heat_load(slope, aspect, lat))
                        if want_convergence:
                            ci = _convergence_index(patch, dx_m, dy_m)[c, c]
                            if np.isfinite(ci):
                                results["convergence"][i] = float(ci)
                        if need_curvature:
                            plan, profile = _curvatures(patch, dx_m, dy_m, dzdx, dzdy)
                            if want_plan_curvature and np.isfinite(plan[c, c]):
                                results["plan_curvature"][i] = float(plan[c, c])
                            if want_profile_curvature and np.isfinite(profile[c, c]):
                                results["profile_curvature"][i] = float(profile[c, c])
                    else:
                        # elevation only — single-pixel read is enough
                        if row < 0 or col < 0 or row >= h or col >= w:
                            continue
                        win = rasterio.windows.Window(col, row, 1, 1)
                        val = ds.read(1, window=win)
                        v = float(val.flat[0])
                        if nodata is not None and v == nodata:
                            continue
                        if np.isfinite(v):
                            results["elevation"][i] = v
                except Exception:
                    continue
    except Exception:
        pass
    return results


def sample_slope_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute slope (degrees) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_slope=True).get("slope", [None] * len(lats))


def compute_slope_at_point(lat: float, lon: float) -> float | None:
    """Compute slope (degrees) at a single lat/lon."""
    return sample_slope_batch(np.array([lat]), np.array([lon]))[0]


def sample_aspect_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute aspect (°, N=0 clockwise) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_aspect=True).get("aspect", [None] * len(lats))


def compute_aspect_at_point(lat: float, lon: float) -> float | None:
    """Compute aspect (°, N=0 clockwise) at a single lat/lon."""
    return sample_aspect_batch(np.array([lat]), np.array([lon]))[0]


def sample_heat_load_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute heat load (MJ cm⁻² yr⁻¹) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_heat_load=True).get("heat_load", [None] * len(lats))


def compute_heat_load_at_point(lat: float, lon: float) -> float | None:
    """Compute heat load (MJ cm⁻² yr⁻¹) at a single lat/lon."""
    return sample_heat_load_batch(np.array([lat]), np.array([lon]))[0]


def sample_convergence_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute the convergence index (±100) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_convergence=True).get(
        "convergence", [None] * len(lats),
    )


def compute_convergence_at_point(lat: float, lon: float) -> float | None:
    """Compute the convergence index (±100) at a single lat/lon."""
    return sample_convergence_batch(np.array([lat]), np.array([lon]))[0]


def sample_plan_curvature_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute plan curvature (m⁻¹) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_plan_curvature=True).get(
        "plan_curvature", [None] * len(lats),
    )


def compute_plan_curvature_at_point(lat: float, lon: float) -> float | None:
    """Compute plan curvature (m⁻¹) at a single lat/lon."""
    return sample_plan_curvature_batch(np.array([lat]), np.array([lon]))[0]


def sample_profile_curvature_batch(lats: np.ndarray, lons: np.ndarray) -> list[float | None]:
    """Compute profile curvature (m⁻¹) for many points with a single file open."""
    return sample_elevation_terrain_batch(lats, lons, want_profile_curvature=True).get(
        "profile_curvature", [None] * len(lats),
    )


def compute_profile_curvature_at_point(lat: float, lon: float) -> float | None:
    """Compute profile curvature (m⁻¹) at a single lat/lon."""
    return sample_profile_curvature_batch(np.array([lat]), np.array([lon]))[0]


def _row_latitudes(dem: np.ndarray, transform) -> np.ndarray:
    rows = np.arange(dem.shape[0], dtype=np.float64)
    return transform.f + (rows + 0.5) * transform.e


def _array_cell_sizes(dem: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (dx_m, dy_m) as (rows, 1) arrays for a north-up EPSG:4326 grid."""
    row_lat = _row_latitudes(dem, transform)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree_np(row_lat)
    return (abs(transform.a) * m_per_deg_lon)[:, None], (abs(transform.e) * m_per_deg_lat)[:, None]


def _array_gradient(dem: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Horn gradient of a north-up EPSG:4326 DEM array, per-row cell sizes."""
    return _horn_gradient(dem, *_array_cell_sizes(dem, transform))


def derive_slope_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive slope (degrees) from a 2-D EPSG:4326 DEM array (Horn kernel).

    NaN cells in `dem` propagate to NaN in the output.
    Returns a float32 array of the same shape as `dem`.
    """
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    dzdx, dzdy = _array_gradient(dem, transform)
    return _slope_deg(dzdx, dzdy).astype(np.float32)


def derive_aspect_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive aspect (°, N=0 clockwise) from a 2-D EPSG:4326 DEM array (Horn kernel).

    NaN cells and perfectly flat cells (aspect undefined) are NaN.
    Returns a float32 array of the same shape as `dem`.
    """
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    dzdx, dzdy = _array_gradient(dem, transform)
    return _aspect_deg(dzdx, dzdy).astype(np.float32)


def derive_heat_load_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive heat load (MJ cm⁻² yr⁻¹) from a 2-D EPSG:4326 DEM array, using
    each row's own latitude. NaN cells propagate to NaN in the output.
    Returns a float32 array of the same shape as `dem`.
    """
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    dzdx, dzdy = _array_gradient(dem, transform)
    lat = _row_latitudes(dem, transform)[:, None]
    return _heat_load(_slope_deg(dzdx, dzdy), _aspect_deg(dzdx, dzdy), lat).astype(np.float32)


def _array_curvatures(dem: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    dx_m, dy_m = _array_cell_sizes(dem, transform)
    dzdx, dzdy = _horn_gradient(dem, dx_m, dy_m)
    return _curvatures(dem, dx_m, dy_m, dzdx, dzdy)


def derive_convergence_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive the convergence index (±100, negative = converging) from a 2-D
    EPSG:4326 DEM array, with per-row ground-metre neighbour offsets.
    Returns float32."""
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    return _convergence_index(dem, *_array_cell_sizes(dem, transform)).astype(np.float32)


def derive_plan_curvature_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive plan curvature (m⁻¹, negative = converging) from a 2-D EPSG:4326
    DEM array. NaN where nodata or the gradient is zero. Returns float32."""
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    return _array_curvatures(dem, transform)[0].astype(np.float32)


def derive_profile_curvature_array(dem: np.ndarray, transform) -> np.ndarray:
    """Derive profile curvature (m⁻¹, negative = concave/decelerating) from a
    2-D EPSG:4326 DEM array. NaN where nodata or the gradient is zero.
    Returns float32."""
    if not np.any(np.isfinite(dem)):
        return np.full(dem.shape, np.nan, dtype=np.float32)
    return _array_curvatures(dem, transform)[1].astype(np.float32)


# Array deriver for each DERIVED_FROM_ELEVATION layer (map tiles, chaining).
ELEVATION_DERIVERS = {
    "slope": derive_slope_array,
    "aspect": derive_aspect_array,
    "heat_load": derive_heat_load_array,
    "convergence": derive_convergence_array,
    "plan_curvature": derive_plan_curvature_array,
    "profile_curvature": derive_profile_curvature_array,
}


# ---------------------------------------------------------------------------
# Soil texture derivation helpers
# ---------------------------------------------------------------------------

# Layers derived on-the-fly from sand/silt/clay COGs (no separate COG stored).
DERIVED_FROM_SOIL: frozenset[str] = frozenset({"soil_texture"})

# SoilGrids stores sand/silt/clay as integer g/kg *10 — must match catalog.json's
# scale_factor for the sand/silt/clay layers.
_SOILGRIDS_SCALE = 0.1
_SOIL_TEXTURE_INPUT_FILES = {"sand": "sand.tif", "silt": "silt.tif", "clay": "clay.tif"}

# Canonical USDA texture classes, ordered to match config/gis/legends/soil_texture_legend.json.
USDA_TEXTURE_CLASSES: list[tuple[int, str]] = [
    (1, "clay"), (2, "sandy clay"), (3, "silty clay"),
    (4, "clay loam"), (5, "sandy clay loam"), (6, "silty clay loam"),
    (7, "loam"), (8, "sandy loam"), (9, "silt loam"),
    (10, "silt"), (11, "loamy sand"), (12, "sand"),
]
_USDA_TEXTURE_LABEL_TO_ID: dict[str, int] = {label: cid for cid, label in USDA_TEXTURE_CLASSES}


def usda_texture_class(sand: float, silt: float, clay: float) -> str:
    """Classify a sand/silt/clay percentage triple into a USDA texture class.

    Conditions taken directly from NRCS Soil Texture Calculator
    (MultiPointTriangle_100pt_508.xlsx, Calc1 sheet).
    """
    if clay >= 40:
        if silt >= 40:
            return "silty clay"
        if sand <= 45:
            return "clay"
        return "sandy clay"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 27:
        if sand <= 20:
            return "silty clay loam"
        if sand <= 45:
            return "clay loam"
        return "sandy clay loam"
    if clay >= 20 and sand > 45 and silt < 28:
        return "sandy clay loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if silt >= 50:
        return "silt loam"
    if clay >= 7 and silt >= 28 and sand <= 52:
        return "loam"
    if (silt + 1.5 * clay) < 15:
        return "sand"
    if (silt + 1.5 * clay) >= 15 and (silt + 2 * clay) < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52 and (silt + 2 * clay) >= 30) or \
       (clay < 7 and (silt + 2 * clay) >= 30):
        return "sandy loam"
    return "sandy loam"


def derive_soil_texture_array(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray) -> np.ndarray:
    """Vectorized USDA texture classification, returning class ids (see USDA_TEXTURE_CLASSES).

    Mirrors usda_texture_class's if/elif chain via np.select, whose first-match-wins
    semantics reproduce the same priority order. NaN inputs produce NaN output.
    Returns a float32 array of the same shape as the inputs.
    """
    s, si, c = sand, silt, clay
    label_order = [
        "silty clay", "clay", "sandy clay", "sandy clay", "silty clay loam",
        "clay loam", "sandy clay loam", "sandy clay loam", "silt", "silt loam",
        "loam", "sand", "loamy sand", "sandy loam",
    ]
    conditions = [
        (c >= 40) & (si >= 40),
        (c >= 40) & (s <= 45),
        (c >= 40),
        (c >= 35) & (s > 45),
        (c >= 27) & (s <= 20),
        (c >= 27) & (s <= 45),
        (c >= 27),
        (c >= 20) & (s > 45) & (si < 28),
        (si >= 80) & (c < 12),
        (si >= 50),
        (c >= 7) & (si >= 28) & (s <= 52),
        (si + 1.5 * c) < 15,
        ((si + 1.5 * c) >= 15) & ((si + 2 * c) < 30),
        (((c >= 7) & (c < 20) & (s > 52) & ((si + 2 * c) >= 30)) | ((c < 7) & ((si + 2 * c) >= 30))),
    ]
    choices = [_USDA_TEXTURE_LABEL_TO_ID[label] for label in label_order]
    ids = np.select(conditions, choices, default=_USDA_TEXTURE_LABEL_TO_ID["sandy loam"])
    result = ids.astype(np.float32)
    finite = np.isfinite(s) & np.isfinite(si) & np.isfinite(c)
    result[~finite] = np.nan
    return result


# Maps a catalog `composition_group` id (see config/gis/catalog.json) to the
# classifier function used to build its ternary classification overlay (see
# util.ternary.build_ternary_classification_overlay). A compositional
# variable with no entry here just gets a density blob, no class ids/legend —
# this is the only soil-specific line in that path.
COMPOSITION_CLASSIFIERS = {"soil_texture": derive_soil_texture_array}


def sample_soil_texture_batch(lats: np.ndarray, lons: np.ndarray) -> list[int | None]:
    """Compute the USDA texture class id for many points.

    Points should be hilbert-sorted for GDAL block-cache locality — sand/silt/clay
    are large global SoilGrids COGs, so each is opened once and sampled with
    ds.sample(), matching the large-raster path used elsewhere for these same files.
    """
    n = len(lats)
    out: list[int | None] = [None] * n
    if n == 0:
        return out
    paths = {k: resolve_layer_path(LAYERS_DIR, v) for k, v in _SOIL_TEXTURE_INPUT_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return out
    coords = list(zip(lons.tolist(), lats.tolist()))
    try:
        readings: dict[str, np.ndarray] = {}
        for key, path in paths.items():
            vals = np.full(n, np.nan, dtype=np.float64)
            with rasterio.open(path) as ds:
                nodata = ds.nodata
                for i, point in enumerate(ds.sample(coords)):
                    raw = float(point[0])
                    if nodata is not None and raw == nodata:
                        continue
                    vals[i] = raw * _SOILGRIDS_SCALE
            readings[key] = vals
        classes = derive_soil_texture_array(readings["sand"], readings["silt"], readings["clay"])
        out = [int(v) if np.isfinite(v) else None for v in classes]
    except Exception:
        return [None] * n
    return out


def compute_soil_texture_at_point(lat: float, lon: float) -> int | None:
    """Compute the USDA texture class id at a single lat/lon."""
    return sample_soil_texture_batch(np.array([lat]), np.array([lon]))[0]


# ---------------------------------------------------------------------------
# Hilbert curve order for spatial indexing.
# Order 13 → 2^13 × 2^13 grid → ~4.9km cells at equator.
# Smaller than a 256-pixel tile at 30m resolution (~7.68km), so observations
# in the same COG internal tile get consecutive indices. Trivially holds for
# all coarser rasters. Index fits in int32 (max value 2^26 - 1 ≈ 67M).
_HILBERT_ORDER = 13


def hilbert_index(latitude: float, longitude: float) -> int:
    """Return a Hilbert curve index for a coordinate (order 13, ~4.9km cells).

    Sort observations by this value before COG raster sampling to maximise
    spatial cache locality across all raster resolutions ≥ 30m.
    """
    n = 1 << _HILBERT_ORDER
    x = min(max(int((longitude + 180.0) / 360.0 * n), 0), n - 1)
    y = min(max(int((latitude + 90.0) / 180.0 * n), 0), n - 1)

    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d
