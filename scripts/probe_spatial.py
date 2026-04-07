"""Quick probe for reading data_spatial .om files from Open-Meteo S3."""
from __future__ import annotations

import json
import sys
from datetime import datetime

import fsspec
import numpy as np
from omfiles import OmFileReader
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds as rasterio_from_bounds
from rasterio.warp import reproject as rasterio_reproject, Resampling

# Add workspace root so we can reuse tiles.py helpers
sys.path.insert(0, "/workspace")
from util.tiles import TileSpec, tile_bounds_wgs84, tile_bounds_mercator, WEB_MERCATOR, NUMERIC_COLOR_STOPS

MODEL = "ncep_gfs013"
GRID_LAT_MAX =  89.912125
GRID_LAT_MIN = -89.912125
GRID_LON_MIN = -180.0
GRID_LON_MAX =  179.88281
LAT_SLC =  40.7596
LON_SLC = -111.8882

# --- 1. Fetch latest.json ---
fs = fsspec.filesystem("s3", anon=True)
with fs.open(f"s3://openmeteo/data_spatial/{MODEL}/latest.json") as f:
    meta = json.load(f)

ref = meta["reference_time"]
valid_time = meta["valid_times"][0]
print(f"Run:        {ref}")
print(f"Valid time: {valid_time}")

r = datetime.fromisoformat(ref.replace("Z", "+00:00"))
v = datetime.fromisoformat(valid_time.replace("Z", "+00:00"))
run_dir = f"{r.year:04d}/{r.month:02d}/{r.day:02d}/{r.hour:02d}{r.minute:02d}Z"
fname = v.strftime("%Y-%m-%dT%H%M") + ".om"
path = f"s3://openmeteo/data_spatial/{MODEL}/{run_dir}/{fname}"
print(f"Path:       {path}\n")

# --- 2. Load full temperature_2m grid into memory ---
backend = fsspec.open(path, mode="rb", s3={"anon": True})
root = OmFileReader(backend)
temp_node = root.get_child_by_name("temperature_2m")
ny, nx = temp_node.shape
print(f"Grid: {ny} x {nx}  ({ny*nx/1e6:.1f}M pixels)")

arr = temp_node.read_array((slice(0, ny), slice(0, nx)))
print(f"Loaded. dtype={arr.dtype}  global range: {arr.min():.1f} to {arr.max():.1f} C")

# --- 3. Verify SLC ---
row = int((GRID_LAT_MAX - LAT_SLC) / (GRID_LAT_MAX - GRID_LAT_MIN) * ny)
col = int((LON_SLC - GRID_LON_MIN) / (GRID_LON_MAX - GRID_LON_MIN) * nx)
val_c = arr[row, col]
val_f = val_c * 9 / 5 + 32
print(f"SLC (row={row}, col={col}): {val_c:.1f} C = {val_f:.1f} F\n")

# --- 4. Spot-check known cities ---
cities = {
    "SLC":    (40.7596,  -111.8882),
    "London": (51.5074,   -0.1278),
    "Tokyo":  (35.6762,  139.6503),
    "Sydney": (-33.8688, 151.2093),
}
for city, (lat, lon) in cities.items():
    row = int((GRID_LAT_MAX - lat) / (GRID_LAT_MAX - GRID_LAT_MIN) * ny)
    col = int((lon - GRID_LON_MIN) / (GRID_LON_MAX - GRID_LON_MIN) * nx)
    val = arr[row, col]
    print(f"  {city:8s} lat={lat:7.2f} lon={lon:8.2f}  row={row:4d} col={col:4d}  temp={val:.1f} C")

# --- 5. Render a tile ---
# z=4 x=8 y=5 covers Western Europe — Italy should be visible
TILE_Z, TILE_X, TILE_Y = 4, 8, 5
TILE_SIZE = 256
TEMP_LO, TEMP_HI = -50.0, 50.0  # colormap range in C

spec = TileSpec(z=TILE_Z, x=TILE_X, y=TILE_Y, tile_size=TILE_SIZE)
lon_w, lat_s, lon_e, lat_n = tile_bounds_wgs84(spec)
print(f"Tile {TILE_Z}/{TILE_X}/{TILE_Y} bounds: lat [{lat_s:.2f}, {lat_n:.2f}]  lon [{lon_w:.2f}, {lon_e:.2f}]")

# Reproject WGS84 grid → Web Mercator tile (same as server)
src_transform = rasterio_from_bounds(GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX, nx, ny)
src_crs = CRS.from_epsg(4326)
minx, miny, maxx, maxy = tile_bounds_mercator(spec)
dst_transform = rasterio_from_bounds(minx, miny, maxx, maxy, TILE_SIZE, TILE_SIZE)
dst_crs = CRS.from_string(WEB_MERCATOR)

dest = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
rasterio_reproject(
    source=arr,
    destination=dest,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=np.nan,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear,
)
print(f"Reprojected dest: min={np.nanmin(dest):.1f}  max={np.nanmax(dest):.1f}  nan_pct={np.isnan(dest).mean()*100:.1f}%")

# Colorize
norm = np.clip((dest - TEMP_LO) / (TEMP_HI - TEMP_LO), 0.0, 1.0)
positions = np.linspace(0.0, 1.0, NUMERIC_COLOR_STOPS.shape[0], dtype=np.float32)
rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
rgba[..., 0] = np.interp(norm, positions, NUMERIC_COLOR_STOPS[:, 0]).astype(np.uint8)
rgba[..., 1] = np.interp(norm, positions, NUMERIC_COLOR_STOPS[:, 1]).astype(np.uint8)
rgba[..., 2] = np.interp(norm, positions, NUMERIC_COLOR_STOPS[:, 2]).astype(np.uint8)
rgba[..., 3] = np.where(np.isfinite(dest), 220, 0).astype(np.uint8)

out_path = "/workspace/test_tile.png"
Image.fromarray(rgba, mode="RGBA").save(out_path)
print(f"Tile saved -> {out_path}")
