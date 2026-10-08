# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Terrain shade index and sky view factor from GEDTM30 (data/gis/layers/elevation.tif),
on the GPU, as two global COGs:

  terrain_shade.tif  share of a cell's yearly direct-beam energy blocked by the
                     surrounding terrain's cast shadows (0 = never shaded). Self-shading
                     (sun behind the cell's own slope) counts in neither total, so this
                     is cast shadow only and doesn't overlap heat load. UInt16, ×0.001.
  sky_view.tif       fraction of the sky dome the cell sees (Dozier & Frew 1990), the
                     share of isotropic diffuse light it gets. UInt8, ×0.004.

Method (both from the same horizons):
  * Horizons (Dozier & Frew 1990) in 64 directions out to MAX_DIST_M, traced per cell
    on the Equi7 30 m grid: half-cell steps near the cell, growing to 2 % of the
    distance, sampled bilinearly from a 2× mean-pooled pyramid (multi-resolution far
    terrain, as in HORAYZON, Steger et al. 2022), with Earth curvature.
  * The year's sun path per 0.1° latitude band — every day, every minute of solar time,
    Spencer (1971) declination and Earth-Sun distance — binned into 64 azimuths × 1°
    elevations. Each bin's beam is the ESRA clear-sky model as in GRASS r.sun
    (Rigollier et al. 2000; Hofierka & Šúri 2002): Linke turbidity 3, Kasten-Young air
    mass with refraction, Rayleigh optical thickness, pressure from the cell's
    elevation; times its incidence on the cell's own slope (Horn gradient).
  * Shade = 1 − Σ(energy with the sun above that direction's horizon) / Σ(energy).
  * Equi7 grid north differs from true north; sun azimuths are rotated per cell.

Checked against analytic cases (an unobstructed 30° plane: sky view (1+cos s)/2,
shade 0; flat cells beside 300 m walls: brute-force sun-by-sun integration) and
against r.sun run with and without terrain shadowing (-p) over the year.

Tiling is download_geomorphons': every Equi7 T6 tile with land in elevation.tif
(warped exactly, see there), BUFFER_CELLS of padding for the horizons, back to
elevation.tif's own 1" grid in 1° cells (bilinear), then one COG per layer.
Resumable like download_geomorphons. Once every 1° cell is written the Equi7
tiles are deleted (they'd double the disk needed, and the cells are all the
COGs are built from); a rerun then goes straight to the COGs. Needs an NVIDIA GPU and CuPy; skipped where
CuPy isn't installed (the docker image has neither — run it on a GPU machine).
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject

from scripts.gis import build_overviews
from scripts.gis import download_geomorphons as geo

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

ELEV_PATH       = geo.ELEV_PATH
SHADE_PATH      = geo.LAYERS_DIR / "terrain_shade.tif"
SVF_PATH        = geo.LAYERS_DIR / "sky_view.tif"
EQUI7_TILES_DIR = Path("data/gis/terrain_light_equi7_tiles")  # <zone>/<tile>_{shade,svf}.tif
WGS84_TILES_DIR = Path("data/gis/terrain_light_1deg_tiles")   # <N00E000>_{shade,svf}.tif
SCRATCH_DIR     = Path("data/gis/terrain_light_scratch")
CELLS_DONE      = WGS84_TILES_DIR / ".complete"   # 1° cells finished; Equi7 tiles no longer needed

RES_M = geo.EQUI7_RES_M
MAX_DIST_M = 40_000.0       # 20 km vs 40 km changed shade < 0.3 points in the Wasatch; 40 for the big ranges
BUFFER_CELLS = 1367          # 41 km: horizons reach MAX_DIST_M from every core cell
K_DIRS = 64                  # horizon directions (grid frame)
K_SUN = 64                   # sun azimuth sectors (true north)
N_ELEV = 90                  # 1° sun elevation bins
STEP_FRAC = 0.02
N_LEVELS = 6
COARSE = 64                  # latitude band / grid convergence sampled every 64 cells
LAT_BAND = 0.1
LINKE_TL = 3.0
MINUTE_STEP = 1              # sun path: every day, every minute of solar time
STRIP_ROWS = 64              # short kernel launches (a desktop GPU's display watchdog)

# download_geomorphons.WARP_THREADS guards GamBase's flaky core; the GPU box can warp on more.
WARP_THREADS = int(os.environ.get("TERRAIN_LIGHT_WARP_THREADS", geo.WARP_THREADS))

SHADE_SCALE, SHADE_NODATA = 0.001, 65535
SVF_SCALE, SVF_NODATA = 0.004, 255

KERNEL = r"""
#define KD %(K_DIRS)d
#define KS %(K_SUN)d
#define NE %(N_ELEV)d
extern "C" __global__ void shade(
    const float* pyr, const long long* off, const int* pw, const int* ph, const int nlev,
    const float res, const float zmax, const float max_dist, const float step_frac,
    const int* lat_idx, const float* gamma, const int cw, const int coarse,
    const float* tW, const float* tSinE, const float* tCosE, const float* tSinA, const float* tCosA,
    const float* tM0, const float linke, float* out_shade, float* out_svf, const int r0, const int r1)
{
    const int W = pw[0], H = ph[0];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = r0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= r1 || i >= H || j >= W) return;
    const long long idx = (long long)i * W + j;
    const float z0 = pyr[idx];
    if (isnan(z0)) { out_shade[idx] = nanf(""); out_svf[idx] = nanf(""); return; }

    // Horn gradient, nodata neighbours -> centre
    float nb[9];
    for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) {
        int ii = min(max(i + di, 0), H - 1), jj = min(max(j + dj, 0), W - 1);
        float v = pyr[(long long)ii * W + jj];
        nb[(di + 1) * 3 + (dj + 1)] = isnan(v) ? z0 : v;
    }
    float dzdx = ((nb[2] + 2*nb[5] + nb[8]) - (nb[0] + 2*nb[3] + nb[6])) / (8.0f * res);
    float dzdy = ((nb[0] + 2*nb[1] + nb[2]) - (nb[6] + 2*nb[7] + nb[8])) / (8.0f * res);
    float nn = rsqrtf(dzdx*dzdx + dzdy*dzdy + 1.0f);
    float nx = -dzdx * nn, ny = -dzdy * nn, nz = nn;      // east, north (grid), up

    // Horizons (tan of elevation angle) per grid direction
    float hz[KD];
    const float two_pi = 6.283185307f;
    for (int k = 0; k < KD; k++) {
        float th = two_pi * k / KD, sx = sinf(th), sy = cosf(th);
        float best = -1e30f, d = 0.5f;                      // d in cells
        while (d * res <= max_dist) {
            float dm = d * res;
            if ((zmax - z0) / dm <= best) break;            // nothing further can be higher
            float step = fmaxf(0.5f, d * step_frac);          // half-cell steps near the cell
            int L = max(0, min(nlev - 1, (int)floorf(log2f(step))));
            float sc = (float)(1 << L);
            // bilinear on level L, cell centres at (k + 0.5) * sc - 0.5 in level-0 units
            float fx = (j + d * sx + 0.5f) / sc - 0.5f, fy = (i - d * sy + 0.5f) / sc - 0.5f;
            int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
            if (x0 < 0 || y0 < 0 || x0 + 1 >= pw[L] || y0 + 1 >= ph[L]) break;
            const float* P = pyr + off[L];
            float a = P[(long long)y0 * pw[L] + x0], b = P[(long long)y0 * pw[L] + x0 + 1];
            float c = P[(long long)(y0 + 1) * pw[L] + x0], e2 = P[(long long)(y0 + 1) * pw[L] + x0 + 1];
            float tx = fx - x0, ty = fy - y0;
            float zs = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + e2 * tx) * ty;   // NaN if any is
            if (!isnan(zs)) {
                float t = (zs - dm * dm / (2.0f * 6371000.0f) - z0) / dm;
                best = fmaxf(best, t);
            }
            d += step;
        }
        hz[k] = fmaxf(best, -1.0f);
    }

    // Sky view factor (Dozier & Frew 1990), horizon zenith angles
    float slope = acosf(nz), aspect = atan2f(nx, ny);      // aspect: downslope direction, grid frame
    float svf = 0.0f;
    for (int k = 0; k < KD; k++) {
        float th = two_pi * k / KD;
        // sky ends at the horizontal (below it is ground, even looking down off
        // a ridge) and at the cell's own tangent plane in that direction
        float Hz = 1.5707963f - atanf(fmaxf(hz[k], 0.0f));
        float plane = 1.5707963f + atanf(tanf(slope) * cosf(th - aspect));
        Hz = fminf(Hz, plane);
        svf += cosf(slope) * sinf(Hz) * sinf(Hz) + sinf(slope) * cosf(th - aspect) * (Hz - sinf(Hz) * cosf(Hz));
    }
    out_svf[idx] = svf / KD;

    // Year sum
    int ci = i / coarse, cj = j / coarse;
    int li = lat_idx[ci * cw + cj];
    float g = gamma[ci * cw + cj], cg = cosf(g), sg = sinf(g);
    float pfac = expf(-z0 / 8434.5f);
    float lit = 0.0f, tot = 0.0f;
    const long long tb = (long long)li * KS * NE;
    for (int b = 0; b < KS * NE; b++) {
        float w = tW[tb + b];
        if (w <= 0.0f) continue;
        float se = tSinE[tb + b], ce = tCosE[tb + b];
        float sa = tSinA[tb + b], ca = tCosA[tb + b];
        // true azimuth -> grid azimuth: a_g = A + gamma (gamma = grid azimuth of true north)
        float sag = sa * cg + ca * sg, cag = ca * cg - sa * sg;
        float cosi = nx * ce * sag + ny * ce * cag + nz * se;
        if (cosi <= 0.0f) continue;
        float m = tM0[tb + b] * pfac;
        float dR = (m <= 20.0f) ? 1.0f / (6.6296f + 1.7513f*m - 0.1202f*m*m + 0.0065f*m*m*m - 0.00013f*m*m*m*m)
                                : 1.0f / (10.4f + 0.718f * m);
        float e = w * expf(-0.8662f * linke * m * dR) * cosi;
        tot += e;
        float ag = atan2f(sag, cag); if (ag < 0) ag += two_pi;
        float f = ag / two_pi * KD; int k0 = (int)floorf(f) %% KD; int k1 = (k0 + 1) %% KD; float fr = f - floorf(f);
        float h = hz[k0] * (1.0f - fr) + hz[k1] * fr;
        if (se / ce > h) lit += e;
    }
    out_shade[idx] = (tot > 0.0f) ? 1.0f - lit / tot : nanf("");
}
""" % dict(K_DIRS=K_DIRS, K_SUN=K_SUN, N_ELEV=N_ELEV)


def mean_pool(a):
    """2x2 mean, ignoring nodata; NaN only where all four are nodata."""
    h, w = (a.shape[0] + 1) // 2, (a.shape[1] + 1) // 2
    p = np.full((h * 2, w * 2), np.nan, np.float32)
    p[: a.shape[0], : a.shape[1]] = a
    blocks = p.reshape(h, 2, w, 2)
    cnt = np.isfinite(blocks).sum(axis=(1, 3))
    tot = np.nansum(blocks, axis=(1, 3), dtype=np.float64)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)


_TABLES: dict[int, dict[str, np.ndarray]] = {}


def _band_tables(band: int) -> dict[str, np.ndarray]:
    """One 0.1° latitude band's year of sun positions, binned [K_SUN × N_ELEV]:
    weight (Earth-Sun distance × time), mean sin/cos elevation, mean sin/cos
    azimuth, mean sea-level air mass (Kasten & Young, refraction-corrected as r.sun)."""
    if band in _TABLES:
        return _TABLES[band]
    lat = -90 + (band + 0.5) * LAT_BAND
    days = np.arange(1, 366)
    G = 2 * np.pi * (days - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(G) + 0.070257 * np.sin(G) - 0.006758 * np.cos(2 * G)
           + 0.000907 * np.sin(2 * G) - 0.002697 * np.cos(3 * G) + 0.00148 * np.sin(3 * G))
    e0 = 1.00011 + 0.034221 * np.cos(G) + 0.00128 * np.sin(G) + 0.000719 * np.cos(2 * G) + 0.000077 * np.sin(2 * G)
    omega = np.radians(np.arange(-180, 180, MINUTE_STEP / 4.0))   # hour angle, 1° = 4 min
    dec, omega = np.meshgrid(dec, omega, indexing="ij")
    w = np.broadcast_to(e0[:, None], dec.shape)
    phi = np.radians(lat)
    se = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega)
    up = se > 0
    se, d, om, w = se[up], dec[up], omega[up], w[up]
    ce = np.sqrt(np.maximum(1 - se ** 2, 1e-12))
    A = np.mod(np.arctan2(-np.cos(d) * np.sin(om) / ce, (np.sin(d) - se * np.sin(phi)) / (ce * np.cos(phi))), 2 * np.pi)
    h0 = np.arcsin(se)
    href = h0 + 0.061359 * (0.1594 + 1.123 * h0 + 0.065656 * h0 ** 2) / (1 + 28.9344 * h0 + 277.3971 * h0 ** 2)
    m0 = 1.0 / (np.sin(href) + 0.50572 * (np.degrees(href) + 6.07995) ** -1.6364)
    b = (np.mod(np.floor(A / (2 * np.pi / K_SUN) + 0.5), K_SUN).astype(int) * N_ELEV
         + np.minimum(np.floor(np.degrees(h0)), N_ELEV - 1).astype(int))
    n = K_SUN * N_ELEV
    Wb = np.bincount(b, w, n)
    out = {"W": Wb}
    for key, v in (("sE", se), ("cE", ce), ("sA", np.sin(A)), ("cA", np.cos(A)), ("M0", m0)):
        s = np.bincount(b, w * v, n)
        out[key] = np.divide(s, Wb, out=np.zeros_like(s), where=Wb > 0)
    norm = np.hypot(out["sA"], out["cA"]); nz = norm > 0
    out["sA"][nz] /= norm[nz]; out["cA"][nz] /= norm[nz]
    _TABLES[band] = {k: v.astype(np.float32) for k, v in out.items()}
    return _TABLES[band]


def _compute(dem: np.ndarray, tf, crs) -> tuple[np.ndarray, np.ndarray]:
    """Shade index and sky view factor for a padded Equi7 DEM (NaN = nodata)."""
    import cupy as cp
    from pyproj import Transformer

    H, W = dem.shape
    levels = [dem]
    for _ in range(N_LEVELS - 1):
        levels.append(mean_pool(levels[-1]))
    off = np.cumsum([0] + [l.size for l in levels[:-1]]).astype(np.int64)

    cw, ch = (W + COARSE - 1) // COARSE, (H + COARSE - 1) // COARSE
    jj, ii = np.meshgrid(np.arange(cw) * COARSE + COARSE / 2, np.arange(ch) * COARSE + COARSE / 2)
    x, y = tf * (jj, ii)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    xn, yn = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, np.minimum(lat + 0.01, 89.99))
    gamma = np.arctan2(xn - x, yn - y).astype(np.float32)
    bands = np.clip(np.floor((lat + 90) / LAT_BAND).astype(int), 0, int(180 / LAT_BAND) - 1)
    uniq, local = np.unique(bands, return_inverse=True)
    tabs = [_band_tables(int(b)) for b in uniq]
    tab = {k: cp.asarray(np.concatenate([t[k] for t in tabs])) for k in ("W", "sE", "cE", "sA", "cA", "M0")}

    pyr = cp.asarray(np.concatenate([l.ravel() for l in levels]))
    out_shade = cp.full((H, W), cp.nan, cp.float32)
    out_svf = cp.full((H, W), cp.nan, cp.float32)
    kern = cp.RawKernel(KERNEL, "shade", options=("--use_fast_math",))
    args = (
        pyr, cp.asarray(off), cp.asarray(np.array([l.shape[1] for l in levels], np.int32)),
        cp.asarray(np.array([l.shape[0] for l in levels], np.int32)), np.int32(N_LEVELS),
        np.float32(RES_M), np.float32(np.nanmax(dem)), np.float32(MAX_DIST_M), np.float32(STEP_FRAC),
        cp.asarray(local.reshape(bands.shape).astype(np.int32).ravel()), cp.asarray(gamma.ravel()),
        np.int32(cw), np.int32(COARSE),
        tab["W"], tab["sE"], tab["cE"], tab["sA"], tab["cA"], tab["M0"], np.float32(LINKE_TL),
        out_shade, out_svf,
    )
    bs = (16, 16)
    for r0 in range(0, H, STRIP_ROWS):
        grid = ((W + bs[0] - 1) // bs[0], (STRIP_ROWS + bs[1] - 1) // bs[1])
        kern(grid, bs, args + (np.int32(r0), np.int32(min(r0 + STRIP_ROWS, H))))
        cp.cuda.Device().synchronize()
    shade, svf = out_shade.get(), out_svf.get()
    del pyr, out_shade, out_svf, tab
    cp.get_default_memory_pool().free_all_blocks()
    return shade, svf


def _encode(a: np.ndarray, scale: float, nodata: int, dtype: str, top: int) -> np.ndarray:
    out = np.full(a.shape, nodata, dtype)
    ok = np.isfinite(a)
    out[ok] = np.clip(np.round(a[ok] / scale), 0, top).astype(dtype)
    return out


# ── Step 1: Equi7 tiles ───────────────────────────────────────────────────────

def _tile_paths(zone_name: str, tile) -> tuple[Path, Path, Path]:
    base = EQUI7_TILES_DIR / zone_name / tile.name
    return Path(f"{base}_shade.tif"), Path(f"{base}_svf.tif"), Path(f"{base}.empty")


def _process_equi7_tile(zone_name: str, epsg: int, tile) -> bool:
    """Shade + sky view for one Equi7 tile; False if it has no DEM data."""
    shade_dest, svf_dest, empty_marker = _tile_paths(zone_name, tile)
    if geo._is_done(shade_dest) and geo._is_done(svf_dest):
        return True
    if empty_marker.exists():
        return False
    shade_dest.parent.mkdir(parents=True, exist_ok=True)

    w, s, e, n = tile.outer_boundary_extent
    buf = BUFFER_CELLS * RES_M
    dem_path = SCRATCH_DIR / f"{tile.name}_dem.tif"
    geo._run([
        "gdalwarp", "-overwrite", "-q", "-t_srs", f"EPSG:{epsg}",
        "-te", str(w - buf), str(s - buf), str(e + buf), str(n + buf),
        "-tr", str(RES_M), str(RES_M), "-r", "bilinear",
        "-et", "0", "-wo", "XSCALE=1", "-wo", "YSCALE=1", "-ovr", "NONE",   # extent-independent, see download_geomorphons
        "-srcnodata", repr(geo._elev_nodata()), "-dstnodata", str(geo.DEM_NODATA), "-ot", "Float32",
        "-wm", "2048", "-multi", "-wo", f"NUM_THREADS={WARP_THREADS}",
        "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
        str(ELEV_PATH), str(dem_path),
    ])
    try:
        with rasterio.open(dem_path) as ds:
            dem = ds.read(1)
            tf, crs, prof = ds.transform, ds.crs, ds.profile
        valid = dem != geo.DEM_NODATA
        core = (slice(BUFFER_CELLS, dem.shape[0] - BUFFER_CELLS), slice(BUFFER_CELLS, dem.shape[1] - BUFFER_CELLS))
        if not valid[core].any():
            empty_marker.touch()
            return False
        dem = np.where(valid, dem, np.nan).astype(np.float32)
        shade, svf = _compute(dem, tf, crs)
        core_tf = tf * tf.translation(BUFFER_CELLS, BUFFER_CELLS)
        for arr, dest, scale, nodata, dtype, top in (
            (shade[core], shade_dest, SHADE_SCALE, SHADE_NODATA, "uint16", 1000),
            (svf[core], svf_dest, SVF_SCALE, SVF_NODATA, "uint8", 250),
        ):
            tmp = dest.with_suffix(".tif.tmp")
            with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                               dtype=dtype, crs=crs, transform=core_tf, nodata=nodata, compress="zstd",
                               predictor=2, tiled=True, blockxsize=512, blockysize=512, bigtiff="if_safer") as ds:
                ds.write(_encode(arr, scale, nodata, dtype, top), 1)
            geo._commit(tmp, dest)
        return True
    finally:
        dem_path.unlink(missing_ok=True)


# ── Step 2: back to WGS84 1° tiles ────────────────────────────────────────────

LAYERS = (("shade", SHADE_NODATA, "uint16"), ("svf", SVF_NODATA, "uint8"))


def _write_wgs84_cell(lon: int, lat: int, zone_srcs: dict, footprints: dict, elev_tf) -> list[Path]:
    dests = {k: WGS84_TILES_DIR / f"{geo._cell_name(lon, lat)}_{k}.tif" for k, _, _ in LAYERS}
    if all(geo._is_done(p) for p in dests.values()):
        return list(dests.values())
    # Same 1" grid as elevation.tif — see download_geomorphons._write_wgs84_cell.
    px = elev_tf.a
    col0 = math.floor((lon - elev_tf.c) / px)
    row0 = math.floor((elev_tf.f - (lat + 1)) / px)
    size = round(1.0 / px)
    dst_tf = from_origin(elev_tf.c + col0 * px, elev_tf.f - row0 * px, px, px)
    shapes = [(footprints[z], i + 1) for i, z in enumerate(geo.EQUI7_ZONES) if z in zone_srcs]
    zone_ids = rasterize(shapes, out_shape=(size, size), transform=dst_tf, fill=0, dtype="uint8")
    written = []
    for key, nodata, dtype in LAYERS:
        out = np.full((size, size), nodata, dtype)
        for i, zone_name in enumerate(geo.EQUI7_ZONES):
            mask = zone_ids == i + 1
            if zone_name not in zone_srcs or not mask.any():
                continue
            warped = np.full((size, size), nodata, dtype)
            reproject(source=rasterio.band(zone_srcs[zone_name][key], 1), destination=warped,
                      src_nodata=nodata, dst_transform=dst_tf, dst_crs="EPSG:4326",
                      dst_nodata=nodata, resampling=Resampling.bilinear)
            out[mask] = warped[mask]
        if not (out != nodata).any():
            continue
        dest = dests[key]
        tmp = dest.with_suffix(".tif.tmp")
        with rasterio.open(tmp, "w", driver="GTiff", width=size, height=size, count=1, dtype=dtype,
                           crs="EPSG:4326", transform=dst_tf, nodata=nodata, compress="zstd", predictor=2,
                           tiled=True, blockxsize=512, blockysize=512) as ds:
            ds.write(out, 1)
        geo._commit(tmp, dest)
        written.append(dest)
    return written


def _reproject_to_wgs84(done: list[tuple[str, int, tuple]]) -> dict[str, list[Path]]:
    footprints = geo._zone_footprints()
    cells = geo._cells_for_tiles(done, footprints)
    WGS84_TILES_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ELEV_PATH) as elev:
        elev_tf = elev.transform
    written: dict[str, list[Path]] = {k: [] for k, _, _ in LAYERS}
    with tempfile.TemporaryDirectory() as tmp_dir:
        zone_srcs: dict[str, dict] = {}
        for zone_name in geo.EQUI7_ZONES:
            for key, _, _ in LAYERS:
                tiles = sorted((EQUI7_TILES_DIR / zone_name).glob(f"*_{key}.tif"))
                if tiles:
                    vrt = Path(tmp_dir) / f"{zone_name}_{key}.vrt"
                    geo._run(["gdalbuildvrt", "-overwrite", "-q", str(vrt), *[str(t) for t in tiles]])
                    zone_srcs.setdefault(zone_name, {})[key] = rasterio.open(vrt)
        try:
            for j, (lon, lat) in enumerate(cells, 1):
                for p in _write_wgs84_cell(lon, lat, zone_srcs, footprints, elev_tf):
                    written["shade" if p.stem.endswith("_shade") else "svf"].append(p)
                if j % 500 == 0 or j == len(cells):
                    print(f"  [{j}/{len(cells)}] 1° cells", flush=True)
        finally:
            for srcs in zone_srcs.values():
                for ds in srcs.values():
                    ds.close()
    return written


# ── Step 3: global COGs ───────────────────────────────────────────────────────

def _build_cog(tile_paths: list[Path], out_path: Path, scale: float) -> None:
    tmp_out = out_path.with_suffix(".tif.tmp")
    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt = Path(tmp_dir) / "global.vrt"
        scaled = Path(tmp_dir) / "scaled.vrt"
        geo._run(["gdalbuildvrt", "-overwrite", "-resolution", "highest", str(vrt), *[str(p) for p in sorted(tile_paths)]])
        geo._run(["gdal_translate", "-q", "-of", "VRT", "-a_scale", str(scale), str(vrt), str(scaled)])
        with rasterio.open(scaled) as ds:
            factors = build_overviews._overview_factors_for_dataset(ds)
        build_overviews._build_cog(scaled, tmp_out, nominal=False, overview_factors=factors)
    if out_path.exists():
        out_path.unlink()
    geo._commit(tmp_out, out_path)


def main(force: bool = False) -> None:
    if not force and VARS_TO_DOWNLOAD is not None and "terrain_shade" not in VARS_TO_DOWNLOAD:
        print("[download_terrain_light] skipped (terrain_shade not in VARS_TO_DOWNLOAD)")
        return
    if SHADE_PATH.exists() and SVF_PATH.exists() and not force:
        print(f"COGs already exist: {SHADE_PATH}, {SVF_PATH} (use --force to rebuild)")
        return
    try:
        import cupy  # noqa: F401
    except ImportError:
        print("[download_terrain_light] skipped: needs an NVIDIA GPU with CuPy (see docstring)")
        return
    if not ELEV_PATH.exists():
        raise RuntimeError(f"{ELEV_PATH} not found — derived from it (download_dem.py)")

    if CELLS_DONE.exists():
        print(f"1° cells already complete ({WGS84_TILES_DIR}/), building the COGs", flush=True)
        _build_cogs()
        return

    tiles = list(geo._land_tiles())
    print(f"Terrain shade + sky view for {len(tiles)} Equi7 tiles ({MAX_DIST_M / 1000:.0f} km horizons, "
          f"{K_DIRS} directions, sun every day every {MINUTE_STEP} min)...", flush=True)
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(parents=True)
    done: list[tuple[str, int, tuple]] = []
    computed, compute_s = 0, 0.0
    try:
        for i, (zone_name, epsg, tile) in enumerate(tiles, 1):
            shade_dest, svf_dest, empty_marker = _tile_paths(zone_name, tile)
            resumed = (geo._is_done(shade_dest) and geo._is_done(svf_dest)) or empty_marker.exists()
            t0 = time.perf_counter()
            if _process_equi7_tile(zone_name, epsg, tile):
                done.append((zone_name, epsg, tile.outer_boundary_extent))
            if resumed:
                continue
            dt = time.perf_counter() - t0
            computed += 1
            compute_s += dt
            eta_h = compute_s / computed * (len(tiles) - i) / 3600
            geo._log(f"  [{i}/{len(tiles)}] {tile.name}: {dt:.0f}s  (ETA {eta_h:.1f} h)")
    finally:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    print("\nReprojecting to WGS84 1° tiles...", flush=True)
    _reproject_to_wgs84(done)
    CELLS_DONE.touch()
    shutil.rmtree(EQUI7_TILES_DIR, ignore_errors=True)
    _build_cogs()


def _build_cogs() -> None:
    for key, path, scale in (("shade", SHADE_PATH, SHADE_SCALE), ("svf", SVF_PATH, SVF_SCALE)):
        cells = sorted(WGS84_TILES_DIR.glob(f"*_{key}.tif"))
        if cells:
            print(f"\nBuilding {path} from {len(cells)} cells...", flush=True)
            _build_cog(cells, path, scale)
    print(f"\nDone. Once these check out, {WGS84_TILES_DIR}/ can be deleted.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Terrain shade index + sky view factor from GEDTM30 (GPU)")
    parser.add_argument("--force", action="store_true", help="Rebuild the COGs even if they exist")
    args = parser.parse_args()
    main(force=args.force)
