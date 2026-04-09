from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.config import load_config
from util import gis_lookup, models
from util import taxa_navigation

CONFIG = load_config("global")

# ---------------------------------------------------------------------------
# Tweak these to control the run
# ---------------------------------------------------------------------------

BASE_FEATURES_DIR = Path("data/gis/temporal/homepage")
FORECAST_HOURS = 0
OUTPUT_PATH = Path("data/gis/temporal/homepage/aggregate_sdm.tif")
TAXON_CAP: int | None = None
N_WORKERS: int = 1
PUSH_TO_B2: bool = True

# ---------------------------------------------------------------------------

_TEMPORAL_WINDOW_LABELS: dict[int, str] = {
    1: "1h",
    8: "8h",
    24: "24h",
    72: "3d",
    168: "7d",
    720: "30d",
    2160: "90d",
}
_TEMPORAL_RASTER_DIR = Path(__file__).resolve().parent.parent / "data" / "gis" / "temporal" / "rasters"

_ERA5_RES = 0.25
_ERA5_TOP_LAT = 90.0
_ERA5_LEFT_LON = -180.0


def _load_temporal_npy(variable_id: str, window_hours: int) -> np.ndarray | None:
    label = _TEMPORAL_WINDOW_LABELS.get(window_hours)
    if label is None:
        return None
    suffix = f"__f{FORECAST_HOURS:03d}h" if FORECAST_HOURS != 0 else ""
    p = _TEMPORAL_RASTER_DIR / f"{variable_id}_{label}{suffix}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def _slice_temporal(arr: np.ndarray, out_transform: Affine, dst_shape: tuple[int, int]) -> np.ndarray:
    """Resample an ERA5 global array to the output grid via nearest-neighbour lookup.

    Works at any output resolution — does not assume output matches ERA5's 0.25°.
    """
    out_h, out_w = dst_shape
    era5_h, era5_w = arr.shape

    # Pixel-centre latitudes and longitudes of the output grid
    out_lats = out_transform.f + out_transform.e * (np.arange(out_h) + 0.5)
    out_lons = out_transform.c + out_transform.a * (np.arange(out_w) + 0.5)

    # Nearest ERA5 row/col for each output pixel
    era5_rows = np.round((_ERA5_TOP_LAT - out_lats) / _ERA5_RES).astype(np.intp)
    era5_cols = np.round((out_lons - _ERA5_LEFT_LON) / _ERA5_RES).astype(np.intp)

    # Mask out-of-bounds pixels
    valid_r = (era5_rows >= 0) & (era5_rows < era5_h)
    valid_c = (era5_cols >= 0) & (era5_cols < era5_w)

    era5_rows = np.clip(era5_rows, 0, era5_h - 1)
    era5_cols = np.clip(era5_cols, 0, era5_w - 1)

    out = arr[np.ix_(era5_rows, era5_cols)].copy()
    # NaN out pixels whose centre falls outside ERA5 coverage
    out[~valid_r, :] = np.nan
    out[:, ~valid_c] = np.nan
    return out


def _hms(seconds: float) -> str:
    s = int(seconds)
    return (
        f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s"
        if s >= 3600
        else f"{s // 60}m {s % 60}s"
        if s >= 60
        else f"{seconds:.1f}s"
    )


def main() -> None:
    import time

    t_total = time.perf_counter()

    npz_path = BASE_FEATURES_DIR / "base_features.npz"
    meta_path = BASE_FEATURES_DIR / "base_features_meta.json"
    if not npz_path.exists():
        raise RuntimeError(f"Base features not found at {npz_path} — run build_base_features.py first")

    t0 = time.perf_counter()
    print(f"[infer] loading base features from {npz_path}")
    base = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    print(f"  → {_hms(time.perf_counter() - t0)}")

    out_height, out_width = meta["shape"]
    t = meta["transform"]
    out_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    out_crs = CRS.from_wkt(meta["crs_wkt"])
    gis_layer_ids: list[str] = meta["layers"]

    t0 = time.perf_counter()
    taxon_ids = models.get_all_sdm_taxon_ids()
    if TAXON_CAP is not None:
        taxon_ids = taxon_ids[:TAXON_CAP]
        print(f"[infer] TAXON_CAP={TAXON_CAP} — using {len(taxon_ids)} taxa")
    all_layer_ids_per_taxon: dict[int, tuple[list[str], str | None]] = {}
    temporal_layer_set: set[str] = set()

    for taxon_id in taxon_ids:
        has_pheno = models.has_phenology_model(taxon_id)
        has_full = models.has_full_model(taxon_id)
        if has_full and not has_pheno:
            cols = models.model_feature_columns(models.AUTO_FULL_MODEL_ID, taxon_id=taxon_id)
            all_layer_ids_per_taxon[taxon_id] = (cols, None)
        else:
            sdm_cols = models.model_feature_columns(models.AUTO_MODEL_ID, taxon_id=taxon_id)
            if has_pheno:
                pheno_cols = models.model_feature_columns(models.AUTO_PHENOLOGY_MODEL_ID, taxon_id=taxon_id)
                merged = list(dict.fromkeys(sdm_cols + pheno_cols))
                all_layer_ids_per_taxon[taxon_id] = (merged, models.AUTO_PHENOLOGY_MODEL_ID)
            else:
                all_layer_ids_per_taxon[taxon_id] = (sdm_cols, None)
        for col in all_layer_ids_per_taxon[taxon_id][0]:
            if gis_lookup.is_temporal_layer_id(col):
                temporal_layer_set.add(col)

    temporal_layer_ids = sorted(temporal_layer_set)
    print(
        f"[infer] {len(gis_layer_ids)} GIS layers, {len(temporal_layer_ids)} temporal layers  ({_hms(time.perf_counter() - t0)})"
    )

    all_layer_ids = gis_layer_ids + temporal_layer_ids
    layer_index = {lid: i for i, lid in enumerate(all_layer_ids)}
    n_layers = len(all_layer_ids)
    all_layer_ids_tuple = tuple(all_layer_ids)

    t0 = time.perf_counter()
    print("[infer] loading temporal arrays...")
    temporal_native: dict[str, np.ndarray | None] = {}
    for layer_id in temporal_layer_ids:
        parsed_temporal = gis_lookup.parse_temporal_layer_id(layer_id)
        if parsed_temporal is None:
            temporal_native[layer_id] = None
            continue
        arr = _load_temporal_npy(parsed_temporal[0], parsed_temporal[2])
        if arr is None:
            print(f"  [warn] temporal '{layer_id}' not found — filling NaN")
        temporal_native[layer_id] = arr
    print(f"  → {_hms(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    stack_gb = out_height * out_width * n_layers * 4 / 1e9
    print(f"[infer] building feature stack {out_width}×{out_height}×{n_layers}  ({stack_gb:.1f} GB)...")
    feature_stack = np.full((out_height, out_width, n_layers), np.nan, dtype=np.float32)
    for layer_id in gis_layer_ids:
        if layer_id in base:
            feature_stack[:, :, layer_index[layer_id]] = base[layer_id]
    for layer_id in temporal_layer_ids:
        arr = temporal_native.get(layer_id)
        if arr is not None:
            feature_stack[:, :, layer_index[layer_id]] = _slice_temporal(arr, out_transform, (out_height, out_width))
    print(f"  → {_hms(time.perf_counter() - t0)}")

    flat = feature_stack.reshape(-1, n_layers)
    valid_mask = np.any(np.isfinite(flat), axis=1)

    # Classify each taxon into a group via its taxonomy path
    def _taxon_group(taxon_id: int) -> str:
        record = taxa_navigation.get_taxon_by_id(str(taxon_id))
        if record is None:
            return "other"
        path_str = str(record.get("path", ""))
        if "Arthropoda_54" in path_str:
            return "arthropods"
        elif "Aves_212" in path_str:
            return "birds"
        elif "Animalia_1" in path_str:
            return "animals"
        elif "Fungi_5" in path_str:
            return "fungi"
        elif "Plantae_6" in path_str:
            return "plants"
        return "other"

    taxon_groups: dict[int, str] = {tid: _taxon_group(tid) for tid in taxon_ids}
    group_names = sorted(set(taxon_groups.values()))
    print(f"[infer] taxon groups: { {g: sum(1 for v in taxon_groups.values() if v == g) for g in group_names} }")

    acc = np.zeros((out_height, out_width), dtype=np.float64)
    pixel_count = np.zeros((out_height, out_width), dtype=np.int32)
    group_acc: dict[str, np.ndarray] = {g: np.zeros((out_height, out_width), dtype=np.float64) for g in group_names}
    group_count: dict[str, np.ndarray] = {g: np.zeros((out_height, out_width), dtype=np.int32) for g in group_names}
    taxon_prob_arrays: dict[int, np.ndarray] = {
        tid: np.full((out_height, out_width), np.nan, dtype=np.float32) for tid in taxon_ids
    }

    n_workers = min(N_WORKERS, max(1, len(taxon_ids)))
    print(f"[infer] running inference  {len(taxon_ids)} taxa  {n_workers} workers")

    def _infer_taxon(taxon_id: int) -> tuple[int, np.ndarray | None]:
        if taxon_id not in all_layer_ids_per_taxon:
            return taxon_id, None
        _, secondary_model_id = all_layer_ids_per_taxon[taxon_id]
        has_pheno = models.has_phenology_model(taxon_id)
        has_full = models.has_full_model(taxon_id)
        active_model = models.AUTO_FULL_MODEL_ID if (has_full and not has_pheno) else models.AUTO_MODEL_ID
        probs = models.predict(
            active_model,
            feature_stack,
            feature_ids=all_layer_ids_tuple,
            taxon_id=taxon_id,
            _preflat=flat,
            _valid_mask=valid_mask,
        )
        if has_pheno and secondary_model_id:
            pheno = models.predict(
                secondary_model_id,
                feature_stack,
                feature_ids=all_layer_ids_tuple,
                taxon_id=taxon_id,
                _preflat=flat,
                _valid_mask=valid_mask,
            )
            probs = probs * pheno
        return taxon_id, probs

    t_infer = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_infer_taxon, tid): tid for tid in taxon_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                _, probs = future.result()
            except Exception as exc:
                print(f"  [warn] taxon={tid} failed: {exc}")
                continue
            if probs is None:
                continue
            finite = np.isfinite(probs)
            acc[finite] += probs[finite]
            pixel_count[finite] += 1
            grp = taxon_groups.get(tid, "other")
            group_acc[grp][finite] += probs[finite]
            group_count[grp][finite] += 1
            taxon_prob_arrays[tid] = probs
            completed += 1
            elapsed = time.perf_counter() - t_infer
            rate = completed / elapsed
            eta = (len(taxon_ids) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(taxon_ids)}] taxon={tid}  {_hms(elapsed)} elapsed  eta {_hms(eta)}")

    print(f"[infer] inference done  ({_hms(time.perf_counter() - t_infer)})")

    if not pixel_count.any():
        print("[infer] no models produced output")
        return

    t0 = time.perf_counter()
    avg = np.full((out_height, out_width), np.nan, dtype=np.float32)
    has_data = pixel_count > 0
    avg[has_data] = (acc[has_data] / pixel_count[has_data]).astype(np.float32)

    _LANDCOVER_WATER_CLASS = 210
    _OCEAN_ELEVATION_THRESHOLD = 1.0
    ocean_mask = np.zeros((out_height, out_width), dtype=bool)
    lc = base["landcover"] if "landcover" in base else None
    elev = base["elevation"] if "elevation" in base else None
    if lc is not None:
        is_water = np.isfinite(lc) & (lc == _LANDCOVER_WATER_CLASS)
        no_lc = ~np.isfinite(lc)
        if elev is not None:
            ocean_mask |= is_water & ((elev < _OCEAN_ELEVATION_THRESHOLD) | ~np.isfinite(elev))
            ocean_mask |= no_lc & ((elev < _OCEAN_ELEVATION_THRESHOLD) | ~np.isfinite(elev))
        else:
            ocean_mask |= is_water | no_lc
    ocean_mask |= ~has_data
    avg[ocean_mask] = np.nan
    masked_px = int(ocean_mask.sum())
    print(f"[infer] averaged + ocean mask ({masked_px:,} px masked)  ({_hms(time.perf_counter() - t0)})")

    valid_vals = avg[np.isfinite(avg)]
    if valid_vals.size > 0:
        stats_path = OUTPUT_PATH.parent / "aggregate_sdm_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "vmin": float(np.percentile(valid_vals, 2)),
                    "vmax": float(np.percentile(valid_vals, 98)),
                }
            )
        )

    t0 = time.perf_counter()
    taxon_probs_path = OUTPUT_PATH.parent / "taxon_probs.npz"
    np.savez_compressed(taxon_probs_path, **{str(k): v for k, v in taxon_prob_arrays.items()})
    print(
        f"[infer] saved {len(taxon_prob_arrays)} per-taxon rasters → {taxon_probs_path}  ({_hms(time.perf_counter() - t0)})"
    )

    t0 = time.perf_counter()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=1,
        dtype=np.float32,
        crs=out_crs,
        transform=out_transform,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(avg, 1)

    valid_px = int(np.isfinite(avg).sum())
    model_count = int(pixel_count.max())
    print(f"[infer] wrote GeoTIFF  ({_hms(time.perf_counter() - t0)})")
    print(f"[infer] done  max {model_count} models/pixel  valid={valid_px:,}/{avg.size:,}  → {OUTPUT_PATH}")

    # Per-group GeoTIFFs
    def _write_raster(path: Path, data: np.ndarray) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=out_height,
            width=out_width,
            count=1,
            dtype=np.float32,
            crs=out_crs,
            transform=out_transform,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data, 1)

    t0 = time.perf_counter()
    all_group_stats: dict[str, dict] = {}
    for grp in group_names:
        g_avg = np.full((out_height, out_width), np.nan, dtype=np.float32)
        g_has = group_count[grp] > 0
        g_avg[g_has] = (group_acc[grp][g_has] / group_count[grp][g_has]).astype(np.float32)
        g_avg[ocean_mask] = np.nan
        g_path = OUTPUT_PATH.parent / f"aggregate_sdm_{grp}.tif"
        _write_raster(g_path, g_avg)
        g_vals = g_avg[np.isfinite(g_avg)]
        if g_vals.size > 0:
            all_group_stats[grp] = {
                "vmin": float(np.percentile(g_vals, 2)),
                "vmax": float(np.percentile(g_vals, 98)),
            }
        print(f"  → {grp}: {int(np.isfinite(g_avg).sum()):,} valid px  {g_path.name}")
    (OUTPUT_PATH.parent / "aggregate_sdm_group_stats.json").write_text(json.dumps(all_group_stats))
    print(f"[infer] wrote {len(group_names)} group rasters  ({_hms(time.perf_counter() - t0)})")
    print(f"[infer] total time: {_hms(time.perf_counter() - t_total)}")

    if PUSH_TO_B2:
        remote = os.environ.get("WW_B2_WRITER_REMOTE", "wherewild-localdev-writer")
        bucket = os.environ.get("WW_B2_BUCKET", "wherewild-data")
        prefix = os.environ.get("WW_B2_PREFIX", "data")
        upload_dir = CONFIG.data_root / "gis" / "temporal" / "homepage"
        rel_path = upload_dir.relative_to(CONFIG.data_root)
        remote_dest = f"{remote}:{bucket}/{prefix}/{rel_path}"
        print(f"[infer] pushing homepage tiles to B2: {remote_dest}")
        subprocess.run(
            ["rclone", "copy", str(upload_dir), remote_dest, "--transfers=4",
             "--include", "aggregate_sdm*.tif",
             "--include", "aggregate_sdm*.json",
             "--include", "taxon_probs.npz"],
            check=True,
        )
        print("[infer] B2 push complete")


if __name__ == "__main__":
    main()
