# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Per-taxon summary statistics and density graphs for GIS layers.

For leaf-rank taxa, exact statistics are computed from the single
occurrence.parquet using pandas describe(). For non-leaf taxa, all
descendant parquets are streamed; a T-Digest accumulates quantile
estimates and a reservoir sample drives the KDE.

Outputs per taxon directory:
  summary_stats.parquet     — wide: one row per variable, metrics as columns
  categorical_stats.parquet — tall: (variable, metric, value) for nominal layers
  density_graph.parquet     — KDE curve rows for continuous layers
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastdigest import TDigest
from KDEpy import FFTKDE
from scipy.optimize import brentq as _brentq
from scipy.special import ive as _bessel_ive
from scipy.stats import circmean, circstd, circvar
from scipy.stats import entropy as _scipy_entropy

from config.config import ValueType, load_config
from util.storage import ParquetStorage, atomic_write_parquet
from util.taxa import TaxonRecord, get_children, iter_descendants

CONFIG = load_config("global")

TREE_ROOT = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "tree"
GLOBAL_STATS_DIR = Path(os.environ.get("WHEREWILD_DATA_ROOT", "data")) / "taxonomy" / "global"
OCCURRENCE_FILE = "occurrence.parquet"
NUMERICAL_STATS_FILE = "numerical_stats.parquet"
NOMINAL_STATS_FILE = "nominal_stats.parquet"
ORDINAL_STATS_FILE = "ordinal_stats.parquet"
CIRCULAR_STATS_FILE = "circular_stats.parquet"
DENSITY_FILE = "density.parquet"
PHENOLOGY_COUNTS_FILE = "phenology_counts.json"

_KDE_MAX_SAMPLES = 100_000
_KDE_N_POINTS = 128

# Non-layer columns required for filtering, deduplication, phenology, and indexing.
_OCC_BASE_COLS: frozenset[str] = frozenset({
    "catalogNumber",
    "obscured",
    "coordinateUncertaintyInMeters",
    "rcs",
    "eventTimestamp",
    "decimalLatitude",
    "decimalLongitude",
})


def _read_occ_table(occ_path: Path, layer_meta: dict[str, dict]) -> pa.Table:
    """Read only the columns needed for stats computation from an occurrence parquet."""
    needed = _OCC_BASE_COLS | layer_meta.keys()
    pf = pq.ParquetFile(occ_path)
    cols = [c for c in pf.schema_arrow.names if c in needed]
    return pf.read(columns=cols)


def apply_phenology_filter(df: pd.DataFrame, phenology: str) -> pd.DataFrame:
    """Keep rows where the rcs column contains phenology (pipe-separated match)."""
    if "rcs" not in df.columns:
        return df.iloc[0:0]
    pheno_lower = phenology.strip().lower()
    mask = df["rcs"].apply(
        lambda val: isinstance(val, str) and pheno_lower in {v.strip().lower() for v in val.split("|")}
    )
    return df.loc[mask]


def apply_timestamp_filter(
    df: pd.DataFrame,
    start_ts: int | None,
    end_ts: int | None,
) -> pd.DataFrame:
    """Keep rows whose eventTimestamp falls within [start_ts, end_ts]."""
    if "eventTimestamp" not in df.columns:
        return df
    col = pd.to_numeric(df["eventTimestamp"], errors="coerce")
    if start_ts is not None:
        df = df[col >= start_ts]
        col = col[col >= start_ts]
    if end_ts is not None:
        df = df[col <= end_ts]
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_value_type(layer: dict) -> ValueType | None:
    try:
        return ValueType(layer.get("value_type", ""))
    except ValueError:
        return None


_legend_valid_ids_cache: dict[str, frozenset[int] | None] = {}

def _valid_class_ids(layer_id: str) -> frozenset[int] | None:
    """Return the set of valid class IDs from the legend file, or None if no legend exists."""
    if layer_id in _legend_valid_ids_cache:
        return _legend_valid_ids_cache[layer_id]
    base_id = re.sub(r'_(avg|sum|mode|mean|min|max)_\d+h$', '', layer_id)
    legend_path = Path("config/gis/legends") / f"{base_id}_legend.json"
    result: frozenset[int] | None = None
    if legend_path.exists():
        try:
            classes = json.loads(legend_path.read_text()).get("classes", [])
            result = frozenset(int(c["id"]) for c in classes if "id" in c)
        except Exception:
            pass
    _legend_valid_ids_cache[layer_id] = result
    return result


def _filter_to_known_classes(counts: Counter, layer_id: str) -> Counter:
    """Remove class IDs not present in the legend. Returns counts unchanged if no legend."""
    valid = _valid_class_ids(layer_id)
    if valid is None:
        return counts
    return Counter({k: v for k, v in counts.items() if k in valid})


def _is_discrete(layer: dict) -> bool:
    return layer.get("domain") == "discrete"


def compute_phenology_counts(df: pd.DataFrame) -> Counter:
    """Count occurrences per phenology value from the pipe-separated rcs column."""
    counts: Counter = Counter()
    if "rcs" not in df.columns:
        return counts
    for val in df["rcs"].dropna():
        if isinstance(val, str):
            for part in val.split("|"):
                part = part.strip().lower()
                if part:
                    counts[part] += 1
    return counts


def write_phenology_counts(taxon_dir: Path, counts: Counter) -> None:
    if not counts:
        return
    taxon_dir.mkdir(parents=True, exist_ok=True)
    (taxon_dir / PHENOLOGY_COUNTS_FILE).write_text(json.dumps(dict(counts)))


def read_phenology_counts(taxon_dir: Path) -> dict[str, int]:
    taxon_key = taxon_dir.name.rsplit("_", 1)[-1]
    global_path = GLOBAL_STATS_DIR / "phenology_counts.parquet"
    if global_path.exists():
        try:
            rows = pq.read_table(
                global_path,
                filters=[("taxon_key", "=", taxon_key)],
            ).to_pylist()
            if rows:
                return {r["phenology_value"]: r["count"] for r in rows}
        except Exception:
            pass
    # fallback: per-node numerical_stats metadata (pre-consolidation)
    p = taxon_dir / NUMERICAL_STATS_FILE
    if p.exists():
        try:
            meta = pq.read_schema(p).metadata or {}
            raw = meta.get(b"phenology_counts")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    # legacy JSON fallback
    p = taxon_dir / PHENOLOGY_COUNTS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        col = df["coordinateUncertaintyInMeters"]
        df = df[col.isna() | (col <= 500)]
    return df


def _reservoir_update(reservoir: list, n_seen: int, values: np.ndarray) -> int:
    """Vitter Algorithm R reservoir sample — updates in place."""
    for val in values.tolist():
        n_seen += 1
        if len(reservoir) < _KDE_MAX_SAMPLES:
            reservoir.append(val)
        else:
            j = random.randrange(n_seen)
            if j < _KDE_MAX_SAMPLES:
                reservoir[j] = val
    return n_seen


_ACC_FILE = ".acc"


def _df_to_acc(df: pd.DataFrame, layer_meta: dict[str, dict]) -> dict:
    """Build an in-memory accumulator dict from a filtered DataFrame."""
    acc: dict = {"continuous": {}, "circular": {}, "nominal": {}, "ordinal": {}, "pheno": {}}
    _total_unique: int | None = None

    def _col_unique(col: str) -> int:
        nonlocal _total_unique
        null_mask = df[col].isna()
        if not null_mask.any():
            if _total_unique is None:
                _total_unique = int(df["catalogNumber"].nunique())
            return _total_unique
        return int(df.loc[~null_mask, "catalogNumber"].nunique())

    for col in df.columns:
        if col not in layer_meta:
            continue
        vtype = _layer_value_type(layer_meta[col])
        if vtype is None:
            continue
        match vtype:
            case ValueType.RATIO | ValueType.INTERVAL:
                raw = df[col]
                series = (raw.dropna() if raw.dtype == np.float64
                          else pd.to_numeric(raw, errors="coerce").dropna())
                if series.empty:
                    continue
                values = series.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                digest = TDigest()
                digest.batch_update(values.tolist())
                reservoir: list = []
                n_seen = _reservoir_update(reservoir, 0, values)
                acc["continuous"][col] = {
                    "digest": digest, "reservoir": reservoir,
                    "n_seen": n_seen, "unique": _col_unique(col),
                }
            case ValueType.NOMINAL:
                series = df[col].dropna()
                if series.empty:
                    continue
                counts_n = _filter_to_known_classes(Counter(int(float(v)) for v in series), col)
                if not counts_n:
                    continue
                acc["nominal"][col] = {"counts": counts_n, "unique": _col_unique(col)}
            case ValueType.ORDINAL:
                series = df[col].dropna()
                if series.empty:
                    continue
                counts_o = _filter_to_known_classes(Counter(int(float(v)) for v in series), col)
                if not counts_o:
                    continue
                acc["ordinal"][col] = {"counts": counts_o, "unique": _col_unique(col)}
            case ValueType.CIRCULAR:
                raw = df[col]
                series = (raw.dropna() if raw.dtype == np.float64
                          else pd.to_numeric(raw, errors="coerce").dropna())
                if series.empty:
                    continue
                values = series.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rad = np.deg2rad(values)
                reservoir = []
                n_seen = _reservoir_update(reservoir, 0, values)
                acc["circular"][col] = {
                    "cos_sum": float(np.sum(np.cos(rad))),
                    "sin_sum": float(np.sum(np.sin(rad))),
                    "n": len(values), "reservoir": reservoir,
                    "n_seen": n_seen, "unique": _col_unique(col),
                }
    acc["pheno"] = dict(compute_phenology_counts(df))
    return acc


def _reservoir_batch_merge(parts: list[tuple[list, int]]) -> tuple[list, int]:
    """Merge N reservoir samples in one proportional draw — O(sum of sizes), not O(N × max)."""
    total_n = sum(n for _, n in parts)
    if total_n == 0:
        return [], 0
    combined_size = sum(len(r) for r, _ in parts)
    if combined_size <= _KDE_MAX_SAMPLES:
        merged = []
        for r, _ in parts:
            merged.extend(r)
        return merged, total_n
    result = []
    for r, n in parts:
        take = max(0, min(round(_KDE_MAX_SAMPLES * n / total_n), len(r)))
        if take == 0:
            continue
        if take >= len(r):
            result.extend(r)
        else:
            arr = np.asarray(r, dtype=np.float64)
            result.extend(arr[np.random.permutation(len(arr))[:take]].tolist())
    return result, total_n


def _merge_accs_batch(accs: list[dict]) -> dict:
    """Merge a list of accumulators efficiently — single proportional reservoir draw per column."""
    merged: dict = {"continuous": {}, "circular": {}, "nominal": {}, "ordinal": {}, "pheno": {}}

    # Gather all per-column contributions, then merge in one shot.
    cont_parts: dict[str, list] = {}
    circ_parts: dict[str, list] = {}

    for acc in accs:
        for col, s in acc.get("continuous", {}).items():
            if col not in cont_parts:
                cont_parts[col] = []
            cont_parts[col].append(s)

        for col, s in acc.get("circular", {}).items():
            if col not in circ_parts:
                circ_parts[col] = []
            circ_parts[col].append(s)

        for col, s in acc.get("nominal", {}).items():
            if col not in merged["nominal"]:
                merged["nominal"][col] = {"counts": Counter(s["counts"]), "unique": s["unique"]}
            else:
                t = merged["nominal"][col]
                t["counts"].update(s["counts"])
                t["unique"] += s["unique"]

        for col, s in acc.get("ordinal", {}).items():
            if col not in merged["ordinal"]:
                merged["ordinal"][col] = {"counts": Counter(s["counts"]), "unique": s["unique"]}
            else:
                t = merged["ordinal"][col]
                t["counts"].update(s["counts"])
                t["unique"] += s["unique"]

        for k, v in acc.get("pheno", {}).items():
            merged["pheno"][k] = merged["pheno"].get(k, 0) + v

    for col, parts in cont_parts.items():
        digest = parts[0]["digest"]
        for p in parts[1:]:
            digest.merge_inplace(p["digest"])
        reservoir, n_seen = _reservoir_batch_merge([(p["reservoir"], p["n_seen"]) for p in parts])
        merged["continuous"][col] = {
            "digest": digest,
            "reservoir": reservoir,
            "n_seen": n_seen,
            "unique": sum(p["unique"] for p in parts),
        }

    for col, parts in circ_parts.items():
        reservoir, n_seen = _reservoir_batch_merge([(p["reservoir"], p["n_seen"]) for p in parts])
        merged["circular"][col] = {
            "cos_sum": sum(p["cos_sum"] for p in parts),
            "sin_sum": sum(p["sin_sum"] for p in parts),
            "n": sum(p["n"] for p in parts),
            "reservoir": reservoir,
            "n_seen": n_seen,
            "unique": sum(p["unique"] for p in parts),
        }

    return merged


def _merge_acc_inplace(target: dict, source: dict) -> None:
    """Merge source accumulator into target in-place (used for own-parquet + children merge)."""
    for col, s in source.get("continuous", {}).items():
        if col not in target["continuous"]:
            target["continuous"][col] = {
                "digest": s["digest"], "reservoir": list(s["reservoir"]),
                "n_seen": s["n_seen"], "unique": s["unique"],
            }
        else:
            t = target["continuous"][col]
            t["digest"].merge_inplace(s["digest"])
            reservoir, n_seen = _reservoir_batch_merge(
                [(t["reservoir"], t["n_seen"]), (s["reservoir"], s["n_seen"])]
            )
            t["reservoir"], t["n_seen"] = reservoir, n_seen
            t["unique"] += s["unique"]

    for col, s in source.get("circular", {}).items():
        if col not in target["circular"]:
            target["circular"][col] = {
                "cos_sum": s["cos_sum"], "sin_sum": s["sin_sum"], "n": s["n"],
                "reservoir": list(s["reservoir"]), "n_seen": s["n_seen"], "unique": s["unique"],
            }
        else:
            t = target["circular"][col]
            t["cos_sum"] += s["cos_sum"]
            t["sin_sum"] += s["sin_sum"]
            t["n"] += s["n"]
            reservoir, n_seen = _reservoir_batch_merge(
                [(t["reservoir"], t["n_seen"]), (s["reservoir"], s["n_seen"])]
            )
            t["reservoir"], t["n_seen"] = reservoir, n_seen
            t["unique"] += s["unique"]

    for col, s in source.get("nominal", {}).items():
        if col not in target["nominal"]:
            target["nominal"][col] = {"counts": Counter(s["counts"]), "unique": s["unique"]}
        else:
            t = target["nominal"][col]
            t["counts"].update(s["counts"])
            t["unique"] += s["unique"]

    for col, s in source.get("ordinal", {}).items():
        if col not in target["ordinal"]:
            target["ordinal"][col] = {"counts": Counter(s["counts"]), "unique": s["unique"]}
        else:
            t = target["ordinal"][col]
            t["counts"].update(s["counts"])
            t["unique"] += s["unique"]

    for k, v in source.get("pheno", {}).items():
        target["pheno"][k] = target["pheno"].get(k, 0) + v


def _save_acc(taxon_dir: Path, acc: dict) -> None:
    data = {
        "continuous": {
            col: {
                "digest_bytes": a["digest"].to_bytes(),
                "reservoir": a["reservoir"], "n_seen": a["n_seen"], "unique": a["unique"],
            }
            for col, a in acc["continuous"].items()
        },
        "circular": {col: dict(a) for col, a in acc["circular"].items()},
        "nominal": {col: {"counts": dict(a["counts"]), "unique": a["unique"]}
                    for col, a in acc["nominal"].items()},
        "ordinal": {col: {"counts": dict(a["counts"]), "unique": a["unique"]}
                    for col, a in acc["ordinal"].items()},
        "pheno": acc["pheno"],
    }
    with open(taxon_dir / _ACC_FILE, "wb") as f:
        pickle.dump(data, f, protocol=4)


def _load_acc(taxon_dir: Path) -> dict | None:
    acc_path = taxon_dir / _ACC_FILE
    if not acc_path.exists():
        return None
    try:
        with open(acc_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    return {
        "continuous": {
            col: {
                "digest": TDigest.from_bytes(a["digest_bytes"]),
                "reservoir": a["reservoir"], "n_seen": a["n_seen"], "unique": a["unique"],
            }
            for col, a in data["continuous"].items()
        },
        "circular": {col: dict(a) for col, a in data["circular"].items()},
        "nominal": {
            col: {"counts": Counter(a["counts"]), "unique": a["unique"]}
            for col, a in data["nominal"].items()
        },
        "ordinal": {
            col: {"counts": Counter(a["counts"]), "unique": a["unique"]}
            for col, a in data.get("ordinal", {}).items()
        },
        "pheno": dict(data["pheno"]),
    }


def _write_stats_from_acc(taxon_dir: Path, acc: dict, layer_meta: dict[str, dict]) -> None:
    """Compute and write stats files from a merged accumulator."""
    numerical_stats: dict[str, dict] = {}
    circular_stats: dict[str, dict] = {}
    nominal_entries: list[dict] = []
    density_rows: list[dict] = []

    for col, a in acc["continuous"].items():
        if col not in layer_meta:
            continue
        layer = layer_meta[col]
        vtype = _layer_value_type(layer)
        digest = a["digest"]
        reservoir = np.array(a["reservoir"], dtype=float)
        reservoir = reservoir[np.isfinite(reservoir)]
        if _is_discrete(layer):
            counts = Counter(int(v) for v in reservoir)
            mode_val = counts.most_common(1)[0][0] if counts else None
            stats = _continuous_stats_streaming(digest, a["unique"], None)
            stats["mode"] = mode_val
            if counts:
                total_c = sum(counts.values())
                probs_c = np.array([c / total_c for c in counts.values()], dtype=float)
                stats["entropy"] = float(_scipy_entropy(probs_c))
            if counts:
                total = sum(counts.values())
                min_val, max_val = min(counts), max(counts)
                all_bins = [(k, counts.get(k, 0)) for k in range(min_val, max_val + 1)]
                density_rows.append({
                    "variable": col,
                    "count": int(digest.n_values),
                    "sampleCount": len(reservoir),
                    "pointCount": len(all_bins),
                    "points": [float(k) for k, _ in all_bins],
                    "density": [float(v / total) for _, v in all_bins],
                    "min": float(min_val),
                    "max": float(max_val),
                    "bandwidth": 0.0,
                })
        else:
            kde = build_density_curve(reservoir, vtype) if vtype is not None and reservoir.size >= 2 else None
            stats = _continuous_stats_streaming(digest, a["unique"], kde)
            if kde is not None:
                xs = np.array(kde["points"])
                dens = np.array(kde["density"])
                mask = dens > 0
                v = float(-np.trapezoid(dens[mask] * np.log(dens[mask]), xs[mask]))
                if math.isfinite(v):
                    stats["entropy"] = v
            if kde:
                density_rows.append({
                    "variable": col,
                    "count": stats["count"],
                    "sampleCount": len(reservoir),
                    "pointCount": len(kde["points"]),
                    "points": kde["points"],
                    "density": kde["density"],
                    "min": kde["min"],
                    "max": kde["max"],
                    "bandwidth": kde["bandwidth"],
                })
        numerical_stats[col] = stats

    for col, a in acc["circular"].items():
        if col not in layer_meta or a["n"] == 0:
            continue
        reservoir = np.array(a["reservoir"], dtype=float)
        reservoir = reservoir[np.isfinite(reservoir)]
        kde = build_density_curve(reservoir, ValueType.CIRCULAR) if reservoir.size >= 2 else None
        stats = _circ_stats_streaming(a["cos_sum"], a["sin_sum"], a["n"], a["unique"], kde)
        if kde:
            density_rows.append({
                "variable": col,
                "count": stats["count"],
                "sampleCount": len(reservoir),
                "pointCount": len(kde["points"]),
                "points": kde["points"],
                "density": kde["density"],
                "min": kde["min"],
                "max": kde["max"],
                "bandwidth": kde["bandwidth"],
            })
        circular_stats[col] = stats

    for col, a in acc["nominal"].items():
        if col not in layer_meta:
            continue
        layer = layer_meta[col]
        counts = a["counts"]
        summary, _ = _nominal_stats(counts, a["unique"])
        if summary:
            nominal_entries.extend(_nominal_cat_entries(col, layer, counts, summary))

    ordinal_entries: list[dict] = []
    for col, a in acc["ordinal"].items():
        if col not in layer_meta:
            continue
        layer = layer_meta[col]
        counts = a["counts"]
        stats = _ordinal_stats(counts, a["unique"])
        if not stats:
            continue
        ordinal_entries.extend(_ordinal_stat_entries(col, layer, counts, stats))

    if not numerical_stats and not nominal_entries and not circular_stats and not ordinal_entries:
        return
    pheno_acc = Counter(acc.get("pheno", {}))
    pheno_meta = {"phenology_counts": json.dumps(dict(pheno_acc))} if pheno_acc else None
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_frame(taxon_dir / NUMERICAL_STATS_FILE, numerical_stats, pheno_meta)
    _write_stats_frame(taxon_dir / CIRCULAR_STATS_FILE, circular_stats)
    _write_nominal_stats(taxon_dir, nominal_entries)
    _write_ordinal_stats(taxon_dir, ordinal_entries)
    _write_density(taxon_dir, density_rows)


def _atomic_write(path: Path, table: pa.Table, custom_metadata: dict[str, str] | None = None) -> None:
    if custom_metadata:
        existing = table.schema.metadata or {}
        merged = {**existing, **{k.encode(): v.encode() for k, v in custom_metadata.items()}}
        table = table.replace_schema_metadata(merged)
    atomic_write_parquet(path, table)


# ---------------------------------------------------------------------------
# KDE / density curve
# ---------------------------------------------------------------------------

_FFT_GRID = 512  # grid size for FFTKDE — power of 2 for efficiency


def _gaussian_kde_curve(values: np.ndarray, bounded_at_zero: bool = False) -> dict | None:
    if values.size < 2:
        return None
    min_val, max_val = float(values.min()), float(values.max())
    if math.isclose(min_val, max_val):
        span = abs(min_val) * 0.1 or 1.0
        min_val -= span
        max_val += span
    try:
        n = len(values)
        std = float(np.std(values, ddof=1))
        if std < 1e-10:
            # All values effectively identical (std may be float noise) — use a small bandwidth
            h = abs(float(values[0])) * 0.01 or 0.1
        else:
            h = 1.06 * std * n ** (-0.2)

        if bounded_at_zero and min_val >= 0.0:
            # Reflection at 0: mirror data into the negative half so the KDE
            # boundary at 0 gets a zero-derivative correction, then fold back.
            work_vals = np.concatenate([-values, values])
            x_fine, density_fine = FFTKDE(bw=h).fit(work_vals).evaluate(_FFT_GRID)
            mask = x_fine >= 0.0
            x_fine, density_fine = x_fine[mask], density_fine[mask] * 2.0
            area = np.trapezoid(density_fine, x_fine)
            if area > 0:
                density_fine /= area
            # Sample output from actual data minimum, not from 0. The boundary
            # reflection is a statistical technique on the fine internal grid;
            # outputting from 0 extends the chart far into unobserved territory
            # for species with min > 0 (e.g. desert plants with precip >> 0mm).
            xs = np.linspace(min_val, max_val, _KDE_N_POINTS)
        else:
            x_fine, density_fine = FFTKDE(bw=h).fit(values).evaluate(_FFT_GRID)
            xs = np.linspace(min_val, max_val, _KDE_N_POINTS)

        density = np.maximum(np.interp(xs, x_fine, density_fine), 0.0)
        return {
            "points": xs.tolist(),
            "density": density.tolist(),
            "min": min_val,
            "max": max_val,
            "bandwidth": h,
            "mode": float(xs[int(np.argmax(density))]),
        }
    except Exception:
        return None


def _von_mises_kde_curve(values_deg: np.ndarray) -> dict | None:
    if values_deg.size < 2:
        return None
    try:
        n = len(values_deg)
        values_rad = np.deg2rad(values_deg)
        cstd_rad = float(circstd(values_rad, high=2 * np.pi, low=0.0, nan_policy="omit"))
        if not np.isfinite(cstd_rad) or cstd_rad < 1e-6:
            return None
        h = max((4.0 / (3.0 * n)) ** 0.2 * cstd_rad, 0.05)
        grid_deg = np.linspace(0.0, 360.0, _KDE_N_POINTS, endpoint=False)
        # FFT-based circular KDE: bin on [0,360) grid, convolve with wrapped Gaussian.
        counts, _ = np.histogram(np.degrees(values_rad) % 360.0,
                                 bins=_FFT_GRID, range=(0.0, 360.0))
        bin_width_deg = 360.0 / _FFT_GRID
        freqs = np.fft.rfftfreq(_FFT_GRID, d=bin_width_deg)
        h_deg = np.degrees(h)
        kernel_fft = np.exp(-2.0 * math.pi ** 2 * freqs ** 2 * h_deg ** 2)
        density_fine = np.fft.irfft(np.fft.rfft(counts.astype(np.float64)) * kernel_fft)[:_FFT_GRID]
        density_fine = np.maximum(density_fine, 0.0)
        area = density_fine.sum() * bin_width_deg
        if area > 0:
            density_fine /= area
        fine_centers = np.linspace(0.0, 360.0, _FFT_GRID, endpoint=False)
        density = np.interp(grid_deg, fine_centers, density_fine)
        mode_deg = float(grid_deg[int(np.argmax(density))])
        return {
            "points": grid_deg.tolist(),
            "density": density.tolist(),
            "min": 0.0,
            "max": 360.0,
            "bandwidth": float(np.degrees(h)),
            "mode": mode_deg,
        }
    except Exception:
        return None


def build_density_curve(values: np.ndarray, value_type: ValueType) -> dict | None:
    """Build a density curve for the given values and value type.

    Returns a dict with points/density/min/max/bandwidth/mode, or None.
    """
    match value_type:
        case ValueType.RATIO:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return _gaussian_kde_curve(arr, bounded_at_zero=True)
        case ValueType.INTERVAL:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return _gaussian_kde_curve(arr)
        case ValueType.CIRCULAR:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return _von_mises_kde_curve(arr)
        case _:
            return None
    return None


# ---------------------------------------------------------------------------
# Stats computation — circular
# ---------------------------------------------------------------------------

_CIRC_KW = dict(high=360.0, low=0.0, nan_policy="omit")


def _circ_stats_exact(series: pd.Series, unique_samples: int, kde: dict | None) -> dict:
    values = series.to_numpy(dtype=float)
    var_ = float(circvar(values, **_CIRC_KW))
    return {
        "count": int(series.size),
        "unique_samples": unique_samples,
        "circular_mean": float(circmean(values, **_CIRC_KW)),
        "rbar": 1.0 - var_,
        "circular_var": var_,
        "circular_std": float(circstd(values, **_CIRC_KW)),
        "mode": kde["mode"] if kde else None,
    }


def _circular_entropy(rbar: float) -> float:
    """Von Mises differential entropy on [0, 2π] from mean resultant length rbar.

    Uses exponentially scaled Bessel functions (ive) so the ratio and entropy
    formula stay numerically stable for arbitrarily large kappa.
    """
    if rbar <= 0.0:
        return math.log(2 * math.pi)   # uniform: maximum entropy
    if rbar >= 1.0 - 1e-9:
        return float("-inf")            # near-point-mass: kappa → ∞, entropy → -∞
    # ive(1,k)/ive(0,k) = I1(k)/I0(k) (exp(-k) cancels) — no overflow at large k.
    # A(κ) ≈ 1 - 1/(2κ) for large κ, so κ ≈ 1/(2*(1-rbar)).
    upper = max(1e6, 1.0 / (1.0 - rbar))
    try:
        kappa = _brentq(lambda k: _bessel_ive(1, k) / _bessel_ive(0, k) - rbar, 0.0, upper)
    except ValueError:
        return float("-inf")
    # log(2π·I0(κ)) - κ·rbar  =  log(2π) + log(ive(0,κ)) + κ·(1 - rbar)
    v = float(math.log(2 * math.pi) + math.log(_bessel_ive(0, kappa)) + kappa * (1.0 - rbar))
    return v if math.isfinite(v) else float("-inf")


def _circ_stats_streaming(
    cos_sum: float, sin_sum: float, n: int, unique_samples: int, kde: dict | None
) -> dict:
    xbar = cos_sum / n
    ybar = sin_sum / n
    rbar = float(np.sqrt(xbar ** 2 + ybar ** 2))
    mean_deg = float(np.degrees(np.arctan2(ybar, xbar)) % 360.0)
    var_ = 1.0 - rbar
    std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(max(rbar, 1e-10)))))
    return {
        "count": n,
        "unique_samples": unique_samples,
        "circular_mean": mean_deg,
        "rbar": rbar,
        "circular_var": var_,
        "circular_std": std_deg,
        "entropy": _circular_entropy(rbar),
        "mode": kde["mode"] if kde else None,
    }


# ---------------------------------------------------------------------------
# Stats computation — exact (leaf taxa)
# ---------------------------------------------------------------------------

def _continuous_stats_exact(
    values: np.ndarray, unique_samples: int, kde: dict | None, *, discrete: bool = False
) -> dict:
    """Exact continuous stats via numpy (faster than pd.describe for small arrays)."""
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    if discrete:
        counts = Counter(int(v) for v in values)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()], dtype=float)
        entropy_val: float | None = float(_scipy_entropy(probs)) if total > 0 else None
    elif kde is not None:
        xs = np.array(kde["points"])
        dens = np.array(kde["density"])
        mask = dens > 0
        entropy_val = float(-np.trapezoid(dens[mask] * np.log(dens[mask]), xs[mask]))
        if not math.isfinite(entropy_val):
            entropy_val = None
    else:
        entropy_val = None
    return {
        "count": len(values),
        "unique_samples": unique_samples,
        "min": float(values.min()),
        "10th_percentile": float(q10),
        "25th_percentile": float(q25),
        "median": float(q50),
        "75th_percentile": float(q75),
        "90th_percentile": float(q90),
        "max": float(values.max()),
        "mean": mean,
        "std": std,
        "variance": std ** 2,
        "iqr": float(q75 - q25),
        "10_90_range": float(q90 - q10),
        "range": float(values.max() - values.min()),
        "entropy": entropy_val,
        "mode": kde["mode"] if kde else None,
    }


# ---------------------------------------------------------------------------
# Stats computation — streaming (non-leaf taxa)
# ---------------------------------------------------------------------------

def _continuous_stats_streaming(digest: TDigest, unique_samples: int, kde: dict | None) -> dict:
    """Approximate continuous stats from a T-Digest accumulator."""
    q10 = float(digest.quantile(0.10))
    q25 = float(digest.quantile(0.25))
    q75 = float(digest.quantile(0.75))
    q90 = float(digest.quantile(0.90))
    return {
        "count": int(digest.n_values),
        "unique_samples": unique_samples,
        "min": float(digest.min()),
        "10th_percentile": q10,
        "25th_percentile": q25,
        "median": float(digest.quantile(0.50)),
        "75th_percentile": q75,
        "90th_percentile": q90,
        "max": float(digest.max()),
        "mean": float(digest.mean()),
        "std": float(digest.std()),
        "variance": float(digest.std()) ** 2,
        "iqr": float(digest.iqr()),
        "10_90_range": q90 - q10,
        "range": float(digest.max() - digest.min()),
        "mode": kde["mode"] if kde else None,
    }


# ---------------------------------------------------------------------------
# Stats computation — ordinal
# ---------------------------------------------------------------------------

def _ordinal_quantile(counts: Counter, p: float) -> float:
    """Exact pth quantile from a Counter of integer class IDs."""
    total = sum(counts.values())
    if total == 0:
        return float(min(counts))
    target = p * total
    cum = 0
    for val in sorted(counts):
        cum += counts[val]
        if cum >= target:
            return float(val)
    return float(max(counts))


def _ordinal_stats(counts: Counter, unique_samples: int) -> dict:
    """Ordinal summary stats: ordered quantiles + nominal distribution metrics."""
    total = sum(counts.values())
    if total == 0:
        return {}
    probs = np.array([counts[k] / total for k in sorted(counts)], dtype=float)
    entropy = float(_scipy_entropy(probs))
    mode_cls = counts.most_common(1)[0][0]

    def q(p: float) -> float:
        return _ordinal_quantile(counts, p)

    return {
        "count": total,
        "unique_samples": unique_samples,
        "total_samples": total,
        "unique_classes": len(counts),
        "entropy": entropy,
        "mode": float(mode_cls),
        "10th_percentile": q(0.10),
        "25th_percentile": q(0.25),
        "median": q(0.50),
        "75th_percentile": q(0.75),
        "90th_percentile": q(0.90),
    }


def _ordinal_stat_entries(layer_id: str, layer: dict, counts: Counter, stats: dict) -> list[dict]:
    """All ordinal_stats.parquet tall rows for one variable: quantile metrics + class fractions."""
    total = stats["total_samples"]
    entries: list[dict] = []
    for metric in (
        "count", "unique_samples", "total_samples", "unique_classes", "entropy",
        "mode", "10th_percentile", "25th_percentile", "median", "75th_percentile", "90th_percentile",
    ):
        entries.append({"variable": layer_id, "metric": metric, "value": float(stats[metric])})
    for cls_id, count in counts.items():
        entries.append({"variable": layer_id, "metric": f"class_{cls_id}", "value": count / total if total else 0.0})
    base_id = re.sub(r'_(avg|sum|mode|mean|min|max)_\d+h$', '', layer_id)
    legend_path = Path("config/gis/legends") / f"{base_id}_legend.json"
    if legend_path.exists():
        try:
            known_ids = {int(c["id"]) for c in json.loads(legend_path.read_text()).get("classes", [])}
            for cls_id in known_ids:
                if cls_id not in counts:
                    entries.append({"variable": layer_id, "metric": f"class_{cls_id}", "value": 0.0})
        except Exception:
            pass
    return entries


# ---------------------------------------------------------------------------
# Stats computation — nominal
# ---------------------------------------------------------------------------

def _nominal_stats(counts: Counter, unique_samples: int) -> tuple[dict, list[dict]]:
    """Nominal summary stats + sorted class distribution."""
    total = sum(counts.values())
    if total == 0:
        return {}, []
    fractions = {k: v / total for k, v in counts.items()}
    probs = np.array(list(fractions.values()), dtype=float)
    entropy = float(_scipy_entropy(probs))
    mode_cls = counts.most_common(1)[0][0]
    summary = {
        "unique_samples": unique_samples,
        "total_samples": total,
        "unique_classes": len(counts),
        "entropy": entropy,
        "mode": mode_cls,
    }
    distribution = sorted(
        [{"class_id": k, "fraction": v} for k, v in fractions.items()],
        key=lambda e: -e["fraction"],
    )
    return summary, distribution


def _nominal_cat_entries(layer_id: str, layer: dict, counts: Counter, summary: dict) -> list[dict]:
    total = summary["total_samples"]
    entries: list[dict] = [
        {"variable": layer_id, "metric": "unique_samples", "value": float(summary["unique_samples"])},
        {"variable": layer_id, "metric": "total_samples",  "value": float(total)},
        {"variable": layer_id, "metric": "unique_classes", "value": float(summary["unique_classes"])},
        {"variable": layer_id, "metric": "entropy",        "value": float(summary["entropy"])},
    ]
    entries.append({"variable": layer_id, "metric": "mode", "value": float(summary["mode"])})
    for cls_id, count in counts.items():
        fraction = count / total if total else 0.0
        entries.append({"variable": layer_id, "metric": f"class_{cls_id}", "value": fraction})
    # Add zero entries for all legend classes not observed, so every class
    # appears in the rank index and search results include this taxon when
    # sorting by that class ascending.
    base_id = re.sub(r'_(avg|sum|mode|mean|min|max)_\d+h$', '', layer_id)
    legend_path = Path("config/gis/legends") / f"{base_id}_legend.json"
    if legend_path.exists():
        try:
            known_ids = {int(c["id"]) for c in json.loads(legend_path.read_text()).get("classes", [])}
            for cls_id in known_ids:
                if cls_id not in counts:
                    entries.append({"variable": layer_id, "metric": f"class_{cls_id}", "value": 0.0})
        except Exception:
            pass
    return entries


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_stats_frame(path: Path, stats: dict[str, dict], custom_metadata: dict[str, str] | None = None) -> None:
    if not stats:
        return
    frame = pd.DataFrame.from_dict(stats, orient="index")
    frame.index.name = "variable"
    frame = frame.reset_index()
    _atomic_write(path, pa.Table.from_pandas(frame, preserve_index=False), custom_metadata)


def _write_nominal_stats(directory: Path, entries: list[dict]) -> None:
    if not entries:
        return
    frame = pd.DataFrame(entries)
    _atomic_write(directory / NOMINAL_STATS_FILE, pa.Table.from_pandas(frame, preserve_index=False))


def _write_ordinal_stats(directory: Path, entries: list[dict]) -> None:
    if not entries:
        return
    frame = pd.DataFrame(entries)
    _atomic_write(directory / ORDINAL_STATS_FILE, pa.Table.from_pandas(frame, preserve_index=False))


def _write_density(directory: Path, rows: list[dict]) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    _atomic_write(directory / DENSITY_FILE, table)


# ---------------------------------------------------------------------------
# Leaf (exact) processing
# ---------------------------------------------------------------------------

def _process_leaf_df(taxon_dir: Path, df: pd.DataFrame, layer_meta: dict[str, dict]) -> None:
    """Compute exact stats from a pre-loaded, pre-filtered DataFrame and write all outputs."""
    gis_cols = [col for col in df.columns if col in layer_meta]
    if not gis_cols:
        return

    numerical_stats: dict[str, dict] = {}
    circular_stats: dict[str, dict] = {}
    nominal_entries: list[dict] = []
    ordinal_entries: list[dict] = []
    density_rows: list[dict] = []

    # Cache total unique count — reused across columns with no nulls (the common case).
    _total_unique: int | None = None

    def _col_unique(col: str) -> int:
        nonlocal _total_unique
        if not df[col].isna().any():
            if _total_unique is None:
                _total_unique = int(df["catalogNumber"].nunique())
            return _total_unique
        return int(df.loc[df[col].notna(), "catalogNumber"].nunique())

    for col in gis_cols:
        layer = layer_meta[col]
        vtype = _layer_value_type(layer)
        if vtype is None:
            continue

        match vtype:
            case ValueType.RATIO | ValueType.INTERVAL:
                raw = df[col]
                values = (raw.to_numpy(dtype=np.float64, na_value=np.nan)
                          if raw.dtype == np.float64
                          else pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64))
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                unique = _col_unique(col)
                if _is_discrete(layer):
                    counts_c = Counter(int(v) for v in values)
                    stats = _continuous_stats_exact(values, unique, None, discrete=True)
                    stats["mode"] = counts_c.most_common(1)[0][0]
                    min_val, max_val = int(values.min()), int(values.max())
                    total = len(values)
                    all_bins = [(k, counts_c.get(k, 0)) for k in range(min_val, max_val + 1)]
                    density_rows.append({
                        "variable": col,
                        "count": stats["count"],
                        "sampleCount": total,
                        "pointCount": len(all_bins),
                        "points": [float(k) for k, _ in all_bins],
                        "density": [float(v / total) for _, v in all_bins],
                        "min": float(min_val),
                        "max": float(max_val),
                        "bandwidth": 0.0,
                    })
                else:
                    kde = build_density_curve(values, vtype)
                    stats = _continuous_stats_exact(values, unique, kde)
                    if kde:
                        density_rows.append({
                            "variable": col,
                            "count": stats["count"],
                            "sampleCount": len(values),
                            "pointCount": len(kde["points"]),
                            "points": kde["points"],
                            "density": kde["density"],
                            "min": kde["min"],
                            "max": kde["max"],
                            "bandwidth": kde["bandwidth"],
                        })
                numerical_stats[col] = stats

            case ValueType.NOMINAL:
                raw = df[col].dropna()
                if raw.empty:
                    continue
                unique = _col_unique(col)
                raw_counts: Counter = _filter_to_known_classes(Counter(int(float(v)) for v in raw), col)
                if not raw_counts:
                    continue
                summary, _ = _nominal_stats(raw_counts, unique)
                nominal_entries.extend(_nominal_cat_entries(col, layer, raw_counts, summary))

            case ValueType.ORDINAL:
                raw = df[col].dropna()
                if raw.empty:
                    continue
                unique = _col_unique(col)
                ord_counts: Counter = _filter_to_known_classes(Counter(int(float(v)) for v in raw), col)
                if not ord_counts:
                    continue
                stats = _ordinal_stats(ord_counts, unique)
                if not stats:
                    continue
                ordinal_entries.extend(_ordinal_stat_entries(col, layer, ord_counts, stats))

            case ValueType.CIRCULAR:
                raw = df[col]
                values = (raw.to_numpy(dtype=np.float64, na_value=np.nan)
                          if raw.dtype == np.float64
                          else pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64))
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                unique = _col_unique(col)
                kde = build_density_curve(values, vtype)
                rad = np.deg2rad(values)
                cos_s = float(np.sum(np.cos(rad)))
                sin_s = float(np.sum(np.sin(rad)))
                stats = _circ_stats_streaming(cos_s, sin_s, len(values), unique, kde)
                if kde:
                    density_rows.append({
                        "variable": col,
                        "count": stats["count"],
                        "sampleCount": len(values),
                        "pointCount": len(kde["points"]),
                        "points": kde["points"],
                        "density": kde["density"],
                        "min": kde["min"],
                        "max": kde["max"],
                        "bandwidth": kde["bandwidth"],
                    })
                circular_stats[col] = stats

            case _:
                raise NotImplementedError(f"Stats not implemented for value type {vtype!r}")

    pheno_counts = compute_phenology_counts(df)
    pheno_meta = {"phenology_counts": json.dumps(dict(pheno_counts))} if pheno_counts else None
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_frame(taxon_dir / NUMERICAL_STATS_FILE, numerical_stats, pheno_meta)
    _write_stats_frame(taxon_dir / CIRCULAR_STATS_FILE, circular_stats)
    _write_nominal_stats(taxon_dir, nominal_entries)
    _write_ordinal_stats(taxon_dir, ordinal_entries)
    _write_density(taxon_dir, density_rows)


def process_observations_df(directory: Path, df: pd.DataFrame, layer_meta: dict[str, dict]) -> None:
    """Compute stats and write all outputs for an arbitrary observations DataFrame.

    Public entry point used by the upload pipeline. Behaves identically to the
    normal per-taxon leaf processing but operates on a caller-supplied DataFrame
    rather than reading from a fixed occurrence.parquet path.
    """
    _process_leaf_df(directory, df, layer_meta)


def _process_leaf(taxon_dir: Path, layer_meta: dict[str, dict]) -> None:
    occ_path = taxon_dir / OCCURRENCE_FILE
    if not occ_path.exists():
        return
    table = _read_occ_table(occ_path, layer_meta)
    if table.num_rows == 0:
        return
    df = _filter_df(table.to_pandas())
    if df.empty:
        return
    _process_leaf_df(taxon_dir, df, layer_meta)


def _collect_species_df(taxon: TaxonRecord, taxon_dir: Path, layer_meta: dict[str, dict]) -> pd.DataFrame | None:
    """Combine occurrence data for a SPECIES and all its subspecies-equivalent descendants.

    Deduplicates by catalogNumber so shared observations are not double-counted.
    Handles the edge case where the species itself has no observations but its
    subspecies do.
    """
    frames = []
    for desc in iter_descendants(taxon, include_self=True):
        occ_path = TREE_ROOT / desc["path"] / OCCURRENCE_FILE
        if not occ_path.exists():
            continue
        table = _read_occ_table(occ_path, layer_meta)
        if table.num_rows == 0:
            continue
        df = _filter_df(table.to_pandas())
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["catalogNumber"])


def _process_species(taxon: TaxonRecord, taxon_dir: Path, layer_meta: dict[str, dict]) -> None:
    """Compute exact stats for a SPECIES, rolling in all subspecies observations."""
    df = _collect_species_df(taxon, taxon_dir, layer_meta)
    if df is None or df.empty:
        return
    _process_leaf_df(taxon_dir, df, layer_meta)
    # Save accumulator so genus (parent) can merge without re-reading parquets.
    # The acc includes all subspecies data (already combined by _collect_species_df).
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _save_acc(taxon_dir, _df_to_acc(df, layer_meta))


def collect_taxon_df(taxon: TaxonRecord, storage: ParquetStorage | None = None) -> pd.DataFrame | None:
    """Quality-filtered occurrence DataFrame for a taxon, deduped by catalogNumber.

    Leaf (subspecies/variety): reads own occurrence file only.
    Species: reads self + descendants (include_self=True), deduplicates.
    Non-leaf: reads all descendants (include_self=False), deduplicates.
    """
    def _read(path: Path):
        if storage is not None:
            if not storage.exists(path):
                return None
            return storage.read_table(path)
        if not path.exists():
            return None
        return pq.read_table(path)

    rank = taxon["rank"]
    taxon_dir = TREE_ROOT / taxon["path"]
    if rank in CONFIG.subspecies_equivalents:
        occ_path = taxon_dir / OCCURRENCE_FILE
        table = _read(occ_path)
        if table is None or table.num_rows == 0:
            return None
        df = _filter_df(table.to_pandas())
        return df if not df.empty else None
    include_self = rank == CONFIG.species_rank
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for desc in iter_descendants(taxon, include_self=include_self):
        occ_path = TREE_ROOT / desc["path"] / OCCURRENCE_FILE
        table = _read(occ_path)
        if table is None or table.num_rows == 0:
            continue
        df = _filter_df(table.to_pandas())
        if df.empty:
            continue
        new = df[~df["catalogNumber"].astype(str).isin(seen)]
        seen.update(new["catalogNumber"].astype(str).tolist())
        frames.append(new)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_location_filtered_stats(
    taxon: TaxonRecord,
    variable_id: str,
    filter_col: str | None,
    gid: str | None,
    layer: dict,
    phenology: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    storage: ParquetStorage | None = None,
) -> dict | None:
    """Compute stats on the fly for variable_id, restricted by location, phenology, and/or timestamp."""
    df = collect_taxon_df(taxon, storage=storage)
    if df is None:
        return None
    if filter_col is not None:
        if filter_col not in df.columns:
            return None
        df = df[df[filter_col].astype(str) == str(gid)]
        if df.empty:
            return None
    if phenology is not None:
        df = apply_phenology_filter(df, phenology)
        if df.empty:
            return None
    if start_ts is not None or end_ts is not None:
        df = apply_timestamp_filter(df, start_ts, end_ts)
        if df.empty:
            return None
    if variable_id not in df.columns:
        return None
    vtype = _layer_value_type(layer)
    if vtype is None:
        return None
    unique = int(df[df[variable_id].notna()]["catalogNumber"].nunique())
    if vtype in (ValueType.RATIO, ValueType.INTERVAL):
        series = pd.to_numeric(df[variable_id], errors="coerce").dropna()
        if series.empty:
            return None
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if _is_discrete(layer):
            stats = _continuous_stats_exact(series[np.isfinite(series)], unique, None, discrete=True)
            stats["mode"] = int(series.value_counts().idxmax())
            bin_counts = series.value_counts().sort_index()
            min_val, max_val = int(values.min()), int(values.max())
            bin_counts = bin_counts.reindex(range(min_val, max_val + 1), fill_value=0)
            total = int(bin_counts.sum())
            density_curve: dict | None = {
                "points": [float(v) for v in bin_counts.index.tolist()],
                "density": [float(c / total) for c in bin_counts.tolist()],
            } if total > 0 else None
        else:
            kde = build_density_curve(values, vtype)
            stats = _continuous_stats_exact(series[np.isfinite(series)], unique, kde)
            density_curve = {"points": kde["points"], "density": kde["density"]} if kde else None
        return {"type": "continuous", "observation_count": stats["count"], "stats": stats, "density_curve": density_curve}
    if vtype == ValueType.NOMINAL:
        series = df[variable_id].dropna()
        if series.empty:
            return None
        raw_counts: Counter = Counter(int(float(v)) for v in series)
        summary, distribution = _nominal_stats(raw_counts, unique)
        return {"type": "nominal", "observation_count": summary["total_samples"], "summary": summary, "distribution": distribution}
    if vtype == ValueType.ORDINAL:
        series = df[variable_id].dropna()
        if series.empty:
            return None
        ord_counts: Counter = Counter(int(float(v)) for v in series)
        stats = _ordinal_stats(ord_counts, unique)
        if not stats:
            return None
        distribution = sorted(
            [{"class_id": k, "fraction": v / stats["total_samples"]} for k, v in ord_counts.items()],
            key=lambda e: e["class_id"],
        )
        return {"type": "ordinal", "observation_count": stats["count"], "stats": stats, "distribution": distribution}
    if vtype == ValueType.CIRCULAR:
        series = pd.to_numeric(df[variable_id], errors="coerce").dropna()
        if series.empty:
            return None
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        kde = build_density_curve(values, ValueType.CIRCULAR)
        rad = np.deg2rad(values)
        stats = _circ_stats_streaming(float(np.sum(np.cos(rad))), float(np.sum(np.sin(rad))), len(values), unique, kde)
        density_curve = {"points": kde["points"], "density": kde["density"]} if kde else None
        return {"type": "circular", "observation_count": stats["count"], "stats": stats, "density_curve": density_curve}
    return None




# ---------------------------------------------------------------------------
# Non-leaf (streaming) processing
# ---------------------------------------------------------------------------

def _process_nonleaf(taxon: TaxonRecord, taxon_dir: Path, layer_meta: dict[str, dict]) -> None:
    child_accs: list[dict] = []

    # Include any direct observations on this taxon (e.g. genus-level GBIF records
    # not identified to species). Rare but valid.
    occ_path = taxon_dir / OCCURRENCE_FILE
    if occ_path.exists():
        table = _read_occ_table(occ_path, layer_meta)
        if table.num_rows > 0:
            df = _filter_df(table.to_pandas())
            if not df.empty:
                child_accs.append(_df_to_acc(df, layer_meta))

    # Collect all direct children's accumulators, then batch-merge in one shot.
    # Each child already accumulated its entire subtree (species acc = species + subspecies).
    for child in get_children(taxon["taxon_key"]):
        child_acc = _load_acc(TREE_ROOT / child["path"])
        if child_acc is not None:
            child_accs.append(child_acc)

    if not child_accs:
        return

    acc = _merge_accs_batch(child_accs)

    taxon_dir.mkdir(parents=True, exist_ok=True)
    _save_acc(taxon_dir, acc)
    _write_stats_from_acc(taxon_dir, acc, layer_meta)




# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def stats_complete(taxon: TaxonRecord) -> bool:
    """True if stats for this taxon have already been written and can be skipped."""
    taxon_dir = TREE_ROOT / taxon["path"]
    if not (taxon_dir / NUMERICAL_STATS_FILE).exists():
        return False
    # Non-leaf taxa must also have their .acc file so the parent can merge it.
    if taxon["rank"] not in CONFIG.subspecies_equivalents:
        return (taxon_dir / _ACC_FILE).exists()
    return True


def compute_taxon_stats(
    taxon: TaxonRecord,
    layers: list[dict],
    layer_meta: dict[str, dict] | None = None,
    resume: bool = False,
) -> None:
    """Compute and write summary stats for one taxon node.

    SUBSPECIES/VARIETY/FORM use exact stats from their own occurrence file.
    SPECIES combine their own observations with any subspecies-equivalent descendants
    before computing exact stats (so a species always reflects all sub-rank obs).
    Higher taxa stream all descendant parquets via T-Digest approximations.
    Must be called in leaf-first (bottom-up) order so non-leaf index builds
    can read from already-completed children's occurrence_index.parquet files.

    ``layer_meta`` may be pre-built and passed in to avoid rebuilding it for every taxon.
    """
    taxon_dir = TREE_ROOT / taxon["path"]
    if resume and stats_complete(taxon):
        return
    if layer_meta is None:
        layer_meta = {layer["id"]: layer for layer in layers}
    rank = taxon["rank"]
    if rank in CONFIG.subspecies_equivalents:
        _process_leaf(taxon_dir, layer_meta)
    elif rank == CONFIG.species_rank:
        _process_species(taxon, taxon_dir, layer_meta)
    else:
        _process_nonleaf(taxon, taxon_dir, layer_meta)


