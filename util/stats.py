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
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastdigest import TDigest
from scipy.stats import circmean, circstd, circvar, gaussian_kde, vonmises
from scipy.stats import entropy as _scipy_entropy

from config.config import ValueType, load_config
from util.taxa import TaxonRecord, get_children, iter_descendants

CONFIG = load_config("global")

TREE_ROOT = Path("data/taxonomy/tree")
OCCURRENCE_FILE = "occurrence.parquet"
OCCURRENCE_INDEX_FILE = "occurrence_index.parquet"
NUMERICAL_STATS_FILE = "numerical_stats.parquet"
NOMINAL_STATS_FILE = "nominal_stats.parquet"
CIRCULAR_STATS_FILE = "circular_stats.parquet"
DENSITY_FILE = "density.parquet"
PHENOLOGY_COUNTS_FILE = "phenology_counts.json"

_KDE_MAX_SAMPLES = 20_000
_KDE_N_POINTS = 128


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

# Columns present in occurrence.parquet that are NOT GIS layer values and should
# be stripped from the slice index (quality-filter cols are applied then dropped).
_INDEX_STRIP_COLS = frozenset([
    "hilbertIdx", "eventTimestamp", "coordinateUncertaintyInMeters", "obscured",
    "gbifRegion", "level0Gid", "level1Gid", "level2Gid", "dp", "vitality", "rcs",
])

# Ranks for which occurrence_index.parquet is built.
# Order and above aggregate too many descendants to be useful for slice queries.
_INDEX_RANKS = CONFIG.leaf_rank_set | frozenset(["GENUS", "FAMILY"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_value_type(layer: dict) -> ValueType | None:
    try:
        return ValueType(layer.get("value_type", ""))
    except ValueError:
        return None


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
    p = taxon_dir / PHENOLOGY_COUNTS_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if "obscured" in df.columns:
        df = df[df["obscured"] == "No"]
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[df["coordinateUncertaintyInMeters"] <= 500]
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


def _atomic_write(path: Path, table: pa.Table) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# KDE / density curve
# ---------------------------------------------------------------------------

def _gaussian_kde_curve(values: np.ndarray) -> dict | None:
    if values.size < 2:
        return None
    min_val, max_val = float(values.min()), float(values.max())
    if math.isclose(min_val, max_val):
        span = abs(min_val) * 0.1 or 1.0
        min_val -= span
        max_val += span
    try:
        kde = gaussian_kde(values, bw_method="silverman")
        xs = np.linspace(min_val, max_val, _KDE_N_POINTS)
        density = kde(xs)
        return {
            "points": xs.tolist(),
            "density": density.tolist(),
            "min": min_val,
            "max": max_val,
            "bandwidth": float(kde.factor * float(values.std())),
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
        kappa = 1.0 / h ** 2
        grid_deg = np.linspace(0.0, 360.0, _KDE_N_POINTS, endpoint=False)
        grid_rad = np.deg2rad(grid_deg)
        # Vectorised: evaluate all sample kernels at once (n × _KDE_N_POINTS broadcast)
        density = vonmises.pdf(
            grid_rad[np.newaxis, :], kappa, loc=values_rad[:, np.newaxis]
        ).sum(axis=0) / n
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
        case ValueType.RATIO | ValueType.INTERVAL:
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
        "mode": kde["mode"] if kde else None,
    }


# ---------------------------------------------------------------------------
# Stats computation — exact (leaf taxa)
# ---------------------------------------------------------------------------

def _continuous_stats_exact(series: pd.Series, unique_samples: int, kde: dict | None) -> dict:
    """Exact continuous stats via pd.describe() + KDE mode."""
    desc = series.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
    q10 = float(desc["10%"])
    q25 = float(desc["25%"])
    q75 = float(desc["75%"])
    q90 = float(desc["90%"])
    return {
        "count": int(desc["count"]),
        "unique_samples": unique_samples,
        "min": float(desc["min"]),
        "10th_percentile": q10,
        "25th_percentile": q25,
        "median": float(desc["50%"]),
        "75th_percentile": q75,
        "90th_percentile": q90,
        "max": float(desc["max"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if math.isfinite(desc["std"]) else 0.0,
        "variance": float(desc["std"]) ** 2 if math.isfinite(desc["std"]) else 0.0,
        "iqr": q75 - q25,
        "10_90_range": q90 - q10,
        "range": float(desc["max"] - desc["min"]),
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
    legend_path = Path("config/gis/legends") / f"{layer_id}_legend.json"
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

def _write_stats_frame(path: Path, stats: dict[str, dict]) -> None:
    if not stats:
        return
    frame = pd.DataFrame.from_dict(stats, orient="index")
    frame.index.name = "variable"
    frame = frame.reset_index()
    _atomic_write(path, pa.Table.from_pandas(frame, preserve_index=False))


def _write_nominal_stats(directory: Path, entries: list[dict]) -> None:
    if not entries:
        return
    frame = pd.DataFrame(entries)
    _atomic_write(directory / NOMINAL_STATS_FILE, pa.Table.from_pandas(frame, preserve_index=False))


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
    density_rows: list[dict] = []

    for col in gis_cols:
        layer = layer_meta[col]
        vtype = _layer_value_type(layer)
        if vtype is None:
            continue

        match vtype:
            case ValueType.RATIO | ValueType.INTERVAL:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if series.empty:
                    continue
                values = series.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                unique = int(df[df[col].notna()]["catalogNumber"].nunique())
                if _is_discrete(layer):
                    stats = _continuous_stats_exact(series[np.isfinite(series)], unique, None)
                    stats["mode"] = int(series.value_counts().idxmax())
                    bin_counts = series.value_counts().sort_index()
                    min_val, max_val = int(values.min()), int(values.max())
                    bin_counts = bin_counts.reindex(range(min_val, max_val + 1), fill_value=0)
                    total = int(bin_counts.sum())
                    density_rows.append({
                        "variable": col,
                        "count": stats["count"],
                        "sampleCount": len(values),
                        "pointCount": len(bin_counts),
                        "points": [float(v) for v in bin_counts.index.tolist()],
                        "density": [float(c / total) for c in bin_counts.tolist()],
                        "min": float(min_val),
                        "max": float(max_val),
                        "bandwidth": 0.0,
                    })
                else:
                    kde = build_density_curve(values, vtype)
                    stats = _continuous_stats_exact(series[np.isfinite(series)], unique, kde)
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
                series = df[col].dropna()
                if series.empty:
                    continue
                unique = int(df[df[col].notna()]["catalogNumber"].nunique())
                raw_counts: Counter = Counter()
                for v in series:
                    raw_counts[int(float(v))] += 1
                summary, _ = _nominal_stats(raw_counts, unique)
                nominal_entries.extend(_nominal_cat_entries(col, layer, raw_counts, summary))

            case ValueType.CIRCULAR:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if series.empty:
                    continue
                values = series.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                unique = int(df[df[col].notna()]["catalogNumber"].nunique())
                kde = build_density_curve(values, vtype)
                stats = _circ_stats_exact(series[np.isfinite(series)], unique, kde)
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

    taxon_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_frame(taxon_dir / NUMERICAL_STATS_FILE, numerical_stats)
    _write_stats_frame(taxon_dir / CIRCULAR_STATS_FILE, circular_stats)
    _write_nominal_stats(taxon_dir, nominal_entries)
    _write_density(taxon_dir, density_rows)
    _write_index_from_df(taxon_dir, df)
    write_phenology_counts(taxon_dir, compute_phenology_counts(df))


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
    table = pq.read_table(occ_path)
    if table.num_rows == 0:
        return
    df = _filter_df(table.to_pandas())
    if df.empty:
        return
    _process_leaf_df(taxon_dir, df, layer_meta)


def _collect_species_df(taxon: TaxonRecord, taxon_dir: Path) -> pd.DataFrame | None:
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
        table = pq.read_table(occ_path)
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
    df = _collect_species_df(taxon, taxon_dir)
    if df is None or df.empty:
        return
    _process_leaf_df(taxon_dir, df, layer_meta)


def collect_taxon_df(taxon: TaxonRecord) -> pd.DataFrame | None:
    """Quality-filtered occurrence DataFrame for a taxon, deduped by catalogNumber.

    Leaf (subspecies/variety): reads own occurrence file only.
    Species: reads self + descendants (include_self=True), deduplicates.
    Non-leaf: reads all descendants (include_self=False), deduplicates.
    """
    rank = taxon["rank"]
    taxon_dir = TREE_ROOT / taxon["path"]
    if rank in CONFIG.subspecies_equivalents:
        occ_path = taxon_dir / OCCURRENCE_FILE
        if not occ_path.exists():
            return None
        table = pq.read_table(occ_path)
        if table.num_rows == 0:
            return None
        df = _filter_df(table.to_pandas())
        return df if not df.empty else None
    include_self = rank == CONFIG.species_rank
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for desc in iter_descendants(taxon, include_self=include_self):
        occ_path = TREE_ROOT / desc["path"] / OCCURRENCE_FILE
        if not occ_path.exists():
            continue
        table = pq.read_table(occ_path)
        if table.num_rows == 0:
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
) -> dict | None:
    """Compute stats on the fly for variable_id, restricted by location, phenology, and/or timestamp."""
    df = collect_taxon_df(taxon)
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
            stats = _continuous_stats_exact(series[np.isfinite(series)], unique, None)
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
    if vtype == ValueType.CIRCULAR:
        series = pd.to_numeric(df[variable_id], errors="coerce").dropna()
        if series.empty:
            return None
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        kde = build_density_curve(values, ValueType.CIRCULAR)
        stats = _circ_stats_exact(series[np.isfinite(series)], unique, kde)
        density_curve = {"points": kde["points"], "density": kde["density"]} if kde else None
        return {"type": "circular", "observation_count": stats["count"], "stats": stats, "density_curve": density_curve}
    return None


def _write_index_from_df(taxon_dir: Path, df: pd.DataFrame) -> None:
    """Write occurrence_index.parquet from an already-filtered DataFrame."""
    idx = df.drop(columns=[c for c in _INDEX_STRIP_COLS if c in df.columns])
    idx = idx.dropna(subset=["catalogNumber", "decimalLatitude", "decimalLongitude"])
    if idx.empty:
        return
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(taxon_dir / OCCURRENCE_INDEX_FILE, pa.Table.from_pandas(idx, preserve_index=False))


# ---------------------------------------------------------------------------
# Non-leaf (streaming) processing
# ---------------------------------------------------------------------------

def _process_nonleaf(taxon: TaxonRecord, taxon_dir: Path, layer_meta: dict[str, dict]) -> None:
    # continuous_acc: layer_id → {digest, reservoir, n_seen, unique}
    continuous_acc: dict[str, dict] = {}
    # nominal_acc: layer_id → {counts, unique}
    nominal_acc: dict[str, dict] = {}
    # circular_acc: layer_id → {cos_sum, sin_sum, n, reservoir, n_seen, unique}
    circular_acc: dict[str, dict] = {}
    pheno_acc: Counter = Counter()
    # index accumulation: deduplicated rows for occurrence_index.parquet
    for desc in iter_descendants(taxon, include_self=True):
        occ_path = TREE_ROOT / desc["path"] / OCCURRENCE_FILE
        if not occ_path.exists():
            continue
        table = pq.read_table(occ_path)
        if table.num_rows == 0:
            continue
        df = _filter_df(table.to_pandas())
        if df.empty:
            continue

        # Cache per-parquet unique catalog number count — reused across columns
        # when the column has no nulls (the common case for fully-enriched data).
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
                    acc = continuous_acc.setdefault(col, {
                        "digest": TDigest(), "reservoir": [], "n_seen": 0, "unique": 0,
                    })
                    acc["digest"].batch_update(values.tolist())
                    acc["n_seen"] = _reservoir_update(acc["reservoir"], acc["n_seen"], values)
                    acc["unique"] += _col_unique(col)

                case ValueType.NOMINAL:
                    series = df[col].dropna()
                    if series.empty:
                        continue
                    acc = nominal_acc.setdefault(col, {"counts": Counter(), "unique": 0})
                    for v in series:
                        acc["counts"][int(float(v))] += 1
                    acc["unique"] += _col_unique(col)

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
                    acc = circular_acc.setdefault(col, {
                        "cos_sum": 0.0, "sin_sum": 0.0, "n": 0,
                        "reservoir": [], "n_seen": 0, "unique": 0,
                    })
                    rad = np.deg2rad(values)
                    acc["cos_sum"] += float(np.sum(np.cos(rad)))
                    acc["sin_sum"] += float(np.sum(np.sin(rad)))
                    acc["n"] += len(values)
                    acc["n_seen"] = _reservoir_update(acc["reservoir"], acc["n_seen"], values)
                    acc["unique"] += _col_unique(col)

                case _:
                    continue  # skip unimplemented types in streaming

        pheno_acc.update(compute_phenology_counts(df))

    numerical_stats: dict[str, dict] = {}
    nominal_entries: list[dict] = []
    density_rows: list[dict] = []

    for col, acc in continuous_acc.items():
        digest = acc["digest"]
        reservoir = np.array(acc["reservoir"], dtype=float)
        reservoir = reservoir[np.isfinite(reservoir)]
        layer = layer_meta[col]
        vtype = _layer_value_type(layer)
        if _is_discrete(layer):
            counts = Counter(int(v) for v in reservoir)
            mode_val = counts.most_common(1)[0][0] if counts else None
            stats = _continuous_stats_streaming(digest, acc["unique"], None)
            stats["mode"] = mode_val
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
            kde = build_density_curve(reservoir, vtype) if vtype is not None else None
            stats = _continuous_stats_streaming(digest, acc["unique"], kde)
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

    for col, acc in nominal_acc.items():
        layer = layer_meta[col]
        summary, _ = _nominal_stats(acc["counts"], acc["unique"])
        nominal_entries.extend(_nominal_cat_entries(col, layer, acc["counts"], summary))

    circular_stats: dict[str, dict] = {}
    for col, acc in circular_acc.items():
        if acc["n"] == 0:
            continue
        layer = layer_meta[col]
        reservoir = np.array(acc["reservoir"], dtype=float)
        reservoir = reservoir[np.isfinite(reservoir)]
        kde = build_density_curve(reservoir, ValueType.CIRCULAR) if reservoir.size >= 2 else None
        stats = _circ_stats_streaming(acc["cos_sum"], acc["sin_sum"], acc["n"], acc["unique"], kde)
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

    if not numerical_stats and not nominal_entries and not circular_stats:
        return
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _write_stats_frame(taxon_dir / NUMERICAL_STATS_FILE, numerical_stats)
    _write_stats_frame(taxon_dir / CIRCULAR_STATS_FILE, circular_stats)
    _write_nominal_stats(taxon_dir, nominal_entries)
    _write_density(taxon_dir, density_rows)
    write_phenology_counts(taxon_dir, pheno_acc)


def _build_nonleaf_index_from_children(taxon: TaxonRecord, taxon_dir: Path) -> None:
    """Build occurrence_index.parquet by concatenating direct children's index files.

    Requires children to already be processed (call in bottom-up / leaf-first order).
    Reads O(children) files instead of O(all leaf descendants).
    """
    frames = []
    for child in get_children(taxon["taxon_key"]):
        child_idx = TREE_ROOT / child["path"] / OCCURRENCE_INDEX_FILE
        if child_idx.exists():
            frames.append(pq.read_table(child_idx).to_pandas())
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["catalogNumber"])
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(taxon_dir / OCCURRENCE_INDEX_FILE, pa.Table.from_pandas(combined, preserve_index=False))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_taxon_stats(
    taxon: TaxonRecord,
    layers: list[dict],
    layer_meta: dict[str, dict] | None = None,
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
    if layer_meta is None:
        layer_meta = {layer["id"]: layer for layer in layers}
    rank = taxon["rank"]
    if rank in CONFIG.subspecies_equivalents:
        _process_leaf(taxon_dir, layer_meta)
    elif rank == CONFIG.species_rank:
        _process_species(taxon, taxon_dir, layer_meta)
    else:
        _process_nonleaf(taxon, taxon_dir, layer_meta)
        if rank in _INDEX_RANKS:
            _build_nonleaf_index_from_children(taxon, taxon_dir)


def _load_occ_for_index(path: Path) -> pd.DataFrame | None:
    """Read and quality-filter one occurrence.parquet, stripping non-index cols."""
    if not path.exists():
        return None
    table = pq.read_table(path)
    if table.num_rows == 0:
        return None
    df = _filter_df(table.to_pandas())
    if df.empty:
        return None
    drop = [c for c in _INDEX_STRIP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df.dropna(subset=["catalogNumber", "decimalLatitude", "decimalLongitude"])


def _build_occurrence_index(taxon: TaxonRecord, taxon_dir: Path, is_leaf: bool) -> None:
    """Build and write occurrence_index.parquet for a taxon.

    Stores catalogNumber, lat, lon, and all GIS layer columns (quality filters
    pre-applied) so slice endpoints need no second lookup pass.
    """
    if is_leaf:
        df = _load_occ_for_index(taxon_dir / OCCURRENCE_FILE)
        frames = [df] if df is not None else []
    else:
        frames = []
        seen: set[str] = set()
        for desc in iter_descendants(taxon, include_self=True):
            df = _load_occ_for_index(TREE_ROOT / desc["path"] / OCCURRENCE_FILE)
            if df is None:
                continue
            df = df[~df["catalogNumber"].astype(str).isin(seen)]
            seen.update(df["catalogNumber"].astype(str).tolist())
            frames.append(df)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    if not is_leaf:
        combined = combined.drop_duplicates(subset=["catalogNumber"])
    taxon_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(taxon_dir / OCCURRENCE_INDEX_FILE, pa.Table.from_pandas(combined, preserve_index=False))
