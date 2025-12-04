from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as rio_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "processed"
SPECIES_DIR = PROCESSED_DIR / "species"
CATALOG_PATH = SPECIES_DIR / "species_catalog.json"
STATS_DIR = SPECIES_DIR / "stats"
LEADERBOARD_DIR = STATS_DIR / "leaderboard"
GIS_CATALOG_PATH = REPO_ROOT / "gis_catalog.json"
WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class SpeciesRecord:
    taxon_id: int
    slug: str
    scientific_name: str
    common_name: str
    parquet_path: Path

    @classmethod
    def from_json(cls, species_root: Path, payload: dict) -> SpeciesRecord:
        return cls(
            taxon_id=int(payload["taxon_id"]),
            slug=payload["slug"],
            scientific_name=payload["scientific_name"],
            common_name=payload["common_name"],
            parquet_path=species_root / payload["parquet_file"],
        )


@dataclass(frozen=True)
class CategoryClass:
    value: int
    name: str
    description: str | None = None

    def to_metadata(self) -> dict:
        return {
            "value": self.value,
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class DirectionBin:
    id: str
    name: str
    start_deg: float
    end_deg: float
    description: str | None = None

    def to_metadata(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_deg": self.start_deg,
            "end_deg": self.end_deg,
            "description": self.description,
        }


@dataclass(frozen=True)
class GISVariable:
    id: str
    name: str
    description: str | None
    path: Path
    units: str | None
    value_type: str | None
    classes: list[CategoryClass] | None = None
    direction_bins: list[DirectionBin] | None = None

    @property
    def class_lookup(self) -> dict[int, CategoryClass]:
        if not self.classes:
            return {}
        return {category.value: category for category in self.classes}


def load_species_catalog() -> list[SpeciesRecord]:
    with CATALOG_PATH.open() as fp:
        raw = json.load(fp)
    return [SpeciesRecord.from_json(SPECIES_DIR, entry) for entry in raw]


def load_gis_catalog() -> list[GISVariable]:
    if not GIS_CATALOG_PATH.exists():
        msg = f"GIS catalog not found at {GIS_CATALOG_PATH}"
        raise FileNotFoundError(msg)
    with GIS_CATALOG_PATH.open() as fp:
        raw = json.load(fp)

    def _load_classes(path: str | None) -> list[CategoryClass] | None:
        if not path:
            return None
        class_path = REPO_ROOT / path
        if not class_path.exists():
            print(f"[WARN] Landcover class legend missing at {class_path}")
            return None
        with class_path.open() as class_file:
            entries = json.load(class_file)
        return [
            CategoryClass(
                value=int(item["value"]),
                name=item.get("name", str(item["value"])),
                description=item.get("description"),
            )
            for item in entries
        ]

    def _load_direction_bins(path: str | None) -> list[DirectionBin] | None:
        if not path:
            return None
        bins_path = REPO_ROOT / path
        if not bins_path.exists():
            print(f"[WARN] Direction bin file missing at {bins_path}")
            return None
        with bins_path.open() as bins_file:
            entries = json.load(bins_file)
        return [
            DirectionBin(
                id=item["id"],
                name=item.get("name", item["id"]),
                start_deg=float(item["start_deg"]),
                end_deg=float(item["end_deg"]),
                description=item.get("description"),
            )
            for item in entries
        ]

    return [
        GISVariable(
            id=item["id"],
            name=item.get("name", item["id"]),
            description=item.get("description"),
            path=REPO_ROOT / item["path"],
            units=item.get("units"),
            value_type=item.get("value_type"),
            classes=_load_classes(item.get("classes_file")),
            direction_bins=_load_direction_bins(item.get("direction_bins_file")),
        )
        for item in raw
    ]


def load_observations(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["id", "latitude", "longitude"])
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return df
    return df.astype({"id": "int64"})


def project_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    dst_crs: CRS | None,
) -> tuple[np.ndarray, np.ndarray]:
    if dst_crs is None or dst_crs == WGS84:
        return lons, lats
    xs, ys = rio_transform(WGS84, dst_crs, lons.tolist(), lats.tolist())
    return np.asarray(xs), np.asarray(ys)


def sample_raster(
    dataset: rasterio.io.DatasetReader,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return observations.assign(value=np.nan).iloc[0:0]
    lats = observations["latitude"].to_numpy()
    lons = observations["longitude"].to_numpy()
    xs, ys = project_coords(lats, lons, dataset.crs)
    coords = list(zip(xs.tolist(), ys.tolist()))
    values = np.fromiter(
        (val[0] for val in dataset.sample(coords)),
        dtype="float64",
        count=len(coords),
    )
    valid = np.isfinite(values)
    if dataset.nodata is not None:
        valid &= values != dataset.nodata
    filtered = observations.loc[valid].copy()
    filtered["value"] = values[valid]
    return filtered


def summarize_values(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "stddev": None,
            "q10": None,
            "q90": None,
        }
    quantiles = np.quantile(values, [0.1, 0.9])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values)),
        "q10": float(quantiles[0]),
        "q90": float(quantiles[1]),
    }


def build_histogram(values: np.ndarray, bins: int) -> dict:
    if values.size == 0:
        return {"bins": [], "counts": []}
    bin_count = min(bins, values.size) or 1
    counts, edges = np.histogram(values, bins=bin_count)
    return {
        "bins": edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def serialize_sorted_records(
    samples: pd.DataFrame, limit: int
) -> tuple[list[dict], bool]:
    if samples.empty:
        return [], False
    sorted_samples = samples.sort_values("value")
    if len(sorted_samples) <= limit:
        selected = sorted_samples
        truncated = False
    else:
        indices = np.linspace(0, len(sorted_samples) - 1, num=limit, dtype=int)
        selected = sorted_samples.iloc[indices]
        truncated = True
    records = [
        {
            "observation_id": int(row.id),
            "value": float(row.value),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in selected.itertuples()
    ]
    return records, truncated


def compute_categorical_distribution(
    samples: pd.DataFrame,
    variable: GISVariable,
) -> tuple[list[dict], list[dict]]:
    if samples.empty:
        return [], []
    counts = Counter(int(value) for value in samples["value"].to_numpy())
    total = sum(counts.values())
    if total == 0:
        return [], []
    lookup = variable.class_lookup
    distribution = []
    for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        cls = lookup.get(value)
        distribution.append(
            {
                "value": value,
                "class_name": cls.name if cls else str(value),
                "description": cls.description if cls else None,
                "count": count,
                "fraction": count / total,
            }
        )
    dominant = distribution[:3]
    return distribution, dominant


def _direction_mask(values: np.ndarray, bin_def: DirectionBin) -> np.ndarray:
    start = bin_def.start_deg % 360
    end = bin_def.end_deg % 360
    if start <= end:
        return (values >= start) & (values < end)
    return (values >= start) | (values < end)


def compute_directional_distribution(
    samples: pd.DataFrame,
    variable: GISVariable,
) -> tuple[list[dict], list[dict]]:
    if not variable.direction_bins or samples.empty:
        return [], []
    values = samples["value"].to_numpy(copy=True)
    if values.size == 0:
        return [], []
    values = np.mod(values, 360.0)
    total = values.size
    distribution = []
    for bin_def in variable.direction_bins:
        mask = _direction_mask(values, bin_def)
        count = int(mask.sum())
        fraction = (count / total) if total else 0.0
        distribution.append(
            {
                "id": bin_def.id,
                "name": bin_def.name,
                "description": bin_def.description,
                "start_deg": bin_def.start_deg,
                "end_deg": bin_def.end_deg,
                "count": count,
                "fraction": fraction,
            }
        )
    distribution.sort(key=lambda item: item["fraction"], reverse=True)
    dominant = distribution[:2]
    return distribution, dominant


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_leaderboard(
    variable: GISVariable,
    entries: list[dict],
) -> dict:
    metrics = {
        "mean": {"field": "mean", "order": "desc"},
        "q90": {"field": "q90", "order": "desc"},
        "q10": {"field": "q10", "order": "desc"},
        "stddev": {"field": "stddev", "order": "desc"},
    }
    metric_payload: dict[str, dict] = {}
    for metric_name, cfg in metrics.items():
        field = cfg["field"]
        valid = [
            entry
            for entry in entries
            if entry["metrics"].get(field) is not None
        ]
        reverse = cfg["order"] == "desc"
        sorted_entries = sorted(
            valid,
            key=lambda item: item["metrics"][field],
            reverse=reverse,
        )
        leaderboard_entries = []
        for idx, entry in enumerate(sorted_entries, start=1):
            leaderboard_entries.append(
                {
                    "rank": idx,
                    "species_id": entry["species_id"],
                    "slug": entry["slug"],
                    "scientific_name": entry["scientific_name"],
                    "common_name": entry["common_name"],
                    "value": entry["metrics"][field],
                    "summary": entry["metrics"],
                }
            )
        metric_payload[metric_name] = {
            "order": cfg["order"],
            "entries": leaderboard_entries,
        }
    class_leaderboards = {}
    if variable.value_type == "categorical":
        class_lookup = variable.class_lookup
        per_class: dict[int, list[dict]] = {}
        for entry in entries:
            for bucket in entry.get("categorical_distribution", []):
                key = int(bucket["value"])
                per_class.setdefault(key, []).append(
                    {
                        "species_id": entry["species_id"],
                        "slug": entry["slug"],
                        "scientific_name": entry["scientific_name"],
                        "common_name": entry["common_name"],
                        "fraction": bucket["fraction"],
                        "count": bucket["count"],
                    }
                )
        for class_value, bucket_entries in per_class.items():
            sorted_bucket = sorted(
                bucket_entries,
                key=lambda item: (item["fraction"], item["count"]),
                reverse=True,
            )
            class_info = class_lookup.get(class_value)
            class_leaderboards[class_value] = {
                "class_name": class_info.name if class_info else str(class_value),
                "description": class_info.description if class_info else None,
                "entries": [
                    {
                        "rank": idx,
                        "species_id": data["species_id"],
                        "slug": data["slug"],
                        "scientific_name": data["scientific_name"],
                        "common_name": data["common_name"],
                        "fraction": data["fraction"],
                        "count": data["count"],
                    }
                    for idx, data in enumerate(sorted_bucket, start=1)
                ],
            }
    directional_leaderboards = {}
    if variable.direction_bins:
        per_bin: dict[str, list[dict]] = {b.id: [] for b in variable.direction_bins}
        for entry in entries:
            for bucket in entry.get("directional_distribution", []):
                per_bin.setdefault(bucket["id"], []).append(
                    {
                        "species_id": entry["species_id"],
                        "slug": entry["slug"],
                        "scientific_name": entry["scientific_name"],
                        "common_name": entry["common_name"],
                        "fraction": bucket["fraction"],
                        "count": bucket["count"],
                    }
                )
        for bin_def in variable.direction_bins:
            bucket_entries = per_bin.get(bin_def.id, [])
            sorted_bucket = sorted(
                bucket_entries,
                key=lambda item: (item["fraction"], item["count"]),
                reverse=True,
            )
            directional_leaderboards[bin_def.id] = {
                "name": bin_def.name,
                "description": bin_def.description,
                "start_deg": bin_def.start_deg,
                "end_deg": bin_def.end_deg,
                "entries": [
                    {
                        "rank": idx,
                        "species_id": data["species_id"],
                        "slug": data["slug"],
                        "scientific_name": data["scientific_name"],
                        "common_name": data["common_name"],
                        "fraction": data["fraction"],
                        "count": data["count"],
                    }
                    for idx, data in enumerate(sorted_bucket, start=1)
                ],
            }
    return {
        "variable": variable.id,
        "variable_name": variable.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metric_payload,
        "categorical": class_leaderboards or None,
        "directional": directional_leaderboards or None,
    }


def run_for_variable(
    variable: GISVariable,
    records: Sequence[SpeciesRecord],
    overwrite: bool,
    hist_bins: int,
    max_values: int,
) -> None:
    raster_path = variable.path
    if not raster_path.exists():
        print(
            f"[WARN] Raster for '{variable.id}' not found at {raster_path}, skipping."
        )
        return
    output_dir = STATS_DIR / variable.id
    ensure_output_dir(output_dir)
    leaderboard_entries: list[dict] = []
    with rasterio.open(raster_path) as dataset:
        for record in records:
            output_path = output_dir / f"{record.taxon_id}.json"
            if output_path.exists() and not overwrite:
                continue
            observations = load_observations(record.parquet_path)
            samples = sample_raster(dataset, observations)
            values = samples["value"].to_numpy()
            summary = summarize_values(values)
            hist = build_histogram(values, hist_bins)
            sorted_values, truncated = serialize_sorted_records(samples, max_values)
            distribution = []
            dominant_categories = []
            if variable.value_type == "categorical":
                distribution, dominant_categories = compute_categorical_distribution(
                    samples, variable
                )
            directional_distribution = []
            dominant_directions = []
            if variable.direction_bins:
                (
                    directional_distribution,
                    dominant_directions,
                ) = compute_directional_distribution(samples, variable)
            payload = {
                "species_id": record.taxon_id,
                "species": {
                    "slug": record.slug,
                    "scientific_name": record.scientific_name,
                    "common_name": record.common_name,
                },
                "variable": variable.id,
                "variable_metadata": {
                    "path": str(raster_path),
                    "name": variable.name,
                    "units": variable.units,
                    "value_type": variable.value_type,
                    "description": variable.description,
                    "classes": [
                        category.to_metadata() for category in (variable.classes or [])
                    ],
                    "direction_bins": [
                        direction.to_metadata()
                        for direction in (variable.direction_bins or [])
                    ],
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "values_sorted": sorted_values,
                "values_truncated": truncated,
                "histogram": hist,
                "categorical_distribution": distribution,
                "dominant_categories": dominant_categories,
                "directional_distribution": directional_distribution,
                "dominant_directions": dominant_directions,
                "source": {
                    "parquet_file": str(record.parquet_path),
                    "num_observations": int(len(observations)),
                    "num_samples": int(len(samples)),
                },
            }
            with output_path.open("w") as fp:
                json.dump(payload, fp, indent=2)
            print(
                f"[OK] {variable.id} · {record.taxon_id} "
                f"({summary['count']} samples) -> {output_path.relative_to(REPO_ROOT)}"
            )
            leaderboard_entries.append(
                {
                    "species_id": record.taxon_id,
                    "slug": record.slug,
                    "scientific_name": record.scientific_name,
                    "common_name": record.common_name,
                    "metrics": summary,
                    "categorical_distribution": distribution,
                    "directional_distribution": directional_distribution,
                }
            )
    ensure_output_dir(LEADERBOARD_DIR)
    leaderboard_path = (
        LEADERBOARD_DIR / f"{variable.id}_leaderboard.json"
    )
    leaderboard_payload = build_leaderboard(variable, leaderboard_entries)
    with leaderboard_path.open("w") as fp:
        json.dump(leaderboard_payload, fp, indent=2)
    print(
        f"[OK] Leaderboard written -> {leaderboard_path.relative_to(REPO_ROOT)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute species × GIS variable statistics."
    )
    parser.add_argument(
        "--species-id",
        type=int,
        nargs="+",
        help="Limit processing to one or more taxon IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate stats even if the output JSON already exists.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Number of bins to use when building histograms.",
    )
    parser.add_argument(
        "--max-values",
        type=int,
        default=10000,
        help="Maximum number of sorted values persisted for histogram rendering.",
    )
    return parser.parse_args()


def filter_species(
    records: Iterable[SpeciesRecord],
    allowed_ids: Sequence[int] | None,
) -> list[SpeciesRecord]:
    if not allowed_ids:
        return list(records)
    allowed = set(allowed_ids)
    return [record for record in records if record.taxon_id in allowed]


def main() -> None:
    args = parse_args()
    species_records = load_species_catalog()
    subset_records = filter_species(species_records, args.species_id)
    variables = load_gis_catalog()
    if not subset_records:
        print("No species matched the provided filters.")
        return
    for variable in variables:
        run_for_variable(
            variable=variable,
            records=subset_records,
            overwrite=args.overwrite,
            hist_bins=args.hist_bins,
            max_values=args.max_values,
        )


if __name__ == "__main__":
    main()
