"""Batch-train SDM (and optionally phenology/full) models for all leaf taxa
observed in a given location with sufficient samples.

Configuration (env vars or edit constants below):

  ML_LOCATION_GID           — location GID to filter by, e.g. "USA.45_1" (required)
  ML_LOCATION_MIN_SAMPLES   — minimum occurrence count in the location (default: 100)
  ML_TRAIN_PHENOLOGY        — also run phenology pass per taxon (default: true)
  ML_TRAIN_FULL             — also run full (GIS+temporal) pass per taxon (default: false)
  ML_PARQUET_STORAGE_MODE   — "local" or "b2" (default: local)
  ML_RASTER_STORAGE_MODE    — "auto", "local", or "b2" (default: auto)
  ML_PUSH_MODEL_TO_B2       — push trained artifact to B2 (default: false for batch runs)

Example — all species in Utah with ≥100 occurrences:

  ML_LOCATION_GID=USA.45_1 pd train_location_models.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from util.config import load_config
import util.gis_lookup as gis_lookup
import util.models as models
import util.taxa_navigation as taxa_navigation
from util.storage import get_parquet_storage_with_mode

CONFIG = load_config("global")

# ── Constants ──────────────────────────────────────────────────────────────────
LOCATION_GID: str = CONFIG.ml_location_gid.strip()
# State/province (1 dot) gets a lower threshold than country (0 dots)
MIN_SAMPLES: int = (
    CONFIG.ml_location_min_samples_state
    if LOCATION_GID.count(".") >= 1
    else CONFIG.ml_location_min_samples_country
)
TRAIN_PHENOLOGY: bool = CONFIG.ml_train_phenology
TRAIN_FULL: bool = CONFIG.ml_train_full
PARQUET_STORAGE_MODE: str = CONFIG.ml_parquet_storage_mode
RASTER_STORAGE_MODE: str = CONFIG.ml_raster_storage_mode
PUSH_TO_B2: bool = CONFIG.ml_push_model_to_b2

TRAIN_SCRIPT = Path(__file__).parent / "train_taxon_model.py"


def _parquet_storage():
    return get_parquet_storage_with_mode(
        CONFIG.data_root,
        CONFIG.project_root,
        PARQUET_STORAGE_MODE,
    )


def _has_parquet(taxon: taxa_navigation.TaxonRecord) -> bool:
    path = Path(taxon["path"]) / CONFIG.occurrence_parquet_filename
    try:
        return _parquet_storage().exists(path)
    except Exception:
        return False


def _run_pass(taxon_id: str, extra_env: dict[str, str]) -> tuple[bool, str]:
    env = {
        **os.environ,
        "ML_TRAIN_TAXON_ID": taxon_id,
        "ML_PARQUET_STORAGE_MODE": PARQUET_STORAGE_MODE,
        "ML_RASTER_STORAGE_MODE": RASTER_STORAGE_MODE,
        "ML_PUSH_MODEL_TO_B2": "true" if PUSH_TO_B2 else "false",
        **extra_env,
    }
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_taxon_model"],
        env=env,
        cwd=str(TRAIN_SCRIPT.parent.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        return False, f"exit code {result.returncode}"
    return True, ""


def main() -> None:
    if not LOCATION_GID:
        print("ERROR: set ML_LOCATION_GID to the location GID to train for (e.g. USA.45_1).")
        sys.exit(1)

    # Resolve scope/target from GID
    try:
        _col, scope, target = gis_lookup.location_lookup_for_gid(LOCATION_GID)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Taxa present in the location
    taxon_ids_in_location = gis_lookup.location_taxa_for(scope, target)
    if not taxon_ids_in_location:
        print(f"ERROR: no taxa found for location {LOCATION_GID!r}. Is the location catalog built?")
        sys.exit(1)

    # Per-taxon occurrence counts in the location for min_samples filtering
    try:
        location_counts = gis_lookup.location_taxon_counts(scope, target, include_species_rollup=True)
    except Exception as e:
        print(f"WARNING: could not load location counts ({e}), skipping min_samples filter")
        location_counts = None

    leaf_ranks = CONFIG.leaf_rank_set

    # Resolve taxon records, filter to leaf rank + min samples + has parquet
    candidates: list[tuple[taxa_navigation.TaxonRecord, int]] = []
    for taxon_id in taxon_ids_in_location:
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
        if taxon is None:
            continue
        if str(taxon.get("rank", "")).upper() not in leaf_ranks:
            continue
        count = (location_counts or {}).get(taxon_id, 0)
        if location_counts is not None and count < MIN_SAMPLES:
            continue
        if not _has_parquet(taxon):
            continue
        candidates.append((taxon, count))

    # Sort by count descending so highest-data taxa train first
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(
        f"[batch] location={LOCATION_GID!r} min_samples={MIN_SAMPLES} "
        f"qualifying_taxa={len(candidates)} "
        f"phenology={TRAIN_PHENOLOGY} full={TRAIN_FULL}"
    )

    results: dict[str, list[str]] = {"ok": [], "failed": [], "skipped": []}
    t0 = time.monotonic()

    for i, (taxon, count) in enumerate(candidates, 1):
        taxon_id = str(taxon["taxon_key"])
        name = taxon.get("scientific_name") or taxon_id
        prefix = f"[{i}/{len(candidates)}] {taxon_id} ({name}, n={count})"

        is_plant = "Plantae_6" in str(taxon.get("path", ""))

        # ── Pass 1: SDM ────────────────────────────────────────────────────────
        if models.has_sdm_model(taxon_id):
            print(f"{prefix} → SDM (already trained, skipping)")
        else:
            print(f"{prefix} → SDM")
            ok, err = _run_pass(taxon_id, {
                "ML_PHENOLOGY_MODE": "false",
                "ML_SDM_INCLUDE_TEMPORAL": "false",
            })
            if not ok:
                print(f"{prefix}   SDM FAILED: {err}")
                results["failed"].append(f"{taxon_id} sdm: {err}")
                continue

        # ── Pass 2: Phenology (plants only) ───────────────────────────────────
        if TRAIN_PHENOLOGY and is_plant:
            if models.has_phenology_model(taxon_id):
                print(f"{prefix} → phenology (already trained, skipping)")
            else:
                print(f"{prefix} → phenology")
                ok, err = _run_pass(taxon_id, {
                    "ML_PHENOLOGY_MODE": "true",
                    "ML_PHENOLOGY_TEMPORAL_ONLY": "true",
                })
                if not ok:
                    print(f"{prefix}   phenology skipped/failed: {err}")
                    results["skipped"].append(f"{taxon_id} phenology: {err}")

        # ── Pass 3: Full (GIS + temporal, non-plants always, plants if ML_TRAIN_FULL) ──
        if not is_plant or TRAIN_FULL:
            if models.has_full_model(taxon_id):
                print(f"{prefix} → full (already trained, skipping)")
            else:
                print(f"{prefix} → full")
                ok, err = _run_pass(taxon_id, {
                    "ML_PHENOLOGY_MODE": "false",
                    "ML_SDM_INCLUDE_TEMPORAL": "true",
                })
                if not ok:
                    print(f"{prefix}   full skipped/failed: {err}")
                    results["skipped"].append(f"{taxon_id} full: {err}")

        results["ok"].append(taxon_id)

    elapsed = time.monotonic() - t0
    print(
        f"\n[batch] done in {elapsed:.0f}s — "
        f"ok={len(results['ok'])} "
        f"failed={len(results['failed'])} "
        f"skipped={len(results['skipped'])}"
    )
    if results["failed"]:
        print("[batch] failures:")
        for entry in results["failed"]:
            print(f"  {entry}")
        sys.exit(1)


if __name__ == "__main__":
    main()
