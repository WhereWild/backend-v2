"""Batch-train SDM (and optionally phenology/full) models for all leaf taxa
under a given subtree root.

Configuration (env vars or edit constants below):

  ML_SUBTREE_ROOT_TAXON_ID  — taxon ID of the subtree root (required)
  ML_TRAIN_PHENOLOGY        — also run phenology pass per taxon (default: true)
  ML_TRAIN_FULL             — also run full (GIS+temporal) pass per taxon (default: false)
  ML_PARQUET_STORAGE_MODE   — "local" or "b2" (default: local)
  ML_RASTER_STORAGE_MODE    — "auto", "local", or "b2" (default: auto)
  ML_PUSH_MODEL_TO_B2       — push trained artifact to B2 (default: false for batch runs)

Leaf ranks trained: whatever is in CONFIG.leaf_ranks (SPECIES, SUBSPECIES, VARIETY, FORM).

For each leaf taxon the script runs up to three passes (each as a subprocess so
CONFIG is re-initialized with the correct taxon ID):

  1. SDM        — always (GIS-only habitat model)
  2. Phenology  — if ML_TRAIN_PHENOLOGY=true; silently skipped if the parquet has no
                  rcs-annotated rows with positive values
  3. Full       — if ML_TRAIN_FULL=true; silently skipped if no temporal columns exist
                  in the parquet
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from util.config import load_config
import util.taxa_navigation as taxa_navigation
from util.storage import get_parquet_storage_with_mode

CONFIG = load_config("global")

SUBTREE_ROOT_TAXON_ID: str = CONFIG.ml_subtree_root_taxon_id.strip()
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
    """Invoke train_taxon_model.py in a subprocess. Returns (success, error_message)."""
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
    if not SUBTREE_ROOT_TAXON_ID:
        print("ERROR: set ML_SUBTREE_ROOT_TAXON_ID to the root taxon ID to train under.")
        sys.exit(1)

    root = taxa_navigation.get_taxon_by_id(SUBTREE_ROOT_TAXON_ID)
    if root is None:
        print(f"ERROR: taxon {SUBTREE_ROOT_TAXON_ID!r} not found in catalog.")
        sys.exit(1)

    leaf_ranks = CONFIG.leaf_rank_set
    descendants = taxa_navigation.iter_descendants(root, include_self=True)
    leaves = [t for t in descendants if str(t.get("rank", "")).upper() in leaf_ranks]

    print(
        f"[batch] root={SUBTREE_ROOT_TAXON_ID} ({root.get('scientific_name')}) "
        f"leaf_taxa={len(leaves)} "
        f"phenology={TRAIN_PHENOLOGY} full={TRAIN_FULL}"
    )

    # Filter to taxa that actually have occurrence data
    with_data = [t for t in leaves if _has_parquet(t)]
    skipped_no_data = len(leaves) - len(with_data)
    print(f"[batch] {len(with_data)} have parquet data, {skipped_no_data} skipped (no parquet)")

    results: dict[str, list[str]] = {"ok": [], "failed": [], "skipped": []}
    t0 = time.monotonic()

    for i, taxon in enumerate(with_data, 1):
        taxon_id = str(taxon["taxon_key"])
        name = taxon.get("scientific_name") or taxon_id
        prefix = f"[{i}/{len(with_data)}] {taxon_id} ({name})"

        # ── Pass 1: SDM ────────────────────────────────────────────────────────
        print(f"{prefix} → SDM")
        ok, err = _run_pass(taxon_id, {
            "ML_PHENOLOGY_MODE": "false",
            "ML_SDM_INCLUDE_TEMPORAL": "false",
        })
        if not ok:
            print(f"{prefix}   SDM FAILED: {err}")
            results["failed"].append(f"{taxon_id} sdm: {err}")
            continue

        is_plant = "Plantae_6" in str(taxon.get("path", ""))

        # ── Pass 2: Phenology (plants only) ───────────────────────────────────
        if TRAIN_PHENOLOGY and is_plant:
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
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
