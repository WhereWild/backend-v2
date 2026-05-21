"""
Full taxonomy rebuild pipeline.

Checks GBIF for new iNat crawl data. If new data is available, wipes the
data directory (preserving sync_state.json) and runs the full pipeline:
  1. sync_gbif     — download taxonomy + occurrence data from GBIF
  2. build_tree    — build catalog, ID maps, names, and images
  3. populate_tree — stream occurrence.txt → per-taxon parquet files
  4. process_gadm  — download GADM GeoPackage + build location tables and catalog
  5. download_gis  — download all GIS layers (runs every scripts/gis/download_*.py)
  6. build_overviews — build COG overviews for all GIS layers
  7. enrich_tree     — sample GIS layer values into per-taxon occurrence parquets
  8. enrich_temporal — enrich occurrences with time-windowed ERA5 weather statistics
  9. process_tree    — compute per-taxon summary statistics and KDE density graphs

Pipeline state is written to sync_state.json["pipeline"] so an external
process (e.g. a Discord bot) can poll it without coupling to this script.
The 4am system reboot is delayed via systemd-inhibit until the pipeline
finishes.
"""

import importlib
import json
import os
import shutil
import subprocess
import traceback as tb
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

import scripts.build_tree as build_tree
import scripts.enrich_temporal as enrich_temporal
import scripts.enrich_tree as enrich_tree
import scripts.gis.build_overviews as build_overviews
import scripts.gis.process_gadm as process_gadm
import scripts.populate_tree as populate_tree
import scripts.process_tree as process_tree
import scripts.sync_gbif as sync_gbif

DATA_DIR = Path("data")
SYNC_STATE_PATH = Path("data/sync_state.json")
NOTIFY_URL = os.environ.get("WHEREWILD_NOTIFY_URL", "")


# ---------------------------------------------------------------------------
# sync_state helpers
# ---------------------------------------------------------------------------

def _read_sync_state() -> dict:
    return json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}


def _write_sync_state(state: dict) -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _update_pipeline(updates: dict) -> None:
    state = _read_sync_state()
    pipeline = state.get("pipeline", {})
    pipeline.update(updates)
    state["pipeline"] = pipeline
    _write_sync_state(state)


def _set_stage(name: str, status: str) -> None:
    state = _read_sync_state()
    pipeline = state.get("pipeline", {})
    pipeline["stage"] = name
    stages = pipeline.get("stages", {})
    entry = stages.get(name, {}) if isinstance(stages.get(name), dict) else {}
    entry["status"] = status
    if status == "in_progress":
        entry["started_at"] = _now()
    elif status == "completed":
        entry["finished_at"] = _now()
    stages[name] = entry
    pipeline["stages"] = stages
    state["pipeline"] = pipeline
    _write_sync_state(state)


# ---------------------------------------------------------------------------
# notifications
# ---------------------------------------------------------------------------

def notify(event: str, payload: dict) -> None:
    """POST an event to WHEREWILD_NOTIFY_URL. Silently drops if URL unset or request fails."""
    if not NOTIFY_URL:
        return
    try:
        httpx.post(NOTIFY_URL, json={"event": event, **payload}, timeout=5)
    except Exception as exc:
        print(f"notify: failed to POST {event!r}: {exc}")


# ---------------------------------------------------------------------------
# systemd-inhibit
# ---------------------------------------------------------------------------

def _acquire_shutdown_inhibitor() -> "subprocess.Popen | None":
    try:
        return subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=shutdown",
                "--who=wherewild-rebuild",
                "--why=Taxonomy rebuild in progress",
                "--mode=delay",
                "sleep", "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def _release_inhibitor(proc: "subprocess.Popen | None") -> None:
    if proc is not None:
        proc.terminate()
        proc.wait()


# ---------------------------------------------------------------------------
# data directory
# ---------------------------------------------------------------------------

def _run_download_gis(gis_dir: Path | None = None) -> None:
    """Discover and run every scripts/gis/download_*.py."""
    if gis_dir is None:  # pragma: no cover
        gis_dir = Path(__file__).parent / "gis"
    for script in sorted(gis_dir.glob("download_*.py")):
        module_name = f"scripts.gis.{script.stem}"
        print(f"  [{script.stem}]")
        mod = importlib.import_module(module_name)
        mod.main()


def wipe_data_dir() -> None:
    """Delete GBIF-derived data in DATA_DIR, preserving sync_state.json and data/gis/."""
    if not DATA_DIR.exists():
        return
    sync_state_backup = SYNC_STATE_PATH.read_bytes() if SYNC_STATE_PATH.exists() else None

    for child in DATA_DIR.iterdir():
        if child.name == "gis":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if sync_state_backup is not None:
        SYNC_STATE_PATH.write_bytes(sync_state_backup)
    print(f"Wiped {DATA_DIR}/ (sync_state.json and data/gis/ preserved)")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Detect a previous crash: if a prior run left status=in_progress it
    # never finished cleanly (OOM, power loss, etc.).
    existing = _read_sync_state().get("pipeline", {})
    if existing.get("status") == "in_progress":
        crashed_at = _now()
        _update_pipeline({"status": "crashed", "finished_at": crashed_at})
        print("WARNING: previous pipeline run did not finish — marked as crashed")
        notify("crashed", {
            "stage": existing.get("stage"),
            "started_at": existing.get("started_at"),
            "crashed_at": crashed_at,
        })

    # Check for new crawl before acquiring inhibitor or touching pipeline state.
    print("Checking for new GBIF crawl...")
    crawl_ts = sync_gbif.latest_crawl_finished()
    state = sync_gbif.load_sync_state()
    taxonomy_current = state.get("gbif_taxonomy", {}).get("crawl_finished") == crawl_ts
    occurrences_current = state.get("gbif_occurrences", {}).get("crawl_finished") == crawl_ts
    if taxonomy_current and occurrences_current:
        print("Already up to date")
        return
    print(f"New crawl detected: {crawl_ts}")

    inhibitor = _acquire_shutdown_inhibitor()
    try:
        started_at = _now()
        _update_pipeline({
            "status": "in_progress",
            "stage": None,
            "started_at": started_at,
            "finished_at": None,
            "stages": {},
            "error": None,
        })

        print("\n--- Wiping data directory ---")
        wipe_data_dir()

        _set_stage("sync_gbif", "in_progress")
        sync_gbif.main()
        sync_gbif.sync_occurrences()
        _set_stage("sync_gbif", "completed")

        print("\n--- Building tree (catalog + ID maps + names + images) ---")
        _set_stage("build_tree", "in_progress")
        build_tree.main()
        _set_stage("build_tree", "completed")

        print("\n--- Populating tree (routing occurrences to parquet) ---")
        _set_stage("populate_tree", "in_progress")
        populate_tree.main()
        _set_stage("populate_tree", "completed")

        print("\n--- Processing GADM (download + location tables + catalog) ---")
        _set_stage("process_gadm", "in_progress")
        process_gadm.main()
        _set_stage("process_gadm", "completed")

        print("\n--- Downloading GIS layers ---")
        _set_stage("download_gis", "in_progress")
        _run_download_gis()
        _set_stage("download_gis", "completed")

        print("\n--- Building COG overviews ---")
        _set_stage("build_overviews", "in_progress")
        build_overviews.main()
        _set_stage("build_overviews", "completed")

        print("\n--- Enriching tree (GIS sampling) ---")
        _set_stage("enrich_tree", "in_progress")
        enrich_tree.main()
        _set_stage("enrich_tree", "completed")

        print("\n--- Enriching tree (temporal ERA5 weather) ---")
        _set_stage("enrich_temporal", "in_progress")
        enrich_temporal.main()
        _set_stage("enrich_temporal", "completed")

        print("\n--- Processing tree (summary stats + KDE) ---")
        _set_stage("process_tree", "in_progress")
        process_tree.main()
        _set_stage("process_tree", "completed")

        finished_at = _now()
        elapsed = int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds())
        _update_pipeline({"status": "completed", "stage": None, "finished_at": finished_at, "duration_s": elapsed})
        final_state = _read_sync_state()
        notify("completed", {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": elapsed,
            "stages": final_state.get("pipeline", {}).get("stages", {}),
        })
        print("\nRebuild complete.")

    except Exception as e:
        current_stage = _read_sync_state().get("pipeline", {}).get("stage")
        error = {
            "stage": current_stage,
            "message": str(e),
            "traceback": tb.format_exc(),
        }
        _update_pipeline({"status": "errored", "finished_at": _now(), "error": error})
        notify("errored", {"error": error})
        raise
    finally:
        _release_inhibitor(inhibitor)


if __name__ == "__main__":  # pragma: no cover
    main()
