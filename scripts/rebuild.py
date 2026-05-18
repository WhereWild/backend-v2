"""
Full taxonomy rebuild pipeline.

Checks GBIF for new iNat crawl data. If new data is available, wipes the
data directory (preserving sync_state.json) and runs the full pipeline:
  1. sync_gbif   — download new GBIF occurrence data
  2. build_tree  — parse taxonomy, build catalog pickle
  3. build_id_maps — build id/slug lookup maps
  4. polish_tree — fetch iNat preferred names/images, update index

Pipeline state is written to sync_state.json["pipeline"] so an external
process (e.g. a Discord bot) can poll it without coupling to this script.
The 4am system reboot is delayed via systemd-inhibit until the pipeline
finishes.
"""

import json
import os
import shutil
import subprocess
import traceback as tb
from datetime import UTC, datetime
from pathlib import Path

import httpx

import scripts.build_id_maps as build_id_maps
import scripts.build_tree as build_tree
import scripts.polish_tree as polish_tree
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

def wipe_data_dir() -> None:
    """Delete everything in DATA_DIR except sync_state.json."""
    if not DATA_DIR.exists():
        return
    sync_state_backup = None
    if SYNC_STATE_PATH.exists():
        sync_state_backup = SYNC_STATE_PATH.read_bytes()

    shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()

    if sync_state_backup is not None:
        SYNC_STATE_PATH.write_bytes(sync_state_backup)
    print(f"Wiped {DATA_DIR}/ (sync_state.json preserved)")


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

        print("\n--- Building taxonomy tree ---")
        _set_stage("build_tree", "in_progress")
        build_tree.main()
        _set_stage("build_tree", "completed")

        print("\n--- Building ID maps ---")
        _set_stage("build_id_maps", "in_progress")
        build_id_maps.main()
        _set_stage("build_id_maps", "completed")

        print("\n--- Polishing tree (iNat preferred names/images) ---")
        _set_stage("polish_tree", "in_progress")
        polish_tree.main()
        _set_stage("polish_tree", "completed")

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
