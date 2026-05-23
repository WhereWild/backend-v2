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

import argparse
import importlib
import json
import os
import shutil
import subprocess
import traceback as tb
from datetime import UTC, datetime
from pathlib import Path

import httpx

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


TAXONOMY_CACHE_DIR = DATA_DIR / "taxonomy" / "cache"

# Directories under DATA_DIR that survive a wipe. data/gis/ is excluded from
# the loop directly; taxonomy/cache is backed up and restored so that ETag-
# cached downloads (inat_dwca.zip, gbif_vernacular.tsv) aren't re-fetched when
# the remote files haven't changed — the ETags live in sync_state.json (also
# preserved) and are useless without the matching local files.
def wipe_data_dir() -> None:
    """Delete GBIF-derived data in DATA_DIR, preserving sync_state.json, data/gis/, and data/taxonomy/cache/."""
    if not DATA_DIR.exists():
        return
    sync_state_backup = SYNC_STATE_PATH.read_bytes() if SYNC_STATE_PATH.exists() else None
    taxonomy_cache_backup: dict[str, bytes] = {}
    if TAXONOMY_CACHE_DIR.exists():
        for f in TAXONOMY_CACHE_DIR.iterdir():
            if f.is_file():
                taxonomy_cache_backup[f.name] = f.read_bytes()

    for child in DATA_DIR.iterdir():
        if child.name == "gis":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if sync_state_backup is not None:
        SYNC_STATE_PATH.write_bytes(sync_state_backup)
    if taxonomy_cache_backup:
        TAXONOMY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in taxonomy_cache_backup.items():
            (TAXONOMY_CACHE_DIR / name).write_bytes(data)
    print(f"Wiped {DATA_DIR}/ (sync_state.json, data/gis/, and data/taxonomy/cache/ preserved)")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    """Return True if a process with the given PID is currently running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _sync_gbif_stage() -> None:
    sync_gbif.main()
    sync_gbif.sync_occurrences()


STAGES: list[tuple[str, str, object]] = [
    ("sync_gbif",       "Syncing GBIF (taxonomy + occurrences)",              lambda: _sync_gbif_stage()),
    ("build_tree",      "Building tree (catalog + ID maps + names + images)", lambda: build_tree.main()),
    ("populate_tree",   "Populating tree (routing occurrences to parquet)",   lambda: populate_tree.main()),
    ("process_gadm",    "Processing GADM (download + location tables)",       lambda: process_gadm.main()),
    ("download_gis",    "Downloading GIS layers",                             lambda: _run_download_gis()),
    ("build_overviews", "Building COG overviews",                             lambda: build_overviews.main()),
    ("enrich_tree",     "Enriching tree (GIS sampling)",                      lambda: enrich_tree.main()),
    ("enrich_temporal", "Enriching tree (temporal ERA5 weather)",              lambda: enrich_temporal.main()),
    ("process_tree",    "Processing tree (summary stats + KDE)",              lambda: process_tree.main()),
]


def main() -> None:
    stage_ids = [s[0] for s in STAGES]
    parser = argparse.ArgumentParser(description="WhereWild rebuild pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the GBIF freshness check and run a full rebuild immediately.",
    )
    parser.add_argument(
        "--stage",
        metavar="STAGE",
        choices=stage_ids,
        help=f"Start the pipeline at this stage, skipping all prior stages. One of: {', '.join(stage_ids)}",
    )
    args = parser.parse_args()

    # Detect a previous crash: if a prior run left status=in_progress but its
    # PID is gone, it never finished cleanly (OOM, power loss, etc.).
    # If the PID is still alive, another instance is already running — exit.
    existing = _read_sync_state().get("pipeline", {})
    resuming = False

    if existing.get("status") == "in_progress":
        running_pid = existing.get("pid")
        if _pid_alive(running_pid):
            print(f"Pipeline already running (pid {running_pid}), exiting.")
            return
        crashed_at = _now()
        _update_pipeline({"status": "crashed", "finished_at": crashed_at})
        print("WARNING: previous pipeline run did not finish — marked as crashed")
        notify("crashed", {
            "stage": existing.get("stage"),
            "started_at": existing.get("started_at"),
            "crashed_at": crashed_at,
        })
        resuming = True

    if args.stage:
        # Jump directly to the given stage — no freshness check, no wipe.
        idx = stage_ids.index(args.stage)
        completed_stages = set(stage_ids[:idx])
        print(f"--stage {args.stage}: skipping {sorted(completed_stages) or 'nothing'}")
    elif not resuming:
        if args.force:
            print("--force: skipping freshness check, running full rebuild")
        else:
            # Only check for new crawl data on a fresh (non-resume) run.
            print("Checking for new GBIF crawl...")
            crawl_ts = sync_gbif.latest_crawl_finished()
            state = sync_gbif.load_sync_state()
            taxonomy_current = state.get("gbif_taxonomy", {}).get("crawl_finished") == crawl_ts
            occurrences_current = state.get("gbif_occurrences", {}).get("crawl_finished") == crawl_ts
            if taxonomy_current and occurrences_current:
                print("Already up to date")
                return
            print(f"New crawl detected: {crawl_ts}")

    if not args.stage:
        completed_stages = set()
        if resuming:
            completed_stages = {
                name
                for name, info in existing.get("stages", {}).items()
                if isinstance(info, dict) and info.get("status") == "completed"
            }
            print(f"Resuming — skipping already-completed stages: {sorted(completed_stages)}\n")

    inhibitor = _acquire_shutdown_inhibitor()
    try:
        started_at = (existing.get("started_at") if resuming else None) or _now()
        _update_pipeline({
            "status": "in_progress",
            "pid": os.getpid(),
            "stage": None,
            "started_at": started_at,
            "finished_at": None,
            "stages": existing.get("stages", {}) if resuming else {},
            "error": None,
        })

        if not resuming and not args.stage:
            print("\n--- Wiping data directory ---")
            wipe_data_dir()
            if args.force:
                # sync_state.json is preserved through the wipe, but its GBIF
                # crawl timestamps would cause sync_gbif to skip the download.
                # Clear them so it re-fetches.
                state = _read_sync_state()
                state.pop("gbif_taxonomy", None)
                state.pop("gbif_occurrences", None)
                _write_sync_state(state)

        for stage_id, label, fn in STAGES:
            if stage_id in completed_stages:
                print(f"\n--- Skipping {label} (already completed) ---")
                continue
            print(f"\n--- {label} ---")
            _set_stage(stage_id, "in_progress")
            fn()
            _set_stage(stage_id, "completed")

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
