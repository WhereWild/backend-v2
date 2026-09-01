# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sync GBIF species list for CONFIG.taxonomy_roots (full taxonomy, all ranks).

Checks the iNaturalist crawl history on GBIF to detect new ingestions.
Only re-downloads if GBIF has processed new data since our last download.
Credentials are read from environment variables loaded from .env

Multiple configured roots (e.g. Plantae + Fungi) are pulled in a single
download via an "in" TAXON_KEY predicate rather than one download per root —
GBIF's TAXON_KEY predicate already matches a taxon and all its descendants,
and "in" unions that match across every listed value (confirmed live: a
combined query's occurrence count exactly equals the sum of each root's own
count, no overlap). This keeps the rest of the pipeline (one species_list.csv,
one occurrence.txt/multimedia.txt) unchanged regardless of how many roots are
configured.
"""

import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from config.config import load_config

GBIF_USER = os.environ.get("GBIF_USER", "")
GBIF_PASSWORD = os.environ.get("GBIF_PASSWORD", "")
GBIF_EMAIL = os.environ.get("GBIF_EMAIL", "")

BASE_URL = "https://api.gbif.org/v1"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
# Catalogue of Life Extended Release — required on TAXON_KEY predicates since
# taxonomy_roots entries are COL XR ids (e.g. "7HS"), not legacy numeric GBIF
# Backbone keys. Without this, GBIF resolves TAXON_KEY against the obsolete
# Backbone by default, where the value doesn't exist.
COL_XR_CHECKLIST_KEY = "7ddf754f-d193-4cc9-b351-99906754a03b"

CONFIG = load_config("global")

CATALOG_DIR = Path("data/taxonomy/catalog")
OCCURRENCES_DIR = Path("data/occurrences")
SYNC_STATE_PATH = Path("data/sync_state.json")


def latest_crawl_finished() -> str:
    """Return finishedCrawling timestamp of the most recent successful iNat crawl."""
    resp = httpx.get(
        f"{BASE_URL}/dataset/{INAT_DATASET_KEY}/process",
        params={"limit": 10},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    for entry in results:
        if entry.get("finishReason") == "NORMAL" and entry.get("finishedCrawling"):
            return entry["finishedCrawling"]
    raise RuntimeError("No successful crawl found in recent history")


def load_sync_state() -> dict:
    if SYNC_STATE_PATH.exists():
        return json.loads(SYNC_STATE_PATH.read_text())
    return {}


def save_sync_state(state: dict) -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))


def _taxon_key_predicate() -> dict:
    """TAXON_KEY predicate matching every configured root (and each root's
    descendants — GBIF's TAXON_KEY predicate already includes descendants,
    "in" unions that across every listed root, confirmed live via matching
    occurrence counts). Works identically for a single configured root."""
    return {
        "type": "in", "key": "TAXON_KEY", "values": [str(r) for r in CONFIG.taxonomy_roots],
        "checklistKey": COL_XR_CHECKLIST_KEY,
    }


def _predicate_matches(request: dict, fmt: str, has_coordinate: bool) -> bool:
    """True if a GBIF download's own recorded request matches what we'd
    submit right now for the given format — same dataset, same taxon roots
    (order-independent), same checklist, same HAS_COORDINATE presence.
    Ignores fields GBIF injects that we never set (e.g. matchCase)."""
    if request.get("format") != fmt:
        return False
    predicates = request.get("predicate", {}).get("predicates", [])
    taxon_pred = next((p for p in predicates if p.get("key") == "TAXON_KEY"), None)
    if taxon_pred is None:
        return False
    if taxon_pred.get("checklistKey") != COL_XR_CHECKLIST_KEY:
        return False
    values = taxon_pred.get("values")
    if values is None and "value" in taxon_pred:
        values = [taxon_pred["value"]]
    expected = sorted(str(r) for r in CONFIG.taxonomy_roots)
    if sorted(str(v) for v in (values or [])) != expected:
        return False
    has_hc = any(p.get("key") == "HAS_COORDINATE" for p in predicates)
    return has_hc == has_coordinate


def _parse_gbif_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _find_existing_download(
    fmt: str, has_coordinate: bool, min_created: str | None = None
) -> str | None:
    """Return the key of an existing GBIF download (not FAILED/KILLED/
    CANCELLED) whose predicate matches what we'd request right now, or None.
    A SUCCEEDED match is always preferred over a still-preparing one (usable
    immediately, no wait) — results come back newest-first, and the newest
    matching job is often still PREPARING while an older identical one has
    already finished, so a naive first-match scan would pick the slower one.

    GBIF does not deduplicate download requests on its own — submitting the
    same predicate twice just creates two separate ~1hr jobs (confirmed live:
    identical predicates minutes apart got distinct keys and both ran their
    full prep time independently). Without this check, any crash/retry
    before local state is persisted burns a fresh ~1hr download every time,
    and can exhaust GBIF's small per-account concurrent-download quota.

    min_created (the crawl_finished timestamp of the crawl we're syncing for)
    rejects any export GBIF built *before* that crawl finished: the predicate
    alone can't tell a fresh export from a weeks-old one for the same taxa,
    and reusing a stale one silently rebuilds the whole tree from data that
    predates the "new crawl" the freshness check just detected (real
    incident: an Aug-8 export got reused for an Aug-28 crawl, so every
    downstream stage saw zero new/changed observations). An entry with no
    parseable `created` is rejected too when min_created is set — better a
    redundant fresh request than another silent stale rebuild.
    """
    resp = httpx.get(
        f"{BASE_URL}/occurrence/download/user/{GBIF_USER}",
        auth=(GBIF_USER, GBIF_PASSWORD),
        params={"limit": 50},
        timeout=30,
    )
    resp.raise_for_status()
    min_created_dt = _parse_gbif_ts(min_created)
    fallback: str | None = None
    for entry in resp.json().get("results", []):
        status = entry.get("status")
        if status in ("FAILED", "KILLED", "CANCELLED"):
            continue
        if not _predicate_matches(entry.get("request", {}), fmt, has_coordinate):
            continue
        if min_created_dt is not None:
            created_dt = _parse_gbif_ts(entry.get("created"))
            if created_dt is None or created_dt < min_created_dt:
                continue
        if status == "SUCCEEDED":
            return entry["key"]
        if fallback is None:
            fallback = entry["key"]
    return fallback


def request_download(crawl_finished: str | None = None) -> str:
    existing = _find_existing_download("SPECIES_LIST", has_coordinate=False, min_created=crawl_finished)
    if existing:
        print(f"Reusing existing download: {existing}")
        return existing

    payload = {
        "creator": GBIF_USER,
        "notificationAddresses": [GBIF_EMAIL],
        "sendNotification": True,
        "format": "SPECIES_LIST",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY},
                _taxon_key_predicate(),
                {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
            ],
        },
    }
    resp = httpx.post(
        f"{BASE_URL}/occurrence/download/request",
        json=payload,
        auth=(GBIF_USER, GBIF_PASSWORD),
        timeout=30,
    )
    resp.raise_for_status()
    key = resp.text.strip().strip('"')
    print(f"Download requested: {key}")
    return key


def _download_state_entry(crawl_finished: str, download_key: str, gbif_meta: dict) -> dict:
    return {
        "crawl_finished": crawl_finished,
        "download_key": download_key,
        "download_link": gbif_meta.get("downloadLink"),
        "doi": gbif_meta.get("doi"),
        "citation": _build_citation(gbif_meta),
        "created": gbif_meta.get("created"),
        "erase_after": gbif_meta.get("eraseAfter"),
        "total_records": gbif_meta.get("totalRecords"),
        "number_datasets": gbif_meta.get("numberDatasets"),
        "size_bytes": gbif_meta.get("size"),
    }


def _build_citation(gbif_meta: dict) -> str:
    if gbif_meta.get("citation"):
        return gbif_meta["citation"]
    doi = gbif_meta.get("doi", "")
    created = gbif_meta.get("created", "")
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(UTC)
        date_str = dt.strftime("%-d %B %Y")
    except (ValueError, AttributeError):
        date_str = created
    return f"GBIF.org ({date_str}) GBIF Occurrence Download https://doi.org/{doi}"


def poll_until_ready(download_key: str, interval: int = 30, timeout: int = 7200) -> dict:
    """Poll until SUCCEEDED and return the full GBIF download metadata."""
    elapsed = 0
    while elapsed < timeout:
        resp = httpx.get(f"{BASE_URL}/occurrence/download/{download_key}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        print(f"  [{elapsed}s] {status}")
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "KILLED", "CANCELLED"):
            raise RuntimeError(f"Download failed: {status}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Download not ready after {timeout}s")


def download_zip(url: str, dest_dir: Path | None = None) -> None:
    d = dest_dir if dest_dir is not None else CATALOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {d}/download.zip")
    subprocess.run(
        [
            "aria2c",
            f"--http-user={GBIF_USER}",
            f"--http-passwd={GBIF_PASSWORD}",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={d}",
            "--out=download.zip",
            url,
        ],
        check=True,
    )
    print(f"Saved to {d}/download.zip")


def extract(src_dir: Path | None = None, output_name: str = "species_list.csv") -> None:
    d = src_dir if src_dir is not None else CATALOG_DIR
    zip_file = d / "download.zip"
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(d)
    for f in d.glob("*.csv"):
        if f.name != output_name:
            f.rename(d / output_name)
            break
    zip_file.unlink(missing_ok=True)
    files = [f.name for f in d.iterdir()]
    print(f"Extracted to {d}/: {files}")


def _cleanup_occurrences_dir() -> None:
    """Delete everything in the occurrences dir except occurrence.txt and multimedia.txt."""
    if not OCCURRENCES_DIR.exists():
        return
    keep = {"occurrence.txt", "multimedia.txt"}
    for item in OCCURRENCES_DIR.iterdir():
        if item.name in keep:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def request_occurrence_download(crawl_finished: str | None = None) -> str:
    existing = _find_existing_download("DWCA", has_coordinate=True, min_created=crawl_finished)
    if existing:
        print(f"Reusing existing occurrence download: {existing}")
        return existing

    payload = {
        "creator": GBIF_USER,
        "notificationAddresses": [GBIF_EMAIL],
        "sendNotification": True,
        "format": "DWCA",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY},
                _taxon_key_predicate(),
                {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
                {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            ],
        },
    }
    resp = httpx.post(
        f"{BASE_URL}/occurrence/download/request",
        json=payload,
        auth=(GBIF_USER, GBIF_PASSWORD),
        timeout=30,
    )
    resp.raise_for_status()
    key = resp.text.strip().strip('"')
    print(f"Occurrence download requested: {key}")
    return key


def sync_occurrences() -> bool:
    """Download iNat occurrence records. Returns True if new data was downloaded."""
    if not all([GBIF_USER, GBIF_PASSWORD, GBIF_EMAIL]):
        raise OSError("GBIF_USER, GBIF_PASSWORD, and GBIF_EMAIL must be set")

    print("Checking GBIF iNat crawl history (occurrences)...")
    crawl_finished = latest_crawl_finished()
    state = load_sync_state()

    if state.get("gbif_occurrences", {}).get("crawl_finished") == crawl_finished:
        print(f"Already up to date (last crawl: {crawl_finished})")
        return False

    print(f"New crawl detected: {crawl_finished}")

    download_key = request_occurrence_download(crawl_finished)
    gbif_meta = poll_until_ready(download_key)
    download_zip(gbif_meta["downloadLink"], OCCURRENCES_DIR)
    extract(OCCURRENCES_DIR)
    _cleanup_occurrences_dir()

    state["gbif_occurrences"] = _download_state_entry(crawl_finished, download_key, gbif_meta)
    save_sync_state(state)
    print("Done.")
    return True


def main() -> bool:
    """Return True if new data was downloaded, False if already up to date."""
    if not all([GBIF_USER, GBIF_PASSWORD, GBIF_EMAIL]):
        raise OSError("GBIF_USER, GBIF_PASSWORD, and GBIF_EMAIL must be set")

    print("Checking GBIF iNat crawl history...")
    crawl_finished = latest_crawl_finished()
    state = load_sync_state()

    if state.get("gbif_taxonomy", {}).get("crawl_finished") == crawl_finished:
        print(f"Already up to date (last crawl: {crawl_finished})")
        return False

    print(f"New crawl detected: {crawl_finished}")

    download_key = request_download(crawl_finished)
    gbif_meta = poll_until_ready(download_key)
    download_zip(gbif_meta["downloadLink"])
    extract()

    state["gbif_taxonomy"] = _download_state_entry(crawl_finished, download_key, gbif_meta)
    save_sync_state(state)
    print("Done.")
    return True


def sync_all() -> bool:
    """Sync both the taxonomy species list and occurrence records, requesting
    both GBIF downloads up front before waiting on either.

    GBIF downloads routinely take on the order of an hour to prepare, and
    that preparation happens entirely on GBIF's side once requested — polling
    doesn't influence it. main() then sync_occurrences() run sequentially
    would request the occurrence download only after the species-list one
    has fully finished preparing, doubling the wall-clock wait for no reason.
    Requesting both immediately lets them prepare concurrently: by the time
    the first poll loop finishes, the second download is often already done
    or close to it.

    Returns True if anything was downloaded.
    """
    if not all([GBIF_USER, GBIF_PASSWORD, GBIF_EMAIL]):
        raise OSError("GBIF_USER, GBIF_PASSWORD, and GBIF_EMAIL must be set")

    print("Checking GBIF iNat crawl history...")
    crawl_finished = latest_crawl_finished()
    state = load_sync_state()

    need_taxonomy = state.get("gbif_taxonomy", {}).get("crawl_finished") != crawl_finished
    need_occurrences = state.get("gbif_occurrences", {}).get("crawl_finished") != crawl_finished

    if not need_taxonomy and not need_occurrences:
        print(f"Already up to date (last crawl: {crawl_finished})")
        return False

    print(f"New crawl detected: {crawl_finished}")

    taxonomy_key = request_download(crawl_finished) if need_taxonomy else None
    occurrences_key = request_occurrence_download(crawl_finished) if need_occurrences else None

    if taxonomy_key:
        gbif_meta = poll_until_ready(taxonomy_key)
        download_zip(gbif_meta["downloadLink"])
        extract()
        state["gbif_taxonomy"] = _download_state_entry(crawl_finished, taxonomy_key, gbif_meta)
        save_sync_state(state)
        print("Taxonomy sync done.")

    if occurrences_key:
        gbif_meta = poll_until_ready(occurrences_key)
        download_zip(gbif_meta["downloadLink"], OCCURRENCES_DIR)
        extract(OCCURRENCES_DIR)
        _cleanup_occurrences_dir()
        state["gbif_occurrences"] = _download_state_entry(crawl_finished, occurrences_key, gbif_meta)
        save_sync_state(state)
        print("Occurrences sync done.")

    return True


if __name__ == "__main__":  # pragma: no cover
    sync_all()
