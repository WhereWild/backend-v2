# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sync GBIF species list for Plantae (full taxonomy, all ranks).

Checks the iNaturalist crawl history on GBIF to detect new ingestions.
Only re-downloads if GBIF has processed new data since our last download.
Credentials are read from environment variables loaded from .env
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


def request_download() -> str:
    payload = {
        "creator": GBIF_USER,
        "notificationAddresses": [GBIF_EMAIL],
        "sendNotification": True,
        "format": "SPECIES_LIST",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY},
                {"type": "equals", "key": "TAXON_KEY", "value": str(CONFIG.plantae_key)},
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


def request_occurrence_download() -> str:
    payload = {
        "creator": GBIF_USER,
        "notificationAddresses": [GBIF_EMAIL],
        "sendNotification": True,
        "format": "DWCA",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY},
                {"type": "equals", "key": "TAXON_KEY", "value": str(CONFIG.plantae_key)},
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

    download_key = request_occurrence_download()
    gbif_meta = poll_until_ready(download_key)
    download_zip(gbif_meta["downloadLink"], OCCURRENCES_DIR)
    extract(OCCURRENCES_DIR)
    _cleanup_occurrences_dir()

    state["gbif_occurrences"] = {
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

    download_key = request_download()
    gbif_meta = poll_until_ready(download_key)
    download_zip(gbif_meta["downloadLink"])
    extract()

    state["gbif_taxonomy"] = {
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
    save_sync_state(state)
    print("Done.")
    return True


if __name__ == "__main__":  # pragma: no cover
    main()
    sync_occurrences()
