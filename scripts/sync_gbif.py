"""
Sync GBIF species list for Plantae (full taxonomy, all ranks).

Checks the iNaturalist crawl history on GBIF to detect new ingestions.
Only re-downloads if GBIF has processed new data since our last download.
Credentials are read from environment variables loaded from .env
"""

import json
import os
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


def download_zip(url: str) -> None:
    print(f"Downloading {url}...")
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    dest = CATALOG_DIR / "download.zip"
    with httpx.stream(
        "GET", url, auth=(GBIF_USER, GBIF_PASSWORD), timeout=300, follow_redirects=True
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    print(f"Saved to {dest}")


def extract() -> None:
    zip_file = CATALOG_DIR / "download.zip"
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(CATALOG_DIR)
    for f in CATALOG_DIR.glob("*.csv"):
        if f.name != "species_list.csv":
            f.rename(CATALOG_DIR / "species_list.csv")
            break
    files = [f.name for f in CATALOG_DIR.iterdir()]
    print(f"Extracted to {CATALOG_DIR}/: {files}")


def main() -> None:
    if not all([GBIF_USER, GBIF_PASSWORD, GBIF_EMAIL]):
        raise OSError("GBIF_USER, GBIF_PASSWORD, and GBIF_EMAIL must be set")

    print("Checking GBIF iNat crawl history...")
    crawl_finished = latest_crawl_finished()
    state = load_sync_state()

    if state.get("gbif_taxonomy", {}).get("crawl_finished") == crawl_finished:
        print(f"Already up to date (last crawl: {crawl_finished})")
        return

    print(f"New crawl detected: {crawl_finished}")

    download_key = request_download()
    gbif_meta = poll_until_ready(download_key)
    download_zip(gbif_meta["downloadLink"])
    extract()

    state["gbif_taxonomy"] = {
        "crawl_finished": crawl_finished,
        "download_key": download_key,
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


if __name__ == "__main__":  # pragma: no cover
    main()
