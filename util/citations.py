from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_SOURCES_PATH = Path("config/data_sources.json")
SYNC_STATE_PATH = Path("data/sync_state.json")


def _load_sync_state() -> dict:
    return json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}


@lru_cache(maxsize=1)
def load_data_sources() -> dict:
    sources = json.loads(DATA_SOURCES_PATH.read_text())
    gbif_taxonomy = _load_sync_state().get("gbif_taxonomy", {})
    sources["gbif_occurrence"]["citation"] = gbif_taxonomy.get("citation", "")
    sources["gbif_occurrence"]["doi"] = (
        f"https://doi.org/{gbif_taxonomy['doi']}" if gbif_taxonomy.get("doi") else ""
    )
    sources["gbif_occurrence"]["download_url"] = gbif_taxonomy.get("download_link", "") or ""
    return sources
