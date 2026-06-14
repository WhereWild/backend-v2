# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_SOURCES_PATH = Path("config/data_sources.json")
SYNC_STATE_PATH = Path("data/sync_state.json")

INAT_OBSERVATIONS_DATASET_DOI = "https://doi.org/10.15468/ab3s5x"


def _load_sync_state() -> dict:
    return json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}


@lru_cache(maxsize=1)
def load_data_sources() -> dict:
    sources = json.loads(DATA_SOURCES_PATH.read_text())
    state = _load_sync_state()

    # Populate GBIF occurrence entry from the occurrence download sync state.
    gbif_occ_state = state.get("gbif_occurrences", {})
    if gbif_occ_state and "gbif_occurrence" in sources:
        occ = sources["gbif_occurrence"]
        doi_url = f"https://doi.org/{gbif_occ_state['doi']}" if gbif_occ_state.get("doi") else ""
        citation = gbif_occ_state.get("citation", "")
        occ["citation"] = citation
        occ["doi"] = doi_url
        occ["download_url"] = gbif_occ_state.get("download_link", "") or ""
        ref: dict = {"title": citation, "authors": "GBIF.org"}
        if doi_url:
            ref["doi"] = doi_url
        occ["references"] = [ref] if citation or doi_url else []

    # Populate iNaturalist observations entry with dynamic access date.
    if "inat_observations" in sources:
        crawl_finished = gbif_occ_state.get("crawl_finished", "")
        if crawl_finished:
            access_date = crawl_finished[:10]  # YYYY-MM-DD
            access_year = int(access_date[:4])
            # Full citation text per iNaturalist/GBIF recommended format.
            # DOI is embedded in the title string so it appears inline rather
            # than as a duplicate separate link.
            title = (
                f"iNaturalist Research-grade Observations. iNaturalist.org. "
                f"Occurrence dataset {INAT_OBSERVATIONS_DATASET_DOI} "
                f"accessed via GBIF.org on {access_date}"
            )
            sources["inat_observations"]["references"] = [
                {
                    "authors": "iNaturalist contributors, iNaturalist",
                    "year": access_year,
                    "title": title,
                }
            ]

    return sources
