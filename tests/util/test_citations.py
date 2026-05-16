import json

import pytest

import util.citations as cit


@pytest.fixture(autouse=True)
def clear_cache():
    cit.load_data_sources.cache_clear()
    yield
    cit.load_data_sources.cache_clear()


def test_load_data_sources_keys():
    sources = cit.load_data_sources()
    assert set(sources) == {"gbif_occurrence", "gbif_backbone", "inat_taxonomy"}


def test_gbif_backbone_has_doi():
    src = cit.load_data_sources()["gbif_backbone"]
    assert "10.15468/39omei" in src["doi"]
    assert src["license"] == "CC BY 4.0"


def test_inat_taxonomy_fields():
    src = cit.load_data_sources()["inat_taxonomy"]
    assert "inaturalist" in src["citation"].lower()
    assert src["doi"] == ""


def test_gbif_occurrence_reads_sync_state(monkeypatch, tmp_path):
    state = {
        "gbif_taxonomy": {
            "doi": "10.15468/dl.test",
            "citation": "GBIF.org (01 Jan 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.test",
            "download_link": "https://api.gbif.org/v1/occurrence/download/request/0000000-000000.zip",
        }
    }
    state_file = tmp_path / "sync_state.json"
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr(cit, "SYNC_STATE_PATH", state_file)
    src = cit.load_data_sources()["gbif_occurrence"]
    assert "10.15468/dl.test" in src["doi"]
    assert "GBIF.org (01 Jan 2026)" in src["citation"]
    assert "0000000-000000.zip" in src["download_url"]


def test_gbif_occurrence_missing_sync_state(monkeypatch, tmp_path):
    monkeypatch.setattr(cit, "SYNC_STATE_PATH", tmp_path / "nonexistent.json")
    src = cit.load_data_sources()["gbif_occurrence"]
    assert src["citation"] == ""
    assert src["doi"] == ""
    assert src["download_url"] == ""
