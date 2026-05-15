import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest_httpx import HTTPXMock

import scripts.sync_gbif as sync_gbif

CRAWL_TS = "2026-05-15T15:54:14.220+00:00"
DOWNLOAD_KEY = "0020579-260507073636908"
DOWNLOAD_LINK = "https://api.gbif.org/v1/occurrence/download/request/0020579-260507073636908.zip"


@pytest.fixture(autouse=True)
def patch_catalog_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sync_gbif, "CATALOG_DIR", tmp_path / "catalog")


@pytest.fixture(autouse=True)
def patch_creds(monkeypatch):
    monkeypatch.setattr(sync_gbif, "GBIF_USER", "testuser")
    monkeypatch.setattr(sync_gbif, "GBIF_PASSWORD", "testpass")
    monkeypatch.setattr(sync_gbif, "GBIF_EMAIL", "test@example.com")


def _crawl_response(finish_reason="NORMAL", ts=CRAWL_TS):
    return {"results": [{"finishReason": finish_reason, "finishedCrawling": ts}]}


def _make_zip(catalog_dir: Path, content: bytes = b"data") -> None:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(catalog_dir / "download.zip", "w") as z:
        z.writestr("species.tsv", content)


# --- _build_citation ---

def test_build_citation_from_api():
    meta = {"citation": "Already provided", "doi": "x", "created": "2026-01-01T00:00:00+00:00"}
    assert sync_gbif._build_citation(meta) == "Already provided"


def test_build_citation_constructed():
    meta = {"doi": "10.15468/dl.abc", "created": "2026-05-15T22:02:36.884+00:00"}
    assert sync_gbif._build_citation(meta) == "GBIF.org (15 May 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.abc"


def test_build_citation_bad_date():
    meta = {"doi": "10.15468/dl.abc", "created": "not-a-date"}
    result = sync_gbif._build_citation(meta)
    assert "not-a-date" in result


# --- latest_crawl_finished ---

def test_latest_crawl_finished(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    assert sync_gbif.latest_crawl_finished() == CRAWL_TS


def test_latest_crawl_finished_skips_non_normal(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={
        "results": [
            {"finishReason": "ABORT", "finishedCrawling": CRAWL_TS},
            {"finishReason": "NORMAL", "finishedCrawling": "2026-01-01T00:00:00.000+00:00"},
        ]
    })
    assert sync_gbif.latest_crawl_finished() == "2026-01-01T00:00:00.000+00:00"


def test_latest_crawl_finished_none_normal(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"results": [{"finishReason": "ABORT", "finishedCrawling": CRAWL_TS}]})
    with pytest.raises(RuntimeError, match="No successful crawl"):
        sync_gbif.latest_crawl_finished()


# --- load_meta / save_meta ---

def test_load_meta_missing():
    assert sync_gbif.load_meta() == {}


def test_load_meta_existing():
    sync_gbif.CATALOG_DIR.mkdir(parents=True)
    (sync_gbif.CATALOG_DIR / "meta.json").write_text(json.dumps({"crawl_finished": CRAWL_TS}))
    assert sync_gbif.load_meta() == {"crawl_finished": CRAWL_TS}


def test_save_meta():
    sync_gbif.save_meta({"crawl_finished": CRAWL_TS, "download_key": DOWNLOAD_KEY})
    saved = json.loads((sync_gbif.CATALOG_DIR / "meta.json").read_text())
    assert saved["crawl_finished"] == CRAWL_TS
    assert saved["download_key"] == DOWNLOAD_KEY


# --- request_download ---

def test_request_download(httpx_mock: HTTPXMock):
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    assert sync_gbif.request_download() == DOWNLOAD_KEY


# --- poll_until_ready ---

GBIF_META = {
    "status": "SUCCEEDED",
    "downloadLink": DOWNLOAD_LINK,
    "doi": "10.15468/dl.7xvnxe",
    "citation": "GBIF.org (15 May 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.7xvnxe",
    "created": "2026-05-15T22:02:36.000+00:00",
    "eraseAfter": "2026-11-15T00:00:00.000+00:00",
    "totalRecords": 1122173,
    "numberDatasets": 19918,
    "size": 96796672,
}


def test_poll_until_ready_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"status": "PREPARING"})
    httpx_mock.add_response(json=GBIF_META)
    with patch("time.sleep"):
        result = sync_gbif.poll_until_ready(DOWNLOAD_KEY, interval=1)
    assert result["downloadLink"] == DOWNLOAD_LINK
    assert result["doi"] == "10.15468/dl.7xvnxe"


def test_poll_until_ready_failed(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"status": "FAILED"})
    with patch("time.sleep"), pytest.raises(RuntimeError, match="FAILED"):
        sync_gbif.poll_until_ready(DOWNLOAD_KEY, interval=1)


def test_poll_until_ready_timeout():
    with patch("time.sleep"), pytest.raises(TimeoutError):
        sync_gbif.poll_until_ready(DOWNLOAD_KEY, interval=1, timeout=0)


# --- download_zip ---

def test_download_zip(httpx_mock: HTTPXMock):
    redirect_url = "https://occurrence-download.gbif.org/occurrence/download/request/0020579.zip"
    httpx_mock.add_response(status_code=302, headers={"location": redirect_url})
    httpx_mock.add_response(content=b"zipdata")
    sync_gbif.download_zip(DOWNLOAD_LINK)
    assert (sync_gbif.CATALOG_DIR / "download.zip").read_bytes() == b"zipdata"


# --- extract ---

def test_extract():
    _make_zip(sync_gbif.CATALOG_DIR)
    sync_gbif.extract()
    assert (sync_gbif.CATALOG_DIR / "species.tsv").exists()


# --- main ---

def test_main_missing_creds(monkeypatch):
    monkeypatch.setattr(sync_gbif, "GBIF_USER", "")
    with pytest.raises(EnvironmentError):
        sync_gbif.main()


def test_main_already_up_to_date(httpx_mock: HTTPXMock, capsys):
    httpx_mock.add_response(json=_crawl_response())
    sync_gbif.save_meta({"crawl_finished": CRAWL_TS})
    sync_gbif.main()
    assert "Already up to date" in capsys.readouterr().out


def test_main_new_crawl(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')

    with patch("scripts.sync_gbif.poll_until_ready", return_value=GBIF_META), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"):
        sync_gbif.main()

    meta = json.loads((sync_gbif.CATALOG_DIR / "meta.json").read_text())
    assert meta["crawl_finished"] == CRAWL_TS
    assert meta["download_key"] == DOWNLOAD_KEY
    assert meta["doi"] == "10.15468/dl.7xvnxe"
    assert meta["total_records"] == 1122173
    assert "GBIF.org" in meta["citation"]
