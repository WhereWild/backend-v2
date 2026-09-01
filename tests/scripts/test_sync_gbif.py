# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    monkeypatch.setattr(sync_gbif, "SYNC_STATE_PATH", tmp_path / "sync_state.json")


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
    httpx_mock.add_response(
        json={"results": [{"finishReason": "ABORT", "finishedCrawling": CRAWL_TS}]}
    )
    with pytest.raises(RuntimeError, match="No successful crawl"):
        sync_gbif.latest_crawl_finished()


# --- load_sync_state / save_sync_state ---

def test_load_sync_state_missing():
    assert sync_gbif.load_sync_state() == {}


def test_load_sync_state_existing():
    data = {"gbif_taxonomy": {"crawl_finished": CRAWL_TS}}
    sync_gbif.SYNC_STATE_PATH.write_text(json.dumps(data))
    assert sync_gbif.load_sync_state() == data


def test_save_sync_state():
    sync_gbif.save_sync_state({
        "gbif_taxonomy": {"crawl_finished": CRAWL_TS, "download_key": DOWNLOAD_KEY},
    })
    saved = json.loads(sync_gbif.SYNC_STATE_PATH.read_text())
    assert saved["gbif_taxonomy"]["crawl_finished"] == CRAWL_TS
    assert saved["gbif_taxonomy"]["download_key"] == DOWNLOAD_KEY


# --- _predicate_matches / _find_existing_download ---

def _species_list_request(values=("7HS", "CXQ")):
    return {
        "format": "SPECIES_LIST",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": sync_gbif.INAT_DATASET_KEY},
                {
                    "type": "in", "key": "TAXON_KEY", "values": list(values),
                    "checklistKey": sync_gbif.COL_XR_CHECKLIST_KEY,
                },
                {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
            ],
        },
    }


def _dwca_request(values=("7HS", "CXQ")):
    req = _species_list_request(values)
    req["format"] = "DWCA"
    req["predicate"]["predicates"].append(
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"}
    )
    return req


def test_predicate_matches_species_list(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    assert sync_gbif._predicate_matches(_species_list_request(), "SPECIES_LIST", has_coordinate=False)


def test_predicate_matches_order_independent(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    assert sync_gbif._predicate_matches(
        _species_list_request(values=("CXQ", "7HS")), "SPECIES_LIST", has_coordinate=False
    )


def test_predicate_matches_false_wrong_format(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    assert not sync_gbif._predicate_matches(_species_list_request(), "DWCA", has_coordinate=False)


def test_predicate_matches_false_different_roots(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    assert not sync_gbif._predicate_matches(
        _species_list_request(values=("7HS",)), "SPECIES_LIST", has_coordinate=False
    )


def test_predicate_matches_false_wrong_checklist(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    req = _species_list_request()
    req["predicate"]["predicates"][1]["checklistKey"] = "some-other-checklist"
    assert not sync_gbif._predicate_matches(req, "SPECIES_LIST", has_coordinate=False)


def test_predicate_matches_dwca_requires_has_coordinate(monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    assert sync_gbif._predicate_matches(_dwca_request(), "DWCA", has_coordinate=True)
    assert not sync_gbif._predicate_matches(_dwca_request(), "DWCA", has_coordinate=False)
    assert not sync_gbif._predicate_matches(_species_list_request(), "SPECIES_LIST", has_coordinate=True)


def test_find_existing_download_prefers_succeeded_over_newer_preparing(httpx_mock: HTTPXMock, monkeypatch):
    # Results come back newest-first; the newest match is still PREPARING
    # while an older identical one already SUCCEEDED — must pick the
    # succeeded one, not just the first match in list order.
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "new-preparing", "status": "PREPARING", "request": _species_list_request()},
        {"key": "old-succeeded", "status": "SUCCEEDED", "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download("SPECIES_LIST", has_coordinate=False) == "old-succeeded"


def test_find_existing_download_falls_back_to_preparing_if_none_succeeded(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "preparing-1", "status": "PREPARING", "request": _species_list_request()},
        {"key": "preparing-2", "status": "RUNNING", "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download("SPECIES_LIST", has_coordinate=False) == "preparing-1"


def test_find_existing_download_skips_terminal_failures(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "cancelled", "status": "CANCELLED", "request": _species_list_request()},
        {"key": "failed", "status": "FAILED", "request": _species_list_request()},
        {"key": "killed", "status": "KILLED", "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download("SPECIES_LIST", has_coordinate=False) is None


def test_find_existing_download_no_match_returns_none(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "wrong-format", "status": "SUCCEEDED", "request": _dwca_request()},
    ]})
    assert sync_gbif._find_existing_download("SPECIES_LIST", has_coordinate=False) is None


def test_find_existing_download_skips_export_older_than_crawl(httpx_mock: HTTPXMock, monkeypatch):
    # An export built before the crawl we're syncing for can't contain that
    # crawl's data — must be ignored so a fresh request goes out.
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "stale", "status": "SUCCEEDED", "created": "2026-08-08T05:38:55.898+00:00",
         "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download(
        "SPECIES_LIST", has_coordinate=False, min_created="2026-08-28T18:55:15.733+00:00"
    ) is None


def test_find_existing_download_reuses_export_newer_than_crawl(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "fresh", "status": "SUCCEEDED", "created": "2026-08-28T19:10:00.000+00:00",
         "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download(
        "SPECIES_LIST", has_coordinate=False, min_created="2026-08-28T18:55:15.733+00:00"
    ) == "fresh"


def test_find_existing_download_skips_export_missing_created_when_crawl_given(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(json={"results": [
        {"key": "no-created", "status": "SUCCEEDED", "request": _species_list_request()},
    ]})
    assert sync_gbif._find_existing_download(
        "SPECIES_LIST", has_coordinate=False, min_created="2026-08-28T18:55:15.733+00:00"
    ) is None


# --- request_download ---

def test_request_download_reuses_existing(httpx_mock: HTTPXMock, capsys):
    with patch("scripts.sync_gbif._find_existing_download", return_value="old-succeeded"):
        assert sync_gbif.request_download() == "old-succeeded"
    assert "Reusing existing download" in capsys.readouterr().out
    assert len(httpx_mock.get_requests()) == 0  # no POST made


def test_request_download(httpx_mock: HTTPXMock):
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    with patch("scripts.sync_gbif._find_existing_download", return_value=None):
        assert sync_gbif.request_download() == DOWNLOAD_KEY
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    predicates = body["predicate"]["predicates"]
    taxon_key_pred = next(p for p in predicates if p["key"] == "TAXON_KEY")
    assert taxon_key_pred["type"] == "in"
    assert taxon_key_pred["values"] == list(sync_gbif.CONFIG.taxonomy_roots)
    assert taxon_key_pred["checklistKey"] == sync_gbif.COL_XR_CHECKLIST_KEY


def test_request_download_multiple_roots(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setattr(sync_gbif.CONFIG, "taxonomy_roots", ("7HS", "CXQ"))
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    with patch("scripts.sync_gbif._find_existing_download", return_value=None):
        assert sync_gbif.request_download() == DOWNLOAD_KEY
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    predicates = body["predicate"]["predicates"]
    taxon_key_pred = next(p for p in predicates if p["key"] == "TAXON_KEY")
    assert taxon_key_pred["values"] == ["7HS", "CXQ"]


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

def test_download_zip(tmp_path):
    with patch("subprocess.run") as mock_run:
        sync_gbif.download_zip(DOWNLOAD_LINK)
    args = mock_run.call_args[0][0]
    assert args[0] == "aria2c"
    assert f"--http-user={sync_gbif.GBIF_USER}" in args
    assert f"--http-passwd={sync_gbif.GBIF_PASSWORD}" in args
    assert "--continue=true" in args
    assert "--max-tries=12" in args
    assert f"--dir={sync_gbif.CATALOG_DIR}" in args
    assert "--out=download.zip" in args
    assert DOWNLOAD_LINK in args
    assert mock_run.call_args[1].get("check") is True


def test_download_zip_custom_dest(tmp_path):
    dest = tmp_path / "custom"
    with patch("subprocess.run") as mock_run:
        sync_gbif.download_zip(DOWNLOAD_LINK, dest)
    args = mock_run.call_args[0][0]
    assert f"--dir={dest}" in args
    assert dest.exists()  # mkdir should have been called


# --- extract ---

def test_extract():
    _make_zip(sync_gbif.CATALOG_DIR)
    sync_gbif.extract()
    assert (sync_gbif.CATALOG_DIR / "species.tsv").exists()


def test_extract_renames_csv():
    catalog_dir = sync_gbif.CATALOG_DIR
    catalog_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(catalog_dir / "download.zip", "w") as z:
        z.writestr("0020579-260507073636908.csv", "taxon\tdata")
    sync_gbif.extract()
    assert (catalog_dir / "species_list.csv").exists()
    assert not (catalog_dir / "0020579-260507073636908.csv").exists()


def test_extract_dwca(tmp_path):
    with zipfile.ZipFile(tmp_path / "download.zip", "w") as z:
        z.writestr("occurrence.txt", "gbifID\tspecies\n1\tRosa")
        z.writestr("multimedia.txt", "gbifID\tidentifier\n1\thttps://example.com/img.jpg")
        z.writestr("meta.xml", "<archive/>")
        z.writestr("citations.txt", "cite me")
        z.writestr("rights.txt", "cc by")
    sync_gbif.extract(tmp_path)
    assert (tmp_path / "occurrence.txt").exists()
    assert (tmp_path / "multimedia.txt").exists()
    assert (tmp_path / "citations.txt").exists()


# --- request_occurrence_download ---

def test_request_occurrence_download_reuses_existing(httpx_mock: HTTPXMock, capsys):
    with patch("scripts.sync_gbif._find_existing_download", return_value="old-succeeded"):
        assert sync_gbif.request_occurrence_download() == "old-succeeded"
    assert "Reusing existing occurrence download" in capsys.readouterr().out
    assert len(httpx_mock.get_requests()) == 0


def test_request_occurrence_download(httpx_mock: HTTPXMock):
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    with patch("scripts.sync_gbif._find_existing_download", return_value=None):
        key = sync_gbif.request_occurrence_download()
    assert key == DOWNLOAD_KEY
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["format"] == "DWCA"
    predicates = body["predicate"]["predicates"]
    keys = {p["key"]: p["value"] for p in predicates if p["key"] != "TAXON_KEY"}
    assert keys["DATASET_KEY"] == sync_gbif.INAT_DATASET_KEY
    assert keys["HAS_COORDINATE"] == "true"
    assert keys["OCCURRENCE_STATUS"] == "PRESENT"
    taxon_key_pred = next(p for p in predicates if p["key"] == "TAXON_KEY")
    assert taxon_key_pred["type"] == "in"
    assert taxon_key_pred["values"] == list(sync_gbif.CONFIG.taxonomy_roots)
    assert taxon_key_pred["checklistKey"] == sync_gbif.COL_XR_CHECKLIST_KEY


# --- sync_occurrences ---

def test_sync_occurrences_missing_creds(monkeypatch):
    monkeypatch.setattr(sync_gbif, "GBIF_USER", "")
    with pytest.raises(OSError, match="GBIF_USER"):
        sync_gbif.sync_occurrences()


def test_sync_occurrences_already_up_to_date(httpx_mock: HTTPXMock, capsys):
    httpx_mock.add_response(json=_crawl_response())
    sync_gbif.save_sync_state({"gbif_occurrences": {"crawl_finished": CRAWL_TS}})
    result = sync_gbif.sync_occurrences()
    assert result is False
    assert "Already up to date" in capsys.readouterr().out


def test_sync_occurrences_new_crawl(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')

    with patch("scripts.sync_gbif._find_existing_download", return_value=None), \
         patch("scripts.sync_gbif.poll_until_ready", return_value=GBIF_META), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"):
        result = sync_gbif.sync_occurrences()

    assert result is True

    state = json.loads(sync_gbif.SYNC_STATE_PATH.read_text())
    occ = state["gbif_occurrences"]
    assert occ["crawl_finished"] == CRAWL_TS
    assert occ["download_key"] == DOWNLOAD_KEY
    assert occ["doi"] == "10.15468/dl.7xvnxe"
    assert occ["total_records"] == 1122173
    assert occ["citation"].startswith("GBIF.org")


# --- main ---

def test_main_missing_creds(monkeypatch):
    monkeypatch.setattr(sync_gbif, "GBIF_USER", "")
    with pytest.raises(OSError, match="GBIF_USER"):
        sync_gbif.main()


def test_main_already_up_to_date(httpx_mock: HTTPXMock, capsys):
    httpx_mock.add_response(json=_crawl_response())
    sync_gbif.save_sync_state({"gbif_taxonomy": {"crawl_finished": CRAWL_TS}})
    result = sync_gbif.main()
    assert result is False
    assert "Already up to date" in capsys.readouterr().out


def test_main_new_crawl(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')

    with patch("scripts.sync_gbif._find_existing_download", return_value=None), \
         patch("scripts.sync_gbif.poll_until_ready", return_value=GBIF_META), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"):
        result = sync_gbif.main()

    assert result is True

    state = json.loads(sync_gbif.SYNC_STATE_PATH.read_text())
    assert state["gbif_taxonomy"]["crawl_finished"] == CRAWL_TS
    assert state["gbif_taxonomy"]["download_key"] == DOWNLOAD_KEY
    assert state["gbif_taxonomy"]["doi"] == "10.15468/dl.7xvnxe"
    assert state["gbif_taxonomy"]["download_link"] == DOWNLOAD_LINK
    assert state["gbif_taxonomy"]["total_records"] == 1122173
    assert state["gbif_taxonomy"]["citation"].startswith("GBIF.org")


# --- sync_all ---

OCCURRENCE_DOWNLOAD_KEY = "0020580-260507073636909"


def test_sync_all_missing_creds(monkeypatch):
    monkeypatch.setattr(sync_gbif, "GBIF_USER", "")
    with pytest.raises(OSError, match="GBIF_USER"):
        sync_gbif.sync_all()


def test_sync_all_already_up_to_date(httpx_mock: HTTPXMock, capsys):
    httpx_mock.add_response(json=_crawl_response())
    sync_gbif.save_sync_state({
        "gbif_taxonomy": {"crawl_finished": CRAWL_TS},
        "gbif_occurrences": {"crawl_finished": CRAWL_TS},
    })
    result = sync_gbif.sync_all()
    assert result is False
    assert "Already up to date" in capsys.readouterr().out


def test_sync_all_requests_both_before_polling_either(httpx_mock: HTTPXMock):
    # The whole point of sync_all over main()+sync_occurrences() run
    # sequentially: both GBIF downloads must be *requested* before either is
    # polled, so they prepare concurrently on GBIF's side instead of one
    # only starting after the other has fully finished (~1hr each).
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    httpx_mock.add_response(text=f'"{OCCURRENCE_DOWNLOAD_KEY}"')
    call_order = []

    real_request_download = sync_gbif.request_download
    real_request_occurrence_download = sync_gbif.request_occurrence_download

    def tracked_request_download(*args, **kwargs):
        call_order.append("request_taxonomy")
        return real_request_download(*args, **kwargs)

    def tracked_request_occurrence_download(*args, **kwargs):
        call_order.append("request_occurrences")
        return real_request_occurrence_download(*args, **kwargs)

    def tracked_poll(key, *a, **kw):
        call_order.append(f"poll:{key}")
        return GBIF_META

    with patch("scripts.sync_gbif._find_existing_download", return_value=None), \
         patch("scripts.sync_gbif.request_download", side_effect=tracked_request_download), \
         patch("scripts.sync_gbif.request_occurrence_download", side_effect=tracked_request_occurrence_download), \
         patch("scripts.sync_gbif.poll_until_ready", side_effect=tracked_poll), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"), \
         patch("scripts.sync_gbif._cleanup_occurrences_dir"):
        result = sync_gbif.sync_all()

    assert result is True
    assert call_order == [
        "request_taxonomy",
        "request_occurrences",
        f"poll:{DOWNLOAD_KEY}",
        f"poll:{OCCURRENCE_DOWNLOAD_KEY}",
    ]


def test_sync_all_new_crawl_both(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{DOWNLOAD_KEY}"')
    httpx_mock.add_response(text=f'"{OCCURRENCE_DOWNLOAD_KEY}"')

    with patch("scripts.sync_gbif._find_existing_download", return_value=None), \
         patch("scripts.sync_gbif.poll_until_ready", return_value=GBIF_META), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"), \
         patch("scripts.sync_gbif._cleanup_occurrences_dir"):
        result = sync_gbif.sync_all()

    assert result is True
    state = json.loads(sync_gbif.SYNC_STATE_PATH.read_text())
    assert state["gbif_taxonomy"]["crawl_finished"] == CRAWL_TS
    assert state["gbif_taxonomy"]["download_key"] == DOWNLOAD_KEY
    assert state["gbif_occurrences"]["crawl_finished"] == CRAWL_TS
    assert state["gbif_occurrences"]["download_key"] == OCCURRENCE_DOWNLOAD_KEY


def test_sync_all_only_occurrences_stale(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_crawl_response())
    httpx_mock.add_response(text=f'"{OCCURRENCE_DOWNLOAD_KEY}"')
    sync_gbif.save_sync_state({"gbif_taxonomy": {"crawl_finished": CRAWL_TS}})

    with patch("scripts.sync_gbif._find_existing_download", return_value=None), \
         patch("scripts.sync_gbif.request_download") as mock_request_download, \
         patch("scripts.sync_gbif.poll_until_ready", return_value=GBIF_META), \
         patch("scripts.sync_gbif.download_zip"), \
         patch("scripts.sync_gbif.extract"), \
         patch("scripts.sync_gbif._cleanup_occurrences_dir"):
        result = sync_gbif.sync_all()

    assert result is True
    mock_request_download.assert_not_called()
    state = json.loads(sync_gbif.SYNC_STATE_PATH.read_text())
    assert state["gbif_taxonomy"]["crawl_finished"] == CRAWL_TS
    assert state["gbif_occurrences"]["crawl_finished"] == CRAWL_TS
    assert state["gbif_occurrences"]["download_key"] == OCCURRENCE_DOWNLOAD_KEY
