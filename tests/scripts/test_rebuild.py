import json
import os
from unittest.mock import MagicMock, patch

import pytest

import scripts.rebuild as rebuild


@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(rebuild, "DATA_DIR", data_dir)
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", data_dir / "sync_state.json")
    monkeypatch.setattr(rebuild, "TAXONOMY_CACHE_DIR", data_dir / "taxonomy" / "cache")
    monkeypatch.setattr(rebuild, "NOTIFY_URL", "")
    monkeypatch.setattr(rebuild, "STATUS_PUSH_URL", "")
    monkeypatch.setattr("sys.argv", ["rebuild"])


@pytest.fixture(autouse=True)
def patch_enrich_tree():
    with patch("scripts.enrich_tree.main"):
        yield


@pytest.fixture(autouse=True)
def patch_process_tree():
    with patch("scripts.process_tree.main"):
        yield


def _pipeline(tmp_path) -> dict:
    return json.loads((tmp_path / "data" / "sync_state.json").read_text())["pipeline"]


# ---------------------------------------------------------------------------
# wipe_data_dir
# ---------------------------------------------------------------------------

def test_wipe_data_dir_preserves_sync_state(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "taxonomy").mkdir()
    (data_dir / "taxonomy" / "catalog.pkl").write_bytes(b"data")
    sync_state = data_dir / "sync_state.json"
    sync_state.write_text('{"key": "val"}')
    monkeypatch.setattr(rebuild, "DATA_DIR", data_dir)
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", sync_state)

    rebuild.wipe_data_dir()

    assert data_dir.exists()
    assert sync_state.exists()
    assert sync_state.read_text() == '{"key": "val"}'
    assert not (data_dir / "taxonomy").exists()


def test_wipe_data_dir_preserves_gis(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gis_dir = data_dir / "gis"
    gis_dir.mkdir()
    (gis_dir / "layers").mkdir()
    (gis_dir / "layers" / "bio1.tif").write_bytes(b"raster")
    (data_dir / "taxonomy").mkdir()
    monkeypatch.setattr(rebuild, "DATA_DIR", data_dir)
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", data_dir / "sync_state.json")

    rebuild.wipe_data_dir()

    assert (gis_dir / "layers" / "bio1.tif").exists()
    assert not (data_dir / "taxonomy").exists()


def test_wipe_data_dir_no_sync_state(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "some_file.pkl").write_bytes(b"x")
    monkeypatch.setattr(rebuild, "DATA_DIR", data_dir)
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", data_dir / "sync_state.json")

    rebuild.wipe_data_dir()

    assert data_dir.exists()
    assert not (data_dir / "some_file.pkl").exists()
    assert not (data_dir / "sync_state.json").exists()


def test_wipe_data_dir_missing_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rebuild, "DATA_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", tmp_path / "nonexistent" / "sync_state.json")
    rebuild.wipe_data_dir()  # should not raise


# ---------------------------------------------------------------------------
# _run_download_gis
# ---------------------------------------------------------------------------

def test_run_download_gis_empty_dir(tmp_path):
    gis_dir = tmp_path / "gis"
    gis_dir.mkdir()
    rebuild._run_download_gis(gis_dir=gis_dir)  # should not raise


def test_run_download_gis_calls_each_download_script(tmp_path):
    gis_dir = tmp_path / "gis"
    gis_dir.mkdir()
    (gis_dir / "download_a.py").touch()
    (gis_dir / "download_b.py").touch()
    (gis_dir / "build_overviews.py").touch()  # should be ignored

    called = []
    fake_a, fake_b = MagicMock(), MagicMock()
    fake_a.main.side_effect = lambda: called.append("download_a")
    fake_b.main.side_effect = lambda: called.append("download_b")

    def fake_import(name):
        return {"scripts.gis.download_a": fake_a, "scripts.gis.download_b": fake_b}[name]

    with patch("scripts.rebuild.importlib") as mock_importlib:
        mock_importlib.import_module.side_effect = fake_import
        rebuild._run_download_gis(gis_dir=gis_dir)

    assert called == ["download_a", "download_b"]


# ---------------------------------------------------------------------------
# notify
# ---------------------------------------------------------------------------

def test_notify_skips_when_no_url(monkeypatch):
    monkeypatch.setattr(rebuild, "NOTIFY_URL", "")
    with patch("httpx.post") as mock_post:
        rebuild.notify("errored", {"error": {"message": "boom"}})
    mock_post.assert_not_called()


def test_notify_posts_event(monkeypatch):
    monkeypatch.setattr(rebuild, "NOTIFY_URL", "http://bot/notify")
    with patch("httpx.post") as mock_post:
        rebuild.notify("errored", {"error": {"message": "boom"}})
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["event"] == "errored"
    assert kwargs["json"]["error"]["message"] == "boom"


def test_notify_swallows_request_failure(monkeypatch):
    monkeypatch.setattr(rebuild, "NOTIFY_URL", "http://bot/notify")
    with patch("httpx.post", side_effect=Exception("connection refused")):
        rebuild.notify("crashed", {})  # should not raise


# ---------------------------------------------------------------------------
# _acquire_shutdown_inhibitor / _release_inhibitor
# ---------------------------------------------------------------------------

def test_acquire_inhibitor_missing_binary():
    with patch("subprocess.Popen", side_effect=FileNotFoundError):
        result = rebuild._acquire_shutdown_inhibitor()
    assert result is None


def test_acquire_inhibitor_success():
    mock_proc = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        result = rebuild._acquire_shutdown_inhibitor()
    assert result is mock_proc
    args = mock_popen.call_args[0][0]
    assert "systemd-inhibit" in args
    assert "--what=shutdown" in args
    assert "--mode=delay" in args


def test_release_inhibitor_terminates():
    mock_proc = MagicMock()
    rebuild._release_inhibitor(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()


def test_release_inhibitor_none():
    rebuild._release_inhibitor(None)  # should not raise


# ---------------------------------------------------------------------------
# main — pipeline state transitions
# ---------------------------------------------------------------------------

def _patch_sync_check(new_crawl_ts="2026-05-15T15:54:14.220+00:00", existing_ts=None):
    """Patch sync_gbif pre-check helpers used by rebuild."""
    state = {
        "gbif_taxonomy": {"crawl_finished": existing_ts},
        "gbif_occurrences": {"crawl_finished": existing_ts},
    } if existing_ts else {}
    return (
        patch("scripts.sync_gbif.latest_crawl_finished", return_value=new_crawl_ts),
        patch("scripts.sync_gbif.load_sync_state", return_value=state),
    )


def test_main_already_up_to_date(tmp_path):
    ts = "2026-05-15T15:54:14.220+00:00"
    check1, check2 = _patch_sync_check(new_crawl_ts=ts, existing_ts=ts)
    with check1, check2, \
         patch("scripts.rebuild._acquire_shutdown_inhibitor") as mock_inhibitor:
        rebuild.main()

    mock_inhibitor.assert_not_called()  # inhibitor not acquired when nothing to do
    assert not (tmp_path / "data" / "sync_state.json").exists()  # state untouched


def test_main_full_pipeline_completes(tmp_path):
    call_order = []
    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir", side_effect=lambda: call_order.append("wipe")), \
         patch("scripts.build_tree.main", side_effect=lambda: call_order.append("tree")), \
         patch("scripts.populate_tree.main", side_effect=lambda: call_order.append("populate")), \
         patch("scripts.gis.process_gadm.main", side_effect=lambda: call_order.append("process_gadm")), \
         patch("scripts.rebuild._run_download_gis", side_effect=lambda: call_order.append("download_gis")), \
         patch("scripts.gis.build_overviews.main", side_effect=lambda: call_order.append("build_overviews")), \
         patch("scripts.enrich_tree.main", side_effect=lambda: call_order.append("enrich_tree")), \
         patch("scripts.enrich_temporal.main", side_effect=lambda: call_order.append("enrich_temporal")), \
         patch("scripts.process_tree.main", side_effect=lambda: call_order.append("process_tree")), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify") as mock_notify:
        rebuild.main()

    assert call_order == [
        "wipe", "tree", "populate",
        "process_gadm", "download_gis", "build_overviews", "enrich_tree", "enrich_temporal", "process_tree",
    ]
    p = _pipeline(tmp_path)
    assert p["status"] == "completed"
    assert all(
        p["stages"][s]["status"] == "completed"
        for s in [
            "sync_gbif", "build_tree", "populate_tree",
            "process_gadm", "download_gis", "build_overviews", "enrich_tree", "enrich_temporal", "process_tree",
        ]
    )
    assert p["error"] is None
    mock_notify.assert_called_once()
    event, payload = mock_notify.call_args[0]
    assert event == "completed"
    assert "stages" in payload
    assert "duration_s" in payload
    assert isinstance(payload["duration_s"], int)


def test_main_wipe_happens_before_sync_download(tmp_path):
    """Wipe must precede sync_gbif.main() so the download lands in a clean dir."""
    call_order = []
    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main", side_effect=lambda: call_order.append("sync")), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir", side_effect=lambda: call_order.append("wipe")), \
         patch("scripts.build_tree.main"), \
         patch("scripts.populate_tree.main"), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"):
        rebuild.main()

    assert call_order.index("wipe") < call_order.index("sync")


def test_main_stage_in_progress_written_before_run(tmp_path):
    seen = []

    def capture():
        state = json.loads((tmp_path / "data" / "sync_state.json").read_text())
        seen.append(state["pipeline"]["stages"].get("build_tree"))

    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.build_tree.main", side_effect=capture), \
         patch("scripts.populate_tree.main"), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"):
        rebuild.main()

    assert seen[0]["status"] == "in_progress"


def test_main_errored_on_exception(tmp_path):
    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.build_tree.main", side_effect=RuntimeError("boom in build_tree")), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify") as mock_notify, \
         pytest.raises(RuntimeError, match="boom in build_tree"):
        rebuild.main()

    p = _pipeline(tmp_path)
    assert p["status"] == "errored"
    assert p["error"]["stage"] == "build_tree"
    assert "boom in build_tree" in p["error"]["message"]
    assert "RuntimeError" in p["error"]["traceback"]
    assert p["finished_at"] is not None
    mock_notify.assert_called_once_with("errored", {"error": p["error"]})


def test_main_crash_detected_on_next_run(tmp_path, capsys):
    sync_state_path = tmp_path / "data" / "sync_state.json"
    sync_state_path.parent.mkdir(parents=True, exist_ok=True)
    sync_state_path.write_text(json.dumps({
        "pipeline": {"status": "in_progress", "stage": "build_tree", "stages": {}}
    }))

    with patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.build_tree.main"), \
         patch("scripts.populate_tree.main"), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify") as mock_notify:
        rebuild.main()

    out = capsys.readouterr().out
    assert "crashed" in out
    # notify is called twice: once for "crashed", once for "completed" (resume ran)
    calls = {call[0][0] for call in mock_notify.call_args_list}
    assert "crashed" in calls
    crashed_call = next(c for c in mock_notify.call_args_list if c[0][0] == "crashed")
    assert crashed_call[0][1]["stage"] == "build_tree"


def test_main_crash_overwrites_pipeline_state(tmp_path):
    sync_state_path = tmp_path / "data" / "sync_state.json"
    sync_state_path.parent.mkdir(parents=True, exist_ok=True)
    sync_state_path.write_text(json.dumps({
        "pipeline": {"status": "in_progress", "stage": "build_tree", "stages": {}}
    }))

    with patch("scripts.sync_gbif.main", side_effect=RuntimeError("sync fail")), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify"), \
         pytest.raises(RuntimeError):
        rebuild.main()

    p = _pipeline(tmp_path)
    assert p["status"] == "errored"


def test_main_inhibitor_released_on_error():
    mock_proc = MagicMock()
    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.build_tree.main", side_effect=RuntimeError("fail")), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=mock_proc), \
         patch("scripts.rebuild._release_inhibitor") as mock_release, \
         pytest.raises(RuntimeError):
        rebuild.main()

    mock_release.assert_called_once_with(mock_proc)


def test_main_inhibitor_released_on_success():
    mock_proc = MagicMock()
    check1, check2 = _patch_sync_check()
    with check1, check2, \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.build_tree.main"), \
         patch("scripts.populate_tree.main"), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=mock_proc), \
         patch("scripts.rebuild._release_inhibitor") as mock_release, \
         patch("scripts.rebuild.notify"):
        rebuild.main()

    mock_release.assert_called_once_with(mock_proc)


def test_wipe_data_dir_preserves_taxonomy_cache(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "taxonomy" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "inat_dwca.zip").write_bytes(b"zipdata")
    (cache_dir / "gbif_vernacular.tsv").write_bytes(b"tsvdata")
    (data_dir / "taxonomy" / "catalog.pkl").write_bytes(b"other")
    monkeypatch.setattr(rebuild, "DATA_DIR", data_dir)
    monkeypatch.setattr(rebuild, "SYNC_STATE_PATH", data_dir / "sync_state.json")
    monkeypatch.setattr(rebuild, "TAXONOMY_CACHE_DIR", cache_dir)

    rebuild.wipe_data_dir()

    assert (cache_dir / "inat_dwca.zip").read_bytes() == b"zipdata"
    assert (cache_dir / "gbif_vernacular.tsv").read_bytes() == b"tsvdata"
    assert not (data_dir / "taxonomy" / "catalog.pkl").exists()


def test_main_force_clears_gbif_crawl_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rebuild", "--force"])
    sync_state_path = tmp_path / "data" / "sync_state.json"
    sync_state_path.parent.mkdir(parents=True, exist_ok=True)
    sync_state_path.write_text(json.dumps({
        "gbif_taxonomy": {"crawl_finished": "2026-01-01"},
        "gbif_occurrences": {"crawl_finished": "2026-01-01"},
    }))

    with patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.sync_gbif.main"), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.build_tree.main"), \
         patch("scripts.populate_tree.main"), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify"):
        rebuild.main()

    state = json.loads(sync_state_path.read_text())
    assert "gbif_taxonomy" not in state
    assert "gbif_occurrences" not in state


def test_main_stage_flag_skips_prior_stages(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rebuild", "--stage", "enrich_tree"])
    call_order = []

    with patch("scripts.sync_gbif.main", side_effect=lambda: call_order.append("sync_gbif")), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.rebuild.wipe_data_dir", side_effect=lambda: call_order.append("wipe")), \
         patch("scripts.build_tree.main", side_effect=lambda: call_order.append("build_tree")), \
         patch("scripts.populate_tree.main", side_effect=lambda: call_order.append("populate_tree")), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main", side_effect=lambda: call_order.append("enrich_tree")), \
         patch("scripts.enrich_temporal.main", side_effect=lambda: call_order.append("enrich_temporal")), \
         patch("scripts.process_tree.main", side_effect=lambda: call_order.append("process_tree")), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify"):
        rebuild.main()

    assert "wipe" not in call_order
    assert "sync_gbif" not in call_order
    assert "build_tree" not in call_order
    assert "enrich_tree" in call_order
    assert "enrich_temporal" in call_order
    assert "process_tree" in call_order


def test_main_resume_skips_completed_stages(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rebuild"])
    sync_state_path = tmp_path / "data" / "sync_state.json"
    sync_state_path.parent.mkdir(parents=True, exist_ok=True)
    sync_state_path.write_text(json.dumps({
        "pipeline": {
            "status": "in_progress",
            "stage": "build_tree",
            "stages": {
                "sync_gbif": {"status": "completed"},
                "build_tree": {"status": "completed"},
            },
        }
    }))

    call_order = []
    with patch("scripts.rebuild.wipe_data_dir"), \
         patch("scripts.sync_gbif.main", side_effect=lambda: call_order.append("sync_gbif")), \
         patch("scripts.sync_gbif.sync_occurrences"), \
         patch("scripts.build_tree.main", side_effect=lambda: call_order.append("build_tree")), \
         patch("scripts.populate_tree.main", side_effect=lambda: call_order.append("populate_tree")), \
         patch("scripts.gis.process_gadm.main"), \
         patch("scripts.rebuild._run_download_gis"), \
         patch("scripts.gis.build_overviews.main"), \
         patch("scripts.enrich_tree.main"), \
         patch("scripts.enrich_temporal.main"), \
         patch("scripts.process_tree.main"), \
         patch("scripts.rebuild._acquire_shutdown_inhibitor", return_value=None), \
         patch("scripts.rebuild._release_inhibitor"), \
         patch("scripts.rebuild.notify"):
        rebuild.main()

    assert "sync_gbif" not in call_order
    assert "build_tree" not in call_order
    assert "populate_tree" in call_order
