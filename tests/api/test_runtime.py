"""Tests for app/runtime behavior and startup guards."""

from __future__ import annotations

import asyncio
import runpy
import sys
import types
from pathlib import Path

import pytest

import main


def _run_lifespan_once() -> None:
    async def _inner() -> None:
        async with main.lifespan(main.app):
            return

    asyncio.run(_inner())


def test_lifespan_handles_preload_exceptions(monkeypatch):
    calls = {"count": 0}

    def fake_preload():
        calls["count"] += 1
        if calls["count"] == 1:
            raise FileNotFoundError("missing catalog")
        raise OSError("storage unavailable")

    monkeypatch.setattr(main.gis_lookup, "preload_layer_legends", fake_preload)
    _run_lifespan_once()
    _run_lifespan_once()
    assert calls["count"] == 2


def test_path_exists_uses_remote_storage_exists(monkeypatch):
    class FakeStorage:
        is_remote = True

        @staticmethod
        def exists(path: Path) -> bool:
            return str(path).endswith("ok")

    monkeypatch.setattr(main, "get_parquet_storage", lambda *_args, **_kwargs: FakeStorage())
    assert main._path_exists(Path("/tmp/ok"))
    assert not main._path_exists(Path("/tmp/nope"))


def test_main_guard_runs_uvicorn(monkeypatch):
    calls = {}

    fake_uvicorn = types.SimpleNamespace(
        run=lambda app, host, port, reload: calls.update(
            {"host": host, "port": port, "reload": reload, "app": app}
        )
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    runpy.run_module("main", run_name="__main__")
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8000
    assert calls["reload"] is True


def test_get_species_detail_description_failure_is_swallowed(monkeypatch):
    taxon = {"taxon_key": "1", "path": "/tmp/ok", "scientific_name": "T", "rank": "SPECIES"}
    payload = {"taxon_id": 1, "scientific_name": "T", "rank": "SPECIES", "slug": "t"}
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: taxon)
    monkeypatch.setattr(main.taxa_navigation, "serialize_taxon", lambda _taxon: dict(payload))
    monkeypatch.setattr(
        main.descriptions,
        "build_taxon_description",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    captured = {"printed": False}
    monkeypatch.setattr(main.traceback, "print_exc", lambda: captured.__setitem__("printed", True))

    out = main.get_species_detail(1, location=None, unit_system=None)
    assert out["taxon_id"] == 1
    assert "description_profile" not in out
    assert captured["printed"] is True
