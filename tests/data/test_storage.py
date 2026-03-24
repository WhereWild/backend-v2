"""Unit tests for util.storage."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from util import storage


class _FakeInfo:
    def __init__(self, file_type):
        self.type = file_type


class _CtxBytesIO(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _FakeFS:
    def __init__(self):
        self.requested = []

    def get_file_info(self, key):
        self.requested.append(("info", key))
        return _FakeInfo("found")

    def open_input_file(self, key):
        self.requested.append(("open", key))
        return _CtxBytesIO(b"remote-bytes")


def test_relative_to_root_and_resolve_errors(tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()
    inside_data = data_root / "x" / "a.parquet"
    inside_data.parent.mkdir(parents=True)
    inside_data.write_text("x", encoding="utf-8")
    inside_proj = project_root / "p.txt"
    inside_proj.write_text("y", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("z", encoding="utf-8")

    assert storage._relative_to_root(inside_data, data_root, project_root) == Path("x/a.parquet")
    assert storage._relative_to_root(inside_proj, data_root, project_root) == Path("p.txt")
    assert storage._relative_to_root(outside, data_root, project_root) is None

    local = storage.ParquetStorage("local", data_root, project_root, None, None, None)
    assert local.resolve(inside_data) == str(inside_data)

    remote = storage.ParquetStorage("b2", data_root, project_root, _FakeFS(), "bucket", "prefix")
    assert remote.resolve(inside_data) == "bucket/prefix/x/a.parquet"

    remote_missing_bucket = storage.ParquetStorage("b2", data_root, project_root, _FakeFS(), "", "prefix")
    with pytest.raises(ValueError):
        remote_missing_bucket.resolve(inside_data)

    with pytest.raises(ValueError):
        remote.resolve(outside)


def test_exists_readers_and_parquet_file(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()
    file_path = data_root / "f.parquet"
    file_path.write_text("x", encoding="utf-8")

    local = storage.ParquetStorage("local", data_root, project_root, None, None, None)
    assert local.exists(file_path)

    fs = _FakeFS()
    remote = storage.ParquetStorage("b2", data_root, project_root, fs, "bucket", "prefix")
    assert remote.exists(file_path)

    class _FakeType:
        NotFound = "notfound"

    monkeypatch.setattr(storage.pafs, "FileType", _FakeType)
    fs_missing = _FakeFS()
    fs_missing.get_file_info = lambda _key: _FakeInfo("notfound")
    remote_missing = storage.ParquetStorage("b2", data_root, project_root, fs_missing, "bucket", "prefix")
    assert not remote_missing.exists(file_path)

    calls = []

    def fake_read_table(path, **kwargs):
        calls.append(("table", path, kwargs))
        return "T"

    def fake_read_metadata(path, **kwargs):
        calls.append(("meta", path, kwargs))
        return "M"

    def fake_read_schema(path, **kwargs):
        calls.append(("schema", path, kwargs))
        return "S"

    class _PF:
        def __init__(self, path, **kwargs):
            calls.append(("pf", path, kwargs))

    monkeypatch.setattr(storage.pq, "read_table", fake_read_table)
    monkeypatch.setattr(storage.pq, "read_metadata", fake_read_metadata)
    monkeypatch.setattr(storage.pq, "read_schema", fake_read_schema)
    monkeypatch.setattr(storage.pq, "ParquetFile", _PF)

    assert local.read_table(file_path, columns=["a"], filters=[("x", "=", 1)]) == "T"
    assert local.read_metadata(file_path) == "M"
    assert local.read_schema(file_path) == "S"
    local.parquet_file(file_path)

    assert remote.read_table(file_path, columns=["a"]) == "T"
    assert remote.read_metadata(file_path) == "M"
    assert remote.read_schema(file_path) == "S"
    remote.parquet_file(file_path)

    assert any(item[0] == "table" and "filesystem" in item[2] for item in calls)
    assert any(item[0] == "pf" and "filesystem" in item[2] for item in calls)


def test_open_input_file_local_and_remote_paths(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()
    existing = data_root / "exists.pkl"
    existing.write_bytes(b"abc")

    local = storage.ParquetStorage("local", data_root, project_root, None, None, None)
    with local.open_input_file(existing) as handle:
        assert handle.read() == b"abc"

    fs = _FakeFS()
    remote = storage.ParquetStorage("b2", data_root, project_root, fs, "bucket", "prefix")

    # Prefer local existing file even in remote mode.
    with remote.open_input_file(existing) as handle:
        assert handle.read() == b"abc"

    # Cached blob path.
    missing_blob = data_root / "blob.pkl"
    cached = tmp_path / "cache.pkl"
    cached.write_bytes(b"cached")
    monkeypatch.setattr(storage, "_ensure_cached", lambda *_a, **_k: cached)
    with remote.open_input_file(missing_blob) as handle:
        assert handle.read() == b"cached"

    # Fallback to remote filesystem.
    monkeypatch.setattr(storage, "_ensure_cached", lambda *_a, **_k: None)
    missing_other = data_root / "blob.txt"
    with remote.open_input_file(missing_other) as handle:
        assert handle.read() == b"remote-bytes"


def test_storage_mode_and_get_parquet_storage(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()

    monkeypatch.delenv("WHEREWILD_PARQUET_STORAGE", raising=False)
    monkeypatch.setenv("WHEREWILD_STORAGE", "  ")
    assert storage._storage_mode() == "local"

    monkeypatch.setenv("WHEREWILD_PARQUET_STORAGE", "B2")
    assert storage._storage_mode() == "b2"

    called = {}

    def fake_get(data_root_s, project_root_s, mode):
        called["args"] = (data_root_s, project_root_s, mode)
        return "ST"

    monkeypatch.setattr(storage, "_get_parquet_storage", fake_get)
    assert storage.get_parquet_storage(data_root, project_root) == "ST"
    assert called["args"][2] == "b2"


def test_get_parquet_storage_modes_and_errors(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()

    local = storage._get_parquet_storage(str(data_root), str(project_root), "local")
    assert local.mode == "local" and local.filesystem is None

    with pytest.raises(ValueError):
        storage._get_parquet_storage(str(data_root), str(project_root), "weird")

    monkeypatch.delenv("WW_B2_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("WW_B2_KEY_ID", raising=False)
    monkeypatch.delenv("WW_B2_APP_KEY", raising=False)
    with pytest.raises(RuntimeError):
        storage._get_parquet_storage(str(data_root), str(project_root), "b2")

    storage._get_parquet_storage.cache_clear()
    monkeypatch.setenv("WW_B2_S3_ENDPOINT", "s3.example")
    monkeypatch.setenv("WW_B2_KEY_ID", "key")
    monkeypatch.setenv("WW_B2_APP_KEY", "secret")
    monkeypatch.setenv("WW_B2_BUCKET", "bucket-x")
    monkeypatch.setenv("WW_B2_PREFIX", "prefix-x")
    monkeypatch.setattr(storage.pafs, "S3FileSystem", lambda **_kwargs: _FakeFS())
    remote = storage._get_parquet_storage(str(data_root), str(project_root), "b2")
    assert remote.mode == "b2"
    assert remote.bucket == "bucket-x"
    assert remote.prefix == "prefix-x"
    assert remote.filesystem is not None


def test_blob_cache_helpers_and_ensure_cached(monkeypatch, tmp_path):
    assert storage._should_cache_blob(Path("x.pkl"))
    assert not storage._should_cache_blob(Path("x.txt"))
    monkeypatch.setenv("WHEREWILD_CACHE_EXTS", ".txt,.bin")
    assert storage._should_cache_blob(Path("x.txt"))

    monkeypatch.delenv("WHEREWILD_CACHE_DIR", raising=False)
    assert storage._cache_root() == Path("/workspace/.cache/wherewild")
    monkeypatch.setenv("WHEREWILD_CACHE_DIR", str(tmp_path / "cache"))
    assert storage._cache_root() == tmp_path / "cache"

    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    data_root.mkdir()
    project_root.mkdir()
    fs = _FakeFS()
    remote = storage.ParquetStorage("b2", data_root, project_root, fs, "bucket", "prefix")
    local = storage.ParquetStorage("local", data_root, project_root, None, None, None)
    assert remote.is_remote is True
    assert local.is_remote is False

    # path outside roots => None
    outside = tmp_path / "outside.pkl"
    outside.write_text("x", encoding="utf-8")
    assert storage._ensure_cached(remote, outside) is None

    target = data_root / "blob.pkl"
    monkeypatch.setenv("WHEREWILD_CACHE_DIR", str(tmp_path / "cache2"))
    cache_path = storage._cache_root() / Path("blob.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"existing")
    assert storage._ensure_cached(remote, target) == cache_path

    cache_path.unlink()
    out = storage._ensure_cached(remote, target)
    assert out is not None and out.exists() and out.read_bytes() == b"remote-bytes"

    # force exception path
    class _ExplodingFS(_FakeFS):
        def open_input_file(self, _key):
            raise RuntimeError("boom")

    exploding = storage.ParquetStorage("b2", data_root, project_root, _ExplodingFS(), "bucket", "prefix")
    target_fail = data_root / "blob-fail.pkl"
    assert storage._ensure_cached(exploding, target_fail) is None


def test_parquet_storage_proxy_forwards_calls(monkeypatch, tmp_path):
    class _DummyStorage:
        value = 123

    monkeypatch.setattr(storage, "get_parquet_storage", lambda *_a, **_k: _DummyStorage())
    proxy = storage.ParquetStorageProxy(tmp_path / "data", tmp_path / "project")
    assert proxy.value == 123
