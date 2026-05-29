"""Storage helpers for local vs B2-backed parquet reads."""

from __future__ import annotations

import configparser
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.fs as pafs
import pyarrow.parquet as pq

_DEFAULT_B2_BUCKET = "wherewild-data"
_DEFAULT_B2_PREFIX = "data"
_DEFAULT_B2_READER_REMOTE = "wherewild-localdev-reader"
_DEFAULT_B2_WRITER_REMOTE = "wherewild-localdev-writer"
_DEFAULT_B2_ENDPOINT = "https://s3.us-west-004.backblazeb2.com"
_DEFAULT_RCLONE_CONFIG = "/workspace/docker/rclone.conf"


@dataclass(frozen=True)
class ParquetStorage:
    mode: str
    data_root: Path
    project_root: Path
    filesystem: pafs.FileSystem | None
    bucket: str | None
    prefix: str | None
    endpoint: str | None = None
    key_id: str | None = None
    app_key: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.filesystem is not None

    def resolve(self, path: Path) -> str:
        """Resolve a local-style Path to the underlying filesystem path."""
        if self.filesystem is None:
            return str(path)
        rel = _relative_to_root(path, self.data_root, self.project_root)
        if rel is not None and rel.parts and rel.parts[0] == ".b2-mount":
            rel = Path(*rel.parts[1:])
        if rel is None:
            mount_root = Path(os.environ.get("WW_B2_MOUNT", "/workspace/.b2-mount")).expanduser().resolve()
            try:
                rel = path.expanduser().resolve().relative_to(mount_root)
            except ValueError:
                rel = None
        if rel is None:
            raise ValueError(f"Path {path} is not under {self.data_root} or {self.project_root}")
        key = rel.as_posix().lstrip("/")
        prefix = (self.prefix or "").strip("/")
        if prefix:
            key = f"{prefix}/{key}"
        bucket = self.bucket or ""
        if not bucket:
            raise ValueError("Missing B2 bucket for remote parquet resolution.")
        return f"{bucket}/{key}"

    def exists(self, path: Path) -> bool:
        if self.filesystem is None:
            return path.exists()
        info = self.filesystem.get_file_info(self.resolve(path))
        return info.type != pafs.FileType.NotFound

    def read_table(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list[tuple[str, str, Any]] | None = None,
    ):
        resolved = self.resolve(path)
        kwargs: dict[str, Any] = {}
        if columns is not None:
            kwargs["columns"] = columns
        if filters is not None:
            kwargs["filters"] = filters
        if self.filesystem is None:
            return pq.read_table(resolved, **kwargs)
        return pq.read_table(resolved, filesystem=self.filesystem, **kwargs)

    def read_metadata(self, path: Path):
        resolved = self.resolve(path)
        if self.filesystem is None:
            return pq.read_metadata(resolved)
        return pq.read_metadata(resolved, filesystem=self.filesystem)

    def read_schema(self, path: Path):
        resolved = self.resolve(path)
        if self.filesystem is None:
            return pq.read_schema(resolved)
        return pq.read_schema(resolved, filesystem=self.filesystem)

    def parquet_file(self, path: Path):
        resolved = self.resolve(path)
        if self.filesystem is None:
            return pq.ParquetFile(resolved)
        return pq.ParquetFile(resolved, filesystem=self.filesystem)

    def open_input_file(self, path: Path):
        if self.filesystem is None:
            return path.open("rb")
        if path.exists():
            return path.open("rb")
        if _should_cache_blob(path):
            cached = _ensure_cached(self, path)
            if cached is not None:
                return cached.open("rb")
        return self.filesystem.open_input_file(self.resolve(path))

    def vsis3_path(self, path: Path) -> str:
        """Return GDAL /vsis3/ path for a path resolved against the remote."""
        if self.filesystem is None:
            raise ValueError("vsis3_path is only available for remote storage.")
        return f"/vsis3/{self.resolve(path)}"

    def gdal_env(self) -> dict[str, str]:
        """Return GDAL environment values for S3-compatible B2 reads."""
        if self.mode != "b2":
            return {}
        key_id = (self.key_id or "").strip()
        app_key = (self.app_key or "").strip()
        endpoint = (self.endpoint or "").strip()
        if not key_id or not app_key or not endpoint:
            return {}
        endpoint_host, endpoint_scheme = _normalize_endpoint(endpoint)
        env = {
            "AWS_ACCESS_KEY_ID": key_id,
            "AWS_SECRET_ACCESS_KEY": app_key,
            "AWS_S3_ENDPOINT": endpoint_host,
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "NO" if endpoint_scheme == "http" else "YES",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        }
        region = _b2_region_from_host(endpoint_host)
        if region:
            env["AWS_REGION"] = region
        return env


class ParquetStorageProxy:
    """Dynamic proxy that resolves storage from current environment."""

    def __init__(self, data_root: Path, project_root: Path) -> None:
        self._data_root = Path(data_root)
        self._project_root = Path(project_root)

    def _storage(self) -> ParquetStorage:
        return get_parquet_storage(self._data_root, self._project_root)

    def current(self) -> ParquetStorage:
        return self._storage()

    def __getattr__(self, name: str):
        return getattr(self._storage(), name)


def get_parquet_storage(data_root: Path, project_root: Path) -> ParquetStorage:
    """Return a cached ParquetStorage for the current environment."""
    return get_parquet_storage_with_mode(data_root, project_root, _storage_mode())


def get_parquet_storage_with_mode(data_root: Path, project_root: Path, mode: str) -> ParquetStorage:
    """Return a cached ParquetStorage for an explicit mode."""
    normalized_mode = str(mode or "").strip().lower() or "local"
    return _get_parquet_storage(
        str(data_root.resolve()),
        str(project_root.resolve()),
        normalized_mode,
    )


def _storage_mode() -> str:
    mode = os.environ.get("WHEREWILD_PARQUET_STORAGE") or os.environ.get("WHEREWILD_STORAGE", "local")
    mode = str(mode).strip().lower()
    return mode if mode else "local"


@lru_cache(maxsize=4)
def _get_parquet_storage(data_root: str, project_root: str, mode: str) -> ParquetStorage:
    data_root_path = Path(data_root)
    project_root_path = Path(project_root)
    if mode == "local":
        return ParquetStorage(
            mode="local",
            data_root=data_root_path,
            project_root=project_root_path,
            filesystem=None,
            bucket=None,
            prefix=None,
            endpoint=None,
            key_id=None,
            app_key=None,
        )

    if mode != "b2":
        raise ValueError(f"Unknown parquet storage mode '{mode}' (expected 'local' or 'b2').")

    endpoint, key_id, app_key = _resolve_b2_credentials()
    bucket = os.environ.get("WW_B2_BUCKET", _DEFAULT_B2_BUCKET)
    prefix = os.environ.get("WW_B2_PREFIX", _DEFAULT_B2_PREFIX)

    if not endpoint or not key_id or not app_key:
        raise RuntimeError(
            "B2 parquet storage is enabled but WW_B2_S3_ENDPOINT / WW_B2_KEY_ID / "
            "WW_B2_APP_KEY are not fully set."
        )

    endpoint_override, scheme = _normalize_endpoint(endpoint)
    filesystem = pafs.S3FileSystem(
        access_key=key_id,
        secret_key=app_key,
        endpoint_override=endpoint_override,
        scheme=scheme,
    )

    return ParquetStorage(
        mode="b2",
        data_root=data_root_path,
        project_root=project_root_path,
        filesystem=filesystem,
        bucket=bucket,
        prefix=prefix,
        endpoint=endpoint,
        key_id=key_id,
        app_key=app_key,
    )


def _relative_to_root(path: Path, data_root: Path, project_root: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    for root in (data_root, project_root):
        try:
            return resolved.relative_to(root)
        except ValueError:
            continue
    return None


def _should_cache_blob(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return True
    extra = os.environ.get("WHEREWILD_CACHE_EXTS", "")
    if extra:
        extensions = {ext.strip().lower() for ext in extra.split(",") if ext.strip()}
        if suffix in extensions:
            return True
    return False


def _cache_root() -> Path:
    root = os.environ.get("WHEREWILD_CACHE_DIR", "/workspace/.cache/wherewild")
    return Path(root)


def _ensure_cached(storage: ParquetStorage, path: Path) -> Path | None:
    try:
        rel = _relative_to_root(path, storage.data_root, storage.project_root)
        if rel is None:
            return None
        cache_path = _cache_root() / rel
        if cache_path.exists():
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=cache_path.name + ".", dir=str(cache_path.parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with storage.filesystem.open_input_file(storage.resolve(path)) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return cache_path
    except Exception:
        return None


def _resolve_b2_credentials() -> tuple[str | None, str | None, str | None]:
    endpoint = (os.environ.get("WW_B2_S3_ENDPOINT") or "").strip()
    key_id = (os.environ.get("WW_B2_KEY_ID") or "").strip()
    app_key = (os.environ.get("WW_B2_APP_KEY") or "").strip()
    remotes = (
        os.environ.get("WW_B2_READER_REMOTE", _DEFAULT_B2_READER_REMOTE),
        os.environ.get("WW_B2_WRITER_REMOTE", _DEFAULT_B2_WRITER_REMOTE),
    )
    for remote in remotes:
        if endpoint and key_id and app_key:
            break
        from_config = _load_rclone_remote(remote)
        if from_config is None:
            continue
        cfg_endpoint, cfg_key_id, cfg_app_key = from_config
        if not endpoint and cfg_endpoint:
            endpoint = cfg_endpoint
        if not key_id and cfg_key_id:
            key_id = cfg_key_id
        if not app_key and cfg_app_key:
            app_key = cfg_app_key
    if not endpoint:
        endpoint = os.environ.get("WW_B2_S3_ENDPOINT_DEFAULT", _DEFAULT_B2_ENDPOINT).strip()
    return endpoint or None, key_id or None, app_key or None


def _load_rclone_remote(remote: str | None) -> tuple[str, str, str] | None:
    remote_name = str(remote or "").strip()
    if not remote_name:
        return None
    config_path = Path(os.environ.get("RCLONE_CONFIG", _DEFAULT_RCLONE_CONFIG))
    if not config_path.exists():
        return None
    parser = configparser.RawConfigParser()
    try:
        parser.read(config_path)
    except Exception:
        return None
    if not parser.has_section(remote_name):
        return None
    endpoint = parser.get(remote_name, "endpoint", fallback="").strip()
    key_id = parser.get(remote_name, "account", fallback="").strip()
    app_key = parser.get(remote_name, "key", fallback="").strip()
    if not key_id or not app_key:
        return None
    return endpoint, key_id, app_key


def _normalize_endpoint(endpoint: str) -> tuple[str, str]:
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return "", "https"
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    parsed = urlparse(endpoint)
    host = (parsed.netloc or parsed.path).strip("/")
    scheme = (parsed.scheme or "https").lower()
    if scheme not in {"http", "https"}:
        scheme = "https"
    return host, scheme


def _b2_region_from_host(host: str) -> str | None:
    parts = [part for part in host.split(".") if part]
    if len(parts) >= 2 and parts[0] == "s3":
        return parts[1]
    return None
