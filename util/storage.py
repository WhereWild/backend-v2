"""Storage helpers for local vs B2-backed parquet reads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

import pyarrow.fs as pafs
import pyarrow.parquet as pq


_DEFAULT_B2_BUCKET = "wherewild-data"
_DEFAULT_B2_PREFIX = "data"


@dataclass(frozen=True)
class ParquetStorage:
    mode: str
    data_root: Path
    project_root: Path
    filesystem: Optional[pafs.FileSystem]
    bucket: Optional[str]
    prefix: Optional[str]

    @property
    def is_remote(self) -> bool:
        return self.filesystem is not None

    def resolve(self, path: Path) -> str:
        """Resolve a local-style Path to the underlying filesystem path."""
        if self.filesystem is None:
            return str(path)
        rel = _relative_to_root(path, self.data_root, self.project_root)
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
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple[str, str, Any]]] = None,
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
        # Prefer local path when available (e.g., rclone mount) for non-parquet blobs.
        if path.exists():
            return path.open("rb")
        if _should_cache_blob(path):
            cached = _ensure_cached(self, path)
            if cached is not None:
                return cached.open("rb")
        return self.filesystem.open_input_file(self.resolve(path))


class ParquetStorageProxy:
    """Dynamic proxy that resolves storage from current environment."""

    def __init__(self, data_root: Path, project_root: Path) -> None:
        self._data_root = Path(data_root)
        self._project_root = Path(project_root)

    def _storage(self) -> ParquetStorage:
        return get_parquet_storage(self._data_root, self._project_root)

    def __getattr__(self, name: str):
        return getattr(self._storage(), name)


def get_parquet_storage(data_root: Path, project_root: Path) -> ParquetStorage:
    """Return a cached ParquetStorage for the current environment."""
    return _get_parquet_storage(
        str(data_root.resolve()),
        str(project_root.resolve()),
        _storage_mode(),
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
        )

    if mode != "b2":
        raise ValueError(f"Unknown parquet storage mode '{mode}' (expected 'local' or 'b2').")

    endpoint = os.environ.get("WW_B2_S3_ENDPOINT")
    key_id = os.environ.get("WW_B2_KEY_ID")
    app_key = os.environ.get("WW_B2_APP_KEY")
    bucket = os.environ.get("WW_B2_BUCKET", _DEFAULT_B2_BUCKET)
    prefix = os.environ.get("WW_B2_PREFIX", _DEFAULT_B2_PREFIX)

    if not endpoint or not key_id or not app_key:
        raise RuntimeError(
            "B2 parquet storage is enabled but WW_B2_S3_ENDPOINT / WW_B2_KEY_ID / "
            "WW_B2_APP_KEY are not fully set."
        )

    filesystem = pafs.S3FileSystem(
        access_key=key_id,
        secret_key=app_key,
        endpoint_override=endpoint,
        scheme="https",
    )

    return ParquetStorage(
        mode="b2",
        data_root=data_root_path,
        project_root=project_root_path,
        filesystem=filesystem,
        bucket=bucket,
        prefix=prefix,
    )


def _relative_to_root(path: Path, data_root: Path, project_root: Path) -> Optional[Path]:
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


def _ensure_cached(storage: ParquetStorage, path: Path) -> Optional[Path]:
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
