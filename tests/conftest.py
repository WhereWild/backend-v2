# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from util.storage import _get_parquet_storage


@pytest.fixture(autouse=True)
def force_local_storage(monkeypatch):
    """Force local parquet storage for all tests.

    Without this, WHEREWILD_PARQUET_STORAGE=b2 in the container environment
    causes _storage.resolve() to fail for tmp_path files (they're not under
    /workspace/data), turning every test that writes to tmp_path into a failure.
    """
    monkeypatch.setenv("WHEREWILD_PARQUET_STORAGE", "local")
    _get_parquet_storage.cache_clear()
    yield
    _get_parquet_storage.cache_clear()
