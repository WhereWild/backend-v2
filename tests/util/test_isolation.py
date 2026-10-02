# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from pathlib import Path

import pytest

from util.isolation import run_isolated


def test_run_isolated_runs_target_in_child(tmp_path):
    marker = tmp_path / "ran"
    run_isolated(Path.touch, marker)
    assert marker.exists()


def test_run_isolated_raises_on_child_failure():
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_isolated(sys.exit, 3)
