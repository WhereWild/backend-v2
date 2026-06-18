#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

template="/workspace/docker/rclone.conf.template"
target="/workspace/docker/rclone.conf"

if [[ -f "$template" && ! -f "$target" ]]; then
  cp "$template" "$target"
  echo "Created $target from template; fill in keys locally."
fi

_venv="${UV_PROJECT_ENVIRONMENT:-/opt/venvs/venv}"
uv sync --frozen --quiet
chmod -R a+rx "${UV_PYTHON_INSTALL_DIR:-/opt/uv-python}" 2>/dev/null || true
chmod -R a+rwx "$_venv" 2>/dev/null || true

MODE="${WHEREWILD_MODE:-dev}"

if [[ "$MODE" == "api" ]]; then
  export WHEREWILD_DATA_ROOT="/workspace/data"
  exec uv run --env-file /workspace/.env uvicorn main:app \
    --host 0.0.0.0 --port 8000 --log-level info
fi

exec "$@"
