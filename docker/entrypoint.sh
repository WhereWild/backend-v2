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

# Runs tileserver-gl (installed globally in the image, see Dockerfile) as a
# background process inside this same container, serving from whatever
# data/gis/tiles/basemap.pmtiles this checkout currently has (built by
# scripts/gis/build_basemap_tiles.py, quarterly cron — see
# bash/run_basemap_tiles.sh). main.py's basemap proxy route
# (WW_TILESERVER_BASE, defaulting to http://localhost:8791) talks to it
# over loopback, same container, no networking setup needed in dev or prod.
# config.json's paths.root is "/data" — symlinked here at *runtime* (not
# build time) since data/gis/tiles only exists once /workspace is actually
# mounted/populated, not while building the image.
_start_tileserver() {
  mkdir -p /workspace/data/gis/tiles /workspace/logs
  ln -sfn /workspace/data/gis/tiles /data
  # tileserver-gl's raster rendering needs a real GL context — Xvfb gives it
  # a headless X display to render into (same setup as the official image's
  # own docker-entrypoint.sh: https://github.com/maptiler/tileserver-gl/
  # blob/master/docker-entrypoint.sh).
  rm -f /tmp/.X99-lock
  export DISPLAY=:99
  Xvfb "$DISPLAY" -nolisten unix >>/workspace/logs/tileserver-xvfb.log 2>&1 &
  tileserver-gl --config /workspace/docker/tileserver/config.json --port 8791 \
    >>/workspace/logs/tileserver.log 2>&1 &
}

if [[ "$MODE" == "api" ]]; then
  export WHEREWILD_DATA_ROOT="/workspace/data"
  _start_tileserver
  exec uv run --env-file /workspace/.env uvicorn main:app \
    --host 0.0.0.0 --port 8000 --log-level info
fi

exec "$@"
