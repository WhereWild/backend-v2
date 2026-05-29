#!/usr/bin/env bash
set -euo pipefail

template="/workspace/docker/rclone.conf.template"
target="/workspace/docker/rclone.conf"

if [[ -f "$template" && ! -f "$target" ]]; then
  cp "$template" "$target"
  echo "Created $target from template; fill in keys locally."
fi

MODE="${WHEREWILD_MODE:-dev}"

if [[ "$MODE" == "api" ]]; then
  export WHEREWILD_DATA_ROOT="/workspace/data"
  exec uv run --env-file /workspace/.env uvicorn main:app \
    --host 0.0.0.0 --port 8000 --log-level info
fi

exec "$@"
