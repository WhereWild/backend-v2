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
  # shellcheck source=/dev/null
  . /workspace/docker/aliases.sh

  mount_point="${WW_B2_MOUNT:-/workspace/.b2-mount}"
  export WHEREWILD_DATA_ROOT="$mount_point"

  echo "[entrypoint] mounting B2 at $mount_point..."
  b2-mount || true

  echo "[entrypoint] waiting for mount..."
  _mounted=0
  for _i in $(seq 30); do
    if mountpoint -q "$mount_point" 2>/dev/null; then
      _mounted=1
      break
    fi
    sleep 1
  done

  if [[ "$_mounted" -eq 1 ]]; then
    echo "[entrypoint] B2 mounted"
  else
    echo "[entrypoint] WARNING: B2 not ready after 30s, starting anyway"
  fi

  exec uv run --env-file /workspace/.env uvicorn main:app \
    --host 0.0.0.0 --port 8000 --log-level info
fi

exec "$@"
