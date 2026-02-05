#!/usr/bin/env bash
set -euo pipefail

template="/workspace/docker/rclone.conf.template"
target="/workspace/docker/rclone.conf"

if [[ -f "$template" && ! -f "$target" ]]; then
  cp "$template" "$target"
  echo "Created $target from template; fill in keys locally."
fi

MODE="${WHEREWILD_MODE:-dev}"

case "$MODE" in
  api)
    echo "Starting WhereWild API"
    export WHEREWILD_DATA_ROOT=/data
    exec uvicorn main:app --host 0.0.0.0 --port 8000
    ;;
  dev)
    echo "Entering dev mode"
    exec "$@"
    ;;
  *)
    echo "Unknown WHEREWILD_MODE=$MODE"
    exit 1
    ;;
esac
