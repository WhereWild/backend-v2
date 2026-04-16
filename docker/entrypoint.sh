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
    if [[ -f /etc/wherewild_aliases.sh ]]; then
      # Reuse the helper logic for data-root selection and startup log rotation.
      . /etc/wherewild_aliases.sh
      ww_prepare_api_runtime >/dev/null

      # If we're serving from local data, kick off a background pull to populate it.
      # (When serving from the mounted remote, pulling everything locally is redundant.)
      local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
      # Normalize paths (strip any trailing slash) before comparing to avoid brittle string equality.
      normalized_data_root="${WHEREWILD_DATA_ROOT%/}"
      normalized_local_root="${local_root%/}"
      if [[ "$normalized_data_root" == "$normalized_local_root" ]]; then
        echo "Starting b2-pull-all to populate $local_root"
        # b2-pull-all manages its own rclone log at /workspace/logs/rclone/clone.log
        b2-pull-all
      fi

      ww_prepare_api_log
    else
      # Fallback if aliases weren't baked into the image.
      # Honor WHEREWILD_LOCAL_DATA_ROOT if set, otherwise default to /workspace/data.
      local_root="${WHEREWILD_LOCAL_DATA_ROOT:-/workspace/data}"
      export WHEREWILD_DATA_ROOT="$local_root"
      mkdir -p /workspace/logs
      if [[ -f /workspace/logs/api.log ]]; then
        mv -f /workspace/logs/api.log /workspace/logs/api.previous.log
      fi
    fi

    exec > >(tee /workspace/logs/api.log) 2>&1
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
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
