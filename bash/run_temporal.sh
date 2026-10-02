#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/temporal.log" 2>&1

# Prevent concurrent runs — if already running, exit immediately.
LOCK_FILE="$LOG_DIR/temporal.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [run_temporal] already running, skipping"
    exit 0
fi

# Throttle forecast pool to 1 worker while rebuild pipeline is active.
FORECAST_WORKERS=8
SYNC_STATE="$REPO_DIR/data/sync_state.json"
if [[ -f "$SYNC_STATE" ]] && command -v jq &>/dev/null; then
    pipeline_status=$(jq -r '.pipeline.status // empty' "$SYNC_STATE" 2>/dev/null || true)
    pipeline_stage=$(jq -r '.pipeline.stage // empty' "$SYNC_STATE" 2>/dev/null || true)
    # Skip entirely during process_tree*: an ERA5-advance tick peaks ~45GB,
    # which on top of process_tree's ~16GB gets OOM-killed every tick — and a
    # kill mid-save_raster_state can truncate a window's .sums.npz.
    if [[ "$pipeline_status" == "in_progress" && "$pipeline_stage" == process_tree* ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [run_temporal] rebuild in $pipeline_stage, skipping"
        exit 0
    fi
    if [[ "$pipeline_status" == "in_progress" ]]; then
        FORECAST_WORKERS=1
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [run_temporal] rebuild in progress → forecast workers=1"
    fi
fi

docker compose -f "$REPO_DIR/docker-compose.yml" up -d gdal
docker compose -f "$REPO_DIR/docker-compose.yml" exec -T --user ubuntu \
    -e PYTHONUNBUFFERED=1 \
    -e WW_FORECAST_WORKERS="$FORECAST_WORKERS" \
    gdal \
    bash -lc '. /etc/wherewild_aliases.sh; cd /workspace && uv run --env-file /workspace/.env python -u -m scripts.build_temporal'
