#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/temporal.log" 2>&1

docker compose -f "$REPO_DIR/docker-compose.yml" up -d gdal
docker compose -f "$REPO_DIR/docker-compose.yml" exec -T --user ubuntu -e PYTHONUNBUFFERED=1 gdal \
    bash -lc '. /etc/wherewild_aliases.sh; cd /workspace && uv run --env-file /workspace/.env python -u -m scripts.build_temporal'
