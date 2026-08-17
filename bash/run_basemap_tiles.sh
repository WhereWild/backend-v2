#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Usage: run_basemap_tiles.sh
#
# Rebuilds the self-hosted OpenMapTiles-schema basemap (planet.osm.pbf →
# data/gis/tiles/basemap.pmtiles + styles + fonts, see
# scripts/gis/build_basemap_tiles.py) and pushes the result to gambaby.
#
# Runs quarterly, not on the taxonomy rebuild's monthly/GBIF-crawl cadence —
# OSM data doesn't need that freshness, and a full planet build is a multi-hour
# job (~94GB download, 4-7hrs on this box's specs) that shouldn't ride along
# with run_rebuild.sh's stages. build_basemap_tiles.py pushes its own result
# to gambaby as its final step (_push_basemap_files) — deliberately NOT
# `rebuild.py --stage push`, which is the taxonomy pipeline's own stage
# runner (alert channel included) and has nothing to do with this pipeline.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/basemap_tiles.log" 2>&1

# Multi-hour job — guard against a second cron firing (e.g. a missed run
# catching up) landing on top of one still in progress, same pattern as
# run_temporal.sh's lock.
LOCK_FILE="$LOG_DIR/basemap_tiles.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [run_basemap_tiles] already running, skipping"
    exit 0
fi

docker compose -f "$REPO_DIR/docker-compose.yml" up -d gdal
docker compose -f "$REPO_DIR/docker-compose.yml" exec -T --user ubuntu -e PYTHONUNBUFFERED=1 gdal \
    bash -lc ". /etc/wherewild_aliases.sh; cd /workspace && uv run --env-file /workspace/.env python -u -m scripts.gis.build_basemap_tiles --area planet"
