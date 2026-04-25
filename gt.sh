#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$script_dir"

needs_rebuild() {
    local container_start
    container_start=$(docker inspect wherewild-v2-gdal-1 --format '{{.State.StartedAt}}' 2>/dev/null) || return 0
    local container_epoch
    container_epoch=$(date -d "$container_start" +%s 2>/dev/null) || return 0
    local newest
    newest=$(find "$project_dir" \( -name "Dockerfile" -o -name "docker-compose.yml" -o -name "*.toml" -o -name "*.lock" -o -path "*/docker/*.sh" \) \
        | xargs stat -c %Y 2>/dev/null | sort -n | tail -1)
    [[ -z "$newest" ]] && return 1
    [[ "$newest" -gt "$container_epoch" ]]
}

if ! docker compose --project-directory "$project_dir" ps --status running gdal 2>/dev/null | grep -q gdal || needs_rebuild; then
    docker compose --project-directory "$project_dir" up -d --build gdal
fi
docker compose --project-directory "$project_dir" exec -it gdal bash -lc ". /workspace/docker/aliases.sh 2>/dev/null || true; exec /bin/bash"
