#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"

project_dir="${root}/wherewild"

docker compose --project-directory "$project_dir" up -d gdal
docker compose --project-directory "$project_dir" exec -it gdal bash -lc ". /etc/wherewild_aliases.sh 2>/dev/null || true; b2-mount; exec /bin/bash"
