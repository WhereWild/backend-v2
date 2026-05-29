#!/usr/bin/env bash
# Run locally (or on the server) to build the frontend web export before
# deploying. Outputs to front-end/dist/, which the Docker image copies in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../../front-end"

cd "$FRONTEND_DIR"

export APP_BACKEND_URL="https://api.wherewild.net"
export APP_STADIA_MAPS_API_KEY="${APP_STADIA_MAPS_API_KEY:-}"  # set in env before running

npm ci
npm run export:web

echo "Frontend built → $FRONTEND_DIR/dist"
echo "Now rebuild the frontend Docker image:"
echo "  docker compose -f wherewild-v2/docker-compose.prod.yml build frontend"
