#!/usr/bin/env bash
# Run once on a fresh Ubuntu 24.04 Hetzner server as root.
set -euo pipefail

DEPLOY_DIR="/opt/wherewild"
BACKEND_REPO="https://github.com/YOUR_ORG/wherewild-v2.git"
FRONTEND_REPO="https://github.com/YOUR_ORG/front-end.git"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
apt-get update -qq
apt-get install -y ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------
mkdir -p "$DEPLOY_DIR"
git clone "$BACKEND_REPO"  "$DEPLOY_DIR/wherewild-v2"
git clone "$FRONTEND_REPO" "$DEPLOY_DIR/front-end"

# ---------------------------------------------------------------------------
# Secrets (fill these in before running, or copy files manually)
# ---------------------------------------------------------------------------
# cp /path/to/rclone.conf  $DEPLOY_DIR/wherewild-v2/docker/rclone.conf
# cp /path/to/.env         $DEPLOY_DIR/wherewild-v2/.env

echo ""
echo "=== MANUAL STEPS REQUIRED ==="
echo "1. Copy docker/rclone.conf to $DEPLOY_DIR/wherewild-v2/docker/rclone.conf"
echo "2. Copy .env to $DEPLOY_DIR/wherewild-v2/.env"
echo "3. Build the frontend (see deploy/build-frontend.sh)"
echo "4. Run: systemctl enable --now wherewild"
echo ""

# ---------------------------------------------------------------------------
# systemd service
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/wherewild.service <<EOF
[Unit]
Description=WhereWild
After=docker.service network-online.target
Requires=docker.service

[Service]
WorkingDirectory=$DEPLOY_DIR/wherewild-v2
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo "systemd service installed (not started yet)"
