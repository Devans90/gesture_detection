#!/usr/bin/env bash
# deploy/install_service.sh
# Copy and enable the systemd service on the Pi.
# Run as root:  sudo bash deploy/install_service.sh

set -euo pipefail

SERVICE_FILE="$(dirname "$0")/gesture.service"
DEST="/etc/systemd/system/gesture.service"

cp "$SERVICE_FILE" "$DEST"
systemctl daemon-reload
systemctl enable gesture.service
systemctl start gesture.service

echo "Service installed and started."
echo "Check status with:  systemctl status gesture.service"
echo "View logs with:     journalctl -u gesture.service -f"
