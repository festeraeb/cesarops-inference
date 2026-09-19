#!/usr/bin/env bash
set -euo pipefail
PASS=cesarops
s() { echo "$PASS" | sudo -S "$@"; }

if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh -o /tmp/ts.sh
  s bash /tmp/ts.sh
fi
s tailscale up --hostname=cesarops4 --accept-routes --ssh || true
tailscale status 2>/dev/null | head -5 || echo "tailscale needs auth: sudo tailscale up"
