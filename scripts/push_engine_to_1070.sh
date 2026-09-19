#!/usr/bin/env bash
# Push the just-built cesarops-inference binary from T440 to cesarops2 (1070).
# Uses passwordless SSH (key was deployed to cesarops@10.0.0.129).
# Idempotent — safe to re-run.
set -eu

REMOTE="cesarops@10.0.0.129"
SRC_BIN="/home/cesarops/wreckhunter2000-1/cesarops-inference/target/release/cesarops-inference"
DST_DIR="/home/cesarops/cesarops-engine"
DST_BIN="$DST_DIR/cesarops-inference"

[[ -x "$SRC_BIN" ]] || { echo "FAIL: $SRC_BIN not built. Run: cargo build --release"; exit 1; }

# Make sure remote dir exists
ssh -o BatchMode=yes "$REMOTE" "mkdir -p $DST_DIR" || { echo "FAIL: ssh prep failed"; exit 1; }

# Ship the binary
scp -o BatchMode=yes -q "$SRC_BIN" "$REMOTE:$DST_BIN"

# Verify
ssh -o BatchMode=yes "$REMOTE" "chmod +x $DST_BIN && ls -lh $DST_BIN"
echo "PUSH OK"
