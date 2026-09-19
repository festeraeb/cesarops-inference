#!/usr/bin/env bash
# Run cesarops-inference smoke test on the cesarops2 GTX 1070 (Pascal sm_61).
# This is the cross-card regression gate — every shader/pipeline change
# should be tested here before being committed.
#
# Usage: scripts/regression_test_1070.sh
# Prereqs:
#   - tailscale ssh access to cesarops@cesarops2 must be working
#   - /home/cesarops/cesarops-engine/cesarops-inference exists on cesarops2
#     (run scripts/push_engine_to_1070.sh to refresh)
#   - /mnt/storage/models/qwen2.5-coder-1.5b-instruct-q6_k.gguf must be present
#
# Exit 0 = PASS, non-zero = FAIL.

set -u

REMOTE="cesarops@cesarops2"
BIN="/home/cesarops/cesarops-engine/cesarops-inference"
MODEL="/mnt/storage/models/qwen2.5-coder-1.5b-instruct-q6_k.gguf"

t0=$(date +%s)
out=$(timeout 300 tailscale ssh "$REMOTE" \
    "timeout 240 $BIN generate --model $MODEL --prompt 'What is 2+2?' --max-tokens 12 2>&1 | tail -8" \
    2>&1)
rc=$?
t1=$(date +%s)
elapsed=$((t1 - t0))

if [[ $rc -ne 0 ]]; then
    echo "FAIL: ssh/cesarops-inference exited rc=$rc (elapsed=${elapsed}s)"
    echo "$out"
    exit 1
fi

# Look for the "Generated N tokens in X.XXs" line that the engine emits
if ! grep -q "Generated.*tokens" <<<"$out"; then
    echo "FAIL: engine never reported generation completion (elapsed=${elapsed}s)"
    echo "$out"
    exit 1
fi

# Engine produced output. Print summary line.
gen_line=$(grep "Generated.*tokens" <<<"$out" | head -1)
echo "PASS: cesarops2 GTX 1070 — ${elapsed}s wall"
echo "      $gen_line"
exit 0
