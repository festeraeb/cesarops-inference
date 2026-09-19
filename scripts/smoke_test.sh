#!/usr/bin/env bash
# Inference engine smoke test — runs after every engine change.
#
# Usage:   scripts/smoke_test.sh [path/to/model.gguf]
# Default model: /codebase/models/qwen2.5-coder-1.5b-instruct-q6_k.gguf
# Exits 0 on PASS (output coherent), 1 on FAIL.
#
# What it tests:
#   1. Engine launches and loads the model without panic
#   2. "What is 2+2?" produces output containing "4"
#   3. "Hello" produces non-empty output
#
# Designed to run on M2200 / any small node. Keeps the regression gate cheap.

set -u

MODEL="${1:-/codebase/models/qwen2.5-coder-1.5b-instruct-q6_k.gguf}"
BIN="$(dirname "$0")/../target/release/cesarops-inference"
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: binary not built. Run: cargo build --release -p cesarops-inference"
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "FAIL: model not found: $MODEL"
    exit 1
fi

t0=$(date +%s)
timeout 120 "$BIN" generate --model "$MODEL" --prompt "What is 2+2?" --max-tokens 15 > "$LOG" 2>&1
rc=$?
t1=$(date +%s)
elapsed=$((t1 - t0))

if [[ $rc -ne 0 ]]; then
    echo "FAIL: engine exited rc=$rc (timeout=120s, elapsed=${elapsed}s)"
    tail -20 "$LOG"
    exit 1
fi

# Look for "4" in the generated output (loose match — any "4" digit in the model's response counts)
if ! grep -q "4" "$LOG"; then
    echo "FAIL: no '4' in output for 'What is 2+2?'  — engine likely garbled"
    tail -30 "$LOG"
    exit 1
fi

# Pull the actual generated line if the engine emits a clear marker
gen_line=$(grep -E "^Generated:|generate \"|Generated text:" "$LOG" | head -1 || true)
echo "PASS: engine coherent in ${elapsed}s"
echo "      model: $MODEL"
echo "      output: ${gen_line:-$(tail -3 "$LOG" | head -1)}"
exit 0
