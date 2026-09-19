#!/bin/bash
set -euo pipefail
LOG=/tmp/llama_gemma4_l0_stages.log
echo "=== cb names ==="
grep 'common_debug_cb_eval:' "$LOG" | sed 's/.*common_debug_cb_eval: *//' | awk -F= '{print $1}' | sed 's/ *$//' | sort -u
echo "=== generated / tokens ==="
grep -E 'tokens in batch|eval:|decode|sampled|n_tokens|prompt tokens|token ids|tokens:' "$LOG" | head -30
# also look near end
tail -80 "$LOG"
