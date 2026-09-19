#!/bin/bash
set -euo pipefail
# Capture llama ffn_up / ffn_gate / ffn_geglu / ffn_down for layer 0
BIN=/codebase/repos/wreckhunter2000-1/llama.cpp/build-vulkan/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
export CESAROPS_DUMP_HEAD=1
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
OUT=/tmp/llama_gemma4_l0_ffn_mids.log

"$BIN/llama-debug" \
  -m "$MODEL" -p '3x3=' -n 1 --temp 0 -ngl 99 -fa off --no-warmup -sm none -mg 0 --verbose \
  --tensor-filter 'ffn_norm_1-0$' \
  --tensor-filter 'ffn_up-0$' \
  --tensor-filter 'ffn_gate-0$' \
  --tensor-filter 'ffn_geglu-0$' \
  --tensor-filter 'ffn_mlp-0$' \
  --tensor-filter 'ffn_moe_gate_up-0$' \
  --tensor-filter 'ffn_moe_geglu-0$' \
  --tensor-filter 'ffn_moe_down-0$' \
  --tensor-filter 'ffn_moe-0$' \
  2>&1 | tee "$OUT" | grep -E 'CESAROPS_HEAD|common_debug_cb_eval:' 

echo LOG=$OUT
