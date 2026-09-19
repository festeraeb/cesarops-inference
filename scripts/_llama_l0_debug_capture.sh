#!/bin/bash
set -euo pipefail
BIN=/codebase/repos/wreckhunter2000-1/llama.cpp/build-vulkan/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
OUT=/tmp/llama_gemma4_l0_stages.log

# Layer-0 stage names from gemma4.cpp cb(...)
# Anchored as ^PATTERN by llama-debug.
FILTERS=(
  --tensor-filter 'inp_scaled'
  --tensor-filter 'Qcur-0'
  --tensor-filter 'Qcur_normed-0'
  --tensor-filter 'Kcur-0'
  --tensor-filter 'Kcur_normed-0'
  --tensor-filter 'Vcur-0'
  --tensor-filter 'Vcur_normed-0'
  --tensor-filter 'attn_post_norm-0'
  --tensor-filter 'attn_out-0'
  --tensor-filter 'ffn_norm_1-0'
  --tensor-filter 'ffn_mlp-0'
  --tensor-filter 'ffn_norm_2-0'
  --tensor-filter 'ffn_moe_logits-0'
  --tensor-filter 'ffn_moe-0'
  --tensor-filter 'ffn_moe_combined-0'
  --tensor-filter 'ffn_post_norm'
  --tensor-filter 'out_scaled-0'
  --tensor-filter 'l_out-0'
)

echo "OUT=$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv
# -fa off, greedy, 1 token; single GPU
"$BIN/llama-debug" \
  -m "$MODEL" \
  -p '3x3=' \
  -n 1 \
  --temp 0 \
  -ngl 99 \
  -fa off \
  --no-warmup \
  -sm none \
  -mg 0 \
  --verbose \
  "${FILTERS[@]}" \
  2>&1 | tee "$OUT" | tail -n 100

echo "==== summary ===="
grep -E 'prompt|tokens|token|Qcur|attn_out|ffn_moe|l_out|inp_scaled|sampling|predict' "$OUT" | head -80
echo "LOG=$OUT lines=$(wc -l < "$OUT")"
