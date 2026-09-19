#!/bin/bash
set -euo pipefail
BIN=/codebase/repos/wreckhunter2000-1/llama.cpp/build-vulkan/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
export CESAROPS_DUMP_HEAD=1
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
OUT=/tmp/llama_gemma4_l0_head16.log

FILTERS=(
  --tensor-filter 'inp_scaled$'
  --tensor-filter 'Qcur-0$'
  --tensor-filter 'Qcur_normed-0$'
  --tensor-filter 'Kcur-0$'
  --tensor-filter 'Kcur_normed-0$'
  --tensor-filter 'Vcur-0$'
  --tensor-filter 'Vcur_normed-0$'
  --tensor-filter 'attn_post_norm-0$'
  --tensor-filter 'attn_out-0$'
  --tensor-filter 'ffn_norm_1-0$'
  --tensor-filter 'ffn_mlp-0$'
  --tensor-filter 'ffn_norm_2-0$'
  --tensor-filter 'ffn_moe_logits-0$'
  --tensor-filter 'ffn_moe-0$'
  --tensor-filter 'ffn_moe_combined-0$'
  --tensor-filter 'ffn_post_norm-0$'
  --tensor-filter 'out_scaled-0$'
  --tensor-filter 'l_out-0$'
)

echo "OUT=$OUT"
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
  2>&1 | tee "$OUT" | grep -E 'CESAROPS_HEAD|common_debug_cb_eval:|tokens:|token |sampled|prompt' | head -120

echo "==== compare ===="
python3 /codebase/repos/wreckhunter2000-1/cesarops-inference/scripts/_compare_l0_head16.py
