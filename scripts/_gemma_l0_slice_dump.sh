#!/bin/bash
# Layer-0 stage dump for Gemma-4 MoE (checklist spine).
set -euo pipefail
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
BIN=/data/cargo-target/release/run_gemma4
OUT=/tmp/gemma4_l0_slice_$(date +%Y%m%d_%H%M%S).log

export MOE_SLICE=1
export MOE_SLICE_LAYER=0
export MOE_SLICE_ALL=1
export MOE_SLICE_DIMS=16
export RUST_LOG=warn
export GEMMA4_GPU=0

echo "OUT=$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv
"$BIN" \
  --model "$MODEL" \
  --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 1 \
  --temperature 0.0 \
  2>&1 | tee "$OUT" | grep -E 'MOE_SLICE|token|LOGIT|next|argmax|prompt|error|panic|EXIT' || true

echo "==== SUMMARY (layer0 pos last=4 or pos0) ===="
grep 'MOE_SLICE layer=0' "$OUT" | awk '
  /pos=0 / { if (!seen0[$3]++) print }
  /pos=4 / { if (!seen4[$3]++) print }
' || grep 'MOE_SLICE layer=0 pos=0' "$OUT" | head -80

echo "FULL_LOG=$OUT"
