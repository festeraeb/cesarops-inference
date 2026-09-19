#!/bin/bash
set -euo pipefail
# Layer-0 MOE_SLICE dump on cesarops2 P100 #0 (idle 3xP100 node)
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
set -euo pipefail
export GEMMA4_MODEL=/mnt/t440/models/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
export GEMMA4_GPU=0
BIN=/mnt/t440/data/cargo-target/release/run_gemma4
cd /tmp
echo "=== RUN A: default norms (raw weight) ==="
MOE_SLICE=1 MOE_SLICE_LAYER=0 MOE_SLICE_POS=0 MOE_SLICE_DIMS=16 \
  LOGIT_AUDIT=1 \
  "$BIN" --model "$GEMMA4_MODEL" --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 4 --temperature 0.0 \
  2>&1 | tee /tmp/gemma_c2_slice_A.log | grep -E 'MOE_SLICE|LOGIT_AUDIT|Output token|Loaded in|Generated|panic|Error' || true
echo EXIT_A=$?
EOF
