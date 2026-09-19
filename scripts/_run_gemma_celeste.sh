#!/bin/bash
set -euo pipefail
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
ln -sfn "$MODEL" /codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf
ln -sfn "$MODEL" /mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS.gguf
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
LOGIT_AUDIT=1 /data/cargo-target/release/run_gemma4 \
  --model "$MODEL" \
  --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 8 \
  --temperature 0.0
echo GEMMA_EXIT=$?
