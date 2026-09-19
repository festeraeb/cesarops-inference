#!/bin/bash
set -euo pipefail
MODEL=/mnt/raid0/models/gemma-4-E4B-it-IQ4_XS.gguf
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
/data/cargo-target/release/run_gemma4 \
  --model "$MODEL" \
  --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 4 \
  --temperature 0.0
echo EXIT=$?
