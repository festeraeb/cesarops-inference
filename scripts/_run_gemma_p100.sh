#!/bin/bash
set -euo pipefail
MODEL="${1:-/codebase/models/google_gemma-4-26B-A4B-it-IQ4_XS.gguf}"
echo "MODEL=$MODEL"
ls -la "$MODEL"
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
/data/cargo-target/release/run_gemma4 \
  --model "$MODEL" \
  --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 8 \
  --temperature 0.0
echo "EXIT=$?"
