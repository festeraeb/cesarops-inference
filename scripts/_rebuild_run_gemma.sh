#!/bin/bash
set -euo pipefail
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
CARGO_TARGET_DIR=/data/cargo-target /home/cesarops/.cargo/bin/cargo build --release --bin run_gemma4
echo BUILD_OK
/data/cargo-target/release/run_gemma4 \
  --model /codebase/models/google_gemma-4-26B-A4B-it-IQ4_XS.gguf \
  --gpu 0 \
  --prompt-tokens 2,236800,236781,236800,236784 \
  --max-new 8 \
  --temperature 0.0
echo EXIT=$?
