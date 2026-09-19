#!/bin/bash
set -euo pipefail
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
/data/cargo-target/release/cesarops-inference generate \
  --model /mnt/raid0/models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf \
  --gpu 0 \
  --prompt "2+2=" \
  --max-tokens 8
echo QWEN_EXIT=$?
