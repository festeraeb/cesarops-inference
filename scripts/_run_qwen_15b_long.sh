#!/bin/bash
set -euo pipefail
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
sudo systemctl stop cesarops-ornith 2>/dev/null || true
sleep 1
/data/cargo-target/release/cesarops-inference generate \
  --model /mnt/raid0/models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf \
  --gpu 0 \
  --prompt "Write a short Python function that returns the factorial of n. Include a docstring and one example." \
  --max-tokens 128
echo QWEN_LONG_EXIT=$?
sudo systemctl start cesarops-ornith 2>/dev/null || true
