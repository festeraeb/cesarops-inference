#!/bin/bash
set -euo pipefail
# Run on T440, exec work on cesarops2 via LAN
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
set -euo pipefail
echo MODEL_CHECK
ls -lh /mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf /mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf 2>&1 | head
ls -lh /mnt/t440/data/cargo-target/release/run_gemma4 /mnt/t440/data/cargo-target/release/cesarops-inference
echo LLAMA_DEBUG
find /mnt/t440/repo/llama.cpp /home/cesarops -name 'llama-debug' -type f 2>/dev/null | head
ls /mnt/t440/repo/llama.cpp/build*/bin/llama-debug 2>/dev/null | head
nvidia-smi --query-gpu=index,memory.used --format=csv
EOF
