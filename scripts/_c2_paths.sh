#!/bin/bash
set -euo pipefail
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
ls -la /mnt/t440/models/ | head -30
ls -lh /mnt/t440/models/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf 2>&1 | head
ls -lh /mnt/t440/models/Gemma* 2>&1 | head
ls -lh /mnt/t440/data/cargo-target/release/run_gemma4
# find llama-debug
find /mnt/t440/repo/llama.cpp /home/cesarops/src /home/cesarops/llama* -name 'llama-debug' 2>/dev/null | head
ls /mnt/t440/repo/llama.cpp/examples/debug 2>/dev/null | head
EOF
