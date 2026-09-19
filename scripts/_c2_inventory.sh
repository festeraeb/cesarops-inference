#!/bin/bash
set -euo pipefail
ssh -o ConnectTimeout=15 cesarops@192.168.4.34 'bash -s' <<'EOF'
set -euo pipefail
echo '=== mounts ==='
mount | grep -Ei 'nfs|cifs|raid|t440|models' || true
echo '=== df ==='
df -h
echo '=== models ==='
ls -lh /mnt/storage/models 2>/dev/null | head || true
ls /mnt 2>/dev/null || true
echo '=== llama ==='
ls -lh /home/cesarops/llama-bin/ | head
# check if can reach t440 models via scp size
echo '=== free for model ==='
df -B1 /home | awk 'NR==2{print $4}'
EOF
