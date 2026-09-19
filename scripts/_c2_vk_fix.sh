#!/bin/bash
set -euo pipefail
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
ls -la /etc/vulkan/icd.d/ || true
# install nvidia vulkan ICD if missing (noninteractive)
if [[ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]] && [[ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  echo 'NEED_NVIDIA_VULKAN_ICD'
  apt-cache search nvidia-vulkan 2>/dev/null | head
  apt-cache search libnvidia-gl 2>/dev/null | head
fi
# try creating icd from known path
find /usr -name 'libGLX_nvidia.so*' 2>/dev/null | head
find /usr -name 'libvulkan*' 2>/dev/null | head
EOF
