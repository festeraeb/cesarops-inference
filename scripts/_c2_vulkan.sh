#!/bin/bash
set -euo pipefail
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
echo '=== vulkan ==='
which vulkaninfo; vulkaninfo --summary 2>&1 | head -60 || true
ls /usr/share/vulkan/icd.d/ 2>/dev/null || true
echo '=== nvidia ==='
nvidia-smi -L
ls /dev/nvidia* 2>/dev/null
echo '=== env ==='
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-}"
ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || true
# try with nvidia icd
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary 2>&1 | head -40
fi
EOF
