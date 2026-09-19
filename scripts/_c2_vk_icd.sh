#!/bin/bash
set -euo pipefail
ssh -o ConnectTimeout=20 cesarops@192.168.4.34 'bash -s' <<'EOF'
set -euo pipefail
ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo 'no icd.d'
find /usr -name 'nvidia_icd.json' 2>/dev/null | head
# try apt without install if packages present
dpkg -l | grep -iE 'vulkan|nvidia.*gl' | head -20
# Can we request vulkan via python/wgpu?
python3 - <<'PY'
try:
  import subprocess, os
  # list ICD candidates
  for p in ['/usr/share/vulkan/icd.d','/etc/vulkan/icd.d','/usr/lib/x86_64-linux-gnu']:
    print(p, os.path.exists(p))
except Exception as e:
  print(e)
PY
# nvidia driver version
cat /proc/driver/nvidia/version 2>/dev/null | head -3
EOF
