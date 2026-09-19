#!/bin/bash
set -euo pipefail
DEST=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS.gguf
URL="https://huggingface.co/CelesteImperia/Gemma-4-26B-MoE-GGUF/resolve/main/Gemma-4-26B-MoE-IQ4_XS.gguf"
# Keep mixed-quant google file; write Celeste to a distinct name first.
OUT=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
if [[ -f "$OUT" ]] && [[ $(stat -c%s "$OUT") -gt 13000000000 ]]; then
  echo "already have $OUT"
  ls -lh "$OUT"
  exit 0
fi
echo "Downloading CelesteImperia uniform IQ4_XS to $OUT"
# resume-capable
if command -v hf >/dev/null 2>&1; then
  hf download CelesteImperia/Gemma-4-26B-MoE-GGUF Gemma-4-26B-MoE-IQ4_XS.gguf --local-dir /mnt/raid0/models/_hf_celeste_gemma
  mv -f /mnt/raid0/models/_hf_celeste_gemma/Gemma-4-26B-MoE-IQ4_XS.gguf "$OUT"
else
  wget -c -O "$OUT" "$URL"
fi
ls -lh "$OUT"
echo DOWNLOAD_OK
