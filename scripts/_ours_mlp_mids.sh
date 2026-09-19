#!/bin/bash
set -euo pipefail
MODEL=/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf
BIN=/data/cargo-target/release/run_gemma4
OUT=/tmp/ours_l0_mlp_mids.log
export MOE_SLICE=1 MOE_SLICE_LAYER=0 MOE_SLICE_POS=0 MOE_SLICE_DIMS=16 RUST_LOG=warn GEMMA4_GPU=0
"$BIN" --model "$MODEL" --gpu 0 --prompt-tokens 2,236800,236781,236800,236784 --max-new 1 --temperature 0.0 \
  2>&1 | tee "$OUT" | grep -E 'MOE_SLICE layer=0 pos=0 post_mlp|MOE_SLICE layer=0 pos=0 pre_mlp|GEMMA4_CONV|Output token'
echo '==== compare gate/up/geglu ===='
python3 - <<'PY'
import re
from pathlib import Path

def ours(phase):
    for line in Path('/tmp/ours_l0_mlp_mids.log').read_text().splitlines():
        m=re.search(rf'MOE_SLICE layer=0 pos=0 {phase} dims=\d+ \[([^\]]+)\]', line)
        if m: return [float(x) for x in m.group(1).split(',') if x.strip()]
    return None

def llama(name):
    text=Path('/tmp/llama_gemma4_l0_ffn_mids.log').read_text().splitlines()
    cur=None
    for line in text:
        m=re.search(r'common_debug_cb_eval:\s+(\S+)\s+=', line)
        if m: cur=m.group(1); continue
        m=re.search(r'CESAROPS_HEAD dim0\[:16\] \(ne=[^)]+\) \[([^\]]+)\]', line)
        if m and cur==name:
            return [float(x) for x in m.group(1).split(',') if x.strip()]
    return None

pairs=[
 ('pre_mlp_norm','ffn_norm_1-0'),
 ('post_mlp_gate','ffn_gate-0'),
 ('post_mlp_up','ffn_up-0'),
 ('post_mlp_geglu','ffn_geglu-0'),
 ('post_mlp','ffn_mlp-0'),
]
for o,l in pairs:
    a,b=ours(o),llama(l)
    if not a or not b:
        print(f'{o} vs {l}: missing ours={a is not None} llama={b is not None}'); continue
    n=min(len(a),len(b)); d=max(abs(a[i]-b[i]) for i in range(n))
    flag='OK' if d<0.05 else ('CLOSE' if d<1 else 'DIVERGE')
    print(f'{flag:7s} {o:16s} vs {l:14s} maxabs={d:.4f}')
    if flag!='OK':
        print('  ours ', [round(x,4) for x in a[:8]])
        print('  llama', [round(x,4) for x in b[:8]])
PY
