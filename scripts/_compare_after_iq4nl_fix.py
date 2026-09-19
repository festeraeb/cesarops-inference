#!/usr/bin/env python3
import re
from pathlib import Path

def ours(phase):
    for line in Path('/tmp/ours_l0_mlp_mids.log').read_text().splitlines():
        m = re.search(rf'MOE_SLICE layer=0 pos=0 {phase} dims=\d+ \[([^\]]+)\]', line)
        if m:
            return [float(x) for x in m.group(1).split(',') if x.strip()]
    return None

def llama(name, log='/tmp/llama_gemma4_l0_head16.log'):
    text = Path(log).read_text().splitlines()
    cur = None
    for line in text:
        m = re.search(r'common_debug_cb_eval:\s+(\S+)\s+=', line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r'CESAROPS_HEAD dim0\[:16\] \(ne=[^)]+\) \[([^\]]+)\]', line)
        if m and cur == name:
            return [float(x) for x in m.group(1).split(',') if x.strip()]
    return None

pairs = [
    ('post_embed', 'inp_scaled'),
    ('post_attn', 'attn_out-0'),
    ('post_mlp', 'ffn_mlp-0'),
    ('post_moe_expert', 'ffn_moe-0'),
    ('post_ffn_sum', 'ffn_moe_combined-0'),
    ('post_ffn_norm', 'ffn_post_norm-0'),
    ('post_layer', 'l_out-0'),
]
for o, l in pairs:
    a, b = ours(o), llama(l)
    if not a or not b:
        print(f'{o}: missing ours={a is not None} llama={b is not None}')
        continue
    d = max(abs(a[i] - b[i]) for i in range(min(len(a), len(b))))
    flag = 'OK' if d < 0.05 else ('CLOSE' if d < 1 else 'DIVERGE')
    print(f'{flag:7s} {o:18s} maxabs={d:.4f}')

for line in Path('/tmp/ours_l0_mlp_mids.log').read_text().splitlines():
    if 'Output token' in line:
        print(line)
