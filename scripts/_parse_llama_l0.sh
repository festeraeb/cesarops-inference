#!/bin/bash
set -euo pipefail
LOG=/tmp/llama_gemma4_l0_stages.log
echo "=== tokens / sampling ==="
grep -E 'token|prompt|sampling|predict|Output|decoded|piece' "$LOG" | head -40
echo "=== cb names matched ==="
grep 'common_debug_cb_eval:' "$LOG" | sed 's/common_debug_cb_eval: *//' | awk '{print $1}' | sort -u
echo "=== full cb lines ==="
grep 'common_debug_cb_eval:' "$LOG"
echo "=== extract head floats near Qcur-0 and attn_out ==="
# show blocks after key names (next 40 lines)
python3 - <<'PY'
from pathlib import Path
import re
text = Path('/tmp/llama_gemma4_l0_stages.log').read_text(errors='replace')
# Find sections starting with common_debug_cb_eval for interesting tensors
keys = ['inp_scaled', 'Qcur-0 =', 'Qcur_normed', 'Kcur-0 =', 'Vcur-0 =', 'attn_out-0', 'ffn_mlp-0', 'ffn_moe-0', 'ffn_moe_combined', 'out_scaled-0', 'l_out-0', 'ffn_post_norm']
lines = text.splitlines()
for i, line in enumerate(lines):
    if 'common_debug_cb_eval:' not in line:
        continue
    if not any(k in line for k in keys):
        continue
    print('\n' + line)
    # print following value lines until next eval or blank separator of sum
    for j in range(i+1, min(i+80, len(lines))):
        l = lines[j]
        if 'common_debug_cb_eval:' in l:
            break
        if l.strip().startswith('sum =') or re.search(r'\[[\s\d\.,\-]+\]', l) or '...,' in l or l.strip().startswith('['):
            print(l)
            if l.strip().startswith('sum ='):
                break
PY
