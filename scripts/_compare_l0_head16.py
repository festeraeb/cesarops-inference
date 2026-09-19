#!/usr/bin/env python3
"""Compare CESAROPS_HEAD dumps to our MOE_SLICE pos=0 first-16."""
from __future__ import annotations

import re
from pathlib import Path

LLAMA = Path("/tmp/llama_gemma4_l0_head16.log")
OURS = Path(
    "/codebase/repos/wreckhunter2000-1/cesarops-inference/regression/baselines/gemma4_l0_slice_checklist_2026-08-27.log"
)

MAP = [
    ("post_embed", "inp_scaled"),
    ("post_q_proj", "Qcur-0"),
    ("post_k_proj", "Kcur-0"),
    ("post_q_norm", "Qcur_normed-0"),
    ("post_k_norm", "Kcur_normed-0"),
    ("post_v_h0", "Vcur_normed-0"),
    ("post_o_norm", "attn_post_norm-0"),
    ("post_attn", "attn_out-0"),
    ("pre_mlp_norm", "ffn_norm_1-0"),
    ("post_mlp", "ffn_mlp-0"),
    ("pre_moe_norm", "ffn_norm_2-0"),
    ("post_moe_expert", "ffn_moe-0"),
    ("post_ffn_sum", "ffn_moe_combined-0"),
    ("post_ffn_norm", "ffn_post_norm-0"),
    ("post_layer", "l_out-0"),
]


def parse_ours() -> dict[str, list[float]]:
    out = {}
    pat = re.compile(r"MOE_SLICE layer=0 pos=0 (\S+) dims=\d+ \[([^\]]+)\]")
    for line in OURS.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = [float(x) for x in m.group(2).split(",") if x.strip()]
    return out


def parse_llama() -> dict[str, list[float]]:
    text = LLAMA.read_text(errors="replace").splitlines()
    out: dict[str, list[float]] = {}
    cur = None
    head_re = re.compile(r"CESAROPS_HEAD dim0\[:16\] \(ne=[^)]+\) \[([^\]]+)\]")
    name_re = re.compile(r"common_debug_cb_eval:\s+(\S+)\s+=")
    for line in text:
        m = name_re.search(line)
        if m:
            cur = m.group(1)
            continue
        m = head_re.search(line)
        if m and cur:
            out[cur] = [float(x) for x in m.group(1).split(",") if x.strip()]
            cur = None
    return out


def main() -> None:
    ours = parse_ours()
    llama = parse_llama()
    print(f"llama tensors: {sorted(llama)}")
    print()
    for phase, tname in MAP:
        o = ours.get(phase)
        l = llama.get(tname)
        if o is None or l is None:
            print(f"{phase:18s} {tname:22s} missing ours={o is not None} llama={l is not None}")
            continue
        n = min(len(o), len(l))
        diffs = [abs(o[i] - l[i]) for i in range(n)]
        mx = max(diffs)
        flag = "OK" if mx < 0.05 else ("CLOSE" if mx < 1.0 else "DIVERGE")
        print(f"{flag:7s} {phase:18s} vs {tname:22s} maxabs={mx:.4f}")
        if flag != "OK":
            print(f"        ours  {[round(x,4) for x in o[:8]]}")
            print(f"        llama {[round(x,4) for x in l[:8]]}")


if __name__ == "__main__":
    main()
