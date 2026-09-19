#!/usr/bin/env python3
"""Extract first floats for key llama-debug tensors and compare to our MOE_SLICE dump."""
from __future__ import annotations

import re
from pathlib import Path

LLAMA_LOG = Path("/tmp/llama_gemma4_l0_stages.log")
OURS_LOG = Path(
    "/codebase/repos/wreckhunter2000-1/cesarops-inference/regression/baselines/gemma4_l0_slice_checklist_2026-08-27.log"
)

# Our phases to compare (pos=0 first-16 where applicable)
OURS_PHASE = {
    "post_embed": "inp_scaled",
    "post_q_proj": "Qcur-0",
    "post_k_proj": "Kcur-0",
    "post_q_norm": "Qcur_normed-0",
    "post_k_norm": "Kcur_normed-0",
    "post_v_h0": "Vcur_normed-0",
    "post_o_norm": "attn_post_norm-0",
    "post_attn": "attn_out-0",
    "post_mlp": "ffn_mlp-0",
    "post_moe_expert": "ffn_moe-0",
    "post_ffn_sum": "ffn_moe_combined-0",
    "post_ffn_norm": "ffn_post_norm-0",
    "post_layer": "l_out-0",
}


def parse_ours(path: Path) -> dict[str, list[float]]:
    out = {}
    pat = re.compile(
        r"MOE_SLICE layer=0 pos=0 (\S+) dims=\d+ \[([^\]]+)\]"
    )
    for line in path.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        phase, vals = m.group(1), m.group(2)
        out[phase] = [float(x) for x in vals.split(",") if x.strip()]
    return out


def extract_tensor_first_row(text: str, name: str) -> list[float] | None:
    """Grab the first printed innermost row after `name = (` cb line.

    For ne={N, T, ...} the first row is token 0's first/last 3 of dim0.
    We take the contiguous first numbers before `...,`.
    """
    # Prefer exact `name = (` not reshaped variants when name has no paren
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "common_debug_cb_eval:" not in line:
            continue
        # extract tensor name token
        rest = line.split("common_debug_cb_eval:", 1)[1].strip()
        tname = rest.split("=", 1)[0].strip()
        if tname == name:
            start = i
            break
    if start is None:
        return None

    # Find first `[` row with floats
    floats: list[float] = []
    for j in range(start + 1, min(start + 200, len(lines))):
        line = lines[j]
        if "common_debug_cb_eval:" in line:
            break
        if "...," in line and floats:
            # stop at ellipsis within first row
            break
        nums = re.findall(r"[-+]?\d+\.\d+", line)
        if nums and "[" in line:
            floats.extend(float(x) for x in nums)
            # first innermost printed row is enough for head compare
            if floats:
                return floats[:16]
    return floats[:16] if floats else None


def main() -> None:
    text = LLAMA_LOG.read_text(errors="replace")
    ours = parse_ours(OURS_LOG)
    print(f"ours phases: {sorted(ours)}")
    print()
    for phase, tname in OURS_PHASE.items():
        o = ours.get(phase)
        l = extract_tensor_first_row(text, tname)
        if o is None:
            print(f"{phase:22s} OURS missing")
            continue
        if l is None:
            print(f"{phase:22s} LLAMA missing ({tname})")
            continue
        n = min(len(o), len(l), 8)
        diffs = [abs(o[i] - l[i]) for i in range(n)]
        mx = max(diffs) if diffs else float("nan")
        print(f"{phase:22s} vs {tname:22s} n={n} maxabs={mx:.4f}")
        print(f"  ours[:{n}]  {[round(x,4) for x in o[:n]]}")
        print(f"  llama[:{n}] {[round(x,4) for x in l[:n]]}")


if __name__ == "__main__":
    main()
