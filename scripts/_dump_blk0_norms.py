#!/usr/bin/env python3
"""Dump Gemma-4 blk.0 norm weight stats from the Celeste GGUF."""
from __future__ import annotations

import sys

try:
    from gguf import GGUFReader
except Exception as e:
    print("FAIL: gguf import:", e)
    sys.exit(1)

import numpy as np

path = "/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf"
r = GGUFReader(path)
wanted = {
    "blk.0.attn_q_norm.weight",
    "blk.0.attn_k_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.layer_output_scale",
    "blk.0.post_attention_norm.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.post_ffw_norm.weight",
}
for t in r.tensors:
    if t.name not in wanted:
        continue
    x = np.asarray(t.data, dtype=np.float32).ravel()
    print(
        f"{t.name}: n={x.size} rms={float((x * x).mean() ** 0.5):.6f} "
        f"mean={float(x.mean()):.6f} min={float(x.min()):.6f} max={float(x.max()):.6f} "
        f"first8={x[:8].tolist()}"
    )

# sliding pattern
for key, field in r.fields.items():
    name = getattr(field, "name", key)
    if any(s in str(name) for s in ("sliding", "swa", "layer_type")):
        print("META", name, getattr(field, "types", None))
