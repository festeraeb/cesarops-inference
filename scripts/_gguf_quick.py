#!/usr/bin/env python3
import sys
sys.path.insert(0, "/codebase/repos/wreckhunter2000-1/llama.cpp/gguf-py")
from collections import Counter
from gguf import GGUFReader
path = sys.argv[1]
r = GGUFReader(path)
c = Counter(int(t.tensor_type) if not hasattr(t.tensor_type,'value') else t.tensor_type.value for t in r.tensors)
# try both
c2 = Counter()
for t in r.tensors:
    tt = t.tensor_type
    v = tt.value if hasattr(tt, "value") else int(tt)
    c2[v] += 1
print(path)
print("counts", dict(sorted(c2.items())))
for name in ("blk.0.attn_q.weight", "blk.0.ffn_gate_up_exps.weight", "token_embd.weight"):
    for t in r.tensors:
        if t.name == name:
            v = t.tensor_type.value if hasattr(t.tensor_type, "value") else int(t.tensor_type)
            print(name, v)
            break
