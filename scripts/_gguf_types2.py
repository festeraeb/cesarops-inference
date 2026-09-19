#!/usr/bin/env python3
import sys
sys.path.insert(0, "/codebase/repos/wreckhunter2000-1/llama.cpp/gguf-py")
from collections import Counter
from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)
c = Counter(str(t.tensor_type).split(".")[-1] for t in r.tensors)
print("counts:", dict(c))
print("first 30:")
for t in r.tensors[:30]:
    print(" ", t.name, str(t.tensor_type).split(".")[-1])
print("attn samples:")
n = 0
for t in r.tensors:
    if "attn_q.weight" in t.name or "ffn_gate_inp" in t.name or "token_embd" in t.name:
        print(" ", t.name, str(t.tensor_type).split(".")[-1])
        n += 1
        if n > 20:
            break
