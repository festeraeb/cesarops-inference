#!/usr/bin/env python3
import sys
sys.path.insert(0, "/codebase/repos/wreckhunter2000-1/llama.cpp/gguf-py")
from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)
vals = r.fields["tokenizer.ggml.tokens"].contents()
prompt_ids = [2, 236800, 236781, 236800, 236784]
print("prompt decode:", [vals[i] for i in prompt_ids])

# greedy longest-prefix for 3x3=
text = "3x3="
# map token string -> first id
index = {}
for i, t in enumerate(vals):
    if t not in index:
        index[t] = i

# SentencePiece pieces
needles = ["3", "x", "=", "▁3", "3x", "3x3", "3x3="]
for n in needles:
    print(repr(n), index.get(n))

# find tokens containing 3x3
hits = [(i, t) for i, t in enumerate(vals) if "3x3" in t or t in ("3", "x", "=")]
print("hits", hits[:20], "n", len(hits))
