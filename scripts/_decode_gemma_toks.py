#!/usr/bin/env python3
import sys
sys.path.insert(0, "/codebase/repos/wreckhunter2000-1/llama.cpp/gguf-py")
from gguf import GGUFReader
path = sys.argv[1]
ids = [int(x) for x in sys.argv[2].split(",")]
r = GGUFReader(path)
# tokenizer.ggml.tokens
toks = None
for f in r.fields.values():
    name = f.name if hasattr(f, "name") else str(f)
# GGUFReader API
tokens = r.fields.get("tokenizer.ggml.tokens")
if tokens is None:
    # try alternate access
    for k, v in r.fields.items():
        if "tokenizer.ggml.tokens" in k:
            tokens = v
            break
parts = []
# gguf field parts are list of arrays
arr = tokens.parts[-1] if hasattr(tokens, "parts") else None
if arr is None:
    print("no tokens field")
    sys.exit(1)
# tokens.data may work in newer gguf
data = getattr(tokens, "data", None)
if data is None:
    # reconstruct from parts
    # In gguf-py, string arrays: parts[0]=lengths? use contents
    try:
        from gguf.gguf_reader import ReaderField
        contents = tokens.contents()
        for i in ids:
            parts.append(contents[i] if i < len(contents) else f"<{i}>")
    except Exception as e:
        print("decode fail", e)
        # fallback: print ids only
        print("ids", ids)
        sys.exit(0)
else:
    for i in ids:
        parts.append(data[i])
print("".join(parts) if all(isinstance(p, str) for p in parts) else parts)
print(parts)
