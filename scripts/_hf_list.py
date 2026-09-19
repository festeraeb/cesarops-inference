#!/usr/bin/env python3
import json, urllib.request
url = "https://huggingface.co/api/models/CelesteImperia/Gemma-4-26B-MoE-GGUF/tree/main"
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)
for x in d:
    p = x.get("path", "")
    size = x.get("size") or (x.get("lfs") or {}).get("size")
    print(f"{size}\t{p}")
