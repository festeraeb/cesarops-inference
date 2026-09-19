#!/usr/bin/env python3
"""Probe Qdrant codebase_index for Qwen shader hits."""
import json
import urllib.request

BASE = "http://127.0.0.1:6333"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


cols = get("/collections")["result"]["collections"]
print("collections:", [c["name"] for c in cols])

# sample payload keys
for col in ["codebase_index_v3", "codebase_index_v2", "codebase_index"]:
    try:
        sample = post(
            f"/collections/{col}/points/scroll",
            {"limit": 1, "with_payload": True, "with_vector": False},
        )
        pts = sample["result"]["points"]
        if not pts:
            print(f"{col}: empty")
            continue
        keys = list(pts[0]["payload"].keys())
        print(f"{col} keys:", keys)
        print(f"{col} sample:", json.dumps(pts[0]["payload"], indent=2)[:1200])
        break
    except Exception as e:
        print(f"{col} sample err:", e)
