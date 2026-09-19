#!/usr/bin/env python3
"""Search Qdrant codebase indexes for Qwen shader / generate path docs."""
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
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def scroll_filter(col, must, limit=20):
    return post(
        f"/collections/{col}/points/scroll",
        {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": must},
        },
    )


def dump_hits(label, resp, fields=("path", "file_path", "filepath", "source", "text", "content", "chunk", "code")):
    pts = resp.get("result", {}).get("points", [])
    print(f"\n=== {label}: {len(pts)} hits ===")
    for p in pts[:15]:
        pay = p.get("payload") or {}
        # print identifying fields
        idents = {k: pay[k] for k in pay if k.lower() in (
            "path", "file_path", "filepath", "rel_path", "source", "repo",
            "language", "symbol", "name", "title", "module", "type"
        ) or "path" in k.lower() or "file" in k.lower()}
        text = None
        for f in fields:
            if f in pay and isinstance(pay[f], str) and pay[f].strip():
                text = pay[f]
                break
        if text is None:
            # any long string
            for k, v in pay.items():
                if isinstance(v, str) and len(v) > 80:
                    text = v
                    break
        print("-", idents or list(pay.keys())[:8])
        if text:
            print(" ", text[:400].replace("\n", " | "))


# Discover real document payload by scrolling past metadata
for col in ["codebase_index_v3", "codebase_index_v2", "codebase_index"]:
    try:
        resp = post(
            f"/collections/{col}/points/scroll",
            {
                "limit": 5,
                "with_payload": True,
                "with_vector": False,
                "filter": {"must_not": [{"key": "type", "match": {"value": "collection_metadata"}}]},
            },
        )
        pts = resp["result"]["points"]
        print(f"\n{col} non-meta count sample: {len(pts)}")
        if pts:
            print("keys:", sorted(pts[0]["payload"].keys()))
            print(json.dumps(pts[0]["payload"], indent=2)[:2000])
            break
    except Exception as e:
        print(col, "err", e)

# Keyword-ish filters on common path fields
queries = [
    ("path_qwen_shader", [{"key": "path", "match": {"text": "qwen"}}]),
    ("path_dequant_q6k", [{"key": "path", "match": {"text": "dequant_q6k"}}]),
    ("path_matvec_q6k", [{"key": "path", "match": {"text": "matvec_q6k"}}]),
    ("file_qwen", [{"key": "file_path", "match": {"text": "qwen"}}]),
    ("rel_q6k", [{"key": "rel_path", "match": {"text": "q6k"}}]),
    ("content_qwen_shader", [{"key": "content", "match": {"text": "qwen shader"}}]),
    ("text_dequant_q6k", [{"key": "text", "match": {"text": "dequant_q6k"}}]),
    ("chunk_q6k", [{"key": "chunk", "match": {"text": "matvec_q6k_fused"}}]),
    ("code_q6k", [{"key": "code", "match": {"text": "dequant_q6k.wgsl"}}]),
]

for col in ["codebase_index_v3", "codebase_index_v2", "codebase_index"]:
    for name, must in queries:
        try:
            resp = scroll_filter(col, must, limit=10)
            n = len(resp.get("result", {}).get("points", []))
            if n:
                dump_hits(f"{col}/{name}", resp)
        except Exception as e:
            # silently skip bad filters
            err = str(e)
            if "400" not in err and "404" not in err:
                print(f"{col}/{name} err:", err[:200])
