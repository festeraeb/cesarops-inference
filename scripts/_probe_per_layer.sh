#!/usr/bin/env bash
# List Gemma-4 per-layer embedding tensors / metadata from Celeste GGUF.
set -euo pipefail
MODEL="${1:-/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf}"
BIN=/data/cargo-target/release/probe_moe_layout
cd /codebase/repos/wreckhunter2000-1/cesarops-inference

# Use a tiny Rust one-shot via existing loader through dump if available;
# fall back to strings on tensor name table via python struct parse of GGUF.
python3 - <<'PY' "$MODEL"
import sys, struct
path = sys.argv[1]
with open(path, "rb") as f:
    magic = f.read(4)
    assert magic == b"GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    print(f"GGUF v{ver} n_tensors={n_tensors} n_kv={n_kv}")

    def read_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_value(t):
        # GGUF value types (ggml gguf.h)
        # 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array
        # 10 u64 11 i64 12 f64
        if t == 8:  # STRING
            read_str(); return
        if t == 9:  # ARRAY
            (at,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_value(at)
            return
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t not in sizes:
            raise RuntimeError(f"unknown gguf type {t} at {f.tell()}")
        f.read(sizes[t])

    interesting = []
    for _ in range(n_kv):
        key = read_str()
        (vt,) = struct.unpack("<I", f.read(4))
        if "per_layer" in key or "embedding_length_per_layer" in key or key.endswith("embd_length"):
            # peek without full decode for scalars
            pos = f.tell()
            if vt == 4:  # UINT32
                (v,) = struct.unpack("<I", f.read(4)); print(f"META {key}={v}")
            elif vt == 5:  # INT32
                (v,) = struct.unpack("<i", f.read(4)); print(f"META {key}={v}")
            elif vt == 6:  # FLOAT32
                (v,) = struct.unpack("<f", f.read(4)); print(f"META {key}={v}")
            else:
                skip_value(vt); print(f"META {key} type={vt}")
                continue
        else:
            skip_value(vt)

    names = []
    for i in range(n_tensors):
        name = read_str()
        (ndims,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q"*ndims, f.read(8*ndims))
        (ttype,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        if "per_layer" in name or name.startswith("blk.0.per_layer") or "token_embd_per" in name:
            names.append((name, dims, ttype))
    print(f"per_layer tensor count: {len(names)}")
    for n,d,t in names[:40]:
        print(f"  {n} dims={d} type={t}")
    if len(names) > 40:
        print(f"  ... +{len(names)-40} more")
PY
