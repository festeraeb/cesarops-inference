#!/usr/bin/env python3
"""Dump GGUF tensor names + ggml types for a model."""
import struct
import sys

GGML = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
}

def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")

def read_val(f, t):
    if t == 8:  # STRING
        return read_str(f)
    sizes = {0:1,1:2,2:4,3:4,4:8,5:1,6:8,7:8,9:8,10:8,11:4,12:8}
    # ARRAY
    if t == 9:
        (at,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_val(f, at) for _ in range(n)]
    sz = sizes.get(t)
    if sz is None:
        raise ValueError(t)
    return f.read(sz)

def main(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        print(f"ver={ver} tensors={n_tensors} kv={n_kv}")
        for _ in range(n_kv):
            _k = read_str(f)
            (vt,) = struct.unpack("<I", f.read(4))
            _ = read_val(f, vt)
        counts = {}
        samples = []
        for i in range(n_tensors):
            name = read_str(f)
            (ndims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tname = GGML.get(ttype, str(ttype))
            counts[tname] = counts.get(tname, 0) + 1
            if i < 30 or "attn_q" in name or "attn_k" in name:
                samples.append((name, tname, dims))
        print("type counts:", dict(sorted(counts.items(), key=lambda x: -x[1])))
        print("samples:")
        for s in samples[:40]:
            print(" ", s)

if __name__ == "__main__":
    main(sys.argv[1])
