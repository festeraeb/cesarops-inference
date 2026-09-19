#!/bin/bash
# Dump blk.0 FFN tensor types via our Rust loader (reliable).
set -euo pipefail
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
cat > /tmp/dump_ffn_types.rs <<'EOF'
// placeholder - use existing probe if any
EOF
# Use strings + python only on tensor name table via dump_embed_shape style:
/data/cargo-target/release/dump_embed_shape /mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf >/dev/null
python3 - <<'PY'
# Use cesarops? faster: parse with corrected skip, stop at first corrupt by catching
import struct
path='/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf'
with open(path,'rb') as f:
    assert f.read(4)==b'GGUF'
    ver=struct.unpack('<I',f.read(4))[0]
    n_tensors,n_kv=struct.unpack('<QQ',f.read(16))
    print('n_tensors', n_tensors, 'n_kv', n_kv)
    def read_str():
        (n,)=struct.unpack('<Q',f.read(8))
        return f.read(n)
    def skip_value(t):
        if t==8: read_str(); return
        if t==9:
            (at,)=struct.unpack('<I',f.read(4)); (n,)=struct.unpack('<Q',f.read(8))
            for _ in range(n): skip_value(at)
            return
        sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        f.read(sizes[t])
    for i in range(n_kv):
        key=read_str(); (vt,)=struct.unpack('<I',f.read(4)); skip_value(vt)
    print('metadata done at', f.tell())
    for i in range(n_tensors):
        name=read_str(); (ndims,)=struct.unpack('<I',f.read(4))
        dims=struct.unpack('<'+'Q'*ndims, f.read(8*ndims))
        (ttype,)=struct.unpack('<I',f.read(4)); (offset,)=struct.unpack('<Q',f.read(8))
        n=name.decode('utf-8','replace')
        if n.startswith('blk.0.ffn_') or n.startswith('blk.0.attn_q'):
            print(f'{n} dims={dims} type={ttype}')
PY
