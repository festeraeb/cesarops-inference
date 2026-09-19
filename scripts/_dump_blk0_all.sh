#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import struct
path='/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS-celeste.gguf'
with open(path,'rb') as f:
    assert f.read(4)==b'GGUF'
    struct.unpack('<I',f.read(4))
    n_tensors,n_kv=struct.unpack('<QQ',f.read(16))
    def read_str():
        (n,)=struct.unpack('<Q',f.read(8)); return f.read(n)
    def skip_value(t):
        if t==8:
            read_str(); return
        if t==9:
            (at,)=struct.unpack('<I',f.read(4))
            (n,)=struct.unpack('<Q',f.read(8))
            for _ in range(n):
                skip_value(at)
            return
        sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        f.read(sizes[t])
    for _ in range(n_kv):
        read_str(); (vt,)=struct.unpack('<I',f.read(4)); skip_value(vt)
    names=[]
    for i in range(n_tensors):
        name=read_str(); (ndims,)=struct.unpack('<I',f.read(4))
        dims=struct.unpack('<'+'Q'*ndims, f.read(8*ndims))
        (ttype,)=struct.unpack('<I',f.read(4)); struct.unpack('<Q',f.read(8))
        n=name.decode('utf-8','replace')
        if n.startswith('blk.0.'):
            names.append((n,dims,ttype))
    for n,d,t in names:
        print(f'{n} dims={d} type={t}')
    print('count', len(names))
PY
