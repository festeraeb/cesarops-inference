// shaders/kv_cache_write.wgsl
//
// Append the current token's K and V into the per-layer KV cache buffers.
// The cache is laid out as [max_seq, n_kv_heads * head_dim] flat f32.
//
//   k_cache[pos * kv_stride + i] = k_in[i]
//   v_cache[pos * kv_stride + i] = v_in[i]
//
// This is a separate dispatch (not inlined into the matvec) because the K
// and V matvecs write to different scratch buffers; we need a sync point
// before they get folded into the cache.
//
// One workgroup writes one whole token's worth of K + V (kv_dim elements).

struct Push {
    pos:        u32,    // sequence position to write
    kv_dim:     u32,    // n_kv_heads * head_dim
    _pad0:      u32,
    _pad1:      u32,
};

@group(0) @binding(0) var<storage, read>       k_in    : array<f32>;
@group(0) @binding(1) var<storage, read>       v_in    : array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache : array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache : array<f32>;
@group(0) @binding(4) var<uniform>             push    : Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.kv_dim) { return; }
    let off = push.pos * push.kv_dim + i;
    k_cache[off] = k_in[i];
    v_cache[off] = v_in[i];
}
