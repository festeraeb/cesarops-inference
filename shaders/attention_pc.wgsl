// Attention QK^T — push-constant variant of attention.wgsl.
// Same algorithm; params arrive via var<push_constant> instead of uniform.
// Drops one binding entry + one buffer write per head per pos.
//
// Drafted by Qwen3.6-A3B (round 4) — produced 1-token empty response (chat
// template mismatch). Polisher (Claude) wrote this. Lesson captured.

struct Params {
    kv_len: u32,
    head_dim: u32,
    cur_pos: u32,
    scale: f32,
    kv_stride: u32,
    kv_head_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       query:     array<f32>;
@group(0) @binding(1) var<storage, read>       key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores:    array<f32>;

var<push_constant> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.kv_len) { return; }

    // Causal mask
    if (i > params.cur_pos) {
        scores[i] = -3.40282347e38;
        return;
    }

    let k_base = i * params.kv_stride + params.kv_head_offset;

    var dot: f32 = 0.0;
    let hd4 = params.head_dim & ~3u;
    var j: u32 = 0u;
    while (j < hd4) {
        dot = dot + query[j]      * key_cache[k_base + j];
        dot = dot + query[j + 1u] * key_cache[k_base + j + 1u];
        dot = dot + query[j + 2u] * key_cache[k_base + j + 2u];
        dot = dot + query[j + 3u] * key_cache[k_base + j + 3u];
        j = j + 4u;
    }
    while (j < params.head_dim) {
        dot = dot + query[j] * key_cache[k_base + j];
        j = j + 1u;
    }

    scores[i] = dot * params.scale;
}
