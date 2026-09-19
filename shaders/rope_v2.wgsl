// shaders/rope_v2.wgsl
//
// Stride-loop RoPE that works for any head_dim and per-call rope_base.
//
// In:    QK[head_idx, head_dim]    array<f32>, in-place rotated.
// Push:  head_dim, pos, n_heads, rope_base
//
// Pair convention matches `rope.wgsl` (HF/Gemma-style half-split):
//   q[d]               = q[d] * cos - q[d + half] * sin
//   q[d + half]        = q[d] * sin + q[d + half] * cos
// where half = head_dim / 2 and theta = pos / rope_base^(2d/head_dim).
//
// Dispatch one workgroup per head, workgroup_size = 256. Each thread strides
// across pair indices 0..half_dim in steps of 256, so this works for
// head_dim up to 65536 with no host-side change.

struct Push {
    head_dim: u32,    // e.g. 256 (SWA) or 512 (full)
    pos: u32,
    n_heads: u32,
    _pad: u32,
    rope_base: f32,   // 1e6 for full, 1e4 for SWA
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read_write> qk : array<f32>;
@group(0) @binding(1) var<uniform>             push: Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;
    if (head_idx >= push.n_heads) { return; }

    let half_dim = push.head_dim >> 1u;
    let base = head_idx * push.head_dim;
    let pos_f = f32(push.pos);

    // Each thread walks pair indices `p, p+256, p+512, ...` until exhausting half_dim.
    var p: u32 = lid.x;
    while (p < half_dim) {
        let freq_exp = -2.0 * f32(p) / f32(push.head_dim);
        let theta = pos_f * pow(push.rope_base, freq_exp);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let lo = base + p;
        let hi = lo + half_dim;
        let x0 = qk[lo];
        let x1 = qk[hi];

        qk[lo] = x0 * cos_t - x1 * sin_t;
        qk[hi] = x0 * sin_t + x1 * cos_t;

        p = p + 256u;
    }
}
