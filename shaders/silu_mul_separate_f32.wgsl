// shaders/silu_mul_separate_f32.wgsl
//
// SwiGLU when gate and up live in two distinct buffers (the dense FFN
// path; for MoE see silu_mul_split_f32.wgsl).
//
//   y[i] = silu(gate[i]) * up[i]
//   silu(x) = x / (1 + exp(-x))
//
// Pascal-safe (f32 only).

struct Push {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gate : array<f32>;
@group(0) @binding(1) var<storage, read>       up   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y    : array<f32>;
@group(0) @binding(3) var<uniform>             push : Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.n) { return; }
    let g = gate[i];
    let s = g / (1.0 + exp(-g));
    y[i] = s * up[i];
}
