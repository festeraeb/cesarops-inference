// shaders/silu_mul_split_f32.wgsl
//
// SiLU-mul over a packed gate||up f32 buffer.
//
// Input  GU: array<f32> length 2 * inner   (gate first, up second)
// Output H : array<f32> length     inner
//
// out[i] = silu(GU[i]) * GU[inner + i]
// silu(x) = x / (1 + exp(-x))
//
// Pascal-safe: pure f32, no subgroups, no f16. Drop-in replacement for
// the existing `silu_mul.wgsl` (which requires f16 and is unsafe on
// Pascal under wgpu).
//
// Used by the MoE FFN path between the gate_up matvec output and the
// down matvec input.

struct Push {
    inner: u32,    // 704 for Gemma-4 MoE
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gu  : array<f32>;
@group(0) @binding(1) var<storage, read_write> h   : array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.inner) {
        return;
    }
    let g = gu[i];
    let u = gu[push.inner + i];
    let s = g / (1.0 + exp(-g));
    h[i] = s * u;
}
