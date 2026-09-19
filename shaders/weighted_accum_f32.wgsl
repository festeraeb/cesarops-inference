// shaders/weighted_accum_f32.wgsl
//
// y[i] += weight * src[i]   (f32, Pascal-safe)
//
// Drop-in replacement for `weighted_accum.wgsl` (f16-only, unsafe on
// Pascal). Used by the MoE FFN path to fold each expert's down-projection
// output into the per-token accumulator.

struct Push {
    weight: f32,
    len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> y   : array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.len) {
        return;
    }
    y[i] = y[i] + push.weight * src[i];
}
