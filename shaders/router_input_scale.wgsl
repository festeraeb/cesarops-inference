// shaders/router_input_scale.wgsl
//
// Elementwise scale of the hidden-state vector before the router projection.
//
//   gi[i] = x[i] * scale[i]
//
// Used by the MoE FFN path:
//   gate_input = x * ffn_gate_inp.scale       // [hidden]
//   router_logits = gate_input @ ffn_gate_inp.weight   // [128]
//
// The router weight matvec itself runs as a tiny f32 matmul on the
// router_w buffer (full-precision, 2816×128 = 1.4 MB) — handled in
// `moe_iq4_dispatch.rs`.

struct Push {
    hidden: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       x     : array<f32>;
@group(0) @binding(1) var<storage, read>       scale : array<f32>;
@group(0) @binding(2) var<storage, read_write> gi    : array<f32>;
@group(0) @binding(3) var<uniform>             push  : Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.hidden) {
        return;
    }
    gi[i] = x[i] * scale[i];
}
