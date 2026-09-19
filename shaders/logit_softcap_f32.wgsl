// shaders/logit_softcap_f32.wgsl
//
// y[i] = cap * tanh(x[i] / cap)
//
// Used at the LM head output for Gemma's `final_logit_softcapping = 30`.

struct Push {
    n:    u32,
    _pad: u32,
    _pad1: u32,
    _pad2: u32,
    cap:   f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.n) { return; }
    let v = x[i] / push.cap;
    y[i] = push.cap * tanh(v);
}
