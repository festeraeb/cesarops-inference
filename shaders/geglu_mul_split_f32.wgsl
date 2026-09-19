// shaders/geglu_mul_split_f32.wgsl
//
// GEGLU over a packed gate||up f32 buffer (Gemma-4 MoE expert path).
//
// out[i] = gelu(GU[i]) * GU[inner + i]
//
// Matches llama.cpp `ggml_geglu_split` used with LLM_FFN_GELU for gemma4 MoE.

struct Push {
    inner: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gu  : array<f32>;
@group(0) @binding(1) var<storage, read_write> h   : array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

fn gelu_approx(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978845608028654 * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.inner) {
        return;
    }
    let g = gu[i];
    let u = gu[push.inner + i];
    h[i] = gelu_approx(g) * u;
}
