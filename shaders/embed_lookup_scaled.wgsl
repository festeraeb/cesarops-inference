// shaders/embed_lookup_scaled.wgsl
//
// Read one row out of a token embedding table that is already dequantized
// to f32 on the GPU, scale it by sqrt(hidden_dim), and write into the
// per-token hidden state buffer.
//
// E   : array<f32>  layout [vocab_size, hidden_dim] row-major
// H   : array<f32>  output [hidden_dim] (overwrite)
// push: token_id, hidden_dim, embed_scale_flag
//
// `embed_scale_flag = 1` multiplies by sqrt(hidden_dim) (Gemma 1/2/3/4
// convention). `0` disables the scale.

struct Push {
    token_id: u32,
    hidden_dim: u32,
    embed_scale_flag: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       e: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= push.hidden_dim) { return; }

    let row_off = push.token_id * push.hidden_dim;
    var v = e[row_off + i];
    if (push.embed_scale_flag != 0u) {
        v = v * sqrt(f32(push.hidden_dim));
    }
    h[i] = v;
}
