// shaders/rmsnorm_weightless_f32.wgsl
//
// f32 RMSNorm without the multiplicative weight gain — used by Gemma's V
// projection. y[i] = x[i] / sqrt(mean(x^2) + eps).
// One workgroup per row. Pascal-safe.

struct Push {
    hidden_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    eps: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

@group(0) @binding(0) var<storage, read>       x  : array<f32>;
@group(0) @binding(1) var<storage, read_write> y  : array<f32>;
@group(0) @binding(2) var<uniform>             push: Push;

var<workgroup> shared_ss: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let row = wid.x;
    let base = row * push.hidden_dim;

    var local_ss: f32 = 0.0;
    var i: u32 = tid;
    while (i < push.hidden_dim) {
        let v = x[base + i];
        local_ss = local_ss + v * v;
        i = i + 256u;
    }
    shared_ss[tid] = local_ss;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_ss[tid] = shared_ss[tid] + shared_ss[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let mean_sq = shared_ss[0] / f32(push.hidden_dim);
    let inv_rms = inverseSqrt(mean_sq + push.eps);

    i = tid;
    while (i < push.hidden_dim) {
        y[base + i] = x[base + i] * inv_rms;
        i = i + 256u;
    }
}
