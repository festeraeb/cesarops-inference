// shaders/rmsnorm_f32.wgsl
//
// f32 RMSNorm over a single row. Pure scalar buffers (no vec4 alignment
// requirement on hidden_dim). Pascal-safe.
//
//   y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gain(weight[i])
//
// gain(w) = (w + plus_one) where plus_one ∈ {0.0, 1.0} via uniform — picks
// between the standard convention (raw weight) and Gemma's offset convention
// (1.0 + weight). Same shader handles both.
//
// One workgroup processes one row. Workgroup size 256. The weight buffer
// is shared (length = hidden_dim). The input/output buffers can be the
// same (in-place) or distinct.

struct Push {
    hidden_dim: u32,
    plus_one_flag: u32,    // 1u → use (w + 1.0); 0u → raw w
    _pad0: u32,
    _pad1: u32,
    eps: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read>       x      : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> y      : array<f32>;
@group(0) @binding(3) var<uniform>             push   : Push;

var<workgroup> shared_ss: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let row = wid.x;
    let base = row * push.hidden_dim;

    // Phase 1: per-thread sum of squares.
    var local_ss: f32 = 0.0;
    var i: u32 = tid;
    while (i < push.hidden_dim) {
        let v = x[base + i];
        local_ss = local_ss + v * v;
        i = i + 256u;
    }
    shared_ss[tid] = local_ss;
    workgroupBarrier();

    // Phase 2: tree reduction.
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
    let plus_one_active = push.plus_one_flag != 0u;

    // Phase 3: scale.
    i = tid;
    while (i < push.hidden_dim) {
        let xi = x[base + i];
        var w = weight[i];
        if (plus_one_active) {
            w = w + 1.0;
        }
        y[base + i] = xi * inv_rms * w;
        i = i + 256u;
    }
}
