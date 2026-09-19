// shaders/router_matvec_f32.wgsl
//
// Tiny full-precision matvec for the MoE router projection.
//
//   logits[e] = sum_k gi[k] * W[e, k]      e in 0..n_experts (=128)
//
// W is row-major fp32, shape [n_experts, hidden] = [128, 2816] = 1.4 MB.
// One workgroup = one expert row; 256 threads cooperatively sum K=hidden
// elements per row using the same shared-mem reduction pattern as the
// IQ4 matvecs.
//
// We then read back the 128 fp32 logits to CPU for top-k softmax (router
// is too small to benefit from a GPU softmax-topk kernel, and the CPU
// branch is what `forward_pass` already does for sampling).

struct Push {
    hidden: u32,      // K = 2816
    n_experts: u32,   // N rows (128)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       w      : array<f32>;
@group(0) @binding(1) var<storage, read>       gi     : array<f32>;
@group(0) @binding(2) var<storage, read_write> logits : array<f32>;
@group(0) @binding(3) var<uniform>             push   : Push;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    if (row >= push.n_experts) {
        return;
    }
    let row_base = row * push.hidden;

    var acc: f32 = 0.0;
    var k: u32 = tid;
    loop {
        if (k >= push.hidden) { break; }
        acc = acc + w[row_base + k] * gi[k];
        k = k + 256u;
    }

    shared_sum[tid] = acc;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        logits[row] = shared_sum[0];
    }
}
