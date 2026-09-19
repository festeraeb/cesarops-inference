// shaders/matvec_f32_rowmajor.wgsl
//
// Row-major fp32 matvec: y[row] = sum_k W[row * K + k] * x[k]
//
// One workgroup per output row; 256 threads cooperatively reduce K.
// Same shape as router_matvec_f32.wgsl but generic over K and N (not
// hardcoded to 128 router experts), so we can use it for any fp32 matmul
// — most importantly the LM head (W[262144 × 2816] @ hidden[2816]).
//
// Pascal-safe: no f16, no subgroups, shared-memory tree reduction.

struct Push {
    k: u32,        // input vector length (cols of W)
    n: u32,        // number of output rows of W (rows produced)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       w   : array<f32>;
@group(0) @binding(1) var<storage, read>       x   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y   : array<f32>;
@group(0) @binding(3) var<uniform>             push: Push;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    if (row >= push.n) {
        return;
    }
    let row_base = row * push.k;

    var acc: f32 = 0.0;
    var k: u32 = tid;
    loop {
        if (k >= push.k) { break; }
        acc = fma(w[row_base + k], x[k], acc);
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
        y[row] = shared_sum[0];
    }
}
