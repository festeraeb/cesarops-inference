// vec4 + push-constant matvec.
// output[n] = sum_k W[n*K + k] * input[k]
//
// W is [N x K] row-major (GGUF native). Loaded as vec4<f32> for 128-bit
// memory bus utilization on Pascal HBM2 / GDDR5. Params arrive via push
// constant — no uniform buffer write per dispatch.
//
// Constraint: K must be a multiple of 4. Caller routes around this when
// K % 4 != 0 (rare in practice — all transformer hidden_dims align).

struct Params {
    N: u32,       // Output dimension (rows of W)
    K: u32,       // Input dimension (cols of W) — informational
    K_vec4: u32,  // K / 4 (vec4 stride)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;

var<push_constant> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.N) { return; }

    let w_row_base = n * params.K_vec4;

    // 4-way f32 accumulator: lets the compiler issue four independent
    // FMA chains, which on Pascal hides the dependent-add latency.
    var acc: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // 4-iteration unroll (16 elements per pass)
    let kv = params.K_vec4;
    let kv4 = kv & ~3u;
    var i: u32 = 0u;
    while (i < kv4) {
        let x0 = input[i];
        let x1 = input[i + 1u];
        let x2 = input[i + 2u];
        let x3 = input[i + 3u];
        let w0 = weights[w_row_base + i];
        let w1 = weights[w_row_base + i + 1u];
        let w2 = weights[w_row_base + i + 2u];
        let w3 = weights[w_row_base + i + 3u];
        acc = acc + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        i = i + 4u;
    }
    while (i < kv) {
        acc = acc + input[i] * weights[w_row_base + i];
        i = i + 1u;
    }

    output[n] = acc.x + acc.y + acc.z + acc.w;
}
