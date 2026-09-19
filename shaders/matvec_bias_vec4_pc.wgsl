// Fused matvec + bias + vec4 + push-constant.
// output[n] = sum_k W[n*K + k] * input[k] + bias[n]
//
// Pattern mirrors matvec_vec4_pc.wgsl (componentwise vec4 multiply, lane-sum
// at the end). Bias is read as scalar f32 indexed by row n.
//
// Drafted by Gemma-4-MoE (round 4) — spun out on the vec4 accumulator
// pattern and ran out of tokens explaining itself. Polisher (Claude) wrote
// this version; the reasoning is captured in research_log/lessons_learned.md
// under "model-behavior,gemma4,vec4-accumulator".

struct Params {
    N: u32,
    K: u32,
    K_vec4: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;

var<push_constant> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.N) { return; }

    let w_row_base = n * params.K_vec4;

    // 4-way f32 accumulator via vec4 — lets the compiler issue four
    // independent FMA chains, hiding the dependent-add latency on Pascal.
    var acc: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

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

    output[n] = acc.x + acc.y + acc.z + acc.w + bias[n];
}
