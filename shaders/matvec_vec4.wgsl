// WGSL Matrix-Vector Multiply with vec4 loads.
//
// output[n] = sum_k W[n*K + k] * input[k]
//
// W is [N × K] row-major (GGUF native layout, no transpose).
// Input is loaded as vec4<f32> for full 128-bit memory bus utilization.
// Weights are read as vec4<f32> too — requires K to be a multiple of 4 (which it is
// for all transformer hidden_dim values: 1024, 1536, 2048, 4096, 5120, etc.)
//
// Each thread computes one output element. The win over scalar matvec is bandwidth:
// vec4 loads pull 128 bits per access vs 32 bits for scalar f32. On HBM2 (P100),
// this is the difference between ~25% and ~80% of peak bandwidth.
//
// Compatibility: requires K % 4 == 0. The fallback scalar matvec.wgsl handles any K.

struct Params {
    N: u32,        // Output dimension
    K: u32,        // Input dimension
    K_vec4: u32,   // K / 4
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.N) { return; }

    // Each output row n reads K elements from weights starting at n*K.
    // In vec4 indexing: starts at n * K_vec4.
    let w_row_base = n * params.K_vec4;

    // 4-way accumulator (preserves precision better than a single f32 sum
    // and lets the compiler issue 4 independent FMA chains)
    var acc: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < params.K_vec4; i = i + 1u) {
        let x = input[i];
        let w = weights[w_row_base + i];
        // Componentwise multiply-accumulate: acc.x += x.x * w.x, etc.
        // The horizontal sum at the end matches the scalar reference exactly.
        acc = acc + x * w;
    }

    output[n] = acc.x + acc.y + acc.z + acc.w;
}
