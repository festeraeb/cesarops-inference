// Fused IQ4_XS decode + matvec
// One workgroup = one output row
// 256 threads cooperatively accumulate
// Decode nibbles via kvalues_iq4nl[16], immediate FMA into f32
// No intermediate fp16 staging buffer

struct PC {
    row_stride : i32,
    input_stride : i32,
};

@group(0) @binding(0) var<storage, read> q : array<u32>;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> pc : PC;

// IQ4_XS nonlinear codebook — kvalues_iq4nl[16]
// VERIFY against your exact llama.cpp revision
const KVALUES : array<f32, 16> = array<f32, 16>(
    -1.0000, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911,  0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,
     0.4407,  0.5626,  0.7230,  1.0000
);

var<workgroup> partial : array<f32, 256>;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) tid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
) {
    let t = tid.x;
    let row = wid.x;
    var acc : f32 = 0.0;

    let row_stride = u32(pc.row_stride);

    for (var b = t; b < row_stride; b = b + 256u) {

        // Each block = 16 bytes = 32 nibbles = 4 u32s
        let base = (row * row_stride + b) * 4u;

        for (var i = 0u; i < 4u; i = i + 1u) {
            let word = q[base + i];

            // 8 nibbles per u32
            for (var n = 0u; n < 8u; n = n + 1u) {
                let nibble = (word >> (n * 4u)) & 0xFu;
                let col = b * 32u + i * 8u + n;
                acc = acc + KVALUES[nibble] * x[col];
            }
        }
    }

    partial[t] = acc;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            partial[t] = partial[t] + partial[t + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (t == 0u) {
        y[row] = partial[0];
    }
}
