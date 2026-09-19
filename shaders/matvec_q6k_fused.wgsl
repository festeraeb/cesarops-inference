// Fused Q6_K dequant + matvec. Eliminates the dequant -> staging buffer ->
// matvec roundtrip. Per output row: for each Q6_K super-block in the row,
// dequant 256 elements following llama.cpp's two-halves / inner-32 pattern
// AND accumulate the dot product against the matching input slice — never
// writing the dequanted f32s to memory.
//
// W is [N x K] row-major in GGUF native layout. Each row n contains
// K_blocks = K / 256 Q6_K super-blocks. Total bytes per row = K_blocks * 210.
//
// Verified against shaders/dequant_q6k.wgsl — same slot logic, same scale
// indexing. The novelty here is the inline multiply-accumulate.

struct Params {
    N: u32,         // Output dimension (rows of W)
    K: u32,         // Input dimension (cols of W) — informational
    K_blocks: u32,  // K / 256
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:   array<f32>;
@group(0) @binding(1) var<storage, read>       q6k:     array<u32>;  // raw block bytes packed as u32
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;

var<push_constant> params: Params;

// Read a single byte from q6k_data array<u32> at a given byte offset.
fn read_byte(byte_off: u32) -> u32 {
    let w = q6k[byte_off >> 2u];
    let s = (byte_off & 3u) * 8u;
    return (w >> s) & 0xFFu;
}

// IEEE754 half -> single conversion. Matches dequant_q6k.wgsl exactly.
fn fp16_to_f32(bits: u32) -> f32 {
    let s = (bits >> 15u) & 0x1u;
    let e = (bits >> 10u) & 0x1Fu;
    let m =  bits        & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) { return 0.0; }
        return select(-1.0, 1.0, s == 0u) * f32(m) * 5.96046447e-8;
    }
    if (e == 31u) { return select(-1.0, 1.0, s == 0u) * 65504.0; }
    return select(-1.0, 1.0, s == 0u) * pow(2.0, f32(e) - 15.0) * (1.0 + f32(m) / 1024.0);
}

fn signed_byte(b: u32) -> i32 {
    return i32(b) - select(0, 256, b >= 128u);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.N) { return; }

    var acc: f32 = 0.0;

    // Each super-block: 210 bytes covering 256 weights in row n.
    // Row n's first super-block starts at byte offset:
    //   row_byte_base = n * K_blocks * 210
    let row_byte_base = n * params.K_blocks * 210u;

    for (var b: u32 = 0u; b < params.K_blocks; b = b + 1u) {
        let blk = row_byte_base + b * 210u;
        let d = fp16_to_f32(read_byte(blk + 208u) | (read_byte(blk + 209u) << 8u));

        // Input slice for this block: input[b*256 .. b*256 + 256)
        let in_base = b * 256u;

        // Two halves of 128 elements.
        for (var half: u32 = 0u; half < 2u; half = half + 1u) {
            let ql_off = blk + half * 64u;          // 128 ql bytes per block, 64 per half
            let qh_off = blk + 128u + half * 32u;   // 64 qh bytes per block, 32 per half
            let sc_off = blk + 192u + half * 8u;    // 16 scale bytes per block, 8 per half
            let half_in = in_base + half * 128u;

            // Inner-32 loop: produces 4 outputs at l, l+32, l+64, l+96.
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let ql_a = read_byte(ql_off + l);
                let ql_b = read_byte(ql_off + l + 32u);
                let qh_b = read_byte(qh_off + l);

                // is = l / 16 within the half. Scales are interleaved across
                // the four slots: slot 0 -> sc[is], slot 1 -> sc[is+2],
                // slot 2 -> sc[is+4], slot 3 -> sc[is+6].
                let is = l / 16u;
                let s0 = f32(signed_byte(read_byte(sc_off + is)));
                let s1 = f32(signed_byte(read_byte(sc_off + is + 2u)));
                let s2 = f32(signed_byte(read_byte(sc_off + is + 4u)));
                let s3 = f32(signed_byte(read_byte(sc_off + is + 6u)));

                // Slot 0: ql_a low | qh shift 0 -> output position l
                // Slot 1: ql_b low | qh shift 2 -> output position l+32
                // Slot 2: ql_a high| qh shift 4 -> output position l+64
                // Slot 3: ql_b high| qh shift 6 -> output position l+96
                let q0 = i32((ql_a       & 0xFu) | (((qh_b >> 0u) & 0x3u) << 4u)) - 32;
                let q1 = i32((ql_b       & 0xFu) | (((qh_b >> 2u) & 0x3u) << 4u)) - 32;
                let q2 = i32(((ql_a >> 4u) & 0xFu) | (((qh_b >> 4u) & 0x3u) << 4u)) - 32;
                let q3 = i32(((ql_b >> 4u) & 0xFu) | (((qh_b >> 6u) & 0x3u) << 4u)) - 32;

                // Inline multiply-accumulate against input.
                acc = acc + d * s0 * f32(q0) * input[half_in + l];
                acc = acc + d * s1 * f32(q1) * input[half_in + l + 32u];
                acc = acc + d * s2 * f32(q2) * input[half_in + l + 64u];
                acc = acc + d * s3 * f32(q3) * input[half_in + l + 96u];
            }
        }
    }

    output[n] = acc;
}
