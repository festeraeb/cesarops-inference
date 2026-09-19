requires f16;

struct Params {
    num_elements : u32,
};

@group(0) @binding(0)
var<storage, read> quant : array<u32>;

@group(0) @binding(1)
var<storage, read_write> output : array<f16>;

@group(0) @binding(2)
var<uniform> params : Params;

// IQ4_XS nonlinear codebook — from ggml-quants.c kvalues_iq4nl[16]
// These are importance-weighted, NOT linear [-8..7].
// VERIFY against your exact llama.cpp revision.
const IQ4_XS_TABLE : array<f16, 16> = array<f16, 16>(
    f16(-1.0000),
    f16(-0.6962),
    f16(-0.5251),
    f16(-0.3949),
    f16(-0.2844),
    f16(-0.1848),
    f16(-0.0911),
    f16( 0.0000),
    f16( 0.0796),
    f16( 0.1609),
    f16( 0.2461),
    f16( 0.3379),
    f16( 0.4407),
    f16( 0.5626),
    f16( 0.7230),
    f16( 1.0000)
);

// ------------------------------------------------------------------
// Raw byte addressing helpers
// ------------------------------------------------------------------

fn load_u8(byte_offset : u32) -> u32 {
    let word = quant[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn load_u16(byte_offset : u32) -> u32 {
    return load_u8(byte_offset)
        | (load_u8(byte_offset + 1u) << 8u);
}

fn load_f16(byte_offset : u32) -> f16 {
    return unpack2x16float(load_u16(byte_offset)).x;
}

// ------------------------------------------------------------------
// IQ4_XS scale decode
// ------------------------------------------------------------------
//
// block_iq4_xs layout (24 bytes per 32 values):
//   0..1   : d (f16 global scale)
//   2..3   : scales_h (high bits of sub-block scales)
//   4..7   : scales_l (4 bytes, one per sub-block)
//   8..23  : qs (16 bytes, 32 nibbles packed)
//
// Sub-block scale reconstruction:
//   scale_i = (high_bit << 4) | low_nibble
//   scale = scale_i + 1  (positive multiplier)
//
// TODO: Verify exact reconstruction against your llama.cpp commit's
// dequantize_row_iq4_xs(). IQ formats change between revisions.
//
fn decode_scale(
    scales_h : u32,
    scales_l : u32,
    sub_block : u32
) -> f16 {
    let low  = scales_l & 0x0fu;
    let high = (scales_h >> sub_block) & 0x1u;
    let scale_i = (high << 4u) | low;
    // Positive multiplier — NOT signed offset.
    return f16(scale_i + 1u);
}

// ------------------------------------------------------------------
// Main kernel
// ------------------------------------------------------------------

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id)
    gid : vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.num_elements) {
        return;
    }

    let block_idx = idx >> 5u;
    let in_block  = idx & 31u;
    let sub_block = in_block >> 3u;

    let base = block_idx * 24u;

    let d = load_f16(base);

    let scales_h = load_u16(base + 2u);
    let scales_l = load_u8(base + 4u + sub_block);

    let scale = decode_scale(
        scales_h,
        scales_l,
        sub_block
    );

    let qbyte =
        load_u8(base + 8u + (in_block >> 1u));

    let q =
        select(
            qbyte & 0x0fu,
            (qbyte >> 4u) & 0x0fu,
            (in_block & 1u) != 0u
        );

    let value =
        d *
        scale *
        IQ4_XS_TABLE[q];

    output[idx] = value;
}
