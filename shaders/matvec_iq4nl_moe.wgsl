// shaders/matvec_iq4nl_moe.wgsl
//
// IQ4_NL fused matvec with per-expert byte offset — Pascal-safe.
//
// MoE variant of `matvec_iq4nl_correct.wgsl`. Same byte arithmetic; adds
// `expert_offset_bytes` push field that biases the per-row byte offset.
// IQ4_NL block stride is 18 bytes (NOT 4-aligned per block, but the per-
// row stride for the shapes we target — 2816 / 32 * 18 = 1584 — is
// 4-aligned, and per-expert chunks land on 4-byte boundaries because the
// row count is even). We still use byte-level reads via `read_byte` to
// stay tolerant of any sub-row offset.
//
// HONEST GAP: same as the IQ4_XS MoE variant — assumes experts are the
// OUTERMOST dim of `ffn_down_exps[704, 2816, 128]`. If llama.cpp turns
// out to interleave experts at the innermost stride, the host-side
// `expert_offset_bytes` calc must change. Decode math is layout-agnostic.
// TODO: cross-check against llama.cpp `llm_build_moe_ffn` ground truth.

struct Push {
    K: u32,
    N_rows_total: u32,
    row_offset: u32,
    expert_offset_bytes: u32,   // bias into W in u8 units
};

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 18u;

@group(0) @binding(0) var<storage, read>       W   : array<u32>;
@group(0) @binding(1) var<storage, read>       X   : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(3) var<uniform>             lut : array<vec4<f32>, 4>;
@group(0) @binding(4) var<uniform>             push: Push;

var<workgroup> shared_sum: array<f32, 256>;

fn lut_lookup(idx: u32) -> f32 {
    let g = idx >> 2u;
    let l = idx & 3u;
    let v = lut[g];
    if (l == 0u) { return v.x; }
    if (l == 1u) { return v.y; }
    if (l == 2u) { return v.z; }
    return v.w;
}

fn read_byte(byte_idx: u32) -> u32 {
    let word = W[byte_idx >> 2u];
    let shift = (byte_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    let row = wid.x + push.row_offset;
    let tid = lid.x;

    let blocks_per_row = (push.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let row_byte_off = push.expert_offset_bytes + row * blocks_per_row * BYTES_PER_BLOCK;

    var acc: f32 = 0.0;

    let elem_in_block = tid & 31u;
    let block_in_tile = tid >> 5u;
    let tile_count    = (blocks_per_row + 7u) / 8u;

    for (var t: u32 = 0u; t < tile_count; t = t + 1u) {
        let b = t * 8u + block_in_tile;
        if (b >= blocks_per_row) { continue; }

        let block_byte = row_byte_off + b * BYTES_PER_BLOCK;

        let d_byte0 = read_byte(block_byte + 0u);
        let d_byte1 = read_byte(block_byte + 1u);
        let d_bits = d_byte0 | (d_byte1 << 8u);
        let d = unpack2x16float(d_bits).x;

        // IQ4_NL: qs[j] low → elem j; qs[j] high → elem j+16 (llama.cpp).
        var qs_byte_idx: u32;
        if (elem_in_block < 16u) {
            qs_byte_idx = elem_in_block;
        } else {
            qs_byte_idx = elem_in_block - 16u;
        }
        let qs_byte = read_byte(block_byte + 2u + qs_byte_idx);
        var q_nib: u32;
        if (elem_in_block < 16u) {
            q_nib = qs_byte & 0xFu;
        } else {
            q_nib = (qs_byte >> 4u) & 0xFu;
        }

        let w_val = d * lut_lookup(q_nib);
        let k = b * BLOCK_SIZE + elem_in_block;
        if (k < push.K) {
            acc = acc + w_val * X[k];
        }
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
        Y[row] = shared_sum[0];
    }
}
