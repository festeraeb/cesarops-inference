// shaders/matvec_iq4nl_correct.wgsl
//
// IQ4_NL fused matvec — Pascal-safe.
//
// Layout per IQ4_NL block (18 bytes):
//   bytes 0..1   d  fp16 block scale
//   bytes 2..17  qs u8x16 — sixteen bytes, each holding two 4-bit nibbles
//
// Block-size = 32 elements. No sub-block scales, no scales_h. Decode is
// just `d * KVALUES[nibble]`.
//
// 32 elements / block × 8 blocks per "tile" = 256 = workgroup size. We
// accumulate one block's contribution per thread per outer iteration over
// blocks_per_row.

struct Push {
    K: u32,
    N_rows_total: u32,
    row_offset: u32,
};

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 18u;
const THREADS: u32 = 256u;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    let row = wid.x + push.row_offset;
    let tid = lid.x;

    let blocks_per_row = (push.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Block byte offset for this row (in u8, but we read u32 so divide by 4 later).
    // Each row uses blocks_per_row * 18 bytes. To avoid odd alignment headaches
    // we walk in u8 indices and synthesize u32 reads from W (which is the
    // same buffer reinterpreted as u32). Caller MUST ensure each row starts
    // on a 4-byte boundary; for IQ4_NL this requires the per-row block count
    // to make the row stride a multiple of 4 bytes, which is true for all
    // tensor shapes we target (rows of 2816 → 88 blocks → 1584 bytes ✓).
    let row_byte_off = row * blocks_per_row * BYTES_PER_BLOCK;

    var acc: f32 = 0.0;

    // Each thread handles 1 element per block, advancing block by block.
    // 256 threads × 32 elements/block isn't 1:1; instead we let each thread
    // own a fixed `tid % 32` slot and walk blocks at stride 256/32 = 8.
    let elem_in_block = tid & 31u;
    let block_in_tile = tid >> 5u;     // 0..7
    let tile_count    = (blocks_per_row + 7u) / 8u;

    for (var t: u32 = 0u; t < tile_count; t = t + 1u) {
        let b = t * 8u + block_in_tile;
        if (b >= blocks_per_row) { continue; }

        let block_byte = row_byte_off + b * BYTES_PER_BLOCK;
        let block_word = block_byte >> 2u;
        let block_bit  = (block_byte & 3u) * 8u;

        // Word 0 of the block contains: byte 0,1 = d; byte 2,3 = qs[0],qs[1]
        // assuming alignment was preserved; otherwise we have to gather across
        // two u32 reads. Build the two needed values via a generic byte fetch.
        let d_byte0 = read_byte(block_byte + 0u);
        let d_byte1 = read_byte(block_byte + 1u);
        let d_bits = d_byte0 | (d_byte1 << 8u);
        let d = unpack2x16float(d_bits).x;

        // IQ4_NL nibble layout (llama.cpp dequantize_row_iq4_nl):
        //   qs[j] low  → element j
        //   qs[j] high → element j+16
        // (NOT interleaved even/odd within the first 16.)
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

// Read one byte from W[] at an arbitrary byte offset.
// W is the tensor reinterpreted as u32; we synthesize the byte via word load + shift.
fn read_byte(byte_idx: u32) -> u32 {
    let word = W[byte_idx >> 2u];
    let shift = (byte_idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}
