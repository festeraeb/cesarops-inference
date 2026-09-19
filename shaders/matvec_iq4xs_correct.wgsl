// shaders/matvec_iq4xs_correct.wgsl
//
// IQ4_XS fused matvec — Pascal-safe, no f16, no subgroups.
//
// Layout per IQ4_XS block (136 bytes = 34 u32 words):
//   bytes 0..1     d         f16  super-block scale
//   bytes 2..3     scales_h  u16  high 2 bits of 8 sub-block scales
//   bytes 4..7     scales_l  u8x4 low 4 bits of 8 sub-block scales (2 nibbles/byte)
//   bytes 8..135   qs        u8x128 4-bit quant indices (2 per byte)
//
// The 6-bit signed scale for sub-block ib is reconstructed as:
//   scale = ((scale_high(ib) << 4) | scale_low(ib)) - 32   // signed offset
// And the dequantized weight at position i is:
//   w[i] = d * scale_6bit * KVALUES[qs_nibble(i)]
//
// One workgroup = one output row. 256 threads cooperatively walk the K
// dimension. Each thread t handles element-index t inside every block of
// the row, accumulates a partial dot product, then a tree reduction lands
// the final value in Y[row].

struct Push {
    K: u32,           // input vector length (= weights per output row)
    N_rows_total: u32, // total output rows in this dispatch (sanity check)
    row_offset: u32,   // first row index for this dispatch
};

const BLOCK_SIZE: u32 = 256u;
const WORDS_PER_BLOCK: u32 = 34u;   // 136 bytes / 4
const THREADS: u32 = 256u;

@group(0) @binding(0) var<storage, read>       W   : array<u32>;
@group(0) @binding(1) var<storage, read>       X   : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(3) var<uniform>             lut : array<vec4<f32>, 4>;
@group(0) @binding(4) var<uniform>             push: Push;

var<workgroup> shared_sum: array<f32, 256>;

// IQ4 codebook lookup, indexed 0..15. We pack the 16 values into 4×vec4
// uniforms because WGSL uniforms can't hold a plain `array<f32, 16>` with
// the tight stride we'd want — vec4 gives us 16-byte alignment for free.
fn lut_lookup(idx: u32) -> f32 {
    let group = idx >> 2u;
    let lane  = idx & 3u;
    let v = lut[group];
    // Manual scalar select so we don't depend on dynamic indexing of vec4.
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    let row = wid.x + push.row_offset;
    let tid = lid.x;

    // Number of full blocks per row. K is always a multiple of 256 in the
    // tensors we target (Gemma-4 hidden = 2816, inner = 1408, etc.); if it
    // isn't, the trailing partial block is silently truncated, which the
    // host caller is responsible for guarding against.
    let blocks_per_row = (push.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let row_word_off = row * blocks_per_row * WORDS_PER_BLOCK;

    var acc: f32 = 0.0;

    // Each thread t handles position `t` (0..255) inside every block of the row.
    for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
        let bo = row_word_off + b * WORDS_PER_BLOCK;

        // word 0: low half = d (f16), high half = scales_h (u16)
        let w0 = W[bo];
        let d = unpack2x16float(w0).x;
        let scales_h = (w0 >> 16u) & 0xFFFFu;

        // word 1: scales_l[4] — 4 bytes packing 8 nibbles
        let w1 = W[bo + 1u];

        // sub-block index for element t (0..7)
        let ib = tid >> 5u;

        // scale_low: nibble within scales_l. byte = ib/2, nibble = ib%2.
        let sl_byte_idx = ib >> 1u;
        let sl_byte = (w1 >> (sl_byte_idx * 8u)) & 0xFFu;
        let sl_nib_pos = ib & 1u;
        var scale_low: u32;
        if (sl_nib_pos == 0u) {
            scale_low = sl_byte & 0xFu;
        } else {
            scale_low = (sl_byte >> 4u) & 0xFu;
        }

        // scale_high: 2 bits from scales_h
        let scale_high = (scales_h >> (ib * 2u)) & 0x3u;
        let scale_6bit_u = (scale_high << 4u) | scale_low;
        // 6-bit signed: subtract 32 to recenter (matches CPU reference).
        let scale_6bit = i32(scale_6bit_u) - 32;

        // qs layout per ggml: sub-block ib has qs[0..15]; low nibbles → y[0..15],
        // high nibbles → y[16..31] within the 32-element slice.
        let pos = tid & 31u;
        let j = pos & 15u;
        let qs_byte_idx = ib * 16u + j;
        let qs_word = bo + 2u + (qs_byte_idx >> 2u);
        let qs_shift = (qs_byte_idx & 3u) * 8u;
        let qs_byte = (W[qs_word] >> qs_shift) & 0xFFu;
        var q_nib: u32;
        if (pos < 16u) {
            q_nib = qs_byte & 0xFu;
        } else {
            q_nib = (qs_byte >> 4u) & 0xFu;
        }
        let kv = lut_lookup(q_nib);

        // dequantized weight, mapped to global K index = b*256 + tid
        let w_val = d * f32(scale_6bit) * kv;
        let k = b * BLOCK_SIZE + tid;
        // K is required to be a multiple of 256 for our tensors; guard anyway.
        if (k < push.K) {
            acc = acc + w_val * X[k];
        }
    }

    shared_sum[tid] = acc;
    workgroupBarrier();

    // Tree reduction: 256 → 1.
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
