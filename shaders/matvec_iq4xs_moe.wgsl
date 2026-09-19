// shaders/matvec_iq4xs_moe.wgsl
//
// IQ4_XS fused matvec with per-expert offset — Pascal-safe, no f16, no subgroups.
//
// This is the MoE variant of `matvec_iq4xs_correct.wgsl`. The layout and
// decode arithmetic are byte-for-byte identical to the canonical kernel;
// the only difference is an additional `expert_offset_words` push field
// that biases the row index into the W buffer. That lets a single big
// buffer holding all 128 experts of a layer be sliced per dispatch by
// just adjusting the push constant — no per-expert sub-buffer binding,
// no rebinding, no GPU↔CPU round trip.
//
// Per IQ4_XS block (136 bytes = 34 u32 words):
//   bytes 0..1     d         f16  super-block scale
//   bytes 2..3     scales_h  u16  high 2 bits of 8 sub-block scales
//   bytes 4..7     scales_l  u8x4 low 4 bits of 8 sub-block scales
//   bytes 8..135   qs        u8x128 4-bit quant indices (2 per byte)
//
// HONEST GAP: this shader assumes the per-expert chunk is contiguous and
// that experts are the OUTERMOST dim of the packed tensor — i.e. expert e
// of `ffn_gate_up_exps[hidden, 1408, 128]` lives at byte offset
// `e * (1408 * blocks_per_row(2816) * 136)` from the start of W. That
// matches llama.cpp's `llm_build_moe_ffn` layout for `ffn_gate_up_exps`
// (cf. `ggml-cuda/cpy.cu` block-major MoE access). If the GGUF in hand
// turns out to use a different packing (experts as innermost stride, or
// interleaved), the host-side `expert_offset_words` calculation must be
// updated accordingly. Decode math here is layout-agnostic.

struct Push {
    K: u32,                    // input vector length (hidden = 2816)
    N_rows_total: u32,         // total output rows in this dispatch (sanity)
    row_offset: u32,            // first row index in this dispatch
    expert_offset_words: u32,   // bias into W in u32-word units
};

const BLOCK_SIZE: u32 = 256u;
const WORDS_PER_BLOCK: u32 = 34u;   // 136 bytes / 4

@group(0) @binding(0) var<storage, read>       W   : array<u32>;
@group(0) @binding(1) var<storage, read>       X   : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y   : array<f32>;
@group(0) @binding(3) var<uniform>             lut : array<vec4<f32>, 4>;
@group(0) @binding(4) var<uniform>             push: Push;

var<workgroup> shared_sum: array<f32, 256>;

fn lut_lookup(idx: u32) -> f32 {
    let group = idx >> 2u;
    let lane  = idx & 3u;
    let v = lut[group];
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

    let blocks_per_row = (push.K + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    // expert_offset_words biases the WHOLE row offset — every row in the
    // shader effectively sees a sub-buffer starting at expert_offset_words.
    let row_word_off = push.expert_offset_words + row * blocks_per_row * WORDS_PER_BLOCK;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
        let bo = row_word_off + b * WORDS_PER_BLOCK;

        let w0 = W[bo];
        let d = unpack2x16float(w0).x;
        let scales_h = (w0 >> 16u) & 0xFFFFu;

        let w1 = W[bo + 1u];

        let ib = tid >> 5u;
        let sl_byte_idx = ib >> 1u;
        let sl_byte = (w1 >> (sl_byte_idx * 8u)) & 0xFFu;
        let sl_nib_pos = ib & 1u;
        var scale_low: u32;
        if (sl_nib_pos == 0u) {
            scale_low = sl_byte & 0xFu;
        } else {
            scale_low = (sl_byte >> 4u) & 0xFu;
        }
        let scale_high = (scales_h >> (ib * 2u)) & 0x3u;
        let scale_6bit_u = (scale_high << 4u) | scale_low;
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

        let w_val = d * f32(scale_6bit) * kv;
        let k = b * BLOCK_SIZE + tid;
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
