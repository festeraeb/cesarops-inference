// shaders/matvec_iq4xs_opt.wgsl
//
// Optimization variant of matvec_iq4xs_correct.wgsl. Same numeric semantics,
// same bindings, same buffer layout — must land bit-identical results to
// the reference (validate_dequant cross-checks every change).
//
// Changes vs reference:
//   * Hoist all (per-thread, block-invariant) bit-decoding outside the
//     block loop: ib, scales_h_shift, sl_byte_idx, sl_nib_pos, q_byte_lane,
//     q_nib_shift, scale_low_mask/scale_low_shift.
//   * Single-FMA inner loop body: acc = fma(weight, X[k], acc).
//   * Reduction unchanged — already shared-memory tree, Pascal-safe.

struct Push {
    K: u32,
    N_rows_total: u32,
    row_offset: u32,
};

const BLOCK_SIZE: u32 = 256u;
const WORDS_PER_BLOCK: u32 = 34u;

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
    let row_word_off = row * blocks_per_row * WORDS_PER_BLOCK;

    // ── Hoisted per-thread invariants ───────────────────────────────────
    let ib = tid >> 5u;
    let scales_h_shift = ib * 2u;       // bits in scales_h to shift
    let sl_byte_idx = ib >> 1u;          // which byte of scales_l holds this thread's nibble
    let sl_byte_shift = sl_byte_idx * 8u;
    let sl_nib_pos = ib & 1u;            // 0 → low nibble, 1 → high nibble
    var scale_low_shift: u32;
    var scale_low_mask: u32;
    if (sl_nib_pos == 0u) {
        scale_low_shift = 0u;
        scale_low_mask  = 0x0Fu;
    } else {
        scale_low_shift = 4u;
        scale_low_mask  = 0xF0u;
    }
    let q_byte_lane = ib * 16u + (tid & 15u);
    let q_word_offset = q_byte_lane >> 2u;
    let q_word_byte_shift = (q_byte_lane & 3u) * 8u;
    let q_nib_high = (tid & 31u) >= 16u;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
        let bo = row_word_off + b * WORDS_PER_BLOCK;

        let w0 = W[bo];
        let d = unpack2x16float(w0).x;
        let scales_h = (w0 >> 16u) & 0xFFFFu;
        let w1 = W[bo + 1u];

        let sl_byte = (w1 >> sl_byte_shift) & 0xFFu;
        let scale_low = (sl_byte & scale_low_mask) >> scale_low_shift;
        let scale_high = (scales_h >> scales_h_shift) & 0x3u;
        let scale_6bit = i32((scale_high << 4u) | scale_low) - 32;

        let qs_byte = (W[bo + 2u + q_word_offset] >> q_word_byte_shift) & 0xFFu;
        var q_nib: u32;
        if (q_nib_high) {
            q_nib = (qs_byte >> 4u) & 0xFu;
        } else {
            q_nib = qs_byte & 0xFu;
        }

        let weight = d * f32(scale_6bit) * lut_lookup(q_nib);
        let k = b * BLOCK_SIZE + tid;
        if (k < push.K) {
            acc = fma(weight, X[k], acc);
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
