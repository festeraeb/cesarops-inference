//! CPU reference for ggml's IQ4_XS × Q8_K fused dot product.
//!
//! llama.cpp `ggml_mul_mat` quantizes the f32 input to Q8_K, then calls
//! `ggml_vec_dot_iq4_xs_q8_K` per output row. Our WGSL matvec dequantizes
//! IQ4 to f32 and dots against f32 input — numerically different.

const QK_K: usize = 256;
const KVALUES_IQ4NL: [i32; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const BLOCK_IQ4_XS_BYTES: usize = 136; // 2 + 2 + 4 + 128
const BLOCK_Q8_K_BYTES: usize = 4 + QK_K + (QK_K / 16) * 2; // 292

fn f16_bytes_to_f32(lo: u8, hi: u8) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
}

/// Port of `dequantize_row_iq4_xs` from ggml-quants.c — one column (`k` floats).
pub fn dequantize_row_iq4_xs_col(col_bytes: &[u8], k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0);
    let nb = k / QK_K;
    assert_eq!(col_bytes.len(), nb * BLOCK_IQ4_XS_BYTES);
    let mut y = vec![0.0f32; k];
    let mut y_off = 0usize;
    for ibl in 0..nb {
        let off = ibl * BLOCK_IQ4_XS_BYTES;
        let d = f16_bytes_to_f32(col_bytes[off], col_bytes[off + 1]);
        let mut qs_off = 8usize;
        for ib in 0..QK_K / 32 {
            let sl_byte = col_bytes[off + 4 + ib / 2];
            let scale_low = if ib % 2 == 0 {
                sl_byte & 0x0f
            } else {
                (sl_byte >> 4) & 0x0f
            };
            let scales_h = u16::from_le_bytes([col_bytes[off + 2], col_bytes[off + 3]]);
            let scale_high = ((scales_h >> (ib * 2)) & 0x03) as u8;
            let ls = ((scale_high << 4) | scale_low) as i32 - 32;
            let dl = d * ls as f32;
            for j in 0..16 {
                let qs_byte = col_bytes[off + qs_off + j];
                y[y_off + j] = dl * KVALUES_IQ4NL[(qs_byte & 0x0f) as usize] as f32;
                y[y_off + j + 16] = dl * KVALUES_IQ4NL[(qs_byte >> 4) as usize] as f32;
            }
            y_off += 32;
            qs_off += 16;
        }
    }
    y
}

/// Max abs diff between bridge flat dequant and ggml per-column dequant.
pub fn max_dequant_col_diff(bridge_flat: &[f32], raw: &[u8], k: usize, n: usize) -> (f32, usize, usize) {
    let blocks_per_col = (k + QK_K - 1) / QK_K;
    let row_bytes = blocks_per_col * BLOCK_IQ4_XS_BYTES;
    let mut worst = 0.0f32;
    let mut worst_out = 0usize;
    let mut worst_in = 0usize;
    for out in 0..n {
        let col = &raw[out * row_bytes..(out + 1) * row_bytes];
        let ggml = dequantize_row_iq4_xs_col(col, k);
        for in_i in 0..k {
            let b = bridge_flat[out * k + in_i];
            let g = ggml[in_i];
            let d = (b - g).abs();
            if d > worst {
                worst = d;
                worst_out = out;
                worst_in = in_i;
            }
        }
    }
    (worst, worst_out, worst_in)
}

/// Mirrors `quantize_row_q8_K_ref` from ggml-quants.c.
pub fn quantize_row_q8_k(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    let mut out = vec![0u8; nb * BLOCK_Q8_K_BYTES];
    for (ibl, chunk) in x.chunks(QK_K).enumerate() {
        let base = ibl * BLOCK_Q8_K_BYTES;
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &xi in chunk {
            let ax = xi.abs();
            if ax > amax {
                amax = ax;
                max = xi;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127.0f32 / max;
        let mut qs = [0i8; QK_K];
        for (j, &xi) in chunk.iter().enumerate() {
            qs[j] = ((iscale * xi).round() as i32).clamp(-128, 127) as i8;
        }
        let d = 1.0f32 / iscale;
        out[base..base + 4].copy_from_slice(&d.to_le_bytes());
        for (j, &q) in qs.iter().enumerate() {
            out[base + 4 + j] = q as u8;
        }
        for j in 0..QK_K / 16 {
            let mut sum = 0i16;
            for ii in 0..16 {
                sum += qs[j * 16 + ii] as i16;
            }
            out[base + 4 + QK_K + j * 2..base + 4 + QK_K + j * 2 + 2]
                .copy_from_slice(&sum.to_le_bytes());
        }
    }
    out
}

/// One output row: `weight_col` is `k/256` IQ4_XS blocks (136 bytes each).
/// `q8` is `quantize_row_q8_k(input)` for the same `k`.
pub fn vec_dot_iq4_xs_q8_k(weight_col: &[u8], q8: &[u8], k: usize) -> f32 {
    assert_eq!(k % QK_K, 0);
    let nb = k / QK_K;
    assert_eq!(weight_col.len(), nb * BLOCK_IQ4_XS_BYTES);
    assert_eq!(q8.len(), nb * BLOCK_Q8_K_BYTES);

    let mut sumf = 0.0f32;
    for ibl in 0..nb {
        let woff = ibl * BLOCK_IQ4_XS_BYTES;
        let yoff = ibl * BLOCK_Q8_K_BYTES;

        let d4 = f16_bytes_to_f32(weight_col[woff], weight_col[woff + 1]);
        let d8 = f32::from_le_bytes(q8[yoff..yoff + 4].try_into().unwrap());
        let d4d8 = d4 * d8;

        let mut h = u16::from_le_bytes([weight_col[woff + 2], weight_col[woff + 3]]);
        let qs = &weight_col[woff + 8..woff + 136];
        let q8s = &q8[yoff + 4..yoff + 4 + QK_K];

        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        for ib in (0..QK_K / 32).step_by(2) {
            let sl_byte = weight_col[woff + 4 + ib / 2];
            let ls1 = (sl_byte & 0x0f) | (((h << 4) & 0x30) as u8);
            let ls2 = (sl_byte >> 4) | (((h << 2) & 0x30) as u8);
            h >>= 4;

            let d1 = d4d8 * (ls1 as i32 - 32) as f32;
            let d2 = d4d8 * (ls2 as i32 - 32) as f32;

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..16 {
                sumi1 += q8s[q8_off + j] as i8 as i32 * KVALUES_IQ4NL[(qs[qs_off + j] & 0x0f) as usize];
                sumi2 += q8s[q8_off + j + 16] as i8 as i32 * KVALUES_IQ4NL[(qs[qs_off + j] >> 4) as usize];
            }
            sumf += d1 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..16 {
                sumi1 += q8s[q8_off + j] as i8 as i32 * KVALUES_IQ4NL[(qs[qs_off + j] & 0x0f) as usize];
                sumi2 += q8s[q8_off + j + 16] as i8 as i32 * KVALUES_IQ4NL[(qs[qs_off + j] >> 4) as usize];
            }
            sumf += d2 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;
        }
    }
    sumf
}

/// Full matvec matching ggml `mul_mat` for IQ4_XS weights × f32 input.
pub fn matvec_iq4_xs_q8_k(weight: &[u8], input: &[f32], k: usize, n: usize) -> Vec<f32> {
    let blocks_per_row = (k + QK_K - 1) / QK_K;
    let row_bytes = blocks_per_row * BLOCK_IQ4_XS_BYTES;
    let q8 = quantize_row_q8_k(input);
    let mut y = vec![0.0f32; n];
    for out in 0..n {
        let col = &weight[out * row_bytes..(out + 1) * row_bytes];
        y[out] = vec_dot_iq4_xs_q8_k(col, &q8, k);
    }
    y
}
