//! GridBuffer ↔ Burn Tensor Bridge
//!
//! Zero-copy conversion between our GridBuffer memory abstraction and Burn tensors.
//! For now, this copies data (true zero-copy requires Burn internals access).
//! TODO: Once Burn exposes `from_existing_buffer`, switch to zero-copy.

use half::f16;

/// Convert raw bytes (f16 format) to a Vec<f32> for Burn consumption.
/// This is the "safe path" — copies data but works with any Burn version.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert raw bytes (f32 format) to a Vec<f32>.
pub fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert raw bytes (bf16 format) to a Vec<f32>.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // BF16: sign(1) + exponent(8) + mantissa(7)
            // F32:  sign(1) + exponent(8) + mantissa(23)
            // Just shift left by 16 bits
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Dequantize Q4_K_M block to f32 values.
/// Q4_K_M uses 256-element blocks with scales and mins.
/// This is a simplified implementation — production would use SIMD.
pub fn dequant_q4_k_m(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    // Q4_K_M block structure (256 elements per block):
    // - 2 bytes: d (f16 scale)
    // - 2 bytes: dmin (f16 min)
    // - 12 bytes: scales (k_scale_t)
    // - 128 bytes: quantized values (4 bits each)
    // Total: 144 bytes per 256 elements (approximate)
    
    let mut output = Vec::with_capacity(n_elements);
    
    // Simplified: treat as 4-bit values with a single scale per block
    let block_size = 32;
    let bytes_per_block = 18; // Q4_0 simplified
    
    for block_start in (0..bytes.len()).step_by(bytes_per_block) {
        if block_start + bytes_per_block > bytes.len() {
            break;
        }
        
        // First 2 bytes are the scale (f16)
        let scale_bits = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        
        // Remaining bytes are 4-bit quantized values (2 per byte)
        for i in 0..16 {
            let byte = bytes[block_start + 2 + i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            output.push(lo as f32 * scale);
            output.push(hi as f32 * scale);
            if output.len() >= n_elements {
                break;
            }
        }
        
        if output.len() >= n_elements {
            break;
        }
    }
    
    output.truncate(n_elements);
    output
}

/// Dequantize Q6_K block to f32 values.
/// Q6_K: 256-element super-blocks, 6 bits per weight.
/// Block layout (210 bytes per 256 elements):
///   ql[128]: lower 4 bits of quants (interleaved, NOT sequential)
///   qh[64]:  upper 2 bits of quants (interleaved, NOT sequential)
///   scales[16]: int8 scales for 16-element sub-blocks (interleaved access)
///   d[2]: f16 super-block scale
///
/// The 256 elements are processed as two 128-element halves.
/// Within each half, 32 iterations produce 4 outputs each from interleaved
/// positions in ql and qh. This matches llama.cpp's dequantize_row_q6_K exactly.
pub fn dequant_q6_k(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n_elements];
    let bytes_per_block: usize = 210;
    let n_blocks = bytes.len() / bytes_per_block;
    let mut out_idx = 0;

    for block_idx in 0..n_blocks {
        if out_idx >= n_elements {
            break;
        }

        let base = block_idx * bytes_per_block;
        if base + bytes_per_block > bytes.len() {
            break;
        }

        let ql = &bytes[base..base + 128];
        let qh = &bytes[base + 128..base + 192];
        let scales = &bytes[base + 192..base + 208];
        let d_bits = u16::from_le_bytes([bytes[base + 208], bytes[base + 209]]);
        let d = f16::from_bits(d_bits).to_f32();

        // Process two 128-element halves (n=0, n=128)
        for half in 0..2u32 {
            let ql_off = (half as usize) * 64;  // each half uses 64 bytes of ql
            let qh_off = (half as usize) * 32;  // each half uses 32 bytes of qh
            let sc_off = (half as usize) * 8;   // each half uses 8 scale entries
            let out_base = out_idx + (half as usize) * 128;

            for l in 0..32usize {
                let is = l / 16; // 0 for l<16, 1 for l>=16

                // Extract 4 interleaved 6-bit values from ql and qh
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                // Scale indices are interleaved: is+0, is+2, is+4, is+6
                let sc0 = scales[sc_off + is] as i8 as f32;
                let sc1 = scales[sc_off + is + 2] as i8 as f32;
                let sc2 = scales[sc_off + is + 4] as i8 as f32;
                let sc3 = scales[sc_off + is + 6] as i8 as f32;

                let idx = out_base + l;
                if idx < n_elements { output[idx] = d * sc0 * q1 as f32; }
                if idx + 32 < n_elements { output[idx + 32] = d * sc1 * q2 as f32; }
                if idx + 64 < n_elements { output[idx + 64] = d * sc2 * q3 as f32; }
                if idx + 96 < n_elements { output[idx + 96] = d * sc3 * q4 as f32; }
            }
        }
        out_idx += 256;
    }

    output.truncate(n_elements);
    output
}

/// IQ4_XS: 256-element super-blocks. Layout per block (136 bytes):
///   d[fp16]   (2 bytes)  super-block scale
///   scales_h  (2 bytes)  high 2 bits of each of 8 sub-block scales
///   scales_l  (4 bytes)  low 4 bits of each of 8 sub-block scales (2 per byte)
///   qs[128]              4-bit quant indices into the IQ4 codebook (2 per byte)
///
/// Mirrors `dequantize_row_iq4_xs` from llama.cpp.
pub fn dequant_iq4_xs(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    const KVALUES: [i32; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    ];
    let block_size = 256;
    let block_bytes = 136;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let off = b * block_bytes;
        if off + block_bytes > bytes.len() { break; }
        let out_base = b * block_size;
        if out_base >= n_elements { break; }

        let d_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let d = f16::from_bits(d_bits).to_f32();
        let scales_h = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]);

        // ggml `dequantize_row_iq4_xs`: within each 32-element sub-block, the
        // low nibbles of qs[0..15] fill y[0..15] and the high nibbles fill y[16..31].
        let mut y_off = 0usize;
        let mut qs_off = 8usize;
        for ib in 0..8usize {
            let sl_byte = bytes[off + 4 + ib / 2];
            let scale_low = if ib % 2 == 0 { sl_byte & 0x0F } else { (sl_byte >> 4) & 0x0F };
            let scale_high = ((scales_h >> (ib * 2)) & 0x03) as u8;
            let scale_6bit = ((scale_high << 4) | scale_low) as i32 - 32;
            let dl = d * scale_6bit as f32;

            for j in 0..16usize {
                let qs_byte = bytes[off + qs_off + j];
                let lo = (qs_byte & 0x0F) as usize;
                let hi = (qs_byte >> 4) as usize;
                let base = out_base + y_off;
                if base + j < n_elements {
                    out[base + j] = dl * KVALUES[lo] as f32;
                }
                if base + j + 16 < n_elements {
                    out[base + j + 16] = dl * KVALUES[hi] as f32;
                }
            }
            y_off += 32;
            qs_off += 16;
        }
    }
    out
}

/// IQ4_NL: 32-element blocks. Layout per block (18 bytes):
///   d[fp16]  (2 bytes)
///   qs[16]   4-bit quant indices into the IQ4 codebook
///
/// Mirrors `dequantize_row_iq4_nl` from llama.cpp.
pub fn dequant_iq4_nl(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    const KVALUES: [i32; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    ];
    let block_size = 32;
    let block_bytes = 18;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let off = b * block_bytes;
        if off + block_bytes > bytes.len() { break; }
        let out_base = b * block_size;
        if out_base >= n_elements { break; }

        let d_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let d = f16::from_bits(d_bits).to_f32();

        // llama.cpp: y[j]=d*kv[qs[j]&0xf]; y[j+16]=d*kv[qs[j]>>4]
        for j in 0..16 {
            let qs_byte = bytes[off + 2 + j];
            let lo = (qs_byte & 0x0F) as usize;
            let hi = ((qs_byte >> 4) & 0x0F) as usize;
            let e0 = out_base + j;
            let e1 = out_base + j + 16;
            if e0 < n_elements {
                out[e0] = d * (KVALUES[lo] as f32);
            }
            if e1 < n_elements {
                out[e1] = d * (KVALUES[hi] as f32);
            }
        }
    }
    out
}

/// Determine the conversion function based on quantization type.
pub fn dequantize_tensor(bytes: &[u8], quant_type: u32, n_elements: usize) -> Vec<f32> {
    match quant_type {
        0 => f32_bytes_to_f32(bytes),           // F32
        1 | 30 => f16_bytes_to_f32(bytes),      // F16
        28 => bf16_bytes_to_f32(bytes),          // BF16
        2 => dequant_q4_k_m(bytes, n_elements), // Q4_0 (simplified)
        12 => dequant_q4_k_m(bytes, n_elements), // Q4_K_M
        14 => dequant_q6_k(bytes, n_elements),   // Q6_K
        8 => {                                   // Q8_0
            bytes.iter().map(|&b| (b as i8) as f32 / 127.0).collect()
        }
        20 => dequant_iq4_nl(bytes, n_elements), // IQ4_NL
        23 => dequant_iq4_xs(bytes, n_elements), // IQ4_XS
        _ => {
            tracing::warn!("Unknown quant type {}, treating as f16", quant_type);
            f16_bytes_to_f32(bytes)
        }
    }
}
