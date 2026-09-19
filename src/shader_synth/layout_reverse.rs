//! GGML block layout reverse engineering
//!
//! Loads binary tensors, detects quantization type, reconstructs block layout,
//! outputs GPU-friendly schema for shader generation.

#[derive(Debug, Clone, PartialEq)]
pub enum QuantType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q6_K,
    IQ4_XS,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct InferredLayout {
    pub quant: QuantType,
    pub block_size: usize,
    pub scale_offset: usize,
    pub data_offset: usize,
    pub packing: String,
}

/// Detect quantization type from raw block bytes using entropy analysis
pub fn detect_quant(block: &[u8]) -> QuantType {
    if block.is_empty() {
        return QuantType::Unknown;
    }

    let entropy = shannon_entropy(block);

    if block.len() % 4 == 0 && entropy > 3.5 {
        return QuantType::F16;
    }

    if block.len() == 24 && entropy < 2.5 {
        return QuantType::IQ4_XS;
    }

    if block.len() == 18 || block.len() % 18 == 0 {
        if entropy < 1.5 {
            return QuantType::Q4_0;
        }
    }

    if block.len() == 210 || block.len() % 210 == 0 {
        return QuantType::Q6_K;
    }

    QuantType::Unknown
}

/// Shannon entropy of byte distribution
pub fn shannon_entropy(data: &[u8]) -> f32 {
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let len = data.len() as f32;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / len;
            -p * p.log2()
        })
        .sum()
}

/// Infer block structure from raw bytes
pub fn infer_layout(block: &[u8]) -> InferredLayout {
    let quant = detect_quant(block);
    let block_size = block.len();

    let (scale_offset, data_offset) = match quant {
        QuantType::IQ4_XS => (0, 8),   // d(2) + scales_h(2) + scales_l(4) = 8, then qs
        QuantType::Q4_0 => (0, 2),     // d(2) then packed nibbles
        QuantType::Q6_K => (0, 12),    // complex header then data
        _ => (0, 0),
    };

    let packing = if block_size % 32 == 0 {
        "warp_aligned_32"
    } else if block_size % 16 == 0 {
        "nibble_packed_16"
    } else {
        "unknown"
    };

    InferredLayout {
        quant,
        block_size,
        scale_offset,
        data_offset,
        packing: packing.to_string(),
    }
}

/// Output as JSON schema for shader generator input
pub fn layout_to_json(layout: &InferredLayout) -> String {
    format!(
        r#"{{"quant": "{:?}", "block_size": {}, "layout": "{}", "scale_offset": {}, "data_offset": {}}}"#,
        layout.quant, layout.block_size, layout.packing, layout.scale_offset, layout.data_offset
    )
}
