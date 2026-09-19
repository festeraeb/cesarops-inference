//! verify_expert_layout — sanity-check the "experts are the OUTERMOST dim"
//! assumption baked into `moe_iq4_dispatch.rs` for Gemma-4-26B-MoE.
//!
//! Per the comments in `moe_iq4_dispatch.rs`:
//!   blk.0.ffn_gate_up_exps.weight  shape [2816, 1408, 128]   IQ4_XS (qtype 23)
//!   blk.0.ffn_down_exps.weight     shape [704,  2816, 128]   IQ4_NL (qtype 20)
//!
//! Outermost-expert assumption ⇒
//!   gate_up bytes per expert = 1408 * (2816/256) * 136 = 1408 * 11 * 136 = 2_106_368
//!   down    bytes per expert = 2816 * (704/32)   * 18  = 2816 * 22  * 18  = 1_115_136
//! Total tensor sizes ⇒
//!   gate_up: 128 * 2_106_368 = 269_615_104
//!   down:    128 * 1_115_136 = 142_737_408
//!
//! This binary:
//!   1. Loads the GGUF.
//!   2. Compares the per-tensor declared size to the expected size.
//!   3. Dequantizes just the first 256 weights of expert 0 and expert 1 from
//!      the byte slice each assumption claims they live at, and prints them.
//!      If they're identical, the assumption is wrong.

use std::path::PathBuf;

use cesarops_inference::bridge;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const MODEL_PATH: &str = "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf";

const N_EXPERTS: usize = 128;

// gate_up: IQ4_XS, hidden=2816 cols per row, 1408 rows per expert
const GU_HIDDEN: usize = 2816;
const GU_ROWS_PER_EXPERT: usize = 1408;
const GU_ROW_BYTES: usize = (GU_HIDDEN / 256) * 136; // 11 * 136 = 1496
const GU_BYTES_PER_EXPERT: usize = GU_ROWS_PER_EXPERT * GU_ROW_BYTES; // 2_106_368
const GU_TOTAL_EXPECTED: usize = N_EXPERTS * GU_BYTES_PER_EXPERT;     // 269_615_104

// down: IQ4_NL, expert_inner=704 cols per row, 2816 rows per expert
const DN_INNER: usize = 704;
const DN_ROWS_PER_EXPERT: usize = 2816;
const DN_ROW_BYTES: usize = (DN_INNER / 32) * 18; // 22 * 18 = 396
const DN_BYTES_PER_EXPERT: usize = DN_ROWS_PER_EXPERT * DN_ROW_BYTES; // 1_115_136
const DN_TOTAL_EXPECTED: usize = N_EXPERTS * DN_BYTES_PER_EXPERT;     // 142_737_408

fn main() -> anyhow::Result<()> {
    // Quiet logging — we want clean side-by-side output.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let path = PathBuf::from(MODEL_PATH);
    let profile = hardware::audit_system();
    let weights = loader::load(&path, &profile)?;

    println!("=== Expert layout verification ===");

    let gate_up_ok = check_iq4xs_tensor(
        &weights,
        "blk.0.ffn_gate_up_exps.weight",
        &[GU_HIDDEN, GU_ROWS_PER_EXPERT, N_EXPERTS],
        23,
        GU_BYTES_PER_EXPERT,
        GU_TOTAL_EXPECTED,
    );

    println!();

    let down_ok = check_iq4nl_tensor(
        &weights,
        "blk.0.ffn_down_exps.weight",
        &[DN_INNER, DN_ROWS_PER_EXPERT, N_EXPERTS],
        20,
        DN_BYTES_PER_EXPERT,
        DN_TOTAL_EXPECTED,
    );

    println!();
    if gate_up_ok && down_ok {
        println!("OVERALL: assumption VALID for both tensors");
    } else {
        println!(
            "OVERALL: assumption INVALID (gate_up={}, down={})",
            yesno(gate_up_ok),
            yesno(down_ok)
        );
    }

    Ok(())
}

fn yesno(b: bool) -> &'static str {
    if b { "VALID" } else { "INVALID" }
}

/// Sanity-check an IQ4_XS expert-packed tensor.
/// Returns true iff declared size matches expected AND expert 0 and expert 1
/// dequantize to different sample weights.
fn check_iq4xs_tensor(
    weights: &loader::ModelWeights,
    name: &str,
    expected_shape: &[usize],
    expected_qtype: u32,
    bytes_per_expert: usize,
    total_expected: usize,
) -> bool {
    let region = match weights.tensors.get(name) {
        Some(r) => r,
        None => {
            println!("Tensor: {} — NOT FOUND in GGUF", name);
            return false;
        }
    };

    println!("Tensor: {}", name);
    println!("  Shape: {:?}", region.shape);
    println!(
        "  qtype: {} ({})",
        region.quant_type,
        qname(region.quant_type)
    );
    println!("  Declared size: {} bytes", region.size);
    println!(
        "  Expected (outermost): {} bytes  -> {}",
        total_expected,
        size_match_label(region.size, total_expected),
    );

    let shape_ok = region.shape == expected_shape;
    let qtype_ok = region.quant_type == expected_qtype;
    let size_ok = region.size == total_expected;

    if !shape_ok {
        println!(
            "  WARNING: shape mismatch (got {:?}, expected {:?})",
            region.shape, expected_shape
        );
    }
    if !qtype_ok {
        println!(
            "  WARNING: qtype mismatch (got {}, expected {})",
            region.quant_type, expected_qtype
        );
    }

    let bytes = match weights.tensor_bytes(name) {
        Some(b) => b,
        None => {
            println!("  ERROR: tensor_bytes returned None");
            return false;
        }
    };

    if !size_ok {
        report_actual_stride("IQ4_XS", region.size);
        return false;
    }

    // Pull expert 0 and expert 1 byte slices and dequantize their first 256
    // weights (one IQ4_XS super-block).
    let e0_off = 0usize;
    let e1_off = bytes_per_expert;
    let e0 = &bytes[e0_off..e0_off + bytes_per_expert];
    let e1 = &bytes[e1_off..e1_off + bytes_per_expert];

    let e0_first = bridge::dequant_iq4_xs(&e0[..136], 256);
    let e1_first = bridge::dequant_iq4_xs(&e1[..136], 256);

    let same = sample_eq(&e0_first[..4], &e1_first[..4]);
    println!();
    println!(
        "  Expert 0 first 4 weights: [{:.6}, {:.6}, {:.6}, {:.6}]",
        e0_first[0], e0_first[1], e0_first[2], e0_first[3]
    );
    println!(
        "  Expert 1 first 4 weights: [{:.6}, {:.6}, {:.6}, {:.6}]",
        e1_first[0], e1_first[1], e1_first[2], e1_first[3]
    );
    println!(
        "  Different: {}  -> assumption {}",
        if same { "no" } else { "yes" },
        if same { "INVALID" } else { "VALID" }
    );

    !same
}

/// Sanity-check an IQ4_NL expert-packed tensor.
fn check_iq4nl_tensor(
    weights: &loader::ModelWeights,
    name: &str,
    expected_shape: &[usize],
    expected_qtype: u32,
    bytes_per_expert: usize,
    total_expected: usize,
) -> bool {
    let region = match weights.tensors.get(name) {
        Some(r) => r,
        None => {
            println!("Tensor: {} — NOT FOUND in GGUF", name);
            return false;
        }
    };

    println!("Tensor: {}", name);
    println!("  Shape: {:?}", region.shape);
    println!(
        "  qtype: {} ({})",
        region.quant_type,
        qname(region.quant_type)
    );
    println!("  Declared size: {} bytes", region.size);
    println!(
        "  Expected (outermost): {} bytes  -> {}",
        total_expected,
        size_match_label(region.size, total_expected),
    );

    let shape_ok = region.shape == expected_shape;
    let qtype_ok = region.quant_type == expected_qtype;
    let size_ok = region.size == total_expected;

    if !shape_ok {
        println!(
            "  WARNING: shape mismatch (got {:?}, expected {:?})",
            region.shape, expected_shape
        );
    }
    if !qtype_ok {
        println!(
            "  WARNING: qtype mismatch (got {}, expected {})",
            region.quant_type, expected_qtype
        );
    }

    let bytes = match weights.tensor_bytes(name) {
        Some(b) => b,
        None => {
            println!("  ERROR: tensor_bytes returned None");
            return false;
        }
    };

    if !size_ok {
        report_actual_stride("IQ4_NL", region.size);
        return false;
    }

    let e0_off = 0usize;
    let e1_off = bytes_per_expert;
    let e0 = &bytes[e0_off..e0_off + bytes_per_expert];
    let e1 = &bytes[e1_off..e1_off + bytes_per_expert];

    // IQ4_NL block is 32 weights per 18 bytes. Decode 32 weights from the
    // first block of each expert.
    let e0_first = bridge::dequant_iq4_nl(&e0[..18], 32);
    let e1_first = bridge::dequant_iq4_nl(&e1[..18], 32);

    let same = sample_eq(&e0_first[..4], &e1_first[..4]);
    println!();
    println!(
        "  Expert 0 first 4 weights: [{:.6}, {:.6}, {:.6}, {:.6}]",
        e0_first[0], e0_first[1], e0_first[2], e0_first[3]
    );
    println!(
        "  Expert 1 first 4 weights: [{:.6}, {:.6}, {:.6}, {:.6}]",
        e1_first[0], e1_first[1], e1_first[2], e1_first[3]
    );
    println!(
        "  Different: {}  -> assumption {}",
        if same { "no" } else { "yes" },
        if same { "INVALID" } else { "VALID" }
    );

    !same
}

fn size_match_label(declared: usize, expected: usize) -> &'static str {
    if declared == expected { "MATCH" } else { "MISMATCH" }
}

fn sample_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn qname(q: u32) -> &'static str {
    match q {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        20 => "IQ4_NL",
        23 => "IQ4_XS",
        28 => "BF16",
        30 => "F16_alt",
        _ => "UNKNOWN",
    }
}

fn report_actual_stride(qname: &str, declared: usize) {
    let actual_per_expert = declared / N_EXPERTS;
    println!(
        "  -> actual byte stride per expert ({}, declared/{}) = {} bytes",
        qname, N_EXPERTS, actual_per_expert
    );
    println!(
        "  -> replacement formula: expert_offset_bytes = e * {}",
        actual_per_expert
    );
    println!(
        "     (current assumption used e * {} for {} but does not match)",
        if qname == "IQ4_XS" { GU_BYTES_PER_EXPERT } else { DN_BYTES_PER_EXPERT },
        qname
    );
}
