// src/bin/probe_moe_layout.rs
//
// Cheap sanity check on the per-expert byte stride assumption used by the
// MoE matvec shaders. Loads `blk.0.ffn_gate_up_exps.weight` from a GGUF
// file, slices off the first two experts under the SLAB hypothesis
// (expert e = bytes[e * stride .. (e+1) * stride]) and dequantizes both
// to fp32. Reports:
//
//   * shape + bytes-per-expert math
//   * stats on expert 0 and expert 1 (mean, var, |max|)
//   * cosine similarity between expert 0 and expert 1
//
// The slab assumption is correct iff:
//   * shape arithmetic matches reality (no remainder bytes)
//   * each expert's stats look like a trained quantized layer
//     (non-zero, not all-equal, magnitudes in the IQ4 range)
//   * cosine similarity is well below 1.0 (experts must be distinct)
//   * cosine similarity is well above near-zero (same architecture, similar
//     weight distribution; if it's effectively orthogonal that's a flag)
//
// Anything else means the slab packing is wrong and we have to revisit.

use std::env;
use std::path::PathBuf;

use cesarops_inference::bridge;
use cesarops_inference::hardware;
use cesarops_inference::loader;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let mut model = PathBuf::new();
    let mut tensor = "blk.0.ffn_gate_up_exps.weight".to_string();
    let mut n_experts: usize = 128;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = PathBuf::from(&args[i + 1]); i += 2; }
            "--tensor" => { tensor = args[i + 1].clone(); i += 2; }
            "--n-experts" => { n_experts = args[i + 1].parse()?; i += 2; }
            other => {
                eprintln!("unknown arg: {}", other);
                eprintln!("usage: probe_moe_layout --model PATH [--tensor NAME] [--n-experts 128]");
                std::process::exit(1);
            }
        }
    }
    if model.as_os_str().is_empty() {
        eprintln!("--model is required");
        std::process::exit(1);
    }

    println!("Loading: {:?}", model);
    let profile = hardware::audit_system();
    let weights = loader::load(&model, &profile)?;

    let region = weights.tensors.get(&tensor)
        .ok_or_else(|| anyhow::anyhow!("tensor not found: {}", tensor))?;
    let bytes = weights.tensor_bytes(&tensor)
        .ok_or_else(|| anyhow::anyhow!("tensor bytes missing"))?;

    println!("Tensor   : {}", tensor);
    println!("Shape    : {:?}  (GGUF [ne0, ne1, ne2])", region.shape);
    println!("Qtype    : {} (23 = IQ4_XS, 20 = IQ4_NL)", region.quant_type);
    println!("Total    : {} bytes", region.size);

    if region.shape.len() != 3 {
        anyhow::bail!("expected 3-D MoE tensor; got {:?}", region.shape);
    }

    // GGUF dimensions for `*_exps` tensors are stored in inner-first order.
    // For `ffn_gate_up_exps` of Gemma-4 MoE the recorded shape is
    // [hidden, gate_up_rows, n_experts] — confirm n_experts is what the user
    // told us.
    let outermost = region.shape[2];
    if outermost != n_experts {
        eprintln!("WARN: tensor outermost dim is {} but --n-experts is {}",
            outermost, n_experts);
    }

    let stride = region.size / n_experts;
    let remainder = region.size % n_experts;
    println!();
    println!("=== Slab hypothesis ===");
    println!("  bytes_per_expert = total / n_experts = {} / {} = {}", region.size, n_experts, stride);
    println!("  remainder        = {}  (must be 0)", remainder);
    if remainder != 0 {
        println!();
        println!("  [FAIL] slab hypothesis cannot hold — total bytes is not divisible by n_experts.");
        println!("  Either n_experts is wrong or the tensor uses a non-slab packing.");
        return Ok(());
    }

    // Dequant first two experts under the slab hypothesis.
    let e0_bytes = &bytes[0..stride];
    let e1_bytes = &bytes[stride..2 * stride];

    // n_elements per expert: shape ne0 * ne1 (the inner two dims).
    let n_elements_per_expert = region.shape[0] * region.shape[1];
    let q = region.quant_type;

    println!();
    println!("Dequantizing expert 0 ({} bytes -> {} elements)...", stride, n_elements_per_expert);
    let e0 = bridge::dequantize_tensor(e0_bytes, q, n_elements_per_expert);
    println!("Dequantizing expert 1 ...");
    let e1 = bridge::dequantize_tensor(e1_bytes, q, n_elements_per_expert);

    let s0 = stats(&e0);
    let s1 = stats(&e1);
    println!();
    println!("=== Expert stats ===");
    println!("  expert 0: n={}  mean={:+.4e}  var={:.4e}  abs_max={:.4}", e0.len(), s0.0, s0.1, s0.2);
    println!("  expert 1: n={}  mean={:+.4e}  var={:.4e}  abs_max={:.4}", e1.len(), s1.0, s1.1, s1.2);

    let cos = cosine(&e0, &e1);
    println!();
    println!("=== Cosine similarity (expert 0 vs expert 1) ===");
    println!("  cos = {:.4}", cos);

    println!();
    println!("=== Verdict ===");
    let mut verdict_pass = true;
    if remainder != 0 {
        println!("  [FAIL] non-divisible byte count");
        verdict_pass = false;
    }
    if s0.2 < 1e-6 || s1.2 < 1e-6 {
        println!("  [FAIL] one expert is all zeros — slab offset is wrong, or this layer has dead experts");
        verdict_pass = false;
    }
    if !s0.0.is_finite() || !s1.0.is_finite() {
        println!("  [FAIL] non-finite values — dequant produced NaN/Inf");
        verdict_pass = false;
    }
    if cos > 0.999 {
        println!("  [FAIL] experts identical (cos > 0.999) — same expert was read twice");
        verdict_pass = false;
    }
    if cos < -0.5 {
        println!("  [WARN] experts strongly anti-correlated — possible sign-bit packing flip");
    }
    if cos.abs() > 0.5 {
        println!("  [WARN] experts unusually similar (|cos| > 0.5) — possible byte aliasing");
    }
    if verdict_pass {
        println!("  [PASS] slab hypothesis is consistent with the data.");
        println!("         expert e is at bytes[{} * e .. {} * (e+1)]", stride, stride);
    }

    Ok(())
}

fn stats(v: &[f32]) -> (f64, f64, f32) {
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| {
        let d = *x as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let abs_max = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    (mean, var, abs_max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}
