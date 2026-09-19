//! Element-wise IQ4_XS dequant + matvec path comparison for attn_q.
//!
//!   cargo test --release -p cesarops-inference --test test_iq4_dequant_compare -- --ignored --nocapture

use std::path::Path;

use cesarops_inference::bridge;
use cesarops_inference::gemma4_runner::Gemma4Runner;
use cesarops_inference::ggml_iq4_dot;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const LLAMA_Q0: f32 = 21.563601;
const NATIVE_Q0: f32 = 8.259674;

fn model_path() -> String {
    std::env::var("GEMMA4_MODEL").unwrap_or_else(|_| {
        for p in [
            "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
            "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
        ] {
            if Path::new(p).exists() {
                return p.to_string();
            }
        }
        "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string()
    })
}

fn attn_norm_bos(runner: &Gemma4Runner) -> Vec<f32> {
    let h = runner.config.hidden_size;
    let eps = runner.config.rms_norm_eps;
    let bos_id = 2usize;
    let mut hidden = runner.token_embeddings[bos_id * h..(bos_id + 1) * h].to_vec();
    let scale = (h as f32).sqrt();
    for v in &mut hidden {
        *v *= scale;
    }
    let layer = &runner.layers[0].dense;
    let mut out = hidden;
    let n = out.len();
    let mut sum_sq = 0.0f32;
    for &xi in out.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = out[i] * inv_rms * layer.attn_norm[i];
    }
    out
}

fn matvec_f32(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for out in 0..n {
        let mut acc = 0.0f64;
        for inp in 0..k {
            acc += (w[out * k + inp] as f64) * (x[inp] as f64);
        }
        y[out] = acc as f32;
    }
    y
}

#[test]
#[ignore = "loads model; IQ4 dequant + matvec path compare"]
fn iq4_dequant_and_matvec_paths() {
    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load");
    let runner = Gemma4Runner::load(Path::new(&model), 0).expect("runner");

    let name = "blk.0.attn_q.weight";
    let region = weights.tensors.get(name).expect("tensor");
    let raw = weights.tensor_bytes(name).expect("bytes");
    let k = region.shape[0];
    let n = region.shape[1];
    let x = attn_norm_bos(&runner);

    let attn_rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let attn_max = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("attn_norm rms={attn_rms:.6} maxabs={attn_max:.6}");

    // ── 1. Weight dequant: bridge flat vs ggml per-column ───────────────
    let bridge = bridge::dequantize_tensor(raw, region.quant_type, k * n);
    let (worst, w_out, w_in) = ggml_iq4_dot::max_dequant_col_diff(&bridge, raw, k, n);
    println!(
        "dequant bridge vs ggml_col: max_abs={worst:.9} at (out={w_out}, in={w_in})"
    );

    // Spot-check column 0 first 8 weights
    let col0 = &raw[0..11 * 136];
    let ggml0 = ggml_iq4_dot::dequantize_row_iq4_xs_col(col0, k);
    println!("col0 bridge[:8] {:?}", &bridge[..8.min(k)]);
    println!("col0 ggml  [:8] {:?}", &ggml0[..8.min(k)]);

    // ── 2. Matvec paths ─────────────────────────────────────────────────
    let y_bridge = matvec_f32(&bridge, &x, k, n);
    let y_ggml_dequant = matvec_f32(
        &{
            let mut flat = vec![0.0f32; k * n];
            let blocks_per_col = (k + 255) / 256;
            let row_bytes = blocks_per_col * 136;
            for out in 0..n {
                let col = &raw[out * row_bytes..(out + 1) * row_bytes];
                let col_f = ggml_iq4_dot::dequantize_row_iq4_xs_col(col, k);
                for inp in 0..k {
                    flat[out * k + inp] = col_f[inp];
                }
            }
            flat
        },
        &x,
        k,
        n,
    );
    let y_ggml_q8 = ggml_iq4_dot::matvec_iq4_xs_q8_k(raw, &x, k, n);

    println!("Q[0] native/gpu path (f32 dequant): {:.6}", y_bridge[0]);
    println!("Q[0] ggml col dequant + f32 dot:    {:.6}", y_ggml_dequant[0]);
    println!("Q[0] ggml iq4×q8_k fused (llama):   {:.6}", y_ggml_q8[0]);
    println!("Q[0] llama MOE_SLICE reference:     {:.6}", LLAMA_Q0);

    let diff_dequant_paths = (y_bridge[0] - y_ggml_dequant[0]).abs();
    let diff_f32_vs_q8 = (y_bridge[0] - y_ggml_q8[0]).abs();
    let diff_q8_vs_llama = (y_ggml_q8[0] - LLAMA_Q0).abs();
    println!("|f32_bridge - f32_ggml_col| = {diff_dequant_paths:.6}");
    println!("|f32_bridge - q8_fused|     = {diff_f32_vs_q8:.6}");
    println!("|q8_fused - llama|          = {diff_q8_vs_llama:.6}");

    assert!(worst < 1e-5, "bridge dequant must match ggml per column (max={worst})");
    assert!(
        diff_dequant_paths < 1e-4,
        "f32 matvec paths must agree"
    );
    assert!(
        (y_bridge[0] - LLAMA_Q0).abs() < 0.5,
        "fixed f32 matvec Q[0] should match llama (got {})",
        y_bridge[0]
    );
}
