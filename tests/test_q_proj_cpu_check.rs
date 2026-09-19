//! CPU matvec orientation check: attn_q @ attn_norm vs native/llama MOE_SLICE dumps.
use std::path::Path;

use cesarops_inference::bridge;
use cesarops_inference::gemma4_runner::Gemma4Runner;
use cesarops_inference::ggml_iq4_dot;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const NATIVE_Q: [f32; 32] = [
    8.259674, 15.588730, 0.499213, -9.702406, -8.610632, 3.448656, -9.324421, 0.377622, 3.441758,
    -2.129112, 3.465098, 3.967266, 4.754221, 5.950785, 5.292104, -13.258488, 2.467796, 3.000110,
    -7.684294, 5.464745, -6.018328, -4.972399, -6.556703, -3.900850, 0.493627, -7.177229,
    15.278490, 6.873722, 0.732064, -4.336679, -12.356001, -6.074178,
];

const LLAMA_Q: [f32; 32] = [
    21.563601, 58.441711, 9.596230, 10.877503, 6.225205, -1.373007, -3.928823, 22.708836, -0.609592,
    -0.542474, -3.435679, 16.036415, -11.390981, 13.621508, -6.252099, -12.248013, 9.910555,
    -35.152802, -15.658803, 8.256136, 5.854598, -2.321296, -12.572491, 5.303913, 51.059006,
    -1.837835, 14.192203, 17.597723, -4.702178, -6.127980, -19.155602, 8.720266,
];

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

fn matvec_row_major(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let mut acc = 0.0f64;
        for kk in 0..k {
            acc += (w[r * k + kk] as f64) * (x[kk] as f64);
        }
        y[r] = acc as f32;
    }
    y
}

fn matvec_ggml_col(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for out in 0..n {
        let mut acc = 0.0f64;
        for inp in 0..k {
            acc += (w[inp + out * k] as f64) * (x[inp] as f64);
        }
        y[out] = acc as f32;
    }
    y
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Gemma-style RMSNorm (matches gemma4_layer default: no +1.0 on weight).
fn attn_norm_vec(layer: &cesarops_inference::gemma4_layer::Gemma4Layer, hidden: &[f32], eps: f32) -> Vec<f32> {
    let mut out = hidden.to_vec();
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

#[test]
#[ignore = "loads model for q_proj orientation check"]
fn q_proj_cpu_orientation_check() {
    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load");
    let runner = Gemma4Runner::load(Path::new(&model), 0).expect("runner");

    let cfg = &runner.config;
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // BOS embed (matches post_embed slice path).
    let bos_id = 2usize;
    let mut hidden = runner.token_embeddings[bos_id * h..(bos_id + 1) * h].to_vec();
    let scale = (h as f32).sqrt();
    for v in &mut hidden {
        *v *= scale;
    }

    let layer0 = &runner.layers[0].dense;
    let x = attn_norm_vec(layer0, &hidden, eps);
    println!("attn_norm[:5] {:?}", &x[..5]);

    let name = "blk.0.attn_q.weight";
    let region = weights.tensors.get(name).expect("tensor");
    let bytes = weights.tensor_bytes(name).expect("bytes");
    let k = region.shape[0];
    let n = region.shape[1];
    assert_eq!(k, h);
    let w = bridge::dequantize_tensor(bytes, region.quant_type, k * n);

    let y_f32 = matvec_row_major(&w, &x, k, n);
    let y_ggml = ggml_iq4_dot::matvec_iq4_xs_q8_k(bytes, &x, k, n);

    let attn_rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let attn_max = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("attn_norm rms={attn_rms:.6} maxabs={attn_max:.6} (llama rms≈5.017 maxabs≈123.25)");

    println!("f32_dequant vs native[:32] maxabs={:.4}", max_abs(&y_f32[..32], &NATIVE_Q));
    println!("ggml_q8     vs native[:32] maxabs={:.4}", max_abs(&y_ggml[..32], &NATIVE_Q));
    println!("f32_dequant vs llama[:32]   maxabs={:.4}", max_abs(&y_f32[..32], &LLAMA_Q));
    println!("ggml_q8     vs llama[:32]   maxabs={:.4}", max_abs(&y_ggml[..32], &LLAMA_Q));
    println!("f32[:5]  {:?}", &y_f32[..5]);
    println!("ggml[:5]  {:?}", &y_ggml[..5]);

    assert!(
        max_abs(&y_f32[..32], &LLAMA_Q) < 0.5,
        "after IQ4_XS qs layout fix, Q should match llama"
    );
}
