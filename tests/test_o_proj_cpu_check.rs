//! CPU matvec check for O-projection: attn_output @ post_attn_core vs native GPU / llama.
//!
//! Run (loads model, dumps post_attn_core via MOE_SLICE_DUMP_DIR):
//!   cargo test --release -p cesarops-inference --test test_o_proj_cpu_check \\
//!     o_proj_cpu_orientation_check -- --ignored --nocapture

use std::path::Path;

use cesarops_inference::bridge;
use cesarops_inference::gemma4_runner::Gemma4Runner;
use cesarops_inference::ggml_iq4_dot;
use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const PROMPT: &str = "3x3=";

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "loads model for O-proj orientation check"]
fn o_proj_cpu_orientation_check() {
    let dump_dir = std::env::var("MOE_SLICE_DUMP_DIR")
        .unwrap_or_else(|_| "/tmp/moe_slice_dump".to_string());
    std::fs::create_dir_all(&dump_dir).expect("dump dir");

    std::env::set_var("MOE_SLICE", "1");
    std::env::set_var("MOE_SLICE_LAYER", "0");
    std::env::set_var("MOE_SLICE_POS", "4");
    std::env::set_var("MOE_SLICE_DIMS", "16");
    std::env::set_var("MOE_SLICE_DUMP_DIR", &dump_dir);
    std::env::set_var("MOE_SLICE_DUMP_PHASE", "post_attn_core,post_o_raw");

    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load");
    let tokenizer = GemmaTokenizer::from_gguf(&weights).expect("tokenizer");
    let prompt_ids = tokenizer.encode(PROMPT, true);
    println!("Prompt {PROMPT:?} ids={prompt_ids:?}");

    let runner = Gemma4Runner::load(Path::new(&model), 0).expect("runner");
    let _ = runner.prefill_last_logits(&prompt_ids).expect("prefill");

    let core_path = format!("{dump_dir}/L0_pos4_post_attn_core.f32");
    let core_bytes = std::fs::read(&core_path).expect("post_attn_core dump — set MOE_SLICE_DUMP_DIR");
    let attn_core: Vec<f32> = bytemuck::cast_slice(&core_bytes).to_vec();
    let layer0 = &runner.layers[0].dense;
    let (head_dim, _) = runner.config.layer_geometry(layer0.is_swa);
    let q_dim = runner.config.num_heads * head_dim;
    assert_eq!(
        attn_core.len(),
        q_dim,
        "attn_core len {} != q_dim {}",
        attn_core.len(),
        q_dim
    );

    let h = runner.config.hidden_size;
    let name = "blk.0.attn_output.weight";
    let region = weights.tensors.get(name).expect("tensor");
    let bytes = weights.tensor_bytes(name).expect("bytes");
    assert_eq!(region.shape[0], q_dim);
    assert_eq!(region.shape[1], h);

    let y_ggml = ggml_iq4_dot::matvec_iq4_xs_q8_k(bytes, &attn_core, q_dim, h);
    let w = bridge::dequantize_tensor(bytes, region.quant_type, q_dim * h);
    let mut y_f32 = vec![0.0f32; h];
    for row in 0..h {
        let mut acc = 0.0f64;
        for col in 0..q_dim {
            acc += (w[row * q_dim + col] as f64) * (attn_core[col] as f64);
        }
        y_f32[row] = acc as f32;
    }

    // Native GPU post_o_raw from same prefill dump.
    let o_path = format!("{dump_dir}/L0_pos4_post_o_raw.f32");
    assert!(
        Path::new(&o_path).exists(),
        "missing post_o_raw dump at {o_path}"
    );
    let o_bytes = std::fs::read(&o_path).expect("o_raw dump");
    let native_o: Vec<f32> = bytemuck::cast_slice(&o_bytes).to_vec();
    let n = 32.min(native_o.len());
    println!(
        "ggml vs native[:32] maxabs={:.4}",
        max_abs(&y_ggml[..n], &native_o[..n])
    );
    println!(
        "f32 vs native[:32] maxabs={:.4}",
        max_abs(&y_f32[..n], &native_o[..n])
    );
    println!("native[:8] {:?}", &native_o[..8.min(n)]);
    println!("ggml[:8]   {:?}", &y_ggml[..8.min(n)]);
    assert!(
        max_abs(&y_ggml[..n], &native_o[..n]) < 1.0,
        "O-proj IQ4 matvec should track native GPU post_o_raw"
    );
}
