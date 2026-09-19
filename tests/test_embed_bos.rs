//! BOS (token 2) embedding must match llama.cpp after sqrt(hidden) scale.
//!
//! Reference from llama-debug MOE_SLICE on Gemma-4-26B-MoE-IQ4_XS @ pos=0:
//!   post_embed dims 6: [0, 0, 0.548983, ..., 2.6231, 2.2952, 0.3279]
//! (head/tail from verbose inp_scaled capture)

use std::path::Path;

use cesarops_inference::gemma4_runner::Gemma4Runner;
use cesarops_inference::hardware;
use cesarops_inference::loader;

fn model_path() -> String {
    std::env::var("GEMMA4_MODEL").unwrap_or_else(|_| {
        for p in [
            "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
            "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
        ] {
            if Path::new(p).exists() {
                return p.to_string();
            }
        }
        "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string()
    })
}

#[test]
#[ignore = "loads 13 GB model; embed gate"]
fn bos_embed_matches_llama_scaled() {
    std::env::set_var("MOE_SLICE", "1");
    std::env::set_var("MOE_SLICE_LAYER", "0");
    std::env::set_var("MOE_SLICE_POS", "0");
    std::env::set_var("MOE_SLICE_DIMS", "8");

    let profile = hardware::audit_system();
    let model = model_path();
    let _ = loader::load(Path::new(&model), &profile).expect("load");

    let runner = Gemma4Runner::load(Path::new(&model), 0).expect("runner");
    let h = runner.config.hidden_size;
    let bos = 2u32;
    let row = bos as usize * h;
    let embed: Vec<f32> = runner.token_embeddings[row..row + h].to_vec();
    let scale = (h as f32).sqrt();
    let scaled: Vec<f32> = embed.iter().map(|v| v * scale).collect();

    // llama inp_scaled @ BOS (llama-debug, dims 0..4 verified from MOE_SLICE capture)
    let llama_head = [0.0f32, 0.0, 0.548983, 0.365988, 1.097965];
    println!("native scaled head: {:?}", &scaled[..8]);
    println!("llama  ref head:     {:?}", llama_head);

    for (i, (&got, &want)) in scaled.iter().zip(llama_head.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err < 0.02,
            "dim {i}: native={got:.6} llama={want:.6} (err={err:.6})"
        );
    }

    let maxabs = scaled.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("native post_embed maxabs={maxabs:.4} (llama ref ≈ 37.73)");
    assert!(
        maxabs > 30.0 && maxabs < 45.0,
        "post_embed maxabs {maxabs} should match llama ~37.7 after transpose fix"
    );
}
