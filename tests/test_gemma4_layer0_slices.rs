//! Layer-0 tensor slice dump: token BOS (id=2), position 0.
//!
//!   MOE_SLICE=1 MOE_SLICE_LAYER=0 MOE_SLICE_POS=0 MOE_SLICE_DIMS=32 \\
//!     cargo test --release -p cesarops-inference --test test_gemma4_layer0_slices -- --ignored --nocapture

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
#[ignore = "loads 13 GB model; layer-0 slice bisect"]
fn layer0_bos_pos0_slices() {
    std::env::set_var("MOE_SLICE", "1");
    std::env::set_var("MOE_SLICE_LAYER", "0");
    std::env::set_var("MOE_SLICE_POS", "0");
    if std::env::var("MOE_SLICE_DIMS").is_err() {
        std::env::set_var("MOE_SLICE_DIMS", "32");
    }

    let profile = hardware::audit_system();
    let model = model_path();
    let _ = loader::load(Path::new(&model), &profile).expect("metadata load");

    println!("Loading Gemma4Runner for BOS token @ pos=0 layer-0 slices...");
    let runner = Gemma4Runner::load(Path::new(&model), 0).expect("runner");

    let bos_id = 2u32;
    let mut kv = cesarops_inference::kv_cache::KvCache::new(runner.layers.len(), 64);
    let _logits = runner.forward_token(bos_id, 0, &mut kv);
    println!("done — grep MOE_SLICE lines above and compare to llama capture");
}
