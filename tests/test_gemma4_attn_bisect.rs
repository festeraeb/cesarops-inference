//! Multi-token attention bisect — compare scores / softmax / context per position.
//!
//! Single-token prefill is degenerate (softmax = [1.0]). Run a short prompt and
//! log layer-0 attention at positions 0..N in **one** prefill pass.
//!
//!   PROMPT='3x3=' MOE_SLICE_LAYER=0 MOE_SLICE_POS_LIST=0,1,2,3,4 \\
//!     cargo test --release -p cesarops-inference --test test_gemma4_attn_bisect -- --ignored --nocapture
//!
//! Layer 5 (first shared-KV global layer, proportional RoPE):
//!   MOE_SLICE_LAYER=5 MOE_SLICE_POS=4 cargo test ... attn_bisect_layer5_equals -- --ignored --nocapture

use std::path::Path;

use cesarops_inference::gemma4_runner::{resolve_gemma4_iq4xs_model, Gemma4Runner};
use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
use cesarops_inference::hardware;
use cesarops_inference::loader;

fn model_path() -> String {
    std::env::var("GEMMA4_MODEL").unwrap_or_else(|_| {
        resolve_gemma4_iq4xs_model().unwrap_or_else(|| {
            "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string()
        })
    })
}

fn prompt() -> String {
    std::env::var("PROMPT").unwrap_or_else(|_| "3x3=".to_string())
}

fn run_attn_bisect_prefill(default_layer: &str) {
    std::env::set_var("MOE_SLICE", "1");
    if std::env::var("MOE_SLICE_LAYER").is_err() {
        std::env::set_var("MOE_SLICE_LAYER", default_layer);
    }
    if std::env::var("MOE_SLICE_DIMS").is_err() {
        std::env::set_var("MOE_SLICE_DIMS", "16");
    }
    if std::env::var("MOE_SLICE_ATTN_LEN").is_err() {
        std::env::set_var("MOE_SLICE_ATTN_LEN", "16");
    }

    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load");
    let tok = GemmaTokenizer::from_gguf(&weights).expect("tokenizer");
    let p = prompt();
    let ids = tok.encode(&p, true);

    if std::env::var("MOE_SLICE_POS_LIST").is_err() && std::env::var("MOE_SLICE_POS").is_err() {
        let list: String = (0..ids.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        std::env::set_var("MOE_SLICE_POS_LIST", &list);
    }

    let layer = std::env::var("MOE_SLICE_LAYER").unwrap_or_else(|_| default_layer.into());
    println!("prompt={p:?} token_ids={ids:?}");
    println!("MOE_SLICE_LAYER={layer}");
    println!("MOE_SLICE_POS_LIST={}", std::env::var("MOE_SLICE_POS_LIST").unwrap_or_default());
    println!("MOE_SLICE_POS={}", std::env::var("MOE_SLICE_POS").unwrap_or_default());
    for (i, &id) in ids.iter().enumerate() {
        println!("  pos={i} id={id}");
    }

    let gpu = cesarops_inference::gemma4_runner::gemma4_gpu_index();
    println!("GEMMA4_GPU index={gpu}");
    let runner = Gemma4Runner::load(Path::new(&model), gpu).expect("runner");
    if let Some(layer) = layer.parse::<usize>().ok() {
        if layer < runner.layers.len() {
            let l = &runner.layers[layer];
            println!(
                "layer {layer}: is_swa={} uses_shared_kv={} n_kv_heads={}",
                l.dense.is_swa, l.dense.uses_shared_kv, l.dense.n_kv_heads
            );
        }
    }
    let mut kv = cesarops_inference::kv_cache::KvCache::new(runner.layers.len(), 64);
    for (pos, &id) in ids.iter().enumerate() {
        let _ = runner.forward_token(id, pos, &mut kv);
    }
    println!("done — grep MOE_SLICE attn_scores_h0_seq / attn_probs_h0_seq / post_layer");
}

#[test]
#[ignore = "loads 13 GB model; multi-pos attention bisect"]
fn attn_bisect_multi_pos_prefill() {
    run_attn_bisect_prefill("0");
}

/// Layer 1: first post-embed block (full RoPE if SWA, else proportional). Bisect at `=` (pos 4).
#[test]
#[ignore = "loads 13 GB model; layer-1 bisect"]
fn attn_bisect_layer1_equals() {
    std::env::set_var("MOE_SLICE_LAYER", "1");
    std::env::set_var("MOE_SLICE_POS", "4");
    run_attn_bisect_prefill("1");
}

/// Layer 5: first shared-KV global layer (proportional RoPE). Bisect at `=` (pos 4).
#[test]
#[ignore = "loads 13 GB model; layer-5 shared-KV bisect"]
fn attn_bisect_layer5_equals() {
    std::env::set_var("MOE_SLICE_LAYER", "5");
    std::env::set_var("MOE_SLICE_POS", "4");
    run_attn_bisect_prefill("5");
}
