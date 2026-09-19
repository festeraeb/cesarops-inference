//! End-to-end smoke test for the GPU-resident Gemma-4 runner.
//!
//! Loads Gemma-4-26B-MoE, builds a `Gemma4Runner` (CPU-bouncing path) for
//! weight uploads, wraps it in `Gemma4GpuRunner`, and generates a few
//! tokens for the "3x3=" prompt with greedy decoding.
//!
//! Run with:
//!     cargo test --release --test test_gemma4_gpu_runner -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use cesarops_inference::gemma4_gpu_runner::Gemma4GpuRunner;
use cesarops_inference::gemma4_runner::Gemma4Runner;
use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
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
const PROMPT: &str = "3x3=";
const MAX_NEW: usize = 8;

#[test]
#[ignore = "loads 13 GB Gemma-4 MoE model; run with --ignored"]
fn gpu_runner_answers_3x3() {
    // 1. Read the GGUF metadata for the tokenizer.
    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load gguf");
    let tokenizer = GemmaTokenizer::from_gguf(&weights).expect("tokenizer");
    drop(weights);

    let prompt_ids = tokenizer.encode(PROMPT, /*add_bos=*/ true);
    println!("Prompt: {:?}", PROMPT);
    println!("Prompt ids: {:?}", prompt_ids);

    // 2. Build the source runner (uploads weights once).
    println!("Loading source Gemma4Runner...");
    let source = Gemma4Runner::load(Path::new(&model), 0).expect("source load");
    let source = Arc::new(source);

    // 3. Wrap in the GPU runner.
    println!("Wrapping in Gemma4GpuRunner (allocating per-layer norms + scratch)...");
    let mut runner = Gemma4GpuRunner::from_source(source.clone(), 4096);

    // 4. Prefill prompt + greedy decode.
    runner.reset_kv();
    let t0 = std::time::Instant::now();
    let mut last_logits = vec![0.0f32; runner.vocab_size];
    for &tok in &prompt_ids {
        last_logits = runner.forward_token(tok);
    }
    let prefill_dt = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut new_ids: Vec<u32> = Vec::new();
    for _ in 0..MAX_NEW {
        let next = argmax(&last_logits);
        if Some(next) == tokenizer.eos_id {
            break;
        }
        new_ids.push(next);
        last_logits = runner.forward_token(next);
    }
    let decode_dt = t1.elapsed();

    let response = tokenizer.decode(&new_ids);
    println!();
    println!("--- Gemma 4 MoE GPU runner ---");
    println!("{}{}", PROMPT, response);
    println!("--- end ---");
    println!(
        "Prefill: {} tok in {:.2?}  ({:.2} t/s)",
        prompt_ids.len(),
        prefill_dt,
        prompt_ids.len() as f64 / prefill_dt.as_secs_f64()
    );
    println!(
        "Decode : {} tok in {:.2?}  ({:.2} t/s)",
        new_ids.len(),
        decode_dt,
        new_ids.len() as f64 / decode_dt.as_secs_f64()
    );
    println!("Token ids: {:?}", new_ids);

    assert!(!new_ids.is_empty(), "GPU runner produced zero new tokens");
}

fn argmax(v: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i as u32;
        }
    }
    best_i
}
