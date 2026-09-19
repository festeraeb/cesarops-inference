//! End-to-end smoke test: load Gemma-4-26B-MoE and ask it "3x3=".
//!
//! We use the real `Gemma4Runner` path (attention + MoE FFN + LM head) plus
//! the GGUF-embedded tokenizer. The point of this test is not correctness of
//! the answer (the model is huge and we may run into accuracy issues with
//! greedy decoding) but to prove the pipeline produces *some* coherent
//! tokens.
//!
//! This test is gated behind `--ignored` because it loads a 13 GB model and
//! takes several minutes on a P100. Run with:
//!     cargo test --release --test test_gemma4_moe -- --ignored --nocapture

use std::path::Path;

use cesarops_inference::gemma4_runner::{resolve_gemma4_iq4xs_model, Gemma4Runner, SamplingParams};
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
const PROMPT: &str = "3x3=";

#[test]
#[ignore = "loads 13 GB Gemma-4 MoE model; run with --ignored"]
fn gemma4_moe_answers_3x3() {
    // 1. Build the tokenizer from the GGUF metadata. This is a cheap
    //    metadata-only pass over the file (no tensor data).
    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile)
        .expect("load Gemma-4 MoE GGUF");
    let tokenizer = GemmaTokenizer::from_gguf(&weights)
        .expect("build Gemma tokenizer from GGUF metadata");

    println!(
        "Vocab size: {} (BOS={:?}, EOS={:?})",
        tokenizer.id_to_token.len(),
        tokenizer.bos_id,
        tokenizer.eos_id,
    );
    drop(weights); // Free the mmap before the runner reopens it.

    // 2. Tokenize prompt.
    let prompt_ids = tokenizer.encode(PROMPT, /*add_bos=*/ true);
    println!("Prompt: {:?}", PROMPT);
    println!("Prompt token ids: {:?}", prompt_ids);
    for &id in &prompt_ids {
        let s = tokenizer
            .id_to_token
            .get(id as usize)
            .cloned()
            .unwrap_or_default();
        println!("  [{:>5}] {:?}", id, s);
    }

    // 3. Build the runner. This is where the GPU upload happens; a few minutes
    //    on first run because we dequant Q6_K embeddings + LM head into fp32.
    let gpu = cesarops_inference::gemma4_runner::gemma4_gpu_index();
    println!("Loading model onto GPU {gpu}...");
    let runner = Gemma4Runner::load(Path::new(&model), gpu).expect("Gemma4Runner::load");
    println!("Model loaded.");

    // 4. Greedy decode 8 new tokens. Greedy is enough to see whether the model
    //    is producing sensible output; sampling we'll check separately.
    let mut params = SamplingParams::default();
    params.max_new_tokens = 8;
    params.temperature = 0.0;
    if let Some(eos) = tokenizer.eos_id {
        params.stop_tokens = vec![eos];
    }

    let t0 = std::time::Instant::now();
    let new_ids = runner
        .generate(&prompt_ids, &params)
        .expect("generate did not fail");
    let dt = t0.elapsed();

    let response = tokenizer.decode(&new_ids);
    println!();
    println!("--- Gemma 4 MoE response ---");
    println!("{}{}", PROMPT, response);
    println!("--- end ---");
    println!(
        "{} new tokens in {:.2?} ({:.2} t/s)",
        new_ids.len(),
        dt,
        new_ids.len() as f64 / dt.as_secs_f64()
    );
    println!("Token ids: {:?}", new_ids);

    assert!(!new_ids.is_empty(), "model produced zero tokens");
}
