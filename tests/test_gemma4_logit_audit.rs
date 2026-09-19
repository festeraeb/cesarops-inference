//! Logit audit for `3x3=`: compare native top-k vs expected math-ish continuations.
//!
//! Run (loads ~13 GB model, several minutes):
//!   cargo test --release -p cesarops-inference --test test_gemma4_logit_audit -- --ignored --nocapture
//!
//! Reference: llama.cpp greedy on same GGUF should rank `9` / `▁9` / `nine` near the top.

use std::path::Path;

use cesarops_inference::gemma4_runner::{resolve_gemma4_iq4xs_model, top_k_logits, Gemma4Runner};
use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const PROMPT: &str = "3x3=";

fn model_path() -> String {
    std::env::var("GEMMA4_MODEL").unwrap_or_else(|_| {
        resolve_gemma4_iq4xs_model().unwrap_or_else(|| {
            "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string()
        })
    })
}

fn decode(tok: &GemmaTokenizer, id: u32) -> String {
    tok.id_to_token
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| format!("<id:{id}>"))
}

/// Token ids that are plausible correct continuations after `3x3=` (llama greedy ≈ `9`).
fn math_continuation_ids(tok: &GemmaTokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    for s in ["9", "\u{2581}9", " nine", "\u{2581}nine", "9.", "\u{2581}9."] {
        if let Some(&id) = tok.vocab.get(s) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[test]
#[ignore = "loads 13 GB Gemma-4 MoE; logit audit gate"]
fn logit_audit_3x3_top5() {
    let profile = hardware::audit_system();
    let model = model_path();
    let weights = loader::load(Path::new(&model), &profile).expect("load gguf");
    let tokenizer = GemmaTokenizer::from_gguf(&weights).expect("tokenizer");
    drop(weights);

    let prompt_ids = tokenizer.encode(PROMPT, true);
    println!("Prompt: {PROMPT:?}");
    println!("Prompt token ids: {prompt_ids:?}");
    for &id in &prompt_ids {
        println!("  [{id:>6}] {:?}", decode(&tokenizer, id));
    }

    println!("\nLoading Gemma4Runner...");
    let gpu = cesarops_inference::gemma4_runner::gemma4_gpu_index();
    println!("GEMMA4_GPU index={gpu}");
    let runner = Gemma4Runner::load(Path::new(&model), gpu).expect("runner load");
    println!(
        "vocab_size={} lm_head_tied={}",
        runner.vocab_size, runner.lm_head_uses_tied_embedding
    );

    let logits = runner
        .prefill_last_logits(&prompt_ids)
        .expect("prefill logits");
    assert_eq!(
        logits.len(),
        runner.vocab_size,
        "lm_head must return full vocab logits"
    );

    let topk = std::env::var("LOGIT_AUDIT_TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let top = top_k_logits(&logits, topk);

    println!("\n=== NATIVE top-{topk} logits after prefill `{PROMPT}` ===");
    for (rank, (id, val)) in top.iter().enumerate() {
        println!(
            "  {:>2}. id={:<7} logit={:>8.4}  {:?}",
            rank + 1,
            id,
            val,
            decode(&tokenizer, *id)
        );
    }

    let greedy = top.first().map(|(id, _)| *id).unwrap_or(0);
    println!("\nGreedy first token: id={greedy} {:?}", decode(&tokenizer, greedy));

    let math_ids = math_continuation_ids(&tokenizer);
    println!("\nReference math-ish token ids: {math_ids:?}");
    for &mid in &math_ids {
        println!("  [{mid:>6}] {:?}", decode(&tokenizer, mid));
    }

    let in_top5: Vec<_> = top
        .iter()
        .take(5)
        .filter(|(id, _)| math_ids.contains(id))
        .collect();
    if in_top5.is_empty() {
        println!(
            "\nVERDICT: FAIL — no math continuation in top-5. Forward pass distribution is wrong."
        );
        println!("Next bisect: layer MOE_TRACE vs llama at pos=0, then LM head if layers match.");
    } else {
        println!("\nVERDICT: PASS (partial) — math token in top-5: {in_top5:?}");
    }

    // Informational only — do not fail CI on wrong logits until gate is green.
}
