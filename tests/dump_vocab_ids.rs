//! Dump specific token ids from the Gemma-4 MoE GGUF vocab.
//!
//! Used during MoE numerics debugging to interpret model output.

use std::path::Path;

use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
use cesarops_inference::hardware;
use cesarops_inference::loader;

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
        "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string()
    })
}

#[test]
#[ignore = "metadata-only quick lookup"]
fn dump_vocab_for_specific_ids() {
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(&model_path()), &profile).expect("load gguf");
    let tok = GemmaTokenizer::from_gguf(&weights).expect("tokenizer");

    let prompt_ids = tok.encode("3x3=", true);
    println!("encode('3x3=') -> {:?}", prompt_ids);
    // llama.cpp reference (greedy prefill): [2, 236800, 236781, 236800, 236784]
    let llama_ref = [2u32, 236800, 236781, 236800, 236784];
    assert_eq!(
        prompt_ids, llama_ref,
        "tokenizer must match llama.cpp tokenization for bisect"
    );

    // Tokens we got out of our runner for the "3x3=" prompt:
    //   [1010, 5596, 2, 1852, 2, 2, 7246, 27077]
    // Ollama (reference) answer was just repeated "3=" so the expected
    // first generated token text is "3" (with or without leading ▁).
    let ours = [1010u32, 5596, 2, 1852, 2, 2, 7246, 27077];
    println!("--- Our output tokens ---");
    for &id in &ours {
        let s = tok.id_to_token.get(id as usize).cloned().unwrap_or_default();
        println!("  [{:>6}] {:?}", id, s);
    }

    // The expected continuation tokens. We need to find the id for "3".
    println!("--- Lookup: token 'before' '3' ---");
    for cand in ["3", "\u{2581}3", "=3", "3=", "\u{2581}=", "="] {
        let id = tok.vocab.get(cand).copied();
        println!("  {:?} -> {:?}", cand, id);
    }
}
