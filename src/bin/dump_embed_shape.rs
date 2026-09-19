//! Print token_embd.weight shape vs metadata vocab_size (fast diagnostic).

use std::env;
use std::path::Path;

use cesarops_inference::hardware;
use cesarops_inference::loader::{self, GgufValue};

fn main() {
    let model = env::args()
        .nth(1)
        .unwrap_or_else(|| "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf".to_string());
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(&model), &profile).expect("load gguf");

    println!("metadata vocab_size: {}", weights.vocab_size);
    println!("hidden_dim: {}", weights.hidden_dim);
    if let Some(v) = weights.metadata.get("gemma4.vocab_size") {
        println!("gemma4.vocab_size meta: {:?}", v);
    }
    if let Some(GgufValue::Array(arr)) = weights.metadata.get("tokenizer.ggml.tokens") {
        println!("tokenizer.ggml.tokens len: {}", arr.len());
    }

    for key in [
        "gemma4.final_logit_softcapping",
        "gemma4.attention.sliding_window",
        "gemma4.attention.key_length",
        "gemma4.attention.key_length_swa",
        "gemma4.attention.shared_kv_layers",
        "gemma4.rope.freq_base",
        "gemma4.rope.freq_base_swa",
        "gemma4.rope.partial_rotary_factor",
        "gemma4.rope.dimension_count",
        "gemma4.rope.dimension_count_swa",
        "gemma4.embedding_length_per_layer",
        "gemma4.embedding_length_per_layer_input",
        "gemma4.expert_count",
        "gemma4.expert_used_count",
    ] {
        if let Some(v) = weights.metadata.get(key) {
            println!("{key}: {v:?}");
        }
    }
    if let Some(GgufValue::Array(arr)) = weights.metadata.get("gemma4.attention.sliding_window_pattern") {
        let bits: Vec<bool> = arr.iter().map(|v| matches!(v, GgufValue::Bool(true))).collect();
        println!("sliding_window_pattern: {:?}", bits);
    }
    if let Some(GgufValue::Array(arr)) = weights.metadata.get("gemma4.attention.head_count_kv") {
        let vals: Vec<u32> = arr
            .iter()
            .map(|v| match v {
                GgufValue::U32(x) => *x,
                GgufValue::I32(x) => *x as u32,
                _ => 0,
            })
            .collect();
        println!("head_count_kv: {:?}", vals);
    }

    if let Some(region) = weights.tensors.get("token_embd.weight") {
        println!("token_embd.weight shape: {:?}", region.shape);
        let n_elem: usize = region.shape.iter().product();
        println!("token_embd n_elem: {}", n_elem);
        if region.shape.len() >= 2 {
            let (a, b) = (region.shape[0], region.shape[1]);
            let vocab_if_hidden_first = if a == weights.hidden_dim { b } else { a };
            let vocab_if_vocab_first = if b == weights.hidden_dim { a } else { b };
            println!("vocab if [hidden,vocab]: {}", vocab_if_hidden_first);
            println!("vocab if [vocab,hidden]: {}", vocab_if_vocab_first);
        }
    } else {
        println!("token_embd.weight: MISSING");
    }
}
