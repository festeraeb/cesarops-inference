//! Minimal Gemma / SentencePiece-style tokenizer that reads the embedded
//! vocab out of a GGUF file's metadata.
//!
//! This is intentionally simple: we use the embedded `tokenizer.ggml.tokens`
//! and `tokenizer.ggml.scores` arrays to do longest-prefix matching with
//! SentencePiece "▁" word-boundary semantics. It is good enough for short
//! smoke-test prompts like "3x3=".
//!
//! Honest gaps (don't pretend otherwise):
//!   * No proper Viterbi decoding — we use greedy longest match. For most
//!     ASCII prompts this matches SentencePiece's unigram output bit-for-bit;
//!     for pathological inputs it can diverge. Smoke-test only.
//!   * No byte-fallback path. If a character is missing from vocab we emit
//!     `<unk>` (or token id 3, the SP convention).
//!   * BOS handling: we prepend <bos> if the model declares one in
//!     `tokenizer.ggml.bos_token_id`.

use std::collections::HashMap;

use crate::loader::{GgufValue, ModelWeights};

const SP_SPACE: &str = "\u{2581}"; // ▁ — SentencePiece word boundary marker

#[derive(Debug)]
pub struct GemmaTokenizer {
    /// Token text (with ▁ prefixes preserved) → id.
    pub vocab: HashMap<String, u32>,
    /// id → token text.
    pub id_to_token: Vec<String>,
    /// Optional BOS / EOS for prompt formatting.
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: u32,
}

impl GemmaTokenizer {
    /// Build a tokenizer from a freshly-loaded GGUF.
    ///
    /// Returns `Err` with a human-readable reason if the metadata lacks the
    /// `tokenizer.ggml.tokens` array. We prefer a real failure to a silent
    /// fallback because Gemma without its vocab is useless.
    pub fn from_gguf(weights: &ModelWeights) -> Result<Self, String> {
        let tokens_arr = match weights.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(a)) => a,
            Some(_) => return Err("tokenizer.ggml.tokens is not an array".into()),
            None => return Err("GGUF has no tokenizer.ggml.tokens metadata".into()),
        };

        let mut id_to_token: Vec<String> = Vec::with_capacity(tokens_arr.len());
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(tokens_arr.len());
        for (i, v) in tokens_arr.iter().enumerate() {
            let s = match v {
                GgufValue::Str(s) => s.clone(),
                _ => format!("<bad_token_{i}>"),
            };
            // First insert wins for duplicates. Some GGUFs have duplicate
            // strings for control tokens; we want the lowest id to win.
            vocab.entry(s.clone()).or_insert(i as u32);
            id_to_token.push(s);
        }

        let bos_id = read_special_id(weights, "tokenizer.ggml.bos_token_id");
        let eos_id = read_special_id(weights, "tokenizer.ggml.eos_token_id");
        let unk_id = read_special_id(weights, "tokenizer.ggml.unknown_token_id").unwrap_or(3);

        Ok(Self {
            vocab,
            id_to_token,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    /// Encode a string into Gemma token ids using greedy longest-prefix match
    /// with SentencePiece "▁" boundary handling. Optionally prepends BOS.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        if add_bos {
            if let Some(b) = self.bos_id {
                out.push(b);
            }
        }

        // SentencePiece convention: ASCII spaces become ▁. Do NOT prepend a
        // standalone ▁ to the whole prompt — that emits spurious token 236743
        // and diverges from llama.cpp (e.g. "3x3=" must be [3,x,3,=] not [▁,3,x,3,=]).
        let mut work = String::new();
        for ch in text.chars() {
            if ch == ' ' {
                work.push_str(SP_SPACE);
            } else {
                work.push(ch);
            }
        }

        // Greedy longest-prefix match.
        let bytes = work.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let mut matched: Option<(usize, u32)> = None;
            // Try the longest possible prefix first. We probe at char
            // boundaries by walking `work` from `i`.
            let remaining = &work[i..];
            let at_word_start = i == 0 || work[..i].ends_with(SP_SPACE);
            let mut end = remaining.len();
            while end > 0 {
                // Snap to char boundary.
                while end > 0 && !remaining.is_char_boundary(end) {
                    end -= 1;
                }
                if end == 0 {
                    break;
                }
                let candidate = &remaining[..end];
                if let Some(&id) = self.vocab.get(candidate) {
                    matched = Some((end, id));
                    break;
                }
                // At a word boundary, also try ▁-prefixed piece (SP convention).
                if at_word_start {
                    let mut prefixed = String::with_capacity(SP_SPACE.len() + end);
                    prefixed.push_str(SP_SPACE);
                    prefixed.push_str(candidate);
                    if let Some(&id) = self.vocab.get(prefixed.as_str()) {
                        matched = Some((end, id));
                        break;
                    }
                }
                end -= 1;
            }

            match matched {
                Some((adv, id)) => {
                    out.push(id);
                    i += adv;
                }
                None => {
                    // Skip one char with <unk>.
                    let ch_len = remaining
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(1);
                    out.push(self.unk_id);
                    i += ch_len;
                }
            }
        }
        out
    }

    /// Decode a slice of token ids back into a string. Strips ▁ markers.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            let idx = id as usize;
            if idx >= self.id_to_token.len() {
                continue;
            }
            let tok = &self.id_to_token[idx];
            if let Some(b) = self.bos_id {
                if id == b {
                    continue;
                }
            }
            // Replace ▁ with a space.
            for ch in tok.chars() {
                if ch == '\u{2581}' {
                    s.push(' ');
                } else {
                    s.push(ch);
                }
            }
        }
        s
    }
}

fn read_special_id(weights: &ModelWeights, key: &str) -> Option<u32> {
    match weights.metadata.get(key)? {
        GgufValue::U32(x) => Some(*x),
        GgufValue::I32(x) => Some(*x as u32),
        GgufValue::U64(x) => Some(*x as u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_vocab_3x3() -> GemmaTokenizer {
        let mut vocab = HashMap::new();
        let id_to_token = vec![
            "<pad>".to_string(),
            "<eos>".to_string(),
            "<bos>".to_string(),
            "<unk>".to_string(),
            "=".to_string(),
            "x".to_string(),
            "3".to_string(),
            SP_SPACE.to_string(),
            format!("{SP_SPACE}3"),
        ];
        for (i, s) in id_to_token.iter().enumerate() {
            vocab.insert(s.clone(), i as u32);
        }
        GemmaTokenizer {
            vocab,
            id_to_token,
            bos_id: Some(2),
            eos_id: Some(1),
            unk_id: 3,
        }
    }

    #[test]
    fn encode_3x3_matches_llama_without_leading_space_token() {
        let tok = toy_vocab_3x3();
        assert_eq!(tok.encode("3x3=", false), vec![6, 5, 6, 4]);
    }

    #[test]
    fn empty_input_returns_only_bos_when_requested() {
        let tok = GemmaTokenizer {
            vocab: HashMap::new(),
            id_to_token: vec!["<unk>".into(); 16],
            bos_id: Some(2),
            eos_id: Some(1),
            unk_id: 3,
        };
        assert_eq!(tok.encode("", true), vec![2]);
        assert_eq!(tok.encode("", false), Vec::<u32>::new());
    }
}
