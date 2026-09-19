Parity test from specialist for speculative decoding. To be saved as
`tests/speculative_parity.rs` AFTER the speculative.rs rewrite lands.

Note: this test depends on:
- `SpecConfig::seed` field (added in rewrite)
- `SpeculativeDecoder::new(draft, main, config, weights)` 4-arg constructor (added in rewrite)
- `cesarops_inference::InferenceArena::default()` (verify exists)
- `cesarops_inference::transformer::ModelWeights::default()` (verify exists)

The test uses a tiny config (vocab=100, hidden=64, 2 layers) so it
runs fast without GPU. May need to gate behind a `#[cfg(feature = "...")]`
if it requires GPU resources.

```rust
// tests/speculative_parity.rs
use cesarops_inference::speculative::{SpecConfig, SpeculativeDecoder};
use cesarops_inference::transformer::{TransformerDecoder, TransformerConfig};
use std::sync::Arc;

#[test]
fn parity_greedy_speculative() {
    let config = TransformerConfig {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 32,
        max_seq_len: 64,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
    };

    let arena = Arc::new(cesarops_inference::InferenceArena::default());
    let weights = Arc::new(cesarops_inference::transformer::ModelWeights::default());
    let decoder = Arc::new(TransformerDecoder::new(config.clone(), arena.clone()));

    let mut spec = SpeculativeDecoder::new(
        decoder.clone(),
        decoder.clone(),
        SpecConfig {
            draft_window: 4,
            temperature: 0.0,
            max_tokens: 32,
            seed: 42,
        },
        weights.clone(),
    );

    let prompt = vec![1u32, 2, 3, 4, 5];
    let spec_out = spec.generate(&prompt);

    // greedy reference
    let mut ctx = prompt.clone();
    let mut greedy = prompt.clone();

    for _ in 0..32 {
        let logits = decoder.forward(
            &mut vec![0.0; config.hidden_size],
            ctx.len(),
            &weights,
            &mut cesarops_inference::kv_cache::KvCache::new(
                config.num_layers,
                config.max_seq_len,
            ),
        );

        let probs = {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut e: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
            let s: f32 = e.iter().sum();
            for v in &mut e {
                *v /= s.max(1e-20);
            }
            e
        };

        let t = probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        ctx.push(t);
        greedy.push(t);
    }

    assert_eq!(
        &spec_out[..greedy.len().min(spec_out.len())],
        &greedy[..greedy.len().min(spec_out.len())]
    );
}
```

POLISH NOTES (apply when integrating):

1. The greedy reference creates a FRESH KvCache per token-step, which
   means it can't actually compare against speculative because spec
   uses persistent KvCache across the loop. Either:
   (a) move the KvCache outside the for loop (cleaner)
   (b) call decoder.forward() with the prompt all-at-once before the loop
   Option (a) is correct.

2. `cesarops_inference::InferenceArena::default()` — verify this Default
   impl exists. Looking at arena.rs is needed.

3. `ModelWeights::default()` — likewise verify. ModelWeights typically
   has GPU buffers + tensor maps, default may not be sensible. Test
   may need a "stub" weights builder instead.

4. The hidden_size=64 with num_heads=2 + head_dim=32 implies
   hidden = num_heads * head_dim = 64 ✓ — math checks out. Note that
   our existing forward expects hidden_dim = num_heads * head_dim.

5. The test asserts spec_out matches greedy on the prefix length, but
   spec_out may include the prompt itself in its output. Verify spec
   semantics — does generate() return tokens INCLUDING prompt or only
   newly generated? If including prompt, the comparison index is right.
   If only new tokens, need to skip prompt.len() in greedy too.

This test goes in tests/speculative_parity.rs after speculative.rs
rewrite lands. Until then, holding here.
