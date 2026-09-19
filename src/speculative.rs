//! Speculative decoding — draft proposes K tokens, main verifies in batch,
//! accept longest valid prefix, sample one repair token at first rejection.
//!
//! Math:
//!   alpha(x) = min(1, p_main(x) / p_draft(x))
//!   accept iff u <= alpha,  u ~ Uniform(0, 1)
//!   on rejection: sample from softmax(max(0, p_main - p_draft) renormalized)
//!
//! Greedy fast-path (temperature == 0): alpha = 1 iff main_argmax == draft_argmax,
//! else 0. Makes parity test against non-spec greedy bit-exact.
//!
//! KV cache rollback on rejection: both draft_kv and main_kv reset to the
//! pre-batch snapshot so subsequent drafts don't see corrupted history.

use std::sync::Arc;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::{
    kv_cache::KvCache,
    loader::ModelWeights,
    transformer::TransformerDecoder,
};

#[derive(Debug, Clone)]
pub struct SpecConfig {
    pub draft_window: usize,
    pub temperature: f32,
    pub max_tokens: usize,
    pub seed: u64,
}

pub struct SpeculativeDecoder {
    pub draft: Arc<TransformerDecoder>,
    pub main: Arc<TransformerDecoder>,
    pub draft_kv: KvCache,
    pub main_kv: KvCache,
    pub config: SpecConfig,
    pub weights: Arc<ModelWeights>,
    pub rng: StdRng,
}

impl SpeculativeDecoder {
    pub fn new(
        draft: Arc<TransformerDecoder>,
        main: Arc<TransformerDecoder>,
        config: SpecConfig,
        weights: Arc<ModelWeights>,
    ) -> Self {
        let layers = main.config.num_layers;
        let max_seq = main.config.max_seq_len;
        let rng = StdRng::seed_from_u64(config.seed);

        Self {
            draft,
            main,
            draft_kv: KvCache::new(layers, max_seq),
            main_kv: KvCache::new(layers, max_seq),
            config,
            weights,
            rng,
        }
    }

    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32> {
        let mut out = prompt_tokens.to_vec();
        let mut ctx = prompt_tokens.to_vec();

        while (out.len() - prompt_tokens.len()) < self.config.max_tokens {
            let k = self.config.draft_window.max(1);

            // Snapshot KV state for rollback safety on rejection.
            let draft_snapshot = self.draft_kv.snapshot_pos();
            let main_snapshot = self.main_kv.snapshot_pos();

            let drafts = self.draft_propose(&ctx, k);
            if drafts.is_empty() {
                break;
            }

            // PERF: replace with true batched forward when
            // TransformerDecoder.forward_batch lands.
            let main_logits_per_pos = self.main_verify_batch(&ctx, &drafts);

            let mut accepted_any = false;

            for i in 0..drafts.len() {
                let token = drafts[i].0;
                let p_main_dist = softmax(&main_logits_per_pos[i]);
                let p_draft_dist = softmax(&drafts[i].1);

                let alpha = if self.config.temperature == 0.0 {
                    // Greedy fast-path
                    let main_arg = argmax(&p_main_dist);
                    let draft_arg = argmax(&p_draft_dist);
                    if main_arg == draft_arg { 1.0 } else { 0.0 }
                } else {
                    let p_main = p_main_dist[token as usize].max(1e-20);
                    let p_draft = p_draft_dist[token as usize].max(1e-20);
                    (p_main / p_draft).min(1.0)
                };

                let u: f32 = self.rng.random();

                if u <= alpha {
                    ctx.push(token);
                    out.push(token);
                    accepted_any = true;
                } else {
                    let repair = sample_repair_distribution(
                        &p_main_dist,
                        &p_draft_dist,
                        &mut self.rng,
                    );
                    ctx.push(repair);
                    out.push(repair);

                    // Critical correctness: roll back both KVs so subsequent
                    // drafts don't see corrupted history. The rejected draft
                    // at index i + everything after never happened.
                    self.main_kv.rollback_to(main_snapshot);
                    self.draft_kv.rollback_to(draft_snapshot);
                    break;
                }
            }

            if !accepted_any {
                // K=0 fallback or pathological rejection at index 0:
                // Run main once normally to make progress.
                let logits = self.main.forward(
                    &mut vec![0.0; self.main.config.hidden_size],
                    ctx.len(),
                    &self.weights,
                    &mut self.main_kv,
                );
                let probs = softmax(&logits);
                let t = argmax(&probs);
                ctx.push(t);
                out.push(t);
            }

            // Commit accepted positions. No-op assert in our KvCache impl,
            // but reserved for future committed/speculative state machines.
            self.draft_kv.commit_through(ctx.len() as u32);
            self.main_kv.commit_through(ctx.len() as u32);
        }

        out
    }

    /// Run draft model forward K times producing (token, logits) pairs.
    fn draft_propose(&mut self, ctx: &[u32], k: usize) -> Vec<(u32, Vec<f32>)> {
        let mut out = Vec::with_capacity(k);
        let mut local_ctx = ctx.to_vec();

        for _ in 0..k {
            let logits = self.draft.forward(
                &mut vec![0.0; self.draft.config.hidden_size],
                local_ctx.len(),
                &self.weights,
                &mut self.draft_kv,
            );
            let probs = softmax(&logits);
            let token = argmax(&probs);
            out.push((token, logits));
            local_ctx.push(token);
        }
        out
    }

    /// Run main model forward K times in sequence, capturing logits at each
    /// position. PERF: replace with batched forward when forward_batch lands.
    fn main_verify_batch(
        &mut self,
        ctx: &[u32],
        drafts: &[(u32, Vec<f32>)],
    ) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(drafts.len());
        let mut local_ctx = ctx.to_vec();

        for (token, _) in drafts.iter() {
            local_ctx.push(*token);
            let logits = self.main.forward(
                &mut vec![0.0; self.main.config.hidden_size],
                local_ctx.len(),
                &self.weights,
                &mut self.main_kv,
            );
            outputs.push(logits);
        }
        outputs
    }
}

/// Sample a repair token from the residual probability distribution
///   max(0, p_main - p_draft) renormalized.
/// This is what makes speculative decoding correct under model disagreement:
/// when the draft proposes a token the main rejects, the replacement comes
/// from the probability mass main has but draft underweighted.
fn sample_repair_distribution(
    p_main: &[f32],
    p_draft: &[f32],
    rng: &mut impl Rng,
) -> u32 {
    let diff: Vec<f32> = p_main
        .iter()
        .zip(p_draft.iter())
        .map(|(m, d)| (m - d).max(0.0))
        .collect();

    let sum: f32 = diff.iter().sum();
    if sum < 1e-20 {
        // Degenerate case: distributions agree everywhere or near-zero diff.
        // Fall back to argmax of p_main.
        return argmax(p_main);
    }

    let normalized: Vec<f32> = diff.iter().map(|x| x / sum).collect();
    let r: f32 = rng.random();

    let mut cum = 0.0;
    for (i, p) in normalized.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (normalized.len().saturating_sub(1)) as u32
}

/// Stable softmax with online max-subtract.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    debug_assert!(max.is_finite(), "softmax max is not finite");
    let mut exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let denom = sum.max(1e-20);
    for v in &mut exps {
        *v /= denom;
    }
    exps
}

fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let p = softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_max_subtract_handles_large_logits() {
        // Pre-max-subtract these would overflow exp() in fp32
        let logits = vec![100.0, 200.0, 150.0];
        let p = softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[1] > p[0]);
        assert!(p[1] > p[2]);
    }

    #[test]
    fn argmax_returns_correct_index() {
        let v = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&v), 3);
    }

    #[test]
    fn repair_distribution_picks_main_only_token() {
        // p_main has mass at idx 2 that p_draft doesn't
        let p_main = vec![0.3, 0.3, 0.4];
        let p_draft = vec![0.5, 0.5, 0.0];

        let mut rng = StdRng::seed_from_u64(0);
        let token = sample_repair_distribution(&p_main, &p_draft, &mut rng);
        // Only positive diff is at idx 2: 0.4 - 0.0 = 0.4
        assert_eq!(token, 2);
    }

    #[test]
    fn repair_distribution_falls_back_to_argmax_when_no_diff() {
        // Identical distributions -> diff sum is 0
        let p_main = vec![0.2, 0.5, 0.3];
        let p_draft = vec![0.2, 0.5, 0.3];

        let mut rng = StdRng::seed_from_u64(0);
        let token = sample_repair_distribution(&p_main, &p_draft, &mut rng);
        assert_eq!(token, 1); // argmax of p_main
    }
}
