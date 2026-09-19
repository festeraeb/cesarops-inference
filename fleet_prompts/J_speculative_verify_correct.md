You are an expert in LLM speculative decoding + Rust + wgpu. Build the correct verify+commit path for our cesarops-inference engine. Earlier cluster output had a wrong verify kernel (compared attention scores instead of log-probs); send the corrected drop-in.

## Hardware + stack context
- Pascal Tesla P100 16 GB (target single-GPU first), GTX 1070 8 GB
- Rust + wgpu 0.20+, axum HTTP server, KoboldCPP-compatible /api/v1/generate
- Existing: TransformerDecoder, KvCache (append-only, fp32), sampling.rs (temp/top_p/rep_pen)
- Models: Qwen2/Qwen2.5 family Q4_K_M / Q6_K. GQA (n_heads=12, n_kv_heads=2 for 1.5B; varies)
- Multi-model registry queued — assume both draft and main models loadable on same GPU concurrently

## Correct math (do not deviate)

For each draft-proposed token t at position i:
- p_main = softmax(main_logits[i])[t]
- p_draft = softmax(draft_logits[i])[t]
- alpha = min(1, p_main / max(p_draft, 1e-20))
- u ~ Uniform(0,1)
- accept iff u <= alpha
- on first rejection: stop chain, sample repair token from softmax(max(0, p_main_dist - p_draft_dist) renormalized)

Both softmaxes MUST use online max-subtract (we have this trap recorded as a recurring cluster bug; fp32 exp() overflows for any logit > ~88).

## Deliverables

### 1. src/speculative.rs (~250 LOC)

Replace existing 148-LOC stub with:

```rust
pub struct SpeculativeDecoder {
    draft: Arc<TransformerDecoder>,
    main: Arc<TransformerDecoder>,
    draft_kv: KvCache,
    main_kv: KvCache,
    config: SpecConfig,
}

pub struct SpecConfig {
    pub draft_window: usize,        // K = 4-8 typical
    pub temperature: f32,
    pub max_tokens: usize,
}

impl SpeculativeDecoder {
    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32>;
    fn draft_propose(&mut self, ctx: &[u32]) -> Vec<(u32, Vec<f32>)>;  // (token, logits) pairs
    fn main_verify_batch(&mut self, ctx: &[u32], drafts: &[u32]) -> Vec<Vec<f32>>;
    fn rejection_sample(&self, draft_token: u32, p_main: &[f32], p_draft: &[f32]) -> Accept;
    fn sample_repair(&self, p_main: &[f32], p_draft: &[f32]) -> u32;
}

enum Accept { Yes, No(u32) }  // No carries the repair token
```

### 2. KvCache extension for rollback (src/kv_cache.rs diff)

New methods:
```rust
impl KvCache {
    pub fn snapshot_pos(&self) -> u32;                  // record current pos
    pub fn rollback_to(&mut self, pos: u32);            // reset internal pos to snapshot
    pub fn commit_through(&mut self, pos: u32);         // mark range [snapshot_pos..pos] as committed
}
```

Implementation: `pos` is just a counter; rollback sets it back. The
buffer bytes from rejected positions stay around but get overwritten
on the next forward. Don't copy or zero-fill — trust they're dead.

### 3. Batched main forward pass

Currently `decoder.forward()` runs 1 token per call. Speculative needs
K candidate tokens through main in one pass. Two options:

(a) Loop the existing forward K times — works correctly, no speedup.
(b) Genuine batched forward — show the changes to TransformerDecoder
    needed to accept `&[u32]` and return K logit vectors.

Ship (a) as the working version with a clear `#[allow(dead_code)]`
stub for (b) noting it's the next perf step. Don't block on (b).

### 4. Wire into server.rs /api/v1/generate

Behind a request flag `use_speculative: bool` (default false until
multi-model registry lands). When true and a draft model is loaded,
route to SpeculativeDecoder; otherwise fall through to current path.

### 5. Test harness

Two parity tests against the non-speculative path:
- Greedy mode (temp=0): speculative output MUST match non-speculative
  exactly (acceptance is deterministic at temp=0, alpha=1 always)
- Stochastic mode (temp=0.7, fixed seed): speculative output should
  have the same distribution properties (mean log-prob within 5% of
  non-speculative on a 256-token sample)

## Constraints
- No async compute queues, single submit queue
- No kernel changes — pure Rust math, calls existing forward()
- Online softmax max-subtract MANDATORY (write the helper, use it)
- p_draft.max(1e-20) clamp instead of (+1e-9) to avoid the fp32 garbage zone
- ~250-350 LOC total across speculative.rs + kv_cache.rs diff + server.rs wiring

## Output format
Three files (full speculative.rs replacement, kv_cache.rs diff, server.rs handler diff) + one test file. Ready to drop into the repo and `cargo build --release`.
