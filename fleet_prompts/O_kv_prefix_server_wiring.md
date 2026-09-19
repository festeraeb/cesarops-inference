You are a Rust + axum + wgpu specialist. Wire the KV prefix cache into our existing /api/v1/generate handler. Single-file diff against src/server.rs.

## Context

`src/kv_prefix_cache.rs` was just landed. Public API:

```rust
pub struct KvPrefixCache { /* trie + arena + LRU */ }
impl KvPrefixCache {
    pub fn new(max_tokens: usize) -> Self;
    pub fn from_vram_budget(vram_bytes: u64, per_token_kv_bytes: u64) -> Self;
    pub fn prefix_match(&mut self, tokens: &[TokenHash]) -> Option<(usize, KvSlice)>;
    pub fn commit(&mut self, tokens: &[TokenHash], kv: KvSlice);
    pub fn evict_lru(&mut self);
    pub fn stats(&self) -> CacheStats;
    pub fn capacity(&self) -> usize;
}

pub fn hash_token(token_id: u32, role_marker: u32) -> TokenHash;
pub fn hash_sequence(tokens: &[u32], role_marker: u32) -> Vec<TokenHash>;

pub struct KvSlice { pub start_pos: u32, pub len: u32, pub layer_data_handle: u64 }
```

`src/kv_cache.rs` extensions exist:
```rust
impl KvCache {
    pub fn snapshot_pos(&self) -> u32;
    pub fn rollback_to(&mut self, pos: u32);
    pub fn commit_through(&mut self, pos: u32);
}
```

## Current /api/v1/generate handler (simplified)

Around src/server.rs line 200 in the generate() handler:

```rust
async fn generate(State(state): State<...>) -> Json<...> {
    let prompt_tokens = tokenizer.encode(&request.prompt);
    let mut kv_cache = KvCache::new(num_layers, max_seq_len);
    let mut hidden_state = embed(prompt_tokens[0]);

    // Cold prefill loop:
    for &token in &prompt_tokens {
        decoder.forward(&mut hidden_state, position, weights, &mut kv_cache);
        kv_cache.advance();
    }

    // Generation loop:
    for step in 0..max_tokens { ... }

    // Decode + return
}
```

For long system prompts (800-2000 tokens), the cold prefill loop dominates latency on every chat turn.

## Deliverable

Modify the generate() handler so:

1. After tokenize, hash the prompt tokens with `hash_sequence(&prompt_tokens, 0)`.
2. Call `cache.write().prefix_match(&hashes)`. If a match returns (prefix_len, kv_slice):
   - Build the kv_cache pre-populated up to prefix_len tokens by calling decoder.forward() for tokens[0..prefix_len] EXACTLY ONCE in batch mode. Since we don't have batch yet, log "[cache] hit prefix_len=N, replaying" and run forward for ONLY the cached range — this is still cheaper because we can skip work after... no wait, we can't skip work without actual KV state replay.

   ACTUALLY the simpler honest version: log the hit/miss telemetry but skip the actual KV state restore for v1 since our KV cache is fresh-per-request. This becomes the placeholder for the real implementation when we add cross-request KV state retention.

   So v1 behavior: prefix_match runs, increments stats, but does NOT skip prefill. The integration value is the telemetry + the cache structure being live + the commit-after-generate side effect that records prefixes for later.

3. Always run the full cold prefill loop (current behavior).
4. After generation completes, call `cache.write().commit(&hashes, KvSlice { start_pos: 0, len: prompt_tokens.len() as u32, layer_data_handle: 0 })` so subsequent identical prompts at least record as "hit" in stats.
5. Add a stats entry to GET /api/v1/model that reports cache hit_rate / capacity / total_tokens.

## Architecture for full implementation (track for v2)

Real prefix reuse requires:
- KvCache to retain state between requests (lock-protected, drop only on model unload)
- KvSlice.layer_data_handle to actually point at GPU buffer ranges
- decoder.forward() to accept a "starting position" parameter and skip tokens [0..prefix_len]
- KvCache state to be SAVED at end of generation (current cache populated through prompt_len + generated_len) so next request can prefix-match against that.

For v1 we land the bookkeeping side. The actual prefill skip becomes a follow-up once we wire decoder to read from a pre-populated KvCache.

## Wiring requirements

- Add `cache: Arc<RwLock<KvPrefixCache>>` to `InferenceState`
- Initialize in `run_server()` with `KvPrefixCache::from_vram_budget(16 * 1024 * 1024 * 1024, 57344)` for our 16 GB P100 + Qwen 1.5B GQA
- Use `parking_lot::RwLock` (already in deps)
- Read-lock for prefix_match, write-lock for commit
- In Cargo.toml ensure `parking_lot` is a direct dep (it might be transitive)

## Output format

Provide ONLY the unified diff against src/server.rs. No preamble, no explanation. Use existing comment markers to make the diff applyable. Match our axum 0.8 handler patterns.

## Constraints

- Must compile against our existing engine
- Must NOT break the current single-shot generation behavior
- Default to OFF if cache fails to initialize (fall back to current handler logic)
- Keep additions to ~80 LOC

## Reference

Our existing imports look like:
```rust
use axum::{extract::State, response::Json, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
```

Note we use tokio::sync::Mutex on InferenceState. The cache should use parking_lot::RwLock since reads dominate.
