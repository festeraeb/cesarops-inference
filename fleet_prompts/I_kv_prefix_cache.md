You are an expert in Rust + wgpu inference engines + KV cache management. Build a prompt-prefix KV cache reuse system for our kobold-compatible inference server.

## Hardware context
- Pascal Tesla P100 16 GB HBM2, GTX 1070 8 GB GDDR5
- Current engine: Rust + wgpu, axum HTTP server, KoboldCPP-compatible /api/v1/generate
- Single GPU per process for now (multi-model registry queued separately)
- Qwen 1.5B Q6_K GQA model: hidden_dim=1536, n_kv_heads=2, head_dim=128, n_layers=28

## What we have
Existing `KvCache` type stores K and V tensors as flat append-only buffers per layer. Position counter `pos` advances on each generated token. No prefix-aware logic today.

`server.rs` generate handler:
1. Tokenize prompt
2. ChatML-wrap (optional)
3. Forward pass each prompt token sequentially (cold prefill)
4. Sample loop until EOS or max_tokens

For chat workloads with long system prompts (800-2000 tokens), the cold prefill dominates latency on every turn. Subsequent turns repeat 90%+ of the prefix.

## Deliverable: KV prefix cache

### Core data structure

Token-hash trie with generational arena backing for O(1) eviction:

```rust
// src/kv_prefix_cache.rs

pub type TokenHash = u64;
pub type NodeId = u32;  // index into arena

pub struct KvPrefixCache {
    arena: Vec<KvNode>,
    free_list: Vec<NodeId>,
    root: NodeId,
    lru: VecDeque<NodeId>,
    total_tokens: usize,
    max_tokens: usize,
}

pub struct KvNode {
    parent: Option<NodeId>,
    children: HashMap<TokenHash, NodeId>,
    kv_slice: Option<KvSlice>,
    depth: u32,
    last_access_epoch: u64,
}

#[derive(Clone)]
pub struct KvSlice {
    pub start_pos: u32,
    pub len: u32,
    pub layer_data_handle: KvDataHandle,  // opaque ref to GPU-side cached bytes
}
```

### API surface

```rust
impl KvPrefixCache {
    pub fn new(max_tokens: usize) -> Self;

    /// Returns (matched_prefix_length, kv_slice) for the longest cached prefix.
    pub fn prefix_match(&self, tokens: &[TokenHash]) -> Option<(usize, KvSlice)>;

    /// After successful generation, commit the new prefix into the cache.
    pub fn commit(&mut self, tokens: &[TokenHash], kv: KvSlice);

    /// LRU eviction triggered when total_tokens > max_tokens.
    pub fn evict_lru(&mut self);

    /// Stats for telemetry.
    pub fn stats(&self) -> CacheStats;
}
```

### Critical behaviors I need addressed

1. **Hash function**: use a fast non-crypto 64-bit hash (xxhash? FxHash?) over the token bytes, NOT just the token id directly. Adjacent tokens with similar IDs shouldn't collide in the trie.

2. **Tokenizer-aware key**: the same text can tokenize differently depending on chat template wrapping. Hash should include the BOS/EOS/role markers as separate tokens. Chat template applied BEFORE hashing.

3. **Arena vs HashMap-of-HashMap**: explicitly use a `Vec<KvNode>` with NodeId indexing. This makes eviction O(1) (return NodeId to free_list) and avoids the orphaned-children problem of nested HashMaps.

4. **Eviction policy**: LRU by node, not by token. Evict leaf nodes with kv_slice first, then internal nodes only when all children are gone. Track via `last_access_epoch: u64` updated on every prefix_match traversal.

5. **Budget calculation for our hardware**:
   - P100 16 GB: max_tokens = 65536 (~67 MB at fp16 KV for Qwen 1.5B GQA)
   - GTX 1070 8 GB: max_tokens = 16384 (~17 MB at fp16 KV)
   - Auto-derive from VRAM in constructor: `max_tokens = (vram_bytes / 4) / kv_bytes_per_token`

6. **KvSlice → GPU memory**: don't copy KV bytes per cache entry. Store opaque handles into the existing KvCache's GPU buffer. Cache holds METADATA + position ranges, GPU buffer holds the actual tensor data. On match, the engine restores `pos` to `start_pos + len` and continues from there.

7. **Concurrency**: parking_lot::RwLock around the whole cache. Reads (prefix_match) take read lock; commit and evict take write lock. Multiple in-flight /api/v1/generate requests on the same model share the cache safely.

### Generate-loop integration

Show the patch to `src/server.rs` `generate()` handler:

```rust
// BEFORE: full prefill on every request
for &token in &prompt_tokens {
    decoder.forward(/* token by token */, &mut kv_cache);
}

// AFTER: prefix-match, then prefill only the tail
let token_hashes: Vec<TokenHash> = prompt_tokens.iter().map(hash_token).collect();
let prefix_len = match cache.prefix_match(&token_hashes) {
    Some((len, kv_slice)) => {
        kv_cache.restore(kv_slice);
        len
    }
    None => 0,
};

for &token in &prompt_tokens[prefix_len..] {
    decoder.forward(/* token by token */, &mut kv_cache);
}

// After generation completes, commit the new prefix span
cache.commit(&token_hashes, kv_cache.snapshot_slice(start_pos, current_pos));
```

### KvCache API additions needed

I need to extend our existing KvCache type with:
- `restore(slice: &KvSlice)` — set internal pos to slice.start_pos + slice.len, mark cached bytes as live
- `snapshot_slice(start, end) -> KvSlice` — record opaque handle to the buffer range without copying
- `rollback_to(pos)` — used by speculative decoding (separate prompt) but conceptually similar; share implementation if practical

Provide the Rust struct extensions for KvCache + the wgpu buffer side-effects (which buffer ranges get retained, how reset interacts with the prefix cache's references).

### Failure modes (be explicit)

What happens when:
- a /generate request with sampling params (seed, temperature) different from the cached prefix's original generation arrives. Answer: doesn't matter — sampling params don't affect KV cache content, only the FINAL token logits sampling. Prefix is reusable across sampling configs. Confirm.
- the model is unloaded mid-cache-entry-lifetime. Cache must be dropped wholesale; can't survive a model swap.
- two concurrent requests both miss the same prefix at the same time and both compute it. Avoid double-work via "in-flight" entry placeholder that subsequent requests await on.
- eviction fires while a request is mid-traversal. The read-lock prevents this; show the lock pattern explicitly.

### Worked example

Show the math for: 800-token system prompt + 50-token user turn, on second request:
- Cold first request: 850 tokens × forward pass time
- Warm second request: 850 - 800 = 50 tokens × forward pass + cache lookup (~10 µs)
- Speedup: ~17× on prefill phase, end-to-end ~5-10× perceived

### Constraints
- ~250-400 LOC total
- Drop into existing axum server (Arc<RwLock<KvPrefixCache>> in InferenceState)
- No new shaders; pure CPU-side bookkeeping over existing GPU buffers
- naga / wgpu changes only if KvCache extension requires them

## Output
Three files:
1. `src/kv_prefix_cache.rs` — full implementation
2. `src/kv_cache.rs` — diff/extension for the new methods
3. `src/server.rs` — diff for the generate handler integration

Plus a stats endpoint patch to `/api/v1/model` exposing cache hit rate, total cached tokens, eviction count.
