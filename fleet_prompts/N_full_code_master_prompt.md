You are a Rust + wgpu + WGSL specialist. We need fully compiling, drop-in code for our cesarops-inference engine — NOT pseudocode, NOT scaffolds, NOT CPU stubs. Every file you produce must:
- compile with `cargo build --release` against our existing crate
- match our existing module signatures (provided below)
- preserve the wgpu (Vulkan) GPU compute path — no CPU-only fallbacks unless explicitly noted
- include all imports, all error handling, and pass naga shader validation

This is a single self-contained brief. Read all sections before writing any code.

---

## CONTEXT — what cesarops-inference IS

A KoboldCPP-compatible LLM inference engine, single-binary axum HTTP server, wgpu/Vulkan backend, targeting Pascal NVIDIA hardware (Tesla P100 sm_60, GTX 1070 sm_61). Competing in the niche of "Pascal+wgpu+Rust+kobold-compat homelab inference."

Current state:
- Working end-to-end on Qwen2.5-Coder-1.5B Q6_K at 2.2 t/s peak
- 28 layers, ~476 dispatches/token, fp32 KV cache
- 7 GGUF quants loaded (F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS)
- 5 with native shader paths (F32, F16, Q4_K + matvec_q6k_fused.wgsl compile-staged but not dispatched)
- KoboldCPP routes wired: POST /api/v1/generate, GET /api/v1/model, GET /health, GET /api/extra/generate/check
- Diagnostics currently CPU-side per-token readbacks (Tier 2 bottleneck, ~30% perf cost)

Hard ceiling on a single P100 (cluster-confirmed): 12-22 t/s pure decode, 25-45 t/s with speculative decoding. 60+ t/s requires model size reduction or batching. We're not chasing TensorRT throughput — we're chasing TensorRT-architectural-maturity on hardware datacenters left behind.

---

## EXISTING MODULE SIGNATURES — match these exactly

### src/kv_cache.rs
```rust
pub struct LayerKvCache {
    pub keys: Vec<Vec<f32>>,    // per-position key vector
    pub values: Vec<Vec<f32>>,
}

pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub current_pos: usize,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self;
    pub fn push(&mut self, layer_idx: usize, k: Vec<f32>, v: Vec<f32>);
    pub fn advance(&mut self);
    pub fn len(&self) -> usize;
    pub fn get_keys(&self, layer_idx: usize) -> &[Vec<f32>];
    pub fn get_values(&self, layer_idx: usize) -> &[Vec<f32>];
}
```

### src/transformer.rs
```rust
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

pub struct TransformerDecoder {
    pub config: TransformerConfig,
    pub arena: Arc<InferenceArena>,
    pub weight_cache: Option<Arc<WeightCache>>,
    pub gpu: Option<Arc<GpuContext>>,
}

impl TransformerDecoder {
    pub fn new(config: TransformerConfig, arena: Arc<InferenceArena>) -> Self;
    pub fn with_weight_cache(mut self, cache: Arc<WeightCache>) -> Self;
    pub fn with_gpu(mut self, gpu: Arc<GpuContext>) -> Self;
    pub fn forward(
        &self,
        hidden_state: &mut Vec<f32>,
        position: usize,
        weights: &ModelWeights,
        kv_cache: &mut KvCache,
    ) -> Vec<f32>;  // returns logits
}
```

### src/tensor_chunker.rs
```rust
pub enum Dtype { F32, F16, Q4K }   // Q6K variant being added in fleet prompt K

pub struct ChunkedMatmulPipeline {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    // pipelines for each dtype
}

impl ChunkedMatmulPipeline {
    pub fn new(device: &wgpu::Device) -> Self;
    pub fn dispatch(&self, /* matmul args */);
}
```

### src/sampling.rs
```rust
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub rep_pen: f32,
    pub rep_pen_range: usize,
    pub stop_sequences: Vec<String>,
    pub banned_tokens: HashSet<u32>,
}

pub fn sample(logits: &mut [f32], params: &SamplingParams, recent_tokens: &[u32]) -> u32;
```

### src/server.rs (axum handlers — call site)
The /api/v1/generate handler currently runs:
```rust
for step in 0..max_tokens {
    let position = prompt_tokens.len() + step;
    let mut logits = decoder.forward(&mut hidden_state, position, weights, &mut kv_cache);
    kv_cache.advance();
    // ... NaN check, logits range log, sampling ...
}
```

This is where new systems plug in.

---

## DELIVERABLES — five fully-implemented files

Produce ALL FIVE files complete and ready to drop in. No placeholders, no `unimplemented!()`, no `// TODO:`. Every function must have a real body.

### File 1: src/diagnostics.rs (~120 LOC)

A diagnostic gate matching the operator's spec:
- `DiagnosticLevel` enum: Off | ErrorOnly | Debug
- `Diagnostics::from_env()` reads CESAROPS_DIAG=off|error|debug, default Off
- Inline functions `check_nan`, `log_logits`, `log_tensor_stats(name, &[f32])` that early-return when level != Debug
- Double-gated with `#[cfg(feature = "engine-debug")]` so Off compiles to no-op
- Replace these existing call sites in src/server.rs around line 220:
  ```rust
  let has_nan = hidden_state.iter().any(|x| x.is_nan() || x.is_infinite());
  let logits_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
  info!("Logits: min={:.4}, max={:.4}, mean={:.6}", ...);
  ```
  Provide the diff for those lines.
- Cargo.toml feature flag declaration.

### File 2: src/uniform_pool.rs (~100 LOC)

Ring-allocated UBO pool:
- Single large `wgpu::Buffer` (8 MB) at construction
- Bump-pointer alloc, returns (offset, &buffer)
- Ring-wrap on overflow (frame-synced)
- Writes via `queue.write_buffer`, NEVER `map_async`
- Alignment cached from `min_uniform_buffer_offset_alignment` (Pascal: 256 bytes)
- `reset_after(submission_idx, device)` uses `Maintain::WaitForSubmissionIndex` before reset
- NO free list (forward-pass-scope reset, no out-of-order lifetimes)
- Single-threaded per GPU context, no Mutex

### File 3: src/gpu_stats.rs (~150 LOC) + WGSL helper

GPU-side stats buffer for NaN/min/max without per-token CPU readback:

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct LogitStats {
    pub min_bits: u32,
    pub max_bits: u32,
    pub has_nan: u32,
    pub _pad: u32,
}
```

Requirements:
- WGSL helper file `shaders/stats_helper.wgsl` containing `fn update_stats(x: f32)` using atomic-on-u32 with the sign-flip-bit-cast trick (WGSL has NO atomic-on-f32). The trick: positive floats sort by bit pattern; for negatives flip sign bit so they sort correctly. atomicMin/atomicMax on the transformed u32, then bitcast back on readback.
- Rust-side `LogitStats::decode() -> (min: f32, max: f32, has_nan: bool)`
- Show how to inject the helper into existing kernels (matvec_pc, attention_pc) via a single binding addition (group=0 binding=5)
- CPU readback only on Debug mode AND only every 16 tokens. Use `queue.submit` + `device.poll(Maintain::Wait)` AFTER the regular forward pass dispatch — never block inside the layer loop.

### File 4: src/kv_prefix_cache.rs (~300 LOC)

Token-hash trie with generational arena backing for O(1) eviction. Match the spec from fleet prompt I exactly:

```rust
pub type TokenHash = u64;
pub type NodeId = u32;

pub struct KvPrefixCache {
    arena: Vec<KvNode>,
    free_list: Vec<NodeId>,
    root: NodeId,
    lru: VecDeque<NodeId>,
    total_tokens: usize,
    max_tokens: usize,
    epoch: u64,
}

pub struct KvNode {
    parent: Option<NodeId>,
    children: HashMap<TokenHash, NodeId>,
    kv_slice: Option<KvSlice>,
    depth: u32,
    last_access_epoch: u64,
}

pub struct KvSlice {
    pub start_pos: u32,
    pub len: u32,
    pub layer_data_handle: KvDataHandle,
}

impl KvPrefixCache {
    pub fn new(max_tokens: usize) -> Self;
    pub fn prefix_match(&mut self, tokens: &[TokenHash]) -> Option<(usize, KvSlice)>;
    pub fn commit(&mut self, tokens: &[TokenHash], kv: KvSlice);
    pub fn evict_lru(&mut self);
    pub fn stats(&self) -> CacheStats;
}

pub fn hash_token(token_id: u32, role_marker: u32) -> TokenHash;
```

Critical correctness:
- Use `ahash` or `rustc_hash::FxHash` (NOT deprecated `SipHasher13`)
- LRU tracks NodeId not single token
- Eviction: evict leaves with kv_slice first, then internal nodes whose children all gone
- Auto-derive max_tokens from VRAM at construction: `(vram_bytes / 4) / kv_bytes_per_token`
- Concurrency: wrap in `parking_lot::RwLock` at the call site (single instance shared across requests)
- For our Qwen 1.5B GQA: kv_bytes_per_token = 2 layers × 2 (K+V) × n_kv_heads(2) × head_dim(128) × 4 bytes = 2048 bytes, wait — recompute correctly per layer count = 28 layers × 2 × 2 × 128 × 4 = 57344 bytes/token at fp32
- Also extend `KvCache` with: `snapshot_slice(start, end) -> KvSlice` and `restore(&KvSlice)`

### File 5: src/speculative.rs (~300 LOC)

Replace existing 148-LOC stub. Match fleet prompt J spec.

Correct math (do not deviate):
- For each draft-proposed token t at position i:
  - p_main = softmax(main_logits[i])[t]
  - p_draft = softmax(draft_logits[i])[t]
  - alpha = min(1, p_main / p_draft.max(1e-20))
  - u ~ Uniform(0,1)
  - accept iff u <= alpha
- On first rejection: stop chain, sample repair token from `softmax(max(0, p_main_dist - p_draft_dist) renormalized)`
- Both softmaxes use online max-subtract (mandatory — fp32 exp() overflows for any logit > ~88)

```rust
pub struct SpeculativeDecoder {
    draft: Arc<TransformerDecoder>,
    main: Arc<TransformerDecoder>,
    draft_kv: KvCache,
    main_kv: KvCache,
    config: SpecConfig,
}

pub struct SpecConfig {
    pub draft_window: usize,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl SpeculativeDecoder {
    pub fn new(draft: Arc<TransformerDecoder>, main: Arc<TransformerDecoder>, config: SpecConfig) -> Self;
    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32>;
    fn draft_propose(&mut self, ctx: &[u32]) -> Vec<(u32, Vec<f32>)>;
    fn main_verify(&mut self, ctx: &[u32], drafts: &[u32]) -> Vec<Vec<f32>>;
    fn rejection_sample(&self, draft_token: u32, p_main: &[f32], p_draft: &[f32]) -> Accept;
    fn sample_repair(&self, p_main: &[f32], p_draft: &[f32]) -> u32;
}

enum Accept { Yes, No(u32) }  // No carries the repair token
```

KvCache extensions needed (extend src/kv_cache.rs):
- `snapshot_pos(&self) -> u32` — record current pos
- `rollback_to(&mut self, pos: u32)` — reset internal pos
- `commit_through(&mut self, pos: u32)` — mark range as committed

Wire into server.rs handler:
- Behind request flag `use_speculative: bool` (default false)
- Loop K times of existing decoder.forward() for the verify-batch (works correctly, no speedup yet — flag the genuine batched forward as next perf step)

---

## ABSOLUTE REQUIREMENTS

1. **Every file compiles standalone.** All imports present. All types resolved.
2. **No CPU mocks for systems with existing GPU paths.** Diagnostic readbacks may be CPU but use the GPU stats buffer pattern, not iter().any().
3. **No async compute queues, no multi-queue submission.** Single submit queue only. Pascal P100 storage-buffer-hazard already worked around by per-operation submit splits.
4. **No persistent kernels, no megakernel patterns.** Tonight is dispatch-per-stage with our existing barrier model.
5. **Test the math.** For speculative: include a 30-LOC parity test that runs greedy-mode (temp=0) and asserts speculative output matches non-speculative output exactly.
6. **Match existing conventions.** Look at how `forward_pass.rs::execute_layer` dispatches kernels for the pattern. Look at `attention_dispatch.rs` for push-constant + bind group structure. Look at `kv_cache.rs` for the existing public API surface.

## CONSTRAINTS YOU MAY HIT

- wgpu 0.20+ (we may be on 0.18 — flag if API differs and provide both)
- `Maintain::WaitForSubmissionIndex` is what fences look like in safe wgpu
- naga validates `var<push_constant>` without `enable chromium_experimental_push_constant;` on our setup
- Workgroup size 128 default for Pascal (subgroup_size=32, 4 subgroups per workgroup)
- Existing shaders in shaders/: matvec_pc, matvec_vec4_pc, matvec_bias_vec4_pc, matvec_q6k_fused, attention_pc, rope, rmsnorm, softmax, swiglu, dequant_q4km, dequant_q6k, dequant_iq4xs

## OUTPUT FORMAT

Five `=== FILE: path ===` sections in this order:
1. `=== FILE: src/diagnostics.rs ===` (full file body, no truncation)
2. `=== FILE: src/uniform_pool.rs ===`
3. `=== FILE: src/gpu_stats.rs ===` followed by `=== FILE: shaders/stats_helper.wgsl ===`
4. `=== FILE: src/kv_prefix_cache.rs ===` followed by `=== DIFF: src/kv_cache.rs ===` for the snapshot/restore additions
5. `=== FILE: src/speculative.rs ===` followed by `=== DIFF: src/server.rs ===` for the use_speculative wiring + `=== FILE: tests/speculative_parity.rs ===`

Plus at the very end:
- `=== DIFF: src/lib.rs ===` declaring the new modules
- `=== DIFF: Cargo.toml ===` for engine-debug feature + ahash/rustc_hash dep

Use `// existing` comments to mark lines that should remain unchanged in DIFF sections so the conductor can apply cleanly.

Do NOT include a long preamble. Do NOT explain the design before the code. Code first, brief notes after if needed.

Begin now.
