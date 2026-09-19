You are an expert in Rust + wgpu + WGSL targeting Pascal NVIDIA GPUs (sm_60 P100, sm_61 GTX 1070) via Vulkan. Build the Tier-2 throughput fixes for our Rust+wgpu inference engine: diagnostic readback elimination + uniform buffer pool + GPU-side stats buffer.

## Hardware context
- Pascal sm_60 / sm_61, Vulkan 1.2-1.3 backend via wgpu 0.20+
- Single command queue, no async compute, conservative barriers
- Current ceiling 2.2 t/s, ~476 dispatches per token
- map_async is a hard sync stall on P100 — use queue.write_buffer for upload

## Current bottleneck
Forward pass currently does per-token CPU readbacks of:
- hidden state (1536 floats) checked for NaN/Inf via .iter().any(...)
- logits (151936 floats) checked for min/max/mean via fold

These force GPU completion + copy + CPU pass on every step of every layer. Tier 2 bottleneck #2 in our internal analysis.

Plus: every dispatch creates a fresh wgpu::Buffer for its uniform/push-constant data. Per-dispatch allocation churn at 476 dispatches/token compounds on Pascal driver allocator.

## Deliverable 1: Compile-time + runtime diagnostic gate

```rust
// src/diagnostics.rs

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DiagnosticLevel {
    Off,
    ErrorOnly,
    Debug,
}

pub struct Diagnostics {
    pub level: DiagnosticLevel,
}

impl Diagnostics {
    pub fn from_env() -> Self { /* read CESAROPS_DIAG env var */ }
}
```

Provide:
1. The full Diagnostics struct with `from_env()` reader (matches CESAROPS_DIAG=off|error|debug, default off)
2. Zero-cost macros / inline functions:
   - `log_logits(d, &logits)` — full statistics, gated
   - `check_nan(d, &slice)` — NaN/Inf scan, gated
   - `dump_buffer(d, label, &slice)` — full hex dump, gated to Debug only
3. The double-gate pattern (`#[cfg(feature = "engine-debug")]` + runtime level check) so Off mode compiles to a no-op
4. Show concrete replacement of these existing call sites in src/server.rs (around line 220):
   ```
   let has_nan = hidden_state.iter().any(|x| x.is_nan() || x.is_infinite());
   let logits_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
   info!("Logits: min={:.4}, max={:.4}, mean={:.6}", ...);
   ```
5. Cargo.toml feature flag declaration for `engine-debug`

## Deliverable 2: Uniform buffer pool

```rust
// src/uniform_pool.rs

pub struct UniformPool {
    buffer: wgpu::Buffer,
    size: usize,
    offset: usize,
    alignment: usize,
}
```

Requirements:
1. Single large wgpu::Buffer allocated at construction (8-16 MB)
2. Bump-pointer suballocation, returns (offset, &buffer)
3. Ring-wrap on overflow (frame-synced — guarantee no in-flight reads at wrap point)
4. Caller writes via `queue.write_buffer(&pool.buffer, offset, bytes)` — NOT map_async
5. Alignment cached from device limits (`min_uniform_buffer_offset_alignment`)
6. Reset method `reset_after(submission_idx, device)` — uses Maintain::WaitForSubmissionIndex to enforce GPU completion before reuse
7. Concrete usage example wired into a single dispatch site showing before/after pattern
8. ~100 LOC max, naga-validated

CRITICAL: free list is dead code under forward-pass-scope reset. Pure bump allocator. No Mutex/Arc needed — pool is single-threaded per GPU context.

## Deliverable 3: GPU-side stats buffer (NaN/min/max without CPU readback)

The trick: instead of CPU scanning logits/hidden state, have the GPU write a tiny stats struct that the CPU only reads on Debug mode (and even then only every N tokens, not every token).

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LogitStats {
    pub min_bits: u32,    // sort-friendly bit-flipped f32
    pub max_bits: u32,    // sort-friendly bit-flipped f32
    pub has_nan: u32,
    pub _pad: u32,
}
```

Requirements:
1. WGSL atomic-on-u32 stats update (WGSL has NO atomic-on-f32 — use the sign-flip bit-cast trick: positive floats sort by bit pattern, flip sign bit on negatives, atomic the bit pattern)
2. Drop-in WGSL helper function `update_stats(x: f32)` callable from existing kernels
3. Rust-side `LogitStats::decode()` that returns (min: f32, max: f32, has_nan: bool)
4. Integration into the existing `attention_pc` and `matvec_*` shaders — show the binding number, the @group/@binding decoration, the WGSL fn that gets injected
5. CPU readback strategy: only read stats buffer on Debug mode, and only every 16 tokens, not every step. Use queue.submit + poll(Wait) AFTER the regular forward pass dispatch, never block inside the layer loop.

## Constraints
- wgpu 0.20+ (we may be on slightly older — flag if API differs)
- Pascal P100 (sm_60) and GTX 1070 (sm_61)
- No async compute queues, single submit queue
- naga must validate everything; no chromium_experimental_push_constant beyond what we already use
- Keep total addition to engine ~250 LOC across the three deliverables

## Expected gain (your estimate)
- Diagnostic strip: +20-40%
- UBO pool: +10-25%
- Stats buffer: subsumes diagnostic strip on debug paths (no extra gain in production, but enables observability without performance regression)

## Output format
Three Rust files (`src/diagnostics.rs`, `src/uniform_pool.rs`, `src/gpu_stats.rs`) + WGSL snippet for the stats helper + Cargo.toml feature flag declaration + 3-5 concrete integration call sites with before/after diff annotations. Keep it copy-paste ready.
