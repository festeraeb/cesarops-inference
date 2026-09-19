You are an expert in WGSL + wgpu + GGUF Q6_K. Wire native Q6_K matvec dispatch in our cesarops-inference engine to unblock loading 7B+ Q6_K models on a 16 GB Pascal P100 without F32 expansion.

## Hardware context
- Pascal P100 (sm_60), 16 GB HBM2, no DP4A on sm_60 specifically
- GTX 1070 (sm_61) — DP4A available but unused for Q6_K
- wgpu 0.20+ via Vulkan
- Single command queue, conservative barriers (P100 storage-buffer-hazard workaround already in place)

## Current state — exact bottleneck

`src/tensor_loader_safe.rs` line 56-65:
```rust
pub fn to_chunker_dtype(&self) -> Dtype {
    match self {
        Self::F32 => Dtype::F32,
        Self::F16 => Dtype::F16,
        Self::Q4_K | Self::Q4_0 | Self::Q4_1 => Dtype::Q4K,  // stays native
        // Everything else gets dequantized to F32 for GPU compute
        _ => Dtype::F32,
    }
}
```

So Q6_K weights get dequanted to F32 at load. For Qwen-Coder-14B Q6_K
that's 12 GB on disk → 48 GB on GPU. OOM by 3× even before KV/scratch.

## What we already have

- `shaders/matvec_q6k_fused.wgsl` — compile-staged from round 6, never
  dispatched. Implements packed Q6_K dequant + dot product directly
  from raw 210-byte block layout (128 ql + 64 qh + 16 scales + f16 d).
- `src/pipeline_init.rs` — pipeline build for matvec_q6k_fused already
  exists, layout matches our other matvec dispatches
- `src/forward_pass.rs::dispatch_matvec_bias` — the dispatch helper
  that needs the routing branch
- Q6_K dequant on CPU (`dequantize_to_f32` for Q6_K) already validated
  as correct against Python reference (DEBUG_LOG bug fixes #1-#5 baseline)

## Deliverables

### 1. Add Dtype::Q6K variant

`src/tensor_chunker.rs` Dtype enum:
```rust
pub enum Dtype {
    F32,
    F16,
    Q4K,
    Q6K,    // NEW
}
```

bytes_per_row:
- Q6_K: 256 elements per block, 210 bytes per block, 0.8203 bytes/elem
- For row of N cols: ceil(N / 256) * 210 bytes

Update all match arms in tensor_chunker.rs to handle Q6K (check
chunked-path eligibility, byte sizing, alignment).

### 2. Update to_chunker_dtype

```rust
pub fn to_chunker_dtype(&self) -> Dtype {
    match self {
        Self::F32 => Dtype::F32,
        Self::F16 => Dtype::F16,
        Self::Q4_K | Self::Q4_0 | Self::Q4_1 => Dtype::Q4K,
        Self::Q6_K => Dtype::Q6K,    // NEW — keep native
        _ => Dtype::F32,
    }
}
```

And `needs_dequant()` returns false for Q6_K.

### 3. Update tensor upload path

`tensor_loader_safe.rs::load_tensor_safe` around line 200-240:
when needs_dequant is false AND data_type is Q6_K, upload raw bytes
directly to a STORAGE buffer at native size. No dequant call. No
transpose (GGUF Q6_K layout matches shader expectation).

Show the exact diff against the current upload path. Preserve the
existing F32/F16/Q4K paths unchanged.

### 4. Wire dispatch routing

`src/forward_pass.rs::dispatch_matvec_bias` (around line 220-280):
add a Dtype match arm that routes Q6K weights to the
`matvec_q6k_fused` pipeline. Show:
- Bind group creation (4 bindings: input, q6k_weights, bias, output —
  bias optional, use 4-byte dummy buffer + push-constant flag like
  the K/V/Q stash design)
- Push constant struct matching matvec_q6k_fused.wgsl's expectation
- Dispatch workgroup count: `(out_dim + 127) / 128` for 128-thread WG
- Bias-fused variant if matvec_q6k_fused doesn't support inline bias,
  fall back to separate add (1 dispatch cost — track for follow-up)

### 5. Sanity-check the existing matvec_q6k_fused.wgsl

The shader was compile-staged but never dispatched against a model.
Read it (file is at `shaders/matvec_q6k_fused.wgsl`), verify:
- 210-byte block layout: ql_base @0, qh_base @128, scales_base @192,
  d (f16) @208
- Sign extension: 6-bit value v becomes (v - 32) signed for the
  -32..31 range
- Half/half scale interleave (is, is+2, is+4, is+6 within the half)
- @group(0) bindings match the 4-binding pattern:
  - 0: input f32 array
  - 1: q6k_weights u32 array (raw bytes as u32)
  - 2: bias f32 array (or 4-byte dummy)
  - 3: output f32 array
- @workgroup_size(128) for Pascal occupancy
- Push constants: hidden_dim, out_dim, k_blocks, flags, ... matching
  the kproj/vproj kernel signature pattern

If you find anything inconsistent, fix it inline and call out in a
comment what changed.

### 6. Smoke test addition

`scripts/smoke_test.sh` already runs Qwen 1.5B Q6_K. After this lands,
that test should still pass — but with native Q6_K dispatch instead
of F32 fallback. Add an env var marker:
```
CESAROPS_QUANT_PATH=native|fallback
```
that the engine prints at startup so we can see which path is active.

### 7. Validate against a 7B model

After the dispatch wire-up, the engine should be able to load
`/codebase/models/DeepSeek-R1-Distill-Qwen-7B-Uncensored.Q4_K_M.gguf`
(4.4 GB, Q4_K already wired) AND a hypothetical Qwen-Coder-14B Q6_K
(12 GB, Q6_K newly wired). For the 7B Q4_K_M case the path is
already native; verify it's still working. For Q6_K, the test target
is `Qwen2.5-Coder-14B-Instruct-abliterated-Q6_K.gguf` (12 GB, fits
P100 16 GB native).

Ship a manual test script that:
1. Loads the 14B Q6_K
2. Runs `--prompt "What is 2+2?" --max-tokens 15`
3. Confirms output contains "4"
4. Reports VRAM usage at peak (via nvidia-smi during run)

## Constraints
- No new shaders. Use existing matvec_q6k_fused.wgsl, fix bugs in
  place if any.
- Q6_K dequant CPU path stays intact (used at load for non-2D-weight
  tensors and as fallback)
- naga must validate everything
- Existing Qwen 1.5B Q6_K smoke test MUST still pass
- Existing GTX 1070 regression test MUST still pass

## Output
Three files modified + one new test script:
1. `src/tensor_chunker.rs` — Dtype::Q6K variant + match arm updates
2. `src/tensor_loader_safe.rs` — diff for to_chunker_dtype + upload path
3. `src/forward_pass.rs` — diff for dispatch_matvec_bias routing
4. `shaders/matvec_q6k_fused.wgsl` — fixes if you find any
5. `scripts/smoke_14b_q6k.sh` — new manual test

Plus a one-page memo on what you found in the existing shader (correctness check).
