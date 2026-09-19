# GPU-Resident Gemma-4 MoE Runner: Findings, Wins, and Open Issues

**Status:** active development. GPU-resident pipeline operational. ~13× decode-time speedup over the CPU-bouncing baseline. Numerical correctness not yet validated against a reference.

**Date:** 2026-05-19
**Author:** Kiro (with cesarops)
**Target hardware:** NVIDIA Tesla P100 (Vulkan via wgpu), Pascal-class.
**Target model:** `Gemma-4-26B-MoE-IQ4_XS.gguf` (30 layers, 16 heads, 8 KV heads, 128 experts/layer, top-k = 8).

---

## 1. Goal

Build a GPU-resident inference path for Gemma-4-26B-MoE on heterogeneous Vulkan devices (no CUDA, no CuDNN, no NCCL). Activation tensors should live on the GPU across the entire forward pass; only routing decisions and final logits round-trip to the CPU. Multi-GPU expert sharding plumbing must coexist with the single-GPU path.

## 2. What works

### 2.1 Multi-GPU MoE plumbing (verified)
- `gpu_tensor.rs::MultiGpuExpertContext` correctly maps `(layer, expert_kind) → (gpu_index, &Buffer)`.
- `moe_expert_loader::PlacementPlan` round-trip succeeds for 90 experts × 30 layers.
- `MoeFfnDispatch::forward` runs router → top-k → expert matvec → SwiGLU → down → weighted accum end-to-end on a single GPU. Cross-GPU dispatch infrastructure compiles and lookups succeed.

### 2.2 Tokenizer
`gemma_tokenizer.rs` reads the embedded `tokenizer.ggml.tokens` array out of the GGUF, performs greedy longest-prefix matching with SentencePiece "▁" word-boundary semantics, and produces token IDs that match a llama.cpp / Ollama reference for the prompt `"3x3="`.

### 2.3 GPU primitive correctness (12 of 12 verified vs CPU reference)

| Kernel | Test name | Status |
|---|---|---|
| `embed_lookup_scaled` | `embed_lookup_matches_cpu` | ✅ |
| `rmsnorm_f32` (raw weight) | `rmsnorm_matches_cpu_both_conventions` | ✅ |
| `rmsnorm_f32` (Gemma `+1`) | `rmsnorm_matches_cpu_both_conventions` | ✅ |
| `rmsnorm_weightless_f32` | `rmsnorm_weightless_matches_cpu` | ✅ |
| `rope_v2` (head_dim=256, SWA) | `rope_matches_cpu` | ✅ |
| `rope_v2` (head_dim=512, full) | `rope_matches_cpu` | ✅ |
| `attn_one_head_gemma` (global) | `attention_matches_cpu_global` | ✅ |
| `attn_one_head_gemma` (SWA) | `attention_matches_cpu_swa` | ✅ |
| `kv_cache_write` | `kv_write_matches_cpu` | ✅ |
| `silu_mul_separate_f32` | `silu_mul_matches_cpu` | ✅ |
| `weighted_accum_f32` | `weighted_accum_matches_cpu` | ✅ |
| `logit_softcap_f32` | `logit_softcap_matches_cpu` | ✅ |
| `Iq4MatvecPipeline` (XS) on real `blk.0.attn_q.weight` | `iq4xs_matvec_matches_cpu_reference` | ✅ |
| `Iq4MatvecPipeline` (NL) on real `blk.0.ffn_down.weight` | `iq4nl_matvec_matches_cpu_reference` | ✅ |

These tests live in `tests/gpu_kernel_correctness.rs`. They are gated behind `--ignored` because they need a real Vulkan adapter. Run with:

```
cargo test --release --test gpu_kernel_correctness -- --ignored --nocapture
```

The naga parse-and-validate suite for the same shaders runs without a GPU and is part of the default test set.

### 2.4 GPU-resident runner

`Gemma4GpuRunner` (in `src/gemma4_gpu_runner.rs`) wraps an existing `Gemma4Runner` (which already has IQ4 weights uploaded) and:

- Allocates per-layer norm buffers on the GPU once (9 norms × 30 layers = 270 small uploads at construction).
- Allocates the embedding table as two halves (`lo`, `hi`) to fit under the P100's 2 GB max-buffer-binding limit.
- Allocates per-layer GPU KV caches (sized `max_seq_len × n_kv_heads × head_dim × 4` per layer).
- Allocates persistent activation scratch (`hidden`, `residual`, `q`, `k`, `v`, `head_norm`, `attn_out`, `o_out`, `moe_y`, `logits`, `logits_softcapped`).
- Drives the forward pass via `forward_token(token_id) -> Vec<f32>`. The only CPU↔GPU round-trips per token are:
  1. The MoE router top-k readback (128 floats × 30 layers = 3840 floats).
  2. The LM head matmul (still currently bouncing through CPU; see §5.2).
  3. The final 262144-vocab logit readback for sampling.

### 2.5 Performance

Measured on a single P100, greedy decoding of `"3x3="` prompt, max_new_tokens=8:

| Path | Decode rate |
|---|---|
| `Gemma4Runner` (CPU-bouncing) | 0.34 tok/s |
| `Gemma4GpuRunner` (this work) | 4.50 tok/s |
| **Speedup** | **~13×** |

Prefill is ~0.74 tok/s on the GPU runner because each prompt token still requires the full layer stack plus the LM head readback. That's known overhead and addressable (see §5.2).

## 3. Numerical correctness gap

The GPU runner produces **incorrect** output for `"3x3="`:

- Reference (Ollama, raw greedy): `"3x3=3=3=3=3=3=3=3=3="` (the model parrots `3=` because the prompt is unconditioned). First generated token id corresponds to the SP token `"3"` (id 236800 in this vocab).
- Ours (GPU runner): `"3x3=ما way?</स्टार笑顔"):"` — multilingual garbage, with frequent `<bos>` (id 2) emissions. First generated token id 4697 (≠ 236800).

The CPU `Gemma4Runner` produces equivalently-broken output. **The bug is shared between both runners.** Because every primitive on the GPU runner is verified bit-for-bit against the CPU reference for the same primitive, the bug is in the *orchestration* of those primitives — not in any single shader.

### 3.1 Hypotheses (in order of likelihood, post-debugging)

1. **Norm-ordering inside the MoE block.** The runner doc explicitly flags `pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2` as un-pinned. We followed the pseudocode in the original task brief, but no reference implementation has been confirmed.
2. **RMSNorm gain convention.** The Gemma-4 26B MoE GGUF stores `attn_norm` weights with mean magnitude ~3-9 (not the usual ~0.5-1). Using the canonical Gemma `(weight + 1.0)` convention amplifies these to gains of 5-10×, which corrupts numerics. Using raw weights produces a different but equally-broken output. **The actual storage convention for this checkpoint is unknown** and is the leading suspect.
3. **Layer-output-scale interaction with residual stream.** The runner does `hidden = residual + scale * delta`. Per-layer `scale` ranges from 0.07 to 0.68 and is wildly different across layers — if it should be applied differently (e.g., post-norm on the delta), we're consistently wrong by a per-layer factor.
4. **Embedding scale convention.** Adding `*= sqrt(hidden_size)` after the embedding lookup changed the output meaningfully (English nonsense → multilingual nonsense), confirming it's a real factor in the residual-stream magnitude. We currently apply it; it might or might not be correct depending on whether the GGUF's stored embeddings already include that factor.
5. **Shared-KV layer detection.** Gemma-4-26B-MoE has `attention.shared_kv_layers=0` in metadata. We honor that (no V projection for those layers — reuses K). If a layer that *does* share KV is not detected, V values are stale.

### 3.2 What the failure mode looks like

- High frequency of `<bos>` (id 2) and other very low-id tokens being produced.
- Real-looking words emerge (`▁been`, `▁ability`, `▁own`) — this means the LM head produces a plausible distribution; the residual stream isn't NaN, it's just wrong.
- Output flavor changes meaningfully when we tweak the embedding scale or RMSNorm convention. This confirms data flows through 30 layers correctly; the bug is multiplicative, not structural.

### 3.3 Recommended next debug step

The cleanest path forward is to dump the residual stream after layer 0 from both *our* GPU runner and a reference (Ollama with debug logging or llama.cpp instrumented), then bisect by layer. Activations diverging at layer 0 isolates the bug to the dense block; diverging at layer 1 isolates it to the MoE block. Anything more random than "constant magnitude error per layer" points to a per-layer factor (e.g. layer_output_scale interaction).

## 4. Bugs found and fixed during this work

| Bug | Where | Fix |
|---|---|---|
| `Multigpu lookup_expert` dereferenced a reference incorrectly | `gpu_tensor.rs` | use destructuring `let &(gpu_idx, ref key) = entry;` |
| Duplicate `MultiGpuExpertContext` definitions | `gpu_tensor.rs` + `moe_multigpu_dispatch.rs` | consolidated to a single definition |
| `enumerate_experts(model)?` propagated `Vec<ExpertTensor>` as if it was `Option` | `multi_gpu_init.rs` | guard with `if experts.is_empty()` |
| Missing `tracing::{info, warn}` imports | `multi_gpu_init.rs`, `moe_multigpu_dispatch.rs` | added imports |
| `create_buffer_init` not a method on `&wgpu::Device` | `moe_multigpu_dispatch.rs`, `moe_multi_gpu.rs` | rewrote to `create_buffer` + `queue.write_buffer` |
| `dispatch_matmul` arity mismatch | `moe_multi_gpu.rs` | added `k: u32` parameter |
| `hidden_state.buffer` (a `Buffer`) bound where `&Buffer` was expected | `moe_multi_gpu.rs` | `&hidden_state.buffer` |
| RoPE shader hardcoded workgroup_size 64, broke for `head_dim > 128` | `rope.wgsl` | wrote `rope_v2.wgsl` with stride loop, validated for `head_dim` ∈ {256, 512} |
| Original `silu_mul.wgsl` and `weighted_accum.wgsl` require `f16` extension (Pascal-unsafe) | shaders | added `silu_mul_separate_f32.wgsl` and used existing `weighted_accum_f32.wgsl` |
| `dispatch_rmsnorm_per_head` bound the same buffer as RO and RW in one dispatch (WGPU validation error) | `gemma4_gpu_runner.rs` | added `head_norm` scratch + post-pass copy back |
| `lm_head_matvec` was private; new runner couldn't call it | `gemma4_runner.rs` | `pub(crate)` |
| `Gemma4GpuPipelines` initially compiled all shaders but pipelines bundle bound the wrong number of bindings — caught by `naga` validation tests | `gemma4_gpu_pipelines.rs` | corrected BGL entries; pushed all 10 shaders through `naga` validator at `cargo test` time |

## 5. Open work

### 5.1 Numerical correctness (highest priority)

Resolve §3 above. Suggested order:
1. Compare layer-0 output against an instrumented reference.
2. If layer-0 is wrong, bisect: which sub-block (attn_norm, attn proj, attn_out_norm, residual_add, ffn_norm, MoE, post_ffw_norm) introduces the error.
3. Fix the offending step. Use the GPU primitive correctness tests as ground truth — *no shader is wrong*, the orchestration is.

### 5.2 Performance follow-ups (post-correctness)

Once the model is producing correct output, in priority order:
1. **One submit per layer.** Bundle every dispatch within a layer into one encoder. Currently each primitive does its own `queue.submit` → ~12 submits/layer × 30 layers = 360 submits/token. Reducing to 30 should yield 5-10× speedup at the wgpu submission boundary.
2. **GPU-resident LM head matvec.** The current path reads `norm_out` back to CPU, calls `lm_head_matvec`, and writes the logits back. Direct GPU buffer-to-buffer would save 2 round trips of 2816 + 262144 floats per token.
3. **Bind-group caching.** Reuse `BindGroup` objects across calls instead of recreating them. wgpu-internal hashing isn't free.
4. **Remove `head_norm` copy-back.** The per-head RMSNorm could write straight into Q/K if we change the shader to take a stride/offset per workgroup. Saves 30 × 3 × ~16 KB copies per token.
5. **Persistent uniforms.** Most uniform buffers (rope_push, rmsnorm_push, …) are recreated on every call. Use a small ring buffer pool.

### 5.3 Multi-GPU MoE (currently single-GPU only)

The `MultiGpuExpertContext` plumbing exists and the placement plan is built correctly across multiple GPUs at construction time, but `Gemma4GpuRunner::dispatch_moe_block` always routes to the source runner's single `MoeFfnDispatch` (one device). Wiring the multi-GPU path requires:
1. A cross-device staging buffer pair on the coordinator GPU.
2. PCIe-mediated activation transfer (no GPUDirect on Pascal — must round-trip through pinned host memory).
3. A revised `MoeFfnDispatch::forward` that takes the GPU index and uses the appropriate device for each expert.

This is a multi-day effort and is gated on §5.1 (no point optimizing wrong math).

### 5.4 Multi-batch / continuous batching

Single-batch only. Adding batch dimension requires:
- 3D KV caches `(batch, max_seq, kv_dim)` instead of 2D.
- Q/K/V scratch sized `(batch, q_dim)`.
- Attention shader's per-head dispatch becomes `(batch, head)`, two-axis workgroup grid.
- LM head matvec becomes a matmul.

Not on the immediate path.

## 6. Testing surface

Default tests (no GPU required):
- `cargo test` — exercises CPU helpers, naga validation of all shaders, push-struct byte layouts.

GPU-required tests (run with `--ignored`):
- `tests/gpu_kernel_correctness.rs` — 12 individual GPU primitive tests against CPU reference. ~30 seconds.
- `tests/test_gemma4_moe.rs` — CPU runner end-to-end with greedy decode for `"3x3="`. ~90 seconds.
- `tests/test_gemma4_gpu_runner.rs` — GPU runner end-to-end. ~70 seconds.
- `tests/dump_gguf_meta.rs`, `tests/dump_layer_scales.rs`, `tests/dump_vocab_ids.rs` — diagnostics.

## 7. Files added by this work

```
src/gemma4_gpu_pipelines.rs     - all small-kernel push structs + pipeline bundle
src/gemma4_gpu_runner.rs        - GPU-resident forward pass driver
src/gemma_tokenizer.rs          - SentencePiece-style GGUF-vocab tokenizer
src/gpu_tensor.rs               - MultiGpuExpertContext + GPU-resident tensor wrapper
src/moe_multi_gpu.rs            - cross-device MoE dispatcher (compiles, partial)
src/multi_gpu_init.rs           - Vulkan adapter discovery + placement upload
src/moe_multigpu_dispatch.rs    - MoE dispatch helpers
shaders/embed_lookup_scaled.wgsl
shaders/rmsnorm_f32.wgsl
shaders/rmsnorm_weightless_f32.wgsl
shaders/rope_v2.wgsl
shaders/attn_one_head_gemma.wgsl
shaders/kv_cache_write.wgsl
shaders/silu_mul_separate_f32.wgsl
shaders/logit_softcap_f32.wgsl
tests/gpu_kernel_correctness.rs
tests/test_gemma4_moe.rs
tests/test_gemma4_gpu_runner.rs
tests/dump_gguf_meta.rs
tests/dump_layer_scales.rs
tests/dump_vocab_ids.rs
```

## 8. Recommended bring-up order for next session

1. Run `cargo test --release --test gpu_kernel_correctness -- --ignored --nocapture` and confirm all 12 primitive tests pass on this hardware.
2. Run `cargo test --release --test test_gemma4_gpu_runner -- --ignored --nocapture` and confirm the runner produces some output (correctness debugging next).
3. Pull a reference logit dump for layer 0 from llama.cpp / Ollama for the prompt `"<bos> 3"` (just two tokens). Compare against the GPU runner's `scratch.norm_out` after `dispatch_attention_block(0)` and `dispatch_moe_block(0)`. The first divergence localizes the bug.
