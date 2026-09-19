# Gemma-4 MoE Rust/WGPU Inference Optimization Plan

**First instruction for any agent picking this up:** profile the current code before acting on historical bottleneck numbers. The engine has moved since earlier benchmarks; re-measure against llama.cpp on the same GGUF, quant, P100, and prompt.

---

## Target

Gemma-4-26B-MoE **IQ4_XS** running correctly and substantially faster on the Rust/WGPU/Vulkan engine. The P100 is the immediate optimization target, but portability remains a first-class requirement.

The mathematical path has crossed the important correctness threshold:

- Coherent multi-token generation after dequantization/indexing fixes (e.g. `"2+2 equals 4."`)
- GPU dequant probe matched llama.cpp CPU reference exactly
- Gemma-4 attention scale fixed to **1.0** (not `1/sqrt(head_dim)`)
- Verified components: Q6_K dequant, embeddings, RMSNorm, gate/up/down/Q/K/V/O matvecs, SwiGLU, attention, residuals, final RMSNorm, LM head, greedy sampling

**Do not redesign model math.** The problem is overwhelmingly execution efficiency.

---

## Current Code Baseline (static audit, 2026-08-28)

Re-profile on hardware before trusting these numbers. They describe the tree as it exists today.

| Area | Status in tree | Notes |
|------|----------------|-------|
| Dense IQ4 matvec | `iq4_pipeline.rs`, `matvec_iq4xs_correct.wgsl` | Correct vs CPU ref; still generic, not Pascal-tuned |
| MoE router | `moe_iq4_dispatch.rs` | GPU matvec → **CPU readback** → `top_k_softmax` → expert dispatch |
| Expert dispatch | Same file | Top-k experts batched in **one** encoder submit, but **serial** gate/up/down/accum per expert |
| GPU-resident runner | `gemma4_gpu_runner.rs` | Hidden state on GPU; ~**12** `queue.submit` calls per layer |
| LM head | `gemma4_gpu_runner.rs` | Still bounces through CPU for matvec; full vocab readback |
| Multi-GPU MoE | `moe_multigpu_dispatch.rs`, `moe_expert_loader.rs` | Placement plan exists; `Gemma4GpuRunner` still single-GPU |
| PAMP | `src/pamp.rs` | Observer hooks wired; placement/prefetch is external |
| Historical decode | `docs/GPU_RESIDENT_RUNNER_REPORT.md` | ~0.34 tok/s CPU-bounce → ~4.5 tok/s GPU-resident (~13×); correctness still open at time of report |

Known per-token sync points today (`gemma4_gpu_runner.rs` module docs):

1. Token ID upload (4 bytes) — acceptable
2. Router logits readback: 128 f32 × 30 layers
3. Full vocabulary logits readback for sampling

---

## 1. Primary Bottleneck: Dense Quantized Matvec

For autoregressive decode, most ops are matrix-vector, not batched GEMM. Memory movement, quantized dot-product implementation, workgroup structure, and dequant overhead dominate.

Historical observation: dense matvec was ~**4–5×** behind llama.cpp. Optimizing MoE scheduling first is premature — every token still passes through dense/shared layers.

**Target pattern evolution:**

```
read packed IQ4 → decode → f32 ops → reduce → write
```

toward:

```
packed IQ4 + FP16/F32 activation → fused unpack + scale + accumulate → minimal intermediate storage
```

### P100 (GP100 / SM60) specifics

- Excellent HBM2 bandwidth, strong FP16
- No Tensor Cores, no DP4A INT8 path like GP10x SM61
- Do **not** copy Turing/Ampere integer kernels blindly

### Kernel work

- Vectorized memory reads; decode multiple quant blocks per invocation
- Scales and LUT in registers/shared memory
- Reduce address arithmetic and workgroup sync
- f16 accumulation/input where numerically safe
- Tune workgroup size for GP100
- Never materialize a dequantized tensor
- Specialize shader variants per quant type

**Principle:** design around HBM2 bandwidth and Pascal arithmetic strengths, not Ampere pretense.

**Code entry points:** `iq4_pipeline.rs`, `shaders/matvec_iq4xs_correct.wgsl`, `device_profile.rs`, `docs/warp_optimized_iq4xs_guide.md`

---

## 2. Move the MoE Router Off the CPU Critical Path

Current path (`moe_iq4_dispatch.rs`):

```
GPU router logits → copy/read logits → CPU softmax/top-k → dispatch experts
```

Target:

```
GPU router logits → GPU softmax/top-k → GPU expert IDs + weights → GPU dispatch
```

CPU should not see routing decisions unless telemetry/debug explicitly requests them. This is prerequisite for PAMP observing selections **asynchronously** without forcing a control-plane sync.

**Required GPU kernel:** router logits → normalization → top-k → indices + weights in GPU-visible buffers.

Even a decent GPU top-k beats a CPU round trip because synchronization latency is the enemy.

**Shader candidates:** new `router_topk.wgsl` (or extend `router_matvec_f32.wgsl` pipeline).

---

## 3. Stop Encoding One Microscopic Operation at a Time

Debug architecture (current `gemma4_gpu_runner.rs`):

```
encode RMSNorm → submit → encode Q → submit → … → encode MoE → submit/readback → …
```

**Phase 1:** one command encoder per transformer layer (RMSNorm → Q/K/V → RoPE → attention → O → residual → FFN norm → router → experts → combine → residual) → **one submit**.

**Phase 2:** encode significantly more of the token forward pass without returning to the host.

Goal: CPU orchestrates coarse work, not per-tensor micromanagement.

---

## 4. Stop Running Selected Experts Serially

Top-k expert computations are independent until weighted combine. Current code loops experts inside one encoder but runs gate/up/down/accum sequentially per expert.

Target:

```
             ┌→ E3  ──┐
router ──────┼→ E11 ──┼→ weighted combine
             └→ E19 ──┘
```

Single-GPU: batch selected experts into one dispatch or enough parallel work for concurrent execution. Multi-GPU comes after single-GPU batching works.

---

## 5. Keep the Entire Forward State GPU-Resident

Production decode path:

```
token ID → embedding → all layers → LM head → sampling → next token ID
```

Debug mode may retain intermediate inspection. Release mode should not copy Q/K/V/router/expert tensors or full logits to CPU.

`Gemma4GpuRunner` is the foundation; finish eliminating remaining CPU bounces (LM head, router).

---

## 6. Put LM-Head Argmax/Sampling on the GPU

Greedy:

```
GPU LM head → GPU argmax → 4-byte token ID → CPU
```

Stochastic: top-k/top-p/temperature on GPU eventually.

Vocabulary layout was corrected (transpose to vocabulary-row layout at load). Production path must not move 262144 floats per token.

---

## 7. Eliminate Hot-Loop Allocations

Preallocate and reuse: hidden, norm, Q/K/V, attention scratch, router logits, top-k IDs/weights, expert output, MoE combine, LM-head scratch, sampling workspace.

Audit: `vec!` in `gemma4_gpu_runner.rs` per-token zero fills, bind-group recreation in `moe_iq4_dispatch.rs`, uniform buffer allocation per expert in the expert loop.

---

## 8. Optimize IQ4_XS Specifically

Sequence:

```
correct generic IQ4 → instrument → specialized IQ4_XS GEMV → P100 tuning → other quants later
```

Not: universal quant framework → hope the compiler optimizes.

---

## 9. Separate P100 and Newer-GPU Kernel Paths

Capability classes (`device_profile.rs`):

| Class | Path |
|-------|------|
| Pascal GP100 / SM60 | Bandwidth/fused GEMV, FP16 where safe |
| Pascal GP10x / SM61 | Possibly integer-oriented where useful |
| Turing+ | Cooperative matrix / Tensor Core |
| AMD | Vulkan/WGPU tuned kernels |

---

## 10. Fix Prefill Separately from Decode

| Workload | Character |
|----------|-----------|
| Decode | batch≈1, GEMV-dominated, bandwidth/latency sensitive |
| Prefill | many tokens, GEMM opportunities, compute throughput |

P100 prefill may benefit from FP16 matmul; decode stays quantized-GEMV dominated.

---

## 11. Preserve Gemma-Specific Correctness Rules

Regression tests before any kernel rewrite:

- Global layers: partial RoPE (`partial_rotary_factor = 0.25`); SWA layers: full RoPE
- Attention scale: **1.0**
- Embedding/vocabulary layout: corrected `token_embd.weight` interpretation
- MoE norm ordering: `pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`

See `regression/baselines/gemma4_l0_checklist_map_2026-08-27.md`, `tests/gpu_kernel_correctness.rs`.

---

## 12. Establish a llama.cpp Performance Oracle

Measure at every optimization stage:

- Prompt tok/s, decode tok/s, time/token
- Per-layer, dense projection, attention, router, MoE, LM-head time
- GPU/CPU utilization, PCIe traffic
- Queue submissions/token, GPU↔CPU sync count

End-to-end tok/s alone is insufficient. Profile downward from the largest delta.

---

## 13. Optimization Order

1. **Dense quantized matvec/GEMV** — largest known deficit
2. **GPU router + top-k** — remove per-layer CPU sync
3. **Whole-layer / whole-token command encoding**
4. **Batch/parallel selected experts**
5. **GPU LM head + sampling**
6. **Preallocate all hot buffers**
7. **Tune IQ4_XS for Pascal**
8. **Multi-GPU expert dispatch** — only after single-GPU baseline is fast

---

## 14. What Not to Optimize First

- Attention rewrite (verified, not clearest disaster)
- New quant format (IQ4_XS works — make it fast)
- Multi-GPU before single-GPU kernels are competitive
- PAMP inside the inference engine (observe/influence, not correctness dependency)
- Replacing WGPU because CUDA is faster (CUDA is reference, WGPU is the product)

---

## 15. MoE Opportunity (Post-Baseline)

Once bottlenecks are fixed:

```
shared/dense → router → top-k → parallel experts → weighted reduction → next layer
```

Heterogeneous placement (hot GPU experts, cold CPU/NUMA banks, async prefetch) sits **on top of** fast local expert kernels.

See `docs/hetero_moe_runtime_architecture.md` for fleet topology context.

---

## 16. Per-Expert Quant Tiering (Future)

Scheduler may eventually assign different quants per expert:

| Tier | Quant | Location |
|------|-------|----------|
| Hot / quality-sensitive | Q8/Q6/FP16 | GPU |
| Normal hot | Q4/IQ4 | GPU |
| Cold | Q3/Q2 | CPU NUMA |

Experimental question: which experts tolerate Q2 on which task classes — ties to model-grader/telemetry, not a new architecture.

---

## 17. End State

```
                       TOKEN
                         │
                         ▼
                 GPU-resident state
                         │
              ┌──────────┴──────────┐
              │                     │
        dense/shared path          router
     fused quant GEMV               │
                                    ▼
                             GPU top-k selection
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                        E1         E7         E24
                     resident   resident   resident/CPU
                         └──────────┼──────────┘
                                    ▼
                             GPU weighted merge
                                    │
                               next layer → … → GPU LM head → GPU sampling → one token ID
```

CPU involvement in the ordinary hot path: minimal.

---

## 18. PAMP Integration (Predictive Async MoE)

PAMP is the **external** placement and prefetch orchestrator. It must not sit on the inference critical path and must not be required for correctness.

### What PAMP is

**P**redictive **A**sync **M**oE **P**lacement — observes expert routing over time, predicts upcoming expert demand, and drives:

- Hot-expert residency on fast GPUs
- Cold-expert offload to CPU NUMA banks
- Async prefetch before the router actually selects an expert
- Optional n8n / fleet corrector workflows (`pamp-route` HTTP endpoint on T440)

### What PAMP is not

- Not a substitute for GPU router or fast IQ4 GEMV
- Not allowed to block token generation waiting for placement decisions
- Not embedded policy inside shaders

### Integration architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Inference engine (hot path)                   │
│  GPU layers → GPU router → GPU top-k → expert dispatch      │
│       │                                                      │
│       │  non-blocking append (ring buffer)                   │
│       ▼                                                      │
│  pamp::ExpertSelectionEvent { layer, token_seq, top-k… }    │
└──────────────────────────┬──────────────────────────────────┘
                           │ async drain (background thread / n8n)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         PAMP service                         │
│  • frequency / Markov / learned predictor per layer          │
│  • emits PlacementHint { expert, layer, target_device, prio }│
│  • triggers prefetch/migration jobs                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ hints (best-effort, stale OK)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              moe_expert_loader / PlacementPlan               │
│  apply_hint() → schedule DMA upload / CPU mmap warm            │
└─────────────────────────────────────────────────────────────┘
```

### Code hooks (`src/pamp.rs`)

| Type | Role |
|------|------|
| `ExpertSelectionEvent` | One router decision: layer, token_seq, indices, weights |
| `PampObserver` trait | `on_expert_selection(&self, event)` — default no-op |
| `RingBufferObserver` | Lock-free SPSC queue for async drain |
| `PlacementHint` | Suggested `(layer, expert) → device` prefetch target |
| `PampHintConsumer` trait | `poll_hints() -> Vec<PlacementHint>` for loader |

Wired from `MoeFfnDispatch::forward` after CPU top-k today; after GPU top-k, wired from a GPU→staging copy that does **not** block dispatch.

### PAMP dependency on optimization order

| Step | Why PAMP cares |
|------|----------------|
| GPU router (§2) | Removes forced CPU sync; telemetry becomes async copy |
| Fast local GEMV (§1) | Prefetch misses must execute on a fast fallback |
| Multi-GPU dispatch (§13.8) | Placement hints become actionable |
| Per-expert quant (§16) | PAMP can tier cold experts independently |

**PAMP enters the workstream at step 8**, not before. Steps 1–7 produce the fast baseline PAMP optimizes around.

### Fleet wiring (T440 / CESARops)

PAMP HTTP surface (external to this crate):

- `POST /pamp-route` — ingest selection batch or return placement suggestion
- Consumes `ExpertSelectionEvent` JSON from ring-buffer drain or periodic export
- Drives expert migration via `moe_expert_loader` budget APIs

Inference engine exports events; PAMP decides. See `w:\AGENTS.md` for node layout (P100s on T440, Lemonade/Ollama lanes).

### Environment flags

| Variable | Effect |
|----------|--------|
| `PAMP_OBSERVE=1` | Enable ring-buffer observer (default off) |
| `PAMP_RING_CAP` | Event ring capacity (default 4096) |
| `MOE_SLICE=1` | Existing debug slice logging (orthogonal) |

---

## Expected Payoff

Sub-1 tok/s on early builds measured a correct but immature path, not P100 hardware ceiling. Removable software overhead:

- Dense GEMV behind llama.cpp
- CPU router synchronization
- Serial expert execution patterns
- Excessive command/submission granularity
- Unnecessary GPU↔CPU transfers
- Hot allocations
- Generic rather than Pascal-specialized quant kernels

Stacking fixes against the llama.cpp oracle on identical inputs is the only honest benchmark.

**Attack dense IQ4 GEMV first. Profile before and after every change.**
