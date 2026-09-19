# Shader spec — what we need to wire `generate.rs` into Gemma‑4‑26B‑MoE

Audience: a Rust + Vulkan + SPIR‑V engineer doing a one‑shot delivery.
Target hardware: Pascal (P100, GTX 10xx) primary; Turing (RTX 20xx) secondary.
Backend: `wgpu` 24.x with `Backends::VULKAN`. WGSL preferred; raw GLSL acceptable
when compiled to SPIR‑V via `glslc` and loaded with `wgpu::ShaderSource::SpirV`.

## Model facts you will be optimizing against

```
arch                gemma4
quant               IQ4_XS (qtype 23) for big tensors, IQ4_NL (qtype 20) for small
file format         GGUF v3, mmaped read‑only
n_layers            30
n_heads             16  (head_dim 512 normal, 256 SWA)
n_kv_heads          8 per layer (varies by layer; full vs sliding window)
hidden              2816
ffn_inner (dense)   2112
ffn_inner (expert)  704                  ← the MoE FFN inside the bypass
n_experts           128
experts_used        8     ← top‑k routing
sliding_window      1024 tokens (alternating SWA / full per `attention.sliding_window_pattern`)
final_softcap       30.0  (logits / 30 → tanh → ×30 before sampling)
rope_freq_base      1_000_000.0  (full attn) / 10_000.0 (SWA)
rms_norm_eps        9.999e-7
```

## Per‑layer tensors we care about

```
blk.{i}.attn_norm.weight                       fp32  [hidden]
blk.{i}.attn_q.weight                          IQ4_XS [hidden, q_dim]
blk.{i}.attn_k.weight                          IQ4_XS [hidden, kv_dim]
blk.{i}.attn_v.weight                          IQ4_XS [hidden, kv_dim]
blk.{i}.attn_q_norm.weight                     fp32  [head_dim]
blk.{i}.attn_k_norm.weight                     fp32  [head_dim]
blk.{i}.attn_output.weight                     IQ4_XS [q_dim, hidden]
blk.{i}.post_attention_norm.weight             fp32  [hidden]

blk.{i}.ffn_norm.weight                        fp32  [hidden]
blk.{i}.ffn_gate.weight                        IQ4_XS [hidden, 2112]   ← dense bypass
blk.{i}.ffn_up.weight                          IQ4_XS [hidden, 2112]
blk.{i}.ffn_down.weight                        IQ4_NL [2112, hidden]

blk.{i}.ffn_gate_inp.weight                    fp32  [hidden, 128]    ← MoE router
blk.{i}.ffn_gate_inp.scale                     fp32  [hidden]          (learned scale)
blk.{i}.ffn_gate_up_exps.weight                IQ4_XS [hidden, 1408, 128]   (gate||up packed)
blk.{i}.ffn_down_exps.weight                   IQ4_NL [704, hidden, 128]
blk.{i}.ffn_down_exps.scale                    fp32  [128]

blk.{i}.layer_output_scale.weight              fp32  [1]               (per-layer residual scale)
blk.{i}.post_ffw_norm.weight                   fp32  [hidden]
blk.{i}.post_ffw_norm_1.weight                 fp32  [hidden]
blk.{i}.post_ffw_norm_2.weight                 fp32  [hidden]
blk.{i}.pre_ffw_norm_2.weight                  fp32  [hidden]
```

`q_dim = n_heads * head_dim` and `kv_dim = n_kv_heads * head_dim`. Both vary per
layer because `gemma4.attention.head_count_kv` is an array of 30 values and the
`sliding_window_pattern` array tells us whether layer `i` uses full attention
(head_dim 512, rope_base 1e6) or SWA (head_dim 256, rope_base 1e4, window 1024).

The model file is at `/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf` (≈13 GB).
We mmap it once and hand you a `&[u8]` slice plus `[start, len]` for each tensor.

## What is already in tree

You can ignore the dense `transformer.rs` and `server.rs` paths — those run
Qwen2.5 and won't be touched here. The relevant files for your work:

* `cesarops-inference/src/generate.rs` — single‑shot CLI generation. Already
  drives a layer loop and `project_lm_head`. Currently dense, no MoE branch,
  no logit softcap, single rope base, no per‑layer head_dim.
* `cesarops-inference/src/forward_pass.rs` — `execute_layer` for one
  transformer block. Dense Qwen-shape today. Needs Gemma‑4 variant.
* `cesarops-inference/src/pipeline_init.rs` — compiles WGSL pipelines once at
  startup. Add Gemma‑specific pipelines here.
* `cesarops-inference/src/moe_dispatch.rs` — partial gate→up→silu→down→accum
  dispatch sketch. Needs to be (re)written against the actual Gemma layout.
* `cesarops-inference/src/moe_expert_loader.rs` — **NEW**, distributes expert
  weights across GPUs. You bind to it via the `PlacementPlan` struct: given
  `(layer, ExpertKind)` it tells you `(gpu_index, buffer_key)`.
* `cesarops-inference/shaders/`
  * `dequant_iq4xs.wgsl`, `dequant_iq4nl` not yet present — please add
  * `matvec_iq4xs_fused.wgsl` and `.glsl` — fused decode+matvec, single‑row
  * `moe_ffn_fused.glsl` — Vulkan reference for fused MoE FFN; not WGSL yet
  * `silu_mul.wgsl`, `weighted_accum.wgsl`, `rmsnorm.wgsl`, `rope.wgsl` —
    untouched, you can call them directly
* `cesarops-inference/src/shader_synth/` — the research stack; you can use
  `gpu_probe::detect_from_gpu_name` to branch Pascal vs Turing.

## What we need from you, ranked

### 1. IQ4_XS / IQ4_NL fused matvec WGSL (Pascal‑safe)

Single workgroup = one output row. 256 threads cooperatively accumulate.
No fp16x4. No subgroup ops. Must validate under wgpu 24 + naga.

Inputs (storage buffers):
* `bufW`: raw IQ4_XS bytes for one matrix. Block layout 136 bytes per 256
  weights: `d:f16 | scales_h:u16 | scales_l:[u8;4] | qs:[u8;128]`. We have a
  CPU reference in `tensor_loader_safe::dequant_iq4_xs` and `bridge::dequant_iq4_xs`
  that produces correct fp32; cross‑check against it.
* `bufX`: input vector fp32, length K.
* `bufY`: output vector fp32, length N (one row per workgroup).
* `bufLut`: 16 fp32 codebook entries (IQ4 nl values).
* push constants: `(K_blocks: u32, N: u32)` plus a workgroup_id bias if you
  want to use 1 dispatch for `chunk_n` rows at a time.

Same shape for IQ4_NL but block is 18 bytes per 32 weights, single d, no sub‑block scales.

Targets: ≥ 200 GB/s on P100 (HBM2 caps at 720 GB/s; achieving 250–300 is great
on Pascal because we cap useful BW at memory channel granularity). Validate
correctness against CPU reference at full‑precision inputs to within 1e‑3 rel
error.

### 2. Fused MoE FFN: gate_up_exps + silu + down_exps + weighted accumulate

Per token:
```
router_logits = x @ ffn_gate_inp.weight                   // [128]
top_k_idx, top_k_w = top_k_softmax(router_logits, 8)      // [8], [8]
y = 0
for j in 0..8:
    e = top_k_idx[j]
    w = top_k_w[j]
    # ffn_gate_up_exps stores gate||up packed along the inner (1408) dim:
    #   first 704 cols = gate, next 704 cols = up
    gu = x @ ffn_gate_up_exps[:, :, e]                    // [1408]
    gate, up = gu[:704], gu[704:]
    h = silu(gate) * up                                   // [704]
    out_e = h @ ffn_down_exps[:, :, e].T                  // [hidden]
    y += w * out_e
```

We want one Vulkan pipeline per (gate_up + silu + down + accum). Today
`moe_ffn_fused.glsl` has the skeleton but it's keyed for hidden=4096,
inner=14336, top_k=2. Rewire it for hidden=2816, inner=704, top_k=8, and
add the IQ4_XS / IQ4_NL decode inline.

Routing decision (top‑k softmax of 128) is fine on CPU; we already read back
the router logits per token. If you want to keep it on GPU, a 128‑lane
softmax‑topk shader is welcome — same buffer pattern as the LM head readback.

### 3. Per‑layer attention with Gemma‑4 quirks

`attention.wgsl` works for Qwen but needs:
* per‑layer rope base (1e6 vs 1e4) selected from a uniform
* per‑layer head_dim (512 vs 256) — make it a push constant
* `attn_q_norm` / `attn_k_norm` applied **per head, per token** before rope
* sliding‑window mask in SWA layers — token at position p attends to
  `[max(0, p-1024), p]`. We already have a causal mask shader; extend with a
  window length param.

Logit softcap on the LM head output: `logits = softcap * tanh(logits / softcap)`
with `softcap = 30.0`. Either fold into the existing matvec output or add a
trivial one‑pass shader.

### 4. Dequant validation utility

A `validate_dequant` binary that, given a tensor name and the GGUF mmap, runs
both the CPU reference dequant and the GPU shader dequant against the same
bytes, then prints max abs err + RMS. We had a `dequant_probe` that targets
Q6_K; copy that pattern for IQ4_XS and IQ4_NL.

## Hardware caveats baked into the design

* **No SHADER_F16 on Pascal under wgpu**. Don't `requires f16;` or use
  `vec4<f16>`. The lab caught two shaders doing this; we'll port them.
* **No subgroup ops on Pascal**. Stick to `workgroupBarrier()` + shared array
  reductions. (Your Turing branch can use `subgroupAdd`.)
* **PUSH_CONSTANTS supported on both**, prefer over uniform buffers for hot
  path params.
* **Per‑buffer cap is 1 GB on wgpu**. The biggest expert tensor is
  `ffn_gate_up_exps` at 2816×1408×128 IQ4_XS ≈ 502 MB raw — fits, but
  `ffn_down_exps` at 704×2816×128 IQ4_NL ≈ 254 MB also fits. Keep planning
  for chunked uploads as a fallback.
* **No GPUDirect on Pascal**. Multi‑GPU expert pinning means activation
  ferrying over PCIe — tolerate that, don't try to hide it.

## How to integrate

Open a branch off `main`. Drop new shaders in `cesarops-inference/shaders/`,
register pipelines in `pipeline_init.rs`, expose dispatch fns alongside
`moe_dispatch.rs`. The integration point in `generate.rs` is `execute_layer`
plus a new `gemma4_layer` variant that:

1. reads `attn_norm`, projects Q/K/V via cached IQ4_XS matvecs
2. applies `attn_q_norm` / `attn_k_norm` per head
3. rope with per‑layer freq base
4. attention dot/mask/softmax/value (your shader)
5. `attn_output` projection + `post_attention_norm` + residual
6. branches into:
   * dense bypass: ffn_norm → gate/up matvecs → silu_mul → down → residual
   * MoE expert path: router → top‑8 dispatch on the placement plan → silu_mul
     → down → weighted accumulate → residual
7. `post_ffw_norm` + `layer_output_scale`

The placement plan is consulted via:
```rust
let plan: PlacementPlan = ...; // from moe_expert_loader::plan_placement
let (gpu_idx, buf_key) = plan.lookup(layer, ExpertKind::GateUp).unwrap();
let device = devices[*gpu_idx].device.clone();
let buffer = devices[*gpu_idx].buffers.get(buf_key).unwrap();
// dispatch on `device` against `buffer`
```

We already have a `shader_lab` binary that scans + benches every shader and
writes results to `target/shader_lab/index.json`. Use it to spot regressions:
```
./target/debug/shader_lab bench --gpu 0 --kind iq4_xs
./target/debug/shader_lab compare matvec_iq4xs_v1.wgsl matvec_iq4xs_v2.wgsl --gpu 0
```

## Acceptance bar

* Smoke run: `generate --model Gemma-4-26B-MoE-IQ4_XS.gguf --prompt "hi"`
  produces a non‑garbled assistant reply within 60s on a single P100.
* Throughput: ≥ 5 tok/s on dual P100 at 1×1 batch with 8 active experts.
* No CPU dequant of any expert tensor at runtime. CPU only handles router
  top‑k and final sampling.
* Engine reuses the existing `Arc<Model>` server pattern (`server.rs` already
  does this for dense models; the MoE state plugs into the same shape).

## What we hand back to you

* Sample mmap layout and tensor offsets for blk.0 — easy to dump from
  `loader.rs::ModelWeights::tensor_bytes`
* The existing fused IQ4_XS WGSL/GLSL we have so you can iterate, not
  rewrite from zero
* `shader_lab` scaffolding that runs your shader through compile + dispatch
  with a noop bind group; you extend it with a real bind group when you wire
  the per‑kind harness
* Direct shell access to a T440 with 2× P100 16 GB, 64 GB RAM, kernel 6.17,
  Vulkan working today
