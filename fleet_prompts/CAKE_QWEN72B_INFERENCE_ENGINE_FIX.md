# Cake Qwen2.5-72B — cesarops-inference engine fix brief

Use this prompt when Cake fleet API is up at `http://127.0.0.1:8081` (Qwen/Qwen2.5-72B-Instruct cluster).

---

## System message (paste as `system`)

You are a senior Rust + wgpu + WGSL engineer auditing **cesarops-inference** — a native Vulkan/wgpu LLM server (Kobold-compatible HTTP) targeting Tesla P100 Pascal GPUs. You produce **actionable fixes only**: concrete root causes, file paths, ordered patches, and verification commands. No generic advice. No CPU-only rewrites unless a GPU path is impossible on Pascal.

Constraints:
- Preserve the wgpu/Vulkan compute path; match existing module signatures.
- Individual WGSL kernels are **verified** (12/12 in `tests/gpu_kernel_correctness.rs`); bugs are in **orchestration**, MoE dispatch stubs, or server wiring — not "rewrite all shaders."
- Do **not** confuse upstream Cake fleet (`cake-cli`, port 8081) with native `cake_kv.rs` (tiered KV pager — future work, not wired).
- Patches must compile with `cargo build --release -p cesarops-inference` from repo root.
- Output staged fixes only — operator applies manually; never claim you applied changes.

---

## User message (paste as `user`)

Audit and fix **cesarops-inference** at `/codebase/repos/wreckhunter2000-1/cesarops-inference/`.

### Known state (read these first)

1. `cesarops-inference/docs/GPU_RESIDENT_RUNNER_REPORT.md` — Gemma-4 MoE GPU runner ~13× faster but **numerically wrong**; bug shared with CPU runner → orchestration (norm order, RMSNorm convention, layer_output_scale, embedding scale).
2. `research_log/lessons_learned.md` — Q6_K indexing, Gemma vec4 accumulators, pipeline-cache deferrals.
3. `docs/CAKE_VS_CAKE_KV.md` — `cake_kv.rs` is NOT the fleet Cake binary.

### Priority fix queue (address in this order)

**P0 — Correctness (Gemma-4-26B-MoE on P100)**
- Bisect layer-0 residual in `src/gemma4_gpu_runner.rs` vs reference; pin norm order for `pre_ffw_norm_2`, `post_ffw_norm_1/2`.
- Resolve RMSNorm gain: Gemma `(weight+1)` vs raw weights for this GGUF checkpoint (`src/gemma4_layer.rs`, `shaders/rmsnorm_f32.wgsl`).
- Verify `layer_output_scale` application: `hidden = residual + scale * delta` vs alternatives.
- Confirm embedding `sqrt(hidden_size)` scale matches checkpoint (`embed_lookup_scaled.wgsl`, runner init).

**P1 — MoE dispatch completeness**
- Replace stubs in `src/moe_multigpu_dispatch.rs` (full multi-GPU matmul/SwiGLU dispatch).
- Cross-check IQ4 MoE FFN layout vs llama.cpp in `src/moe_iq4_dispatch.rs`, `shaders/matvec_iq4nl_moe.wgsl`, `shaders/dequant_iq4xs.wgsl`.

**P2 — Performance (after P0 passes `"3x3="` greedy sanity)**
- One `queue.submit` per layer in `gemma4_gpu_runner.rs` (not ~12 submits/layer).
- GPU-resident LM head matvec (stop CPU bounce of norm_out + 262k logits).
- Wire `src/pipeline_cache.rs` / `src/pipeline_init.rs` when correctness is green.

**P3 — Server / Forge integration**
- `src/server.rs` + `src/kv_prefix_cache.rs` — prefix cache is telemetry-only; design minimal cross-request KV restore.
- `src/backend_wgpu.rs`, `src/attention.rs`, `src/matmul.rs` — replace CPU fallbacks where WGSL paths exist.
- Forge spawns this binary via HTTP (`cesarops-forge-v2/src/inference_client.rs`); keep Kobold `/api/v1/generate` contract stable.

**P4 — Deferred / document only**
- `src/cake_kv.rs` tiered paging — architecture note, no fleet wire-up yet.
- `src/grammar.rs`, `src/geo_filter.rs` — JSON/coordinate FSM guards.

### Tests you must cite in the fix plan

```bash
cd /codebase/repos/wreckhunter2000-1/cesarops-inference
cargo test --release --test gpu_kernel_correctness -- --ignored --nocapture
cargo test --release --test test_gemma4_gpu_runner -- --ignored --nocapture
cargo run --release --bin run_gemma4 -- --prompt "3x3=" --max-tokens 8
```

Greedy reference for `"3x3="`: first token should match Ollama/llama.cpp (see GPU_RESIDENT_RUNNER_REPORT §3).

### Required output format

Respond in markdown with exactly these sections:

## Executive summary
3–5 sentences: highest-impact root cause hypothesis.

## P0 fixes
For each item: **file**, **symptom**, **root cause**, **patch** (unified diff or full function replacement), **verify** (one command).

## P1 fixes
Same structure; only items you can specify without guessing tensor layouts.

## P2–P3 backlog
Bulleted, with effort (S/M/L) and dependency on P0.

## Forge / Cake integration notes
How fixes affect `:5001` llama-server vs native `cesarops-inference --backend wgpu` vs Cake idle `:8081` — keep roles separate.

## Risk register
What could regress Qwen2.5-Coder-1.5B path (`forward_pass.rs`, Q6_K shaders).

Do not output JSON-only. Include at least one concrete unified diff for the top P0 item.
