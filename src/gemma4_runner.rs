//! Single-GPU runner for Gemma-4-26B-MoE.
//!
//! Wires together:
//!   * `Gemma4Layer`        — attention + dense FFN bypass
//!   * `MoeFfnDispatch`     — MoE expert FFN (gate_up + silu·up + down + accum)
//!   * `Iq4MatvecPipeline`  — shared GPU matvec primitives
//!   * GGUF metadata        — per-layer head_count_kv + sliding_window_pattern
//!   * LM head + softcap    — final logit projection with `30 * tanh(x/30)`
//!
//! What this file does NOT do (called out, not silently skipped):
//!   * Multi-GPU dispatch — single GPU only. PlacementPlan integration
//!     is a follow-up.
//!   * Tokenization — `ZeroAllocBpeTokenizer` is Qwen-tuned. The runner
//!     accepts already-tokenized prompts as `Vec<u32>`. A gemma4 tokenizer
//!     is its own task.
//!   * GPU softmax sampling — small CPU readback of 262144 logits per token.
//!
//! Honest gaps that the operator must verify against llama.cpp output:
//!   1. Order of `pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`
//!      in the MoE block. We follow the spec doc as written; the candle
//!      reference doesn't model MoE. If output diverges from llama.cpp,
//!      this is the first place to look.
//!   2. Gate/up split direction inside the 1408 row dim of
//!      `ffn_gate_up_exps`. We assume rows 0..704 = gate, 704..1408 = up.
//!      Flipped via `silu_mul_split_f32.wgsl`.

use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use crate::bridge;
use crate::gemma4_layer::{Gemma4Config, Gemma4Layer, Iq4MatvecScratch, MatvecIo};
use crate::hardware;
use crate::iq4_pipeline::{
    iq4nl_shader_src, iq4xs_shader_src, Iq4MatvecPipeline, MatvecPush, QuantKind,
};
use crate::kv_cache::KvCache;
use crate::loader::{self, GgufValue, ModelWeights};
use crate::moe_iq4_dispatch::{MoeFfnConfig, MoeFfnDispatch};

// ─────────────────────────────────────────────────────────────────────────────
// fp32 row-major matvec pipeline (used by the LM head)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct F32MatvecPush {
    k: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct F32MatvecPipeline {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl F32MatvecPipeline {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec_f32_rowmajor"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/matvec_f32_rowmajor.wgsl").into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f32_matvec_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("f32_matvec_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("f32_matvec_pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { device, queue, pipeline, bgl }
    }

    /// Run y[n] = W[n,k] @ x[k]. All buffers are GPU-resident.
    /// Encodes into the supplied encoder. Caller submits + reads back.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        push: &wgpu::Buffer,
        n_rows: u32,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("f32_matvec_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("f32_matvec_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_rows, 1, 1);
    }

    pub fn make_push_buffer(&self, k: u32, n: u32) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f32_matvec_push"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&F32MatvecPush {
            k, n, _pad0: 0, _pad1: 0,
        }));
        buf
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-token GPU I/O scratch (reused across layers)
// ─────────────────────────────────────────────────────────────────────────────

struct MoeIoScratch {
    expert_buf: wgpu::Buffer,
    router_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    y_staging: wgpu::Buffer,
    y_cpu: Vec<f32>,
}

impl MoeIoScratch {
    fn new(device: &wgpu::Device, hidden: usize) -> Self {
        let nbytes = (hidden * 4) as u64;
        let f32buf = |label: &str, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: nbytes,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            expert_buf: f32buf("moe_expert_x", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            router_buf: f32buf("moe_router_x", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            y_buf: f32buf(
                "moe_y_out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            y_staging: f32buf("moe_y_staging", wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            y_cpu: vec![0.0f32; hidden],
        }
    }
}

struct RunnerScratch {
    matvec: Iq4MatvecScratch,
    io: MatvecIo,
    moe_io: MoeIoScratch,
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level runner
// ─────────────────────────────────────────────────────────────────────────────

pub struct Gemma4Runner {
    pub config: Gemma4Config,
    pub moe_config: MoeFfnConfig,
    pub layers: Vec<Gemma4LayerWithMoe>,

    /// Token embedding table — kept on CPU as f32 for cheap row lookup.
    /// Q6_K dequantized once at load. ~3 GB for vocab=262144 × hidden=2816.
    pub token_embeddings: Vec<f32>,
    pub final_norm: Vec<f32>,
    /// Logit projection. We dequantize the LM head ONCE at load and upload
    /// to the GPU as fp32. Split into two buffers because the full matrix
    /// (262144 × 2816 × 4 = ~2.95 GB) exceeds the P100's 2 GB per-binding
    /// limit. Each half is ~1.47 GB.
    pub lm_head_f32_lo: wgpu::Buffer,  // rows 0..vocab/2
    pub lm_head_f32_hi: wgpu::Buffer,  // rows vocab/2..vocab
    pub lm_head_uses_tied_embedding: bool,
    pub logit_softcap: f32,
    pub vocab_size: usize,

    pub iq4xs_pipe: Arc<Iq4MatvecPipeline>,
    pub iq4nl_pipe: Arc<Iq4MatvecPipeline>,
    pub moe_dispatch: Arc<MoeFfnDispatch>,

    /// Generic fp32 row-major matvec pipeline. Used today by the LM head;
    /// future use cases (router, debug paths) can share it.
    pub f32_matvec: F32MatvecPipeline,

    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,

    /// Reused matvec + MoE I/O buffers — locked for the duration of each token.
    scratch: Mutex<RunnerScratch>,
}

pub struct Gemma4LayerWithMoe {
    pub dense: Gemma4Layer,
    pub pre_ffw_norm_2: Vec<f32>,
    pub post_ffw_norm_1: Vec<f32>,
    pub post_ffw_norm_2: Vec<f32>,

    /// Slab-packed across all 128 experts (verified via `probe_moe_layout`).
    pub gate_up_exps: wgpu::Buffer,
    pub down_exps: wgpu::Buffer,
    /// Per-expert output scale for `ffn_down_exps` (`blk.{i}.ffn_down_exps.scale`,
    /// length = n_experts). llama.cpp applies this in `build_lora_mm_id(..., down_exps_s)`.
    pub down_exps_scale: Vec<f32>,
    pub router_w: wgpu::Buffer,
    pub router_scale: wgpu::Buffer,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.0, // greedy by default for first smoke test
            top_p: 1.0,
            stop_tokens: vec![1, 2], // Gemma-4 EOS=1, BOS=2
        }
    }
}

/// GGUF paths known to work with the IQ4_XS Vulkan MoE path (not Q5_K llama GGUF).
pub fn gemma4_iq4xs_model_candidates() -> &'static [&'static str] {
    &[
        "/mnt/t440/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
        "/mnt/t440/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
        "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
        "/mnt/raid0/models/Gemma-4-26B-MoE-IQ4_XS.gguf",
    ]
}

/// Resolve first existing IQ4_XS Gemma-4 MoE checkpoint, if any.
pub fn resolve_gemma4_iq4xs_model() -> Option<String> {
    gemma4_iq4xs_model_candidates()
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
        .map(str::to_string)
}

fn ggml_quant_name(qtype: u32) -> &'static str {
    match qtype {
        8 => "Q8_0",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        20 => "IQ4_NL",
        23 => "IQ4_XS",
        _ => "unknown",
    }
}

/// Fail fast unless attention + experts are IQ4_XS (ggml type 23).
fn validate_gemma4_iq4xs_weights(weights: &ModelWeights) -> Result<(), String> {
    const IQ4_XS: u32 = 23;
    let probes = [
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate_up_exps.weight",
    ];
    let mut bad = Vec::new();
    for name in probes {
        if let Some(t) = weights.tensors.get(name) {
            if t.quant_type != IQ4_XS {
                bad.push(format!(
                    "{}: type {} ({})",
                    name,
                    t.quant_type,
                    ggml_quant_name(t.quant_type)
                ));
            }
        }
    }
    if bad.is_empty() {
        return Ok(());
    }
    let hint = resolve_gemma4_iq4xs_model()
        .map(|p| format!("Found IQ4_XS model at: {p}"))
        .unwrap_or_else(|| {
            "No IQ4_XS GGUF on this host — download Gemma-4-26B-MoE-IQ4_XS.gguf or use llama-server for Q5_K_M.".into()
        });
    Err(format!(
        "Gemma4Runner (cesarops-inference) requires IQ4_XS weights (ggml type 23).\n\
         Mismatched: {}\n\
         Q5_K / Q6_K GGUFs (e.g. gemma-4-26B-A4B-it-UD-Q5_K_M) work on llama.cpp :5200, not this Vulkan path.\n\
         {}",
        bad.join("; "),
        hint
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

impl Gemma4Runner {
    /// Load a Gemma-4-26B-MoE checkpoint and prepare every weight for inference.
    ///
    /// Walks the GGUF, derives `Gemma4Config` from `gemma4.*` metadata,
    /// validates per-layer KV head counts and SWA pattern, and uploads
    /// every projection / expert tensor to the chosen GPU.
    pub fn load(model_path: &Path, gpu_index: usize) -> Result<Self, String> {
        let profile = hardware::audit_system();
        let weights =
            loader::load(model_path, &profile).map_err(|e| format!("loader::load: {e}"))?;
        Self::from_weights(weights, gpu_index)
    }

    /// Build a runner from an already-mmap'd GGUF (avoids a second full file parse).
    pub fn from_weights(weights: ModelWeights, gpu_index: usize) -> Result<Self, String> {
        validate_gemma4_iq4xs_weights(&weights)?;

        // ── Resolve Gemma-4 config from GGUF metadata ─────────────────────
        let cfg = read_gemma4_config(&weights)?;
        let moe_cfg = read_moe_config(&weights)?;
        let n_layers = weights.n_layers;
        let head_count_kv_per_layer = read_head_count_kv_array(&weights, n_layers)?;
        let swa_pattern = read_swa_pattern(&weights, n_layers)?;
        let logit_softcap =
            get_f32_meta(&weights, "gemma4.final_logit_softcapping").unwrap_or(30.0);

        // Confirm Gemma-4 conventions (not Gemma-2/3 (1+w), not 1/sqrt(d)).
        let norm_plus_one = std::env::var("GEMMA4_NORM_PLUS_ONE").map(|v| v == "1").unwrap_or(false);
        if cfg.attention_scale != 1.0 {
            return Err(format!(
                "Gemma-4 requires attention_scale=1.0, got {}",
                cfg.attention_scale
            ));
        }
        if norm_plus_one {
            eprintln!(
                "WARN: GEMMA4_NORM_PLUS_ONE=1 → RMSNorm uses (1+weight); \
                 Gemma-4 expects raw weight (llama.cpp build_norm)"
            );
        }
        // ── Init GPU + pipelines ─────────────────────────────────────────
        let (device, queue) = init_device(gpu_index)?;

        let iq4xs_pipe = Arc::new(Iq4MatvecPipeline::new(
            device.clone(),
            queue.clone(),
            QuantKind::Iq4Xs,
            iq4xs_shader_src(),
        ));
        let iq4nl_pipe = Arc::new(Iq4MatvecPipeline::new(
            device.clone(),
            queue.clone(),
            QuantKind::Iq4Nl,
            iq4nl_shader_src(),
        ));
        let moe_dispatch =
            Arc::new(MoeFfnDispatch::new(device.clone(), queue.clone()));

        // ── Top-level CPU norms + embeddings ──────────────────────────────
        // GGUF stores token_embd as [hidden, vocab]; our embed lookup + LM head
        // shaders expect [vocab, hidden] row-major (same fix as Qwen in main.rs).
        let token_embeddings = load_vocab_row_matrix(&weights, "token_embd.weight")?;
        let final_norm =
            load_dequantized_or_fail(&weights, "output_norm.weight", cfg.hidden_size)?;

        // LM head: try output.weight (untied), fall back to token_embd.weight (tied).
        // Same [vocab, hidden] row layout as embed lookup — transpose if GGUF is [hidden, vocab].
        let (lm_head_f32_vec, lm_head_uses_tied) =
            if weights.tensors.contains_key("output.weight") {
                tracing::info!("LM head from output.weight (untied)");
                (load_vocab_row_matrix(&weights, "output.weight")?, false)
            } else {
                tracing::info!("LM head tied to token_embd.weight (no separate output.weight)");
                (token_embeddings.clone(), true)
            };

        // Upload LM head as fp32 GPU buffer, split into two halves to stay
        // under the 2 GB per-binding limit on P100.
        let half_vocab = lm_head_f32_vec.len() / (2 * cfg.hidden_size);
        let half_elems = half_vocab * cfg.hidden_size;
        let half_bytes = (half_elems * 4) as u64;

        let lm_head_f32_lo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemma4.lm_head_f32_lo"),
            size: half_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &lm_head_f32_lo,
            0,
            bytemuck::cast_slice(&lm_head_f32_vec[..half_elems]),
        );

        let lm_head_f32_hi = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemma4.lm_head_f32_hi"),
            size: (lm_head_f32_vec.len() - half_elems) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &lm_head_f32_hi,
            0,
            bytemuck::cast_slice(&lm_head_f32_vec[half_elems..]),
        );

        // ── Per-layer construction ───────────────────────────────────────
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let is_swa = swa_pattern[layer_idx];
            let n_kv_heads = head_count_kv_per_layer[layer_idx];

            let dense = Gemma4Layer::load(
                layer_idx,
                is_swa,
                n_kv_heads,
                &cfg,
                &weights,
                &device,
                &queue,
            )?;

            let pre_ffw_norm_2 = load_norm_for_layer(
                &weights,
                layer_idx,
                "pre_ffw_norm_2",
                cfg.hidden_size,
            )?;
            let post_ffw_norm_1 = load_norm_for_layer(
                &weights,
                layer_idx,
                "post_ffw_norm_1",
                cfg.hidden_size,
            )?;
            let post_ffw_norm_2 = load_norm_for_layer(
                &weights,
                layer_idx,
                "post_ffw_norm_2",
                cfg.hidden_size,
            )?;

            let gate_up_exps = upload_raw_quant(
                &weights,
                &format!("blk.{}.ffn_gate_up_exps.weight", layer_idx),
                23, // IQ4_XS
                &device,
                &queue,
                "gemma4.ffn_gate_up_exps",
            )?;
            let down_exps = upload_raw_quant(
                &weights,
                &format!("blk.{}.ffn_down_exps.weight", layer_idx),
                20, // IQ4_NL
                &device,
                &queue,
                "gemma4.ffn_down_exps",
            )?;
            let down_exps_scale = load_dequantized_or_fail(
                &weights,
                &format!("blk.{}.ffn_down_exps.scale", layer_idx),
                moe_cfg.n_experts,
            )?;

            let router_w = upload_fp32_tensor(
                &weights,
                &format!("blk.{}.ffn_gate_inp.weight", layer_idx),
                cfg.hidden_size * moe_cfg.n_experts,
                &device,
                &queue,
                "gemma4.ffn_gate_inp.weight",
            )?;
            let router_scale = upload_fp32_tensor(
                &weights,
                &format!("blk.{}.ffn_gate_inp.scale", layer_idx),
                cfg.hidden_size,
                &device,
                &queue,
                "gemma4.ffn_gate_inp.scale",
            )?;

            layers.push(Gemma4LayerWithMoe {
                dense,
                pre_ffw_norm_2,
                post_ffw_norm_1,
                post_ffw_norm_2,
                gate_up_exps,
                down_exps,
                down_exps_scale,
                router_w,
                router_scale,
            });

            if layer_idx % 5 == 0 {
                tracing::info!(
                    "Loaded layer {}/{} (swa={}, n_kv_heads={})",
                    layer_idx + 1,
                    n_layers,
                    is_swa,
                    n_kv_heads
                );
            }
        }

        let hidden_size = cfg.hidden_size;
        let max_proj = cfg.num_heads * cfg.head_dim_full.max(cfg.head_dim_swa);
        Ok(Self {
            config: cfg,
            moe_config: moe_cfg,
            layers,
            token_embeddings,
            final_norm,
            lm_head_f32_lo,
            lm_head_f32_hi,
            lm_head_uses_tied_embedding: lm_head_uses_tied,
            logit_softcap,
            vocab_size: read_vocab_size(&weights)?,
            iq4xs_pipe,
            iq4nl_pipe,
            moe_dispatch,
            f32_matvec: F32MatvecPipeline::new(device.clone(), queue.clone()),
            device: device.clone(),
            queue: queue.clone(),
            scratch: Mutex::new(RunnerScratch {
                matvec: Iq4MatvecScratch::new(&device, max_proj, max_proj),
                io: MatvecIo::new(),
                moe_io: MoeIoScratch::new(&device, hidden_size),
            }),
        })
    }

    // ── Forward pass for one token ───────────────────────────────────────
    ///
    /// `hidden` is read from `token_embeddings` for the input token id,
    /// then passed through every layer. Returns logits (vocab_size floats)
    /// after final_norm + LM head + softcap.
    pub fn forward_token(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut KvCache,
    ) -> Vec<f32> {
        let h = self.config.hidden_size;
        let vocab = self.vocab_size;
        let mut scratch = self.scratch.lock();

        // 1. Embedding row lookup (CPU).
        let row_start = (token_id as usize) * h;
        let row_end = row_start + h;
        let mut hidden_state = self.token_embeddings[row_start..row_end].to_vec();
        moe_slice_log(0, position, "post_embed_raw", &hidden_state);

        // Gemma embedding scale: x *= sqrt(hidden_size). Matches llama.cpp
        // gemma4.cpp: ggml_scale(inpL, sqrtf(n_embd)) for token embeddings.
        let embed_scale = (h as f32).sqrt();
        for v in hidden_state.iter_mut() {
            *v *= embed_scale;
        }
        moe_slice_log(0, position, "post_embed", &hidden_state);

        // 2. Walk all layers.
        for (li, layer) in self.layers.iter().enumerate() {
            {
                let RunnerScratch { matvec, io, .. } = &mut *scratch;
                layer.dense.forward_attention(
                    &self.config,
                    &mut hidden_state,
                    position,
                    kv_cache,
                    &self.iq4xs_pipe,
                    li,
                    matvec,
                    io,
                );
            }

            self.moe_block(layer, &mut hidden_state, position, &mut *scratch);
        }

        // 3. Final RMSNorm (Gemma-style).
        rmsnorm_gemma_inplace(&mut hidden_state, &self.final_norm, self.config.rms_norm_eps);

        // 4. LM head matvec on GPU (fp32 path).
        let logits = self.lm_head_matvec(&hidden_state, h, vocab);

        // 5. Logit softcap on CPU.
        let cap = self.logit_softcap;
        let mut out = Vec::with_capacity(logits.len());
        for l in &logits {
            let v = (l / cap).tanh() * cap;
            out.push(v);
        }
        out
    }

    /// Run prefill over `prompt_tokens` and return logits at the last prompt position.
    /// This is the correctness gate for greedy decode: compare top-k here vs llama.cpp.
    pub fn prefill_last_logits(&self, prompt_tokens: &[u32]) -> Result<Vec<f32>, String> {
        if prompt_tokens.is_empty() {
            return Err("empty prompt".into());
        }
        let mut kv_cache = KvCache::new(self.layers.len(), 4096);
        let mut logits = vec![0.0f32; self.vocab_size];
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward_token(tok, pos, &mut kv_cache);
            kv_cache.advance();
        }
        Ok(logits)
    }

    /// Gemma-4 MoE FFN sub-block (matches `llama.cpp/src/models/gemma4.cpp`):
    /// - cur_mlp: rmsnorm(attn_out, ffn_norm) → (GEGLU MLP) → rmsnorm(…, ffn_post_norm_1)
    /// - cur_moe: rmsnorm(attn_out, ffn_pre_norm_2) → moe_dispatch → rmsnorm(…, ffn_post_norm_2)
    /// - cur = cur_mlp + cur_moe
    /// - cur = rmsnorm(cur, ffn_post_norm)
    /// - hidden_state = cur + attn_out
    /// - hidden_state *= out_scale
    fn moe_block(
        &self,
        layer: &Gemma4LayerWithMoe,
        hidden_state: &mut Vec<f32>,
        position: usize,
        scratch: &mut RunnerScratch,
    ) {
        let li = layer.dense.layer_idx;
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let scale = layer.dense.layer_output_scale;

        let attn_out = hidden_state.clone(); // post-attention residual stream
        moe_slice_log(li, position, "post_attn_residual", &attn_out);

        // ── Shared dense exp MLP (cur_mlp) ───────────────────────────────
        let mut mlp_in = attn_out.clone();
        rmsnorm_gemma_inplace(&mut mlp_in, &layer.dense.ffn_norm, eps);
        moe_slice_log(li, position, "pre_mlp_norm", &mlp_in);
        let mut mlp_out = {
            let RunnerScratch { matvec, io, .. } = scratch;
            layer.dense.forward_shared_mlp_down_only(
                cfg,
                &mlp_in,
                &self.iq4xs_pipe,
                &self.iq4nl_pipe,
                matvec,
                io,
            )
        };
        // ffn_post_norm_1
        rmsnorm_gemma_inplace(&mut mlp_out, &layer.post_ffw_norm_1, eps);
        moe_slice_log(li, position, "post_mlp", &mlp_out);

        // ── Expert MoE branch (cur_moe) ─────────────────────────────────
        let mut expert_in = attn_out.clone();
        rmsnorm_gemma_inplace(&mut expert_in, &layer.pre_ffw_norm_2, eps);
        moe_slice_log(li, position, "pre_moe_norm", &expert_in);

        // Router logits use weightless RMSNorm(attn_out) / sqrt(hidden), then
        // * ffn_gate_inp.scale — see gemma4.cpp lines 287-290.
        let mut router_in = attn_out.clone();
        crate::gemma4_layer::rmsnorm_weightless_inplace(&mut router_in, eps);
        let inv_sqrt_h = 1.0 / (h as f32).sqrt();
        for v in router_in.iter_mut() {
            *v *= inv_sqrt_h;
        }
        moe_slice_log(li, position, "pre_router", &router_in);

        let queue = &self.queue;
        let moe_io = &mut scratch.moe_io;

        queue.write_buffer(&moe_io.expert_buf, 0, bytemuck::cast_slice(&expert_in));
        queue.write_buffer(&moe_io.router_buf, 0, bytemuck::cast_slice(&router_in));
        let zeros = vec![0u8; (h * 4) as usize];
        queue.write_buffer(&moe_io.y_buf, 0, &zeros);

        // Dispatch the MoE FFN. This blocks on a tiny readback (128
        // logits) for top-k routing inside the dispatcher.
        let meta = crate::pamp::MoeDispatchMeta {
            layer: li as u32,
            token_seq: position as u64,
        };
        if let Err(e) = self.moe_dispatch.forward(
            &self.moe_config,
            &moe_io.expert_buf,
            &moe_io.router_buf,
            &layer.gate_up_exps,
            &layer.down_exps,
            &layer.down_exps_scale,
            &layer.router_w,
            &layer.router_scale,
            &moe_io.y_buf,
            Some(&meta),
        ) {
            tracing::error!("MoE forward failed: {}", e);
            return;
        }

        // Read y_buf back to CPU.
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_y_copy"),
        });
        enc.copy_buffer_to_buffer(&moe_io.y_buf, 0, &moe_io.y_staging, 0, (h * 4) as u64);
        let sub_idx = queue.submit(std::iter::once(enc.finish()));

        let slice = moe_io.y_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
        let _ = rx.recv();
        let mapped = slice.get_mapped_range();
        moe_io.y_cpu.copy_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        moe_io.y_staging.unmap();

        moe_slice_log(li, position, "post_moe_expert_raw", &moe_io.y_cpu);

        // ffn_post_norm_2 (applied to cur_moe before the add)
        rmsnorm_gemma_inplace(&mut moe_io.y_cpu, &layer.post_ffw_norm_2, eps);
        moe_slice_log(li, position, "post_moe_expert", &moe_io.y_cpu);

        // cur = cur_mlp + cur_moe
        for i in 0..h {
            hidden_state[i] = mlp_out[i] + moe_io.y_cpu[i];
        }
        moe_slice_log(li, position, "post_ffn_sum", &hidden_state);

        // ffn_post_norm (shared across all layers)
        rmsnorm_gemma_inplace(hidden_state, &layer.dense.post_ffw_norm, eps);
        moe_slice_log(li, position, "post_ffn_norm", &hidden_state);

        // residual add: cur + attn_out
        for i in 0..h {
            hidden_state[i] += attn_out[i];
        }
        moe_slice_log(li, position, "post_layer_residual", &hidden_state);

        // out_scale
        for i in 0..h {
            hidden_state[i] *= scale;
        }
        moe_slice_log(li, position, "post_layer", &hidden_state);
    }

    /// GPU fp32 LM head matvec from a GPU-resident hidden buffer into `y_buf`.
    /// Single queue submit; caller readbacks `y_buf` when needed.
    pub(crate) fn lm_head_matvec_gpu(
        &self,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        k: usize,
        n: usize,
    ) {
        let device = &self.device;
        let queue = &self.queue;
        const MAX_WG: usize = 65535;

        // Split vocab into chunks that fit both the binding limit and dispatch limit.
        // lm_head_f32_lo covers rows 0..half_n, lm_head_f32_hi covers half_n..n.
        let half_n = n / 2;
        let second_n = n - half_n;

        // Each half may still exceed 65535 workgroups. Sub-chunk each half.
        let lo_chunks = chunk_ranges(half_n, MAX_WG);
        let hi_chunks = chunk_ranges(second_n, MAX_WG);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lm_head_encoder"),
        });

        let mut y_offset: u64 = 0;

        // Dispatch lo chunks.
        for (start, count) in &lo_chunks {
            let chunk_bytes = (*count * 4) as u64;
            let y_chunk = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lm_head_y_chunk"),
                size: chunk_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // The W buffer for this chunk starts at row `start` within lm_head_f32_lo.
            // But our shader always reads W from row 0 of the bound buffer. So we need
            // to bind a sub-range. wgpu doesn't support buffer sub-range bindings easily,
            // so instead we use a push constant `row_offset` trick — but our shader
            // doesn't have that field. Simplest: create a view buffer per chunk.
            //
            // Actually, the shader reads `w[row * k + col]` where `row` = workgroup_id.
            // If we bind the FULL lo buffer and set push.n = count, push.k = k, the
            // shader will read rows 0..count. We need rows start..start+count.
            // The shader doesn't have a row_offset. Let's add one to the push struct.
            //
            // WAIT — the push struct IS `{k, n, _pad0, _pad1}`. We can repurpose _pad0
            // as row_offset. But that changes the shader. Simpler: just bind a buffer
            // slice via offset. wgpu supports `BufferBinding { offset, size }`.
            let w_offset = (*start * k * 4) as u64;
            let w_size = (*count * k * 4) as u64;
            let push_buf = self.f32_matvec.make_push_buffer(k as u32, *count as u32);

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lm_head_chunk_bg"),
                layout: &self.f32_matvec.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.lm_head_f32_lo,
                            offset: w_offset,
                            size: Some(std::num::NonZeroU64::new(w_size).unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y_chunk.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: push_buf.as_entire_binding() },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("lm_head_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.f32_matvec.pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(*count as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&y_chunk, 0, y_buf, y_offset, chunk_bytes);
            y_offset += chunk_bytes;
        }

        // Dispatch hi chunks.
        for (start, count) in &hi_chunks {
            let chunk_bytes = (*count * 4) as u64;
            let y_chunk = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lm_head_y_chunk_hi"),
                size: chunk_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let w_offset = (*start * k * 4) as u64;
            let w_size = (*count * k * 4) as u64;
            let push_buf = self.f32_matvec.make_push_buffer(k as u32, *count as u32);

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lm_head_chunk_bg_hi"),
                layout: &self.f32_matvec.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.lm_head_f32_hi,
                            offset: w_offset,
                            size: Some(std::num::NonZeroU64::new(w_size).unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y_chunk.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: push_buf.as_entire_binding() },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("lm_head_pass_hi"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.f32_matvec.pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(*count as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&y_chunk, 0, y_buf, y_offset, chunk_bytes);
            y_offset += chunk_bytes;
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// GPU fp32 LM head matvec, chunked to stay under both the 2 GB
    /// per-binding limit AND the 65535 max workgroups-per-dimension limit.
    pub(crate) fn lm_head_matvec(&self, hidden: &[f32], k: usize, n: usize) -> Vec<f32> {
        let device = &self.device;
        let queue = &self.queue;

        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lm_head_x"),
            size: (k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(hidden));

        let y_size = (n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lm_head_staging"),
            size: y_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.lm_head_matvec_gpu(&x_buf, &staging, k, n);
        device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let mapped = slice.get_mapped_range();
        let logits: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        logits
    }

    // ── End-to-end token generation ──────────────────────────────────────

    /// Greedy / top-p sampling loop over the prompt and `max_new_tokens`.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
    ) -> Result<Vec<u32>, String> {
        if prompt_tokens.is_empty() {
            return Err("empty prompt".into());
        }
        let mut kv_cache = KvCache::new(self.layers.len(), 4096);
        let mut output = Vec::<u32>::new();

        // 1. Prefill: walk the prompt to seed the KV cache.
        let mut last_logits = vec![0.0f32; self.vocab_size];
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_token(tok, pos, &mut kv_cache);
            kv_cache.advance();
        }

        // 2. Sample new tokens.
        let mut pos = prompt_tokens.len();
        for step in 0..params.max_new_tokens {
            if std::env::var("LOGIT_AUDIT").ok().as_deref() == Some("1") && step == 0 {
                assert_eq!(
                    last_logits.len(),
                    self.vocab_size,
                    "lm_head must return vocab-sized logits, not hidden/router buffer"
                );
                let k = std::env::var("LOGIT_AUDIT_TOPK")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);
                eprintln!(
                    "LOGIT_AUDIT prefill_last_pos={} logits_len={}",
                    pos.saturating_sub(1),
                    last_logits.len(),
                );
                for (rank, (id, val)) in top_k_logits(&last_logits, k).into_iter().enumerate() {
                    eprintln!("LOGIT_AUDIT top{} id={} logit={:.4}", rank + 1, id, val);
                }
            }
            let next = sample_logits(&last_logits, params.temperature, params.top_p);
            if std::env::var("LOGIT_AUDIT").ok().as_deref() == Some("1") && step == 0 {
                assert!(
                    (next as usize) < self.vocab_size,
                    "sampled token id {next} out of vocab range {}",
                    self.vocab_size
                );
                eprintln!("LOGIT_AUDIT greedy_first_token_id={next}");
            }
            if params.stop_tokens.contains(&next) {
                break;
            }
            output.push(next);
            last_logits = self.forward_token(next, pos, &mut kv_cache);
            kv_cache.advance();
            pos += 1;
        }
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config + metadata helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Gemma4Config {
    fn num_layers(&self) -> usize {
        // This method is only called from Gemma4Runner::generate via
        // KvCache::new. The runner passes self.layers.len() directly now.
        // This stub exists only to satisfy the compiler; it should never
        // be called from outside the runner.
        panic!("Gemma4Config::num_layers() called directly — use runner.layers.len()")
    }
}

fn read_gemma4_config(weights: &ModelWeights) -> Result<Gemma4Config, String> {
    let hidden_size =
        get_u32_meta(weights, "gemma4.embedding_length").ok_or("missing embedding_length")? as usize;
    let num_heads = get_u32_meta(weights, "gemma4.attention.head_count")
        .ok_or("missing attention.head_count")? as usize;
    // num_kv_heads is per-layer; default to 8 here, used only as fallback.
    let num_kv_heads = 8;
    let head_dim_full = get_u32_meta(weights, "gemma4.attention.key_length").unwrap_or(512) as usize;
    let head_dim_swa =
        get_u32_meta(weights, "gemma4.attention.key_length_swa").unwrap_or(256) as usize;
    let intermediate_size_dense =
        get_u32_meta(weights, "gemma4.feed_forward_length").ok_or("missing feed_forward_length")? as usize;
    let rope_theta_full =
        get_f32_meta(weights, "gemma4.rope.freq_base").unwrap_or(1_000_000.0);
    let rope_theta_swa =
        get_f32_meta(weights, "gemma4.rope.freq_base_swa").unwrap_or(10_000.0);
    let sliding_window =
        get_u32_meta(weights, "gemma4.attention.sliding_window").unwrap_or(1024) as usize;
    let rms_norm_eps =
        get_f32_meta(weights, "gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
    let partial_rotary_factor =
        get_f32_meta(weights, "gemma4.rope.partial_rotary_factor").unwrap_or(0.25);

    Ok(Gemma4Config {
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim_full,
        head_dim_swa,
        intermediate_size_dense,
        rope_theta_full,
        rope_theta_swa,
        partial_rotary_factor,
        sliding_window,
        rms_norm_eps,
        // gemma4.cpp: hparams.f_attention_scale = 1.0f (no 1/sqrt(head_dim)).
        attention_scale: 1.0,
    })
}

fn read_moe_config(weights: &ModelWeights) -> Result<MoeFfnConfig, String> {
    let hidden = get_u32_meta(weights, "gemma4.embedding_length")
        .ok_or("missing embedding_length")? as usize;
    let expert_inner = get_u32_meta(weights, "gemma4.expert_feed_forward_length")
        .ok_or("missing expert_feed_forward_length")? as usize;
    let n_experts =
        get_u32_meta(weights, "gemma4.expert_count").ok_or("missing expert_count")? as usize;
    let top_k =
        get_u32_meta(weights, "gemma4.expert_used_count").unwrap_or(8) as usize;
    Ok(MoeFfnConfig {
        hidden,
        expert_inner,
        n_experts,
        top_k,
    })
}

fn read_head_count_kv_array(
    weights: &ModelWeights,
    n_layers: usize,
) -> Result<Vec<usize>, String> {
    if let Some(GgufValue::Array(arr)) = weights.metadata.get("gemma4.attention.head_count_kv") {
        let mut out = Vec::with_capacity(n_layers);
        for (i, v) in arr.iter().enumerate() {
            let n = match v {
                GgufValue::U32(x) => *x as usize,
                GgufValue::I32(x) => *x as usize,
                GgufValue::U64(x) => *x as usize,
                _ => return Err(format!("head_count_kv[{i}] has unexpected type")),
            };
            out.push(n);
        }
        if out.len() != n_layers {
            return Err(format!(
                "head_count_kv array length {} != n_layers {}",
                out.len(),
                n_layers
            ));
        }
        Ok(out)
    } else {
        // Fallback: scalar value broadcast.
        let scalar = get_u32_meta(weights, "gemma4.attention.head_count_kv")
            .ok_or("missing attention.head_count_kv")? as usize;
        Ok(vec![scalar; n_layers])
    }
}

fn read_swa_pattern(weights: &ModelWeights, n_layers: usize) -> Result<Vec<bool>, String> {
    if let Some(GgufValue::Array(arr)) =
        weights.metadata.get("gemma4.attention.sliding_window_pattern")
    {
        let mut out = Vec::with_capacity(n_layers);
        for v in arr {
            let b = match v {
                GgufValue::Bool(b) => *b,
                GgufValue::U32(x) => *x != 0,
                GgufValue::I32(x) => *x != 0,
                _ => return Err("sliding_window_pattern entry has unexpected type".into()),
            };
            out.push(b);
        }
        if out.len() != n_layers {
            return Err(format!(
                "sliding_window_pattern length {} != n_layers {}",
                out.len(),
                n_layers
            ));
        }
        Ok(out)
    } else {
        // No SWA pattern → assume all layers are full attention.
        Ok(vec![false; n_layers])
    }
}

fn read_vocab_size(weights: &ModelWeights) -> Result<usize, String> {
    if let Some(region) = weights.tensors.get("token_embd.weight") {
        if region.shape.len() >= 2 {
            if region.shape[0] == weights.hidden_dim {
                return Ok(region.shape[1]);
            }
            if region.shape[1] == weights.hidden_dim {
                return Ok(region.shape[0]);
            }
        }
    }
    Ok(weights.vocab_size)
}

fn get_u32_meta(weights: &ModelWeights, key: &str) -> Option<u32> {
    match weights.metadata.get(key)? {
        GgufValue::U32(x) => Some(*x),
        GgufValue::I32(x) => Some(*x as u32),
        GgufValue::U64(x) => Some(*x as u32),
        _ => None,
    }
}

fn get_f32_meta(weights: &ModelWeights, key: &str) -> Option<f32> {
    match weights.metadata.get(key)? {
        GgufValue::F32(x) => Some(*x),
        GgufValue::U32(x) => Some(*x as f32),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor loaders
// ─────────────────────────────────────────────────────────────────────────────

/// Dequantize a `[vocab, hidden]` or `[hidden, vocab]` matrix and return vocab-row layout
/// so row `token_id * hidden_dim` is the embedding / LM-head row.
fn load_vocab_row_matrix(weights: &ModelWeights, name: &str) -> Result<Vec<f32>, String> {
    let region = weights
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor `{}`", name))?;
    let bytes = weights
        .tensor_bytes(name)
        .ok_or_else(|| format!("tensor `{}` has no mapped bytes", name))?;
    let n_elements: usize = region.shape.iter().product();
    let raw = bridge::dequantize_tensor(bytes, region.quant_type, n_elements);
    if region.shape.len() < 2 {
        return Ok(raw);
    }

    let h = weights.hidden_dim;
    let (d0, d1) = (region.shape[0], region.shape[1]);
    if d1 == h {
        tracing::info!(
            "{} shape [{}, {}] already vocab-row — no transpose",
            name, d0, d1
        );
        return Ok(raw);
    }
    if d0 == h {
        let vocab = d1;
        tracing::info!(
            "{} shape [hidden={}, vocab={}] — transposing to vocab-row layout",
            name, h, vocab
        );
        // GGML/llama layout for shape [hidden, vocab]: offset(h, v) = h + v * hidden.
        // Vocab-row lookup uses offset(v, h) = v * hidden + h.
        let mut out = vec![0.0f32; raw.len()];
        for v_idx in 0..vocab {
            for h_idx in 0..h {
                out[v_idx * h + h_idx] = raw[h_idx + v_idx * h];
            }
        }
        return Ok(out);
    }

    Err(format!(
        "{} shape {:?} does not match hidden_dim={}",
        name, region.shape, h
    ))
}

fn load_dequantized_or_fail(
    weights: &ModelWeights,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let region = weights
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor `{}`", name))?;
    let bytes = weights
        .tensor_bytes(name)
        .ok_or_else(|| format!("tensor `{}` has no mapped bytes", name))?;
    let n_elements: usize = region.shape.iter().product();
    let mut out = bridge::dequantize_tensor(bytes, region.quant_type, n_elements);
    if out.len() < expected {
        return Err(format!(
            "tensor `{}` produced {} elements, expected {}",
            name, out.len(), expected
        ));
    }
    out.truncate(expected);
    Ok(out)
}

fn load_norm_for_layer(
    weights: &ModelWeights,
    layer_idx: usize,
    suffix: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    load_dequantized_or_fail(
        weights,
        &format!("blk.{}.{}.weight", layer_idx, suffix),
        expected,
    )
}

fn upload_raw_quant(
    weights: &ModelWeights,
    name: &str,
    expected_qtype: u32,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    label: &'static str,
) -> Result<wgpu::Buffer, String> {
    let region = weights
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor `{}`", name))?;
    if region.quant_type != expected_qtype {
        return Err(format!(
            "tensor `{}`: expected qtype {} got {}",
            name, expected_qtype, region.quant_type
        ));
    }
    let bytes = weights
        .tensor_bytes(name)
        .ok_or_else(|| format!("tensor `{}` has no mapped bytes", name))?;

    let padded = (bytes.len() + 3) & !3;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: padded as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    if padded > bytes.len() {
        let pad = vec![0u8; padded - bytes.len()];
        queue.write_buffer(&buf, bytes.len() as u64, &pad);
    }
    Ok(buf)
}

fn upload_fp32_tensor(
    weights: &ModelWeights,
    name: &str,
    expected: usize,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    label: &'static str,
) -> Result<wgpu::Buffer, String> {
    let data = load_dequantized_or_fail(weights, name, expected)?;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU helpers (RMSNorm + sampling)
// ─────────────────────────────────────────────────────────────────────────────

/// When `MOE_SLICE=1`, log checkpoints at scoped layer/position.
///
/// Position filter (first match wins):
///   `MOE_SLICE_POS_LIST=0,1,2,3` — log multiple query positions in one prefill
///   `MOE_SLICE_ALL=1`             — log every position at `MOE_SLICE_LAYER`
///   `MOE_SLICE_POS=3`             — single position (default 0)
pub(crate) fn moe_slice_active(layer: usize, pos: usize) -> bool {
    if std::env::var("MOE_SLICE").ok().as_deref() != Some("1") {
        return false;
    }
    let want_layer = std::env::var("MOE_SLICE_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if layer != want_layer {
        return false;
    }
    moe_slice_pos_match(pos)
}

fn moe_slice_pos_match(pos: usize) -> bool {
    if std::env::var("MOE_SLICE_ALL").ok().as_deref() == Some("1") {
        return true;
    }
    if let Ok(list) = std::env::var("MOE_SLICE_POS_LIST") {
        let trimmed = list.trim();
        if !trimmed.is_empty() {
            return trimmed
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .any(|p| p == pos);
        }
    }
    let want_pos = std::env::var("MOE_SLICE_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    pos == want_pos
}

pub(crate) fn moe_slice_log(layer: usize, pos: usize, phase: &str, x: &[f32]) {
    if !moe_slice_active(layer, pos) {
        return;
    }
    if let Ok(dir) = std::env::var("MOE_SLICE_DUMP_DIR") {
        let want = std::env::var("MOE_SLICE_DUMP_PHASE").unwrap_or_default();
        let dump = want.is_empty()
            || want
                .split(',')
                .any(|p| p.trim() == phase);
        if dump {
            let path = format!("{dir}/L{layer}_pos{pos}_{phase}.f32");
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let bytes: &[u8] = bytemuck::cast_slice(x);
            let _ = std::fs::write(&path, bytes);
        }
    }
    let n = moe_slice_dims(x.len());
    let vals: Vec<String> = x[..n].iter().map(|v| format!("{v:.6}")).collect();
    eprintln!(
        "MOE_SLICE layer={layer} pos={pos} {phase} dims={n} [{}]",
        vals.join(",")
    );
}

fn moe_slice_dims(len: usize) -> usize {
    std::env::var("MOE_SLICE_DIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
        .min(len)
}

/// Log attention score/prob vectors (head 0) — may be longer than MOE_SLICE_DIMS.
pub(crate) fn moe_slice_log_attn_vec(layer: usize, pos: usize, phase: &str, x: &[f32]) {
    if !moe_slice_active(layer, pos) {
        return;
    }
    let cap = std::env::var("MOE_SLICE_ATTN_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let n = cap.min(x.len());
    let vals: Vec<String> = x[..n].iter().map(|v| format!("{v:.6}")).collect();
    eprintln!(
        "MOE_SLICE layer={layer} pos={pos} {phase} dims={n} [{}]",
        vals.join(",")
    );
}

fn rmsnorm_gemma_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    let mut sum_sq = 0.0f32;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    let plus_one = std::env::var("GEMMA4_NORM_PLUS_ONE").map(|v| v == "1").unwrap_or(false);
    if plus_one {
        for i in 0..n {
            x[i] = x[i] * inv_rms * (weight[i] + 1.0);
        }
    } else {
        for i in 0..n {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }
}

/// Top-k `(token_id, logit)` pairs by value (LOGIT_AUDIT / layer bisect).
pub fn top_k_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .into_iter()
        .take(k)
        .map(|(i, v)| (i as u32, v))
        .collect()
}

/// Greedy / temperature / top-p sampling matching `generate.rs::sample_next_token`.
fn sample_logits(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature < 0.01 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
    }
    let max_logit = logits
        .iter()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max_logit.is_finite() {
        return 0;
    }
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| {
            if !l.is_finite() {
                (idx, 0.0)
            } else {
                (idx, ((l - max_logit) / temperature).exp())
            }
        })
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 {
        return 0;
    }
    for entry in probs.iter_mut() {
        entry.1 /= sum;
    }
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut cutoff = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    probs.truncate(cutoff);
    let final_sum: f32 = probs.iter().map(|(_, p)| p).sum();
    let r = rand::random::<f32>() * final_sum;
    let mut acc = 0.0f32;
    for (id, p) in &probs {
        acc += p;
        if r <= acc {
            return *id as u32;
        }
    }
    probs.last().map(|(id, _)| *id as u32).unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Device init
// ─────────────────────────────────────────────────────────────────────────────

/// Vulkan discrete-GPU index for Gemma4Runner tests and CLI (`GEMMA4_GPU`, default 0).
pub fn gemma4_gpu_index() -> usize {
    std::env::var("GEMMA4_GPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn init_device(gpu_index: usize) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter()
        .filter(|a| matches!(a.get_info().device_type, wgpu::DeviceType::DiscreteGpu))
        .collect();
    if adapters.is_empty() {
        return Err("no discrete Vulkan GPUs visible".into());
    }
    let adapter = &adapters[gpu_index.min(adapters.len() - 1)];
    let info = adapter.get_info();
    tracing::info!("Selected GPU {}: {} ({:?})", gpu_index, info.name, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("gemma4_runner"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024 - 1, // 2 GB (P100 max)
                max_buffer_size: u64::MAX,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    Ok((Arc::new(device), Arc::new(queue)))
}

// Silence dead_code on the embedded MatvecPush we re-export pattern;
// remove if a future PR uses it directly.
#[allow(dead_code)]
fn _suppress_unused_matvec_push(_p: MatvecPush) {}

/// Split `total` into chunks of at most `max_per_chunk`, returning
/// `(start_row, count)` pairs.
fn chunk_ranges(total: usize, max_per_chunk: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < total {
        let count = (total - offset).min(max_per_chunk);
        out.push((offset, count));
        offset += count;
    }
    out
}
