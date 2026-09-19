//! GPU-resident Gemma-4 forward pass.
//!
//! `Gemma4Runner` (sibling module) keeps the hidden state on the CPU and
//! copies it to the GPU for every matmul. That CPU↔GPU bouncing dominates
//! single-token latency on Pascal where the PCIe round trip is ~10–20 µs
//! per submission and we do 30 layers × ~12 dispatches = ~360 round trips
//! per token.
//!
//! This runner hosts everything the layer needs on the GPU:
//!   * hidden state (single 2816 f32 buffer, ping-pongs in place)
//!   * residual snapshot (same shape)
//!   * Q / K / V scratch (one buffer each, reused across layers)
//!   * FFN scratch (gate, up, fused, down output)
//!   * KV caches per layer (one (K, V) pair)
//!   * pre-uploaded norm weights per layer
//!
//! The only CPU↔GPU traffic per token is:
//!   1. write the token id into a tiny push uniform (4 bytes).
//!   2. ~128 fp32 readbacks for the MoE router top-k (per layer).
//!   3. one final readback of the vocab logits (for sampling).
//!
//! Numerical correctness for every primitive is enforced by
//! `tests/gpu_kernel_correctness.rs`. See `gemma4_gpu_pipelines.rs` for
//! the shader bundle.
//!
//! HONEST CAVEATS (don't pretend otherwise):
//!   * The MoE block ordering of the three Gemma-4 norms
//!     (`pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`) follows
//!     the same pseudocode as `gemma4_runner.rs::moe_block`. If that's
//!     wrong the GPU runner is wrong by the same amount.
//!   * The MoE FFN dispatch (router → top-k → expert matvec → SwiGLU →
//!     down → accum) routes through `MoeFfnDispatch::forward`, which
//!     internally still does ONE small CPU readback of 128 f32 router
//:     logits per layer. That's the unavoidable sync point of top-k.
//!   * The dense bypass FFN (gate_w/up_w/down_w) is dropped here because
//!     Gemma-4-26B-MoE has no dense bypass — the runner uses the MoE FFN
//!     for every layer. If you point this at a non-MoE Gemma-4 variant,
//!     the dense block will need to be added back.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::gemma4_gpu_pipelines::{
    AttnPush, EmbedLookupPush, Gemma4GpuPipelines, KvWritePush, LogitSoftcapPush, RmsNormPush,
    RmsNormWeightlessPush, RopePush, SiluMulPush,
};
use crate::gemma4_layer::Gemma4Config;
use crate::gemma4_runner::Gemma4Runner;
use crate::moe_iq4_dispatch::MoeFfnConfig;
use crate::pamp::MoeDispatchMeta;

const RMS_PLUS_ONE: u32 = 0; // see gemma4_layer.rs::rmsnorm_gemma_inplace

// ────────────────────────────────────────────────────────────────────────────
// Per-layer GPU norms
// ────────────────────────────────────────────────────────────────────────────

/// All per-layer norm weights uploaded once to the GPU.
struct LayerGpuNorms {
    attn_norm: wgpu::Buffer,
    attn_q_norm: wgpu::Buffer,
    attn_k_norm: wgpu::Buffer,
    post_attention_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,            // pre_feedforward_layernorm in HF
    post_ffw_norm: wgpu::Buffer,
    pre_ffw_norm_2: wgpu::Buffer,
    post_ffw_norm_1: wgpu::Buffer,
    post_ffw_norm_2: wgpu::Buffer,
    layer_output_scale: f32,
}

fn upload_norm_buf(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    label: &str,
    data: &[f32],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

// ────────────────────────────────────────────────────────────────────────────
// GPU KV cache (one (K, V) pair per layer)
// ────────────────────────────────────────────────────────────────────────────

struct LayerKvGpu {
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    kv_dim: usize,
    max_seq: usize,
    cur_len: usize,
}

impl LayerKvGpu {
    fn new(device: &Arc<wgpu::Device>, kv_dim: usize, max_seq: usize, layer: usize) -> Self {
        let bytes = (max_seq * kv_dim * 4) as u64;
        let k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gpu_kv_k_{layer}")),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gpu_kv_v_{layer}")),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            k_cache: k,
            v_cache: v,
            kv_dim,
            max_seq,
            cur_len: 0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Token-embedding table on GPU
// ────────────────────────────────────────────────────────────────────────────

struct EmbedTable {
    /// fp32 dense embedding table, possibly split when total size > max_buffer_size.
    /// For Gemma-4 26B MoE, hidden=2816 × vocab=262144 × 4 = ~2.95 GB which is over
    /// the P100 max_storage_buffer_binding_size of 2,147,483,647 bytes. We keep two
    /// halves and pick the right one at lookup time.
    lo: wgpu::Buffer,
    hi: wgpu::Buffer,
    half_vocab: usize,
    hidden_dim: usize,
}

impl EmbedTable {
    fn upload(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        table: &[f32],
        vocab: usize,
        hidden_dim: usize,
    ) -> Self {
        debug_assert_eq!(table.len(), vocab * hidden_dim);
        let half_vocab = vocab / 2;
        let half_elems = half_vocab * hidden_dim;
        let lo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu.embed.lo"),
            size: (half_elems * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&lo, 0, bytemuck::cast_slice(&table[..half_elems]));
        let hi = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu.embed.hi"),
            size: ((table.len() - half_elems) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&hi, 0, bytemuck::cast_slice(&table[half_elems..]));
        Self {
            lo,
            hi,
            half_vocab,
            hidden_dim,
        }
    }

    /// Returns (buffer, local_token_id within that half).
    fn pick(&self, token_id: u32) -> (&wgpu::Buffer, u32) {
        let tid = token_id as usize;
        if tid < self.half_vocab {
            (&self.lo, token_id)
        } else {
            (&self.hi, (tid - self.half_vocab) as u32)
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Scratch buffers reused across all layers
// ────────────────────────────────────────────────────────────────────────────

struct Scratch {
    /// Hidden state (current residual stream). Length = hidden_dim.
    hidden: wgpu::Buffer,
    /// Snapshot of the residual before a sub-block.
    residual: wgpu::Buffer,
    /// Norm output buffer (post-RMSNorm scratch).
    norm_out: wgpu::Buffer,
    /// Q after Q projection (length = q_dim_full = n_heads * head_dim_full).
    /// Sized for the largest layer so SWA reuses the buffer.
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    /// Per-head RMSNorm output scratch (sized to max_q_dim — large enough
    /// for Q, K, V usage). Required because WGPU disallows binding the same
    /// buffer as both read-only and read/write within one dispatch.
    head_norm: wgpu::Buffer,
    /// Attention output (length = q_dim).
    attn_out: wgpu::Buffer,
    /// Output of o_proj (= hidden_dim).
    o_out: wgpu::Buffer,
    /// MoE scratch (intermediate inner = expert_inner per token).
    moe_y: wgpu::Buffer,
    /// Logits scratch for LM head (vocab_size).
    logits: wgpu::Buffer,
    logits_softcapped: wgpu::Buffer,
}

impl Scratch {
    fn new(
        device: &Arc<wgpu::Device>,
        hidden_dim: usize,
        max_q_dim: usize,
        max_kv_dim: usize,
        vocab: usize,
    ) -> Self {
        let f32buf = |label: &'static str, n: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | extra,
                mapped_at_creation: false,
            })
        };
        Self {
            hidden: f32buf("scratch.hidden", hidden_dim, wgpu::BufferUsages::empty()),
            residual: f32buf("scratch.residual", hidden_dim, wgpu::BufferUsages::empty()),
            norm_out: f32buf("scratch.norm_out", hidden_dim, wgpu::BufferUsages::empty()),
            q: f32buf("scratch.q", max_q_dim, wgpu::BufferUsages::empty()),
            k: f32buf("scratch.k", max_kv_dim, wgpu::BufferUsages::empty()),
            v: f32buf("scratch.v", max_kv_dim, wgpu::BufferUsages::empty()),
            head_norm: f32buf("scratch.head_norm", max_q_dim, wgpu::BufferUsages::empty()),
            attn_out: f32buf("scratch.attn_out", max_q_dim, wgpu::BufferUsages::empty()),
            o_out: f32buf("scratch.o_out", hidden_dim, wgpu::BufferUsages::empty()),
            moe_y: f32buf("scratch.moe_y", hidden_dim, wgpu::BufferUsages::empty()),
            logits: f32buf("scratch.logits", vocab, wgpu::BufferUsages::empty()),
            logits_softcapped: f32buf("scratch.logits_capped", vocab, wgpu::BufferUsages::empty()),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The GPU runner
// ────────────────────────────────────────────────────────────────────────────

/// GPU-resident driver for Gemma-4 inference. Constructed from an existing
/// CPU `Gemma4Runner` so we can reuse its IQ4 weight uploads + MoE expert
/// uploads — those buffers are already on the GPU. We just add per-layer
/// norm uploads and the activation scratch.
pub struct Gemma4GpuRunner {
    pub config: Gemma4Config,
    pub moe_config: MoeFfnConfig,
    pub vocab_size: usize,
    pub logit_softcap: f32,

    pipes: Gemma4GpuPipelines,

    /// Reuses every weight buffer already on the GPU inside the source
    /// runner: layer Q/K/V/O/gate/up/down, expert tensors, LM head halves,
    /// final norm. We hold an Arc to keep the buffers alive.
    pub source: Arc<Gemma4Runner>,

    layer_norms: Vec<LayerGpuNorms>,
    final_norm: wgpu::Buffer,
    embed: EmbedTable,
    kv: Vec<LayerKvGpu>,
    scratch: Scratch,

    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Gemma4GpuRunner {
    /// Wrap an existing `Gemma4Runner`, allocating the per-layer norm
    /// buffers + activation scratch on the source runner's device.
    pub fn from_source(source: Arc<Gemma4Runner>, max_seq: usize) -> Self {
        let device = source.device.clone();
        let queue = source.queue.clone();
        let pipes = Gemma4GpuPipelines::new(&device);
        let cfg = source.config.clone();

        // Upload per-layer norms.
        let mut layer_norms = Vec::with_capacity(source.layers.len());
        for (li, layer) in source.layers.iter().enumerate() {
            let dense = &layer.dense;
            let l = LayerGpuNorms {
                attn_norm: upload_norm_buf(&device, &queue, &fmt(li, "attn_norm"), &dense.attn_norm),
                attn_q_norm: upload_norm_buf(&device, &queue, &fmt(li, "attn_q_norm"), &dense.attn_q_norm),
                attn_k_norm: upload_norm_buf(&device, &queue, &fmt(li, "attn_k_norm"), &dense.attn_k_norm),
                post_attention_norm: upload_norm_buf(
                    &device, &queue, &fmt(li, "post_attention_norm"), &dense.post_attention_norm,
                ),
                ffn_norm: upload_norm_buf(&device, &queue, &fmt(li, "ffn_norm"), &dense.ffn_norm),
                post_ffw_norm: upload_norm_buf(
                    &device, &queue, &fmt(li, "post_ffw_norm"), &dense.post_ffw_norm,
                ),
                pre_ffw_norm_2: upload_norm_buf(
                    &device, &queue, &fmt(li, "pre_ffw_norm_2"), &layer.pre_ffw_norm_2,
                ),
                post_ffw_norm_1: upload_norm_buf(
                    &device, &queue, &fmt(li, "post_ffw_norm_1"), &layer.post_ffw_norm_1,
                ),
                post_ffw_norm_2: upload_norm_buf(
                    &device, &queue, &fmt(li, "post_ffw_norm_2"), &layer.post_ffw_norm_2,
                ),
                layer_output_scale: dense.layer_output_scale,
            };
            layer_norms.push(l);
        }

        // Final norm.
        let final_norm = upload_norm_buf(&device, &queue, "final_norm", &source.final_norm);

        // Embedding table.
        let embed = EmbedTable::upload(
            &device,
            &queue,
            &source.token_embeddings,
            source.vocab_size,
            cfg.hidden_size,
        );

        // KV caches: largest possible kv_dim is the max across layers, but
        // each layer has its own dimensions, so we size each cache exactly.
        let mut kv = Vec::with_capacity(source.layers.len());
        let mut max_q_dim = 0usize;
        let mut max_kv_dim = 0usize;
        for (li, layer) in source.layers.iter().enumerate() {
            let head_dim = if layer.dense.is_swa {
                cfg.head_dim_swa
            } else {
                cfg.head_dim_full
            };
            let q_dim = cfg.num_heads * head_dim;
            let kv_dim = layer.dense.n_kv_heads * head_dim;
            max_q_dim = max_q_dim.max(q_dim);
            max_kv_dim = max_kv_dim.max(kv_dim);
            kv.push(LayerKvGpu::new(&device, kv_dim, max_seq, li));
        }

        let scratch = Scratch::new(&device, cfg.hidden_size, max_q_dim, max_kv_dim, source.vocab_size);

        Self {
            moe_config: source.moe_config,
            vocab_size: source.vocab_size,
            logit_softcap: source.logit_softcap,
            source,
            config: cfg,
            pipes,
            layer_norms,
            final_norm,
            embed,
            kv,
            scratch,
            device,
            queue,
        }
    }

    /// Reset KV state. Call between independent prompts.
    pub fn reset_kv(&mut self) {
        for k in &mut self.kv {
            k.cur_len = 0;
        }
    }

    /// Single forward step: write the token's logits (already softcapped)
    /// to a CPU vector and return them. After this call the per-layer KV
    /// caches have been advanced by one position.
    pub fn forward_token(&mut self, token_id: u32) -> Vec<f32> {
        // 1. Embed lookup → scratch.hidden.
        self.dispatch_embed_lookup(token_id);

        // 2. Walk layers.
        let n_layers = self.layer_norms.len();
        for li in 0..n_layers {
            self.dispatch_attention_block(li);
            self.dispatch_moe_block(li);
        }

        // 3. Final norm + LM head + softcap → logits.
        self.dispatch_final_norm();
        let logits = self.dispatch_lm_head_and_softcap();

        // 4. Advance KV.
        for k in &mut self.kv {
            k.cur_len += 1;
        }
        logits
    }

    // ── embedding ────────────────────────────────────────────────────────
    fn dispatch_embed_lookup(&self, token_id: u32) {
        let (table, local_id) = self.embed.pick(token_id);
        let push = self.upload_uniform(
            "embed_push",
            EmbedLookupPush {
                token_id: local_id,
                hidden_dim: self.config.hidden_size as u32,
                embed_scale_flag: 1,
                _pad: 0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed_bg"),
            layout: &self.pipes.embed_lookup.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scratch.hidden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.embed_lookup.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(((self.config.hidden_size + 255) / 256) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    // ── attention block ──────────────────────────────────────────────────
    fn dispatch_attention_block(&mut self, li: usize) {
        let layer = &self.source.layers[li];
        let dense = &layer.dense;
        let cfg = &self.config;
        let norms = &self.layer_norms[li];
        let head_dim = if dense.is_swa { cfg.head_dim_swa } else { cfg.head_dim_full };
        let rope_base = if dense.is_swa { cfg.rope_theta_swa } else { cfg.rope_theta_full };
        let q_dim = cfg.num_heads * head_dim;
        let kv_dim = dense.n_kv_heads * head_dim;
        let pos = self.kv[li].cur_len as u32;

        // Snapshot residual = hidden.
        self.copy_buffer(&self.scratch.hidden, &self.scratch.residual, cfg.hidden_size);

        // attn_norm onto norm_out.
        self.dispatch_rmsnorm(&self.scratch.hidden, &norms.attn_norm, &self.scratch.norm_out);

        // Q/K/V projections. Each is its own dispatch through Iq4MatvecPipeline.
        self.iq4_matvec(&dense.q_w, &self.scratch.norm_out, &self.scratch.q, cfg.hidden_size, q_dim, /*nl=*/ false);
        self.iq4_matvec(&dense.k_w, &self.scratch.norm_out, &self.scratch.k, cfg.hidden_size, kv_dim, false);
        if dense.uses_shared_kv {
            self.copy_buffer(&self.scratch.k, &self.scratch.v, kv_dim);
        } else {
            self.iq4_matvec(&dense.v_w, &self.scratch.norm_out, &self.scratch.v, cfg.hidden_size, kv_dim, false);
        }

        // Per-head Q/K RMSNorm (head_dim each), V RMSNorm weightless.
        self.dispatch_rmsnorm_per_head(&self.scratch.q, &norms.attn_q_norm, head_dim, cfg.num_heads);
        self.dispatch_rmsnorm_per_head(&self.scratch.k, &norms.attn_k_norm, head_dim, dense.n_kv_heads);
        self.dispatch_rmsnorm_weightless_per_head(&self.scratch.v, head_dim, dense.n_kv_heads);

        // RoPE on Q and K.
        self.dispatch_rope(&self.scratch.q, head_dim, cfg.num_heads, pos, rope_base);
        self.dispatch_rope(&self.scratch.k, head_dim, dense.n_kv_heads, pos, rope_base);

        // Append K/V into per-layer cache at position `pos`.
        self.dispatch_kv_write(li, kv_dim, pos);
        let new_seq_len = self.kv[li].cur_len + 1;

        // Single-step attention.
        let window_start = if dense.is_swa {
            (new_seq_len - 1).saturating_sub(cfg.sliding_window)
        } else {
            0
        };
        self.dispatch_attention(
            li,
            head_dim,
            cfg.num_heads,
            dense.n_kv_heads,
            new_seq_len,
            window_start,
            dense.is_swa,
        );

        // O projection: attn_out → o_out.
        self.iq4_matvec(&dense.o_w, &self.scratch.attn_out, &self.scratch.o_out, q_dim, cfg.hidden_size, false);

        // post_attention_norm in place onto o_out (use scratch.norm_out as
        // staging, then back to o_out).
        self.dispatch_rmsnorm(&self.scratch.o_out, &norms.post_attention_norm, &self.scratch.norm_out);
        // hidden = residual + scale * norm_out.
        self.dispatch_residual_scaled(
            &self.scratch.residual,
            &self.scratch.norm_out,
            &self.scratch.hidden,
            norms.layer_output_scale,
        );
    }

    // ── MoE block ────────────────────────────────────────────────────────
    fn dispatch_moe_block(&mut self, li: usize) {
        let cfg = &self.config;
        let norms = &self.layer_norms[li];
        let layer = &self.source.layers[li];
        let h = cfg.hidden_size;

        // residual snapshot.
        self.copy_buffer(&self.scratch.hidden, &self.scratch.residual, h);
        // pre_ffw_norm_2 → norm_out.
        self.dispatch_rmsnorm(&self.scratch.hidden, &norms.pre_ffw_norm_2, &self.scratch.norm_out);

        // MoE FFN: norm_out → moe_y. Internally does router top-k readback.
        // Pre-zero the moe_y accumulator.
        {
            let zeros = vec![0u8; h * 4];
            self.queue.write_buffer(&self.scratch.moe_y, 0, &zeros);
        }
        // TODO: GPU path still lacks shared dense FFN + correct router prep.
        let meta = MoeDispatchMeta {
            layer: li as u32,
            token_seq: self.kv[0].cur_len as u64,
        };
        if let Err(e) = self.source.moe_dispatch.forward(
            &self.moe_config,
            &self.scratch.norm_out,
            &self.scratch.residual,
            &layer.gate_up_exps,
            &layer.down_exps,
            &layer.down_exps_scale,
            &layer.router_w,
            &layer.router_scale,
            &self.scratch.moe_y,
            Some(&meta),
        ) {
            tracing::error!("MoE dispatch failed at layer {}: {}", li, e);
            // Bail: leave hidden untouched. Caller will likely produce gibberish
            // for this token, but better than crashing.
            return;
        }

        // post_ffw_norm_1 onto moe_y (in place via norm_out staging).
        self.dispatch_rmsnorm(&self.scratch.moe_y, &norms.post_ffw_norm_1, &self.scratch.norm_out);
        // hidden = residual + scale * norm_out.
        self.dispatch_residual_scaled(
            &self.scratch.residual,
            &self.scratch.norm_out,
            &self.scratch.hidden,
            norms.layer_output_scale,
        );
        // Final post_ffw_norm_2 in place on hidden.
        self.dispatch_rmsnorm(&self.scratch.hidden, &norms.post_ffw_norm_2, &self.scratch.norm_out);
        self.copy_buffer(&self.scratch.norm_out, &self.scratch.hidden, h);
    }

    // ── final norm + LM head + softcap ───────────────────────────────────
    fn dispatch_final_norm(&self) {
        self.dispatch_rmsnorm(&self.scratch.hidden, &self.final_norm, &self.scratch.norm_out);
    }

    fn dispatch_lm_head_and_softcap(&self) -> Vec<f32> {
        let h = self.config.hidden_size;
        self.source.lm_head_matvec_gpu(
            &self.scratch.norm_out,
            &self.scratch.logits,
            h,
            self.vocab_size,
        );
        self.device.poll(wgpu::Maintain::Wait);

        // Softcap in-place: logits → logits_softcapped (one vocab readback at end).
        let push = self.upload_uniform(
            "softcap_push",
            LogitSoftcapPush {
                n: self.vocab_size as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                cap: self.logit_softcap,
                _pad3: 0.0,
                _pad4: 0.0,
                _pad5: 0.0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softcap_bg"),
            layout: &self.pipes.logit_softcap.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scratch.logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scratch.logits_softcapped.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.logit_softcap.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(((self.vocab_size + 255) / 256) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        let mut out = vec![0.0f32; self.vocab_size];
        self.read_back(&self.scratch.logits_softcapped, &mut out);
        out
    }

    // ── primitive helpers ─────────────────────────────────────────────────

    fn copy_buffer(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, n: usize) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(src, 0, dst, 0, (n * 4) as u64);
        self.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_rmsnorm(&self, x: &wgpu::Buffer, weight: &wgpu::Buffer, y: &wgpu::Buffer) {
        let push = self.upload_uniform(
            "rmsnorm_push",
            RmsNormPush {
                hidden_dim: self.config.hidden_size as u32,
                plus_one_flag: RMS_PLUS_ONE,
                _pad0: 0,
                _pad1: 0,
                eps: self.config.rms_norm_eps,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm_bg"),
            layout: &self.pipes.rmsnorm.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.rmsnorm.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// rmsnorm applied independently to each [head_dim]-slice of the buffer.
    fn dispatch_rmsnorm_per_head(
        &self,
        buf: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        head_dim: usize,
        n_heads: usize,
    ) {
        // The shader hardcodes 256 threads per workgroup and 1 row per workgroup.
        // It indexes `row * hidden_dim + i` so we can dispatch n_heads workgroups
        // with hidden_dim = head_dim.
        //
        // CRITICAL (wgpu usage scope): never bind `buf` as both STORAGE_READ_ONLY
        // and STORAGE_READ_WRITE in one compute pass — that is the scratch.q panic
        // seen on P100 (`forward_pipeline_*` batching). Stage through head_norm,
        // then copy back in a *separate* submit so COPY_DST is not unioned with
        // the prior storage read.
        let total = head_dim * n_heads;
        let push = self.upload_uniform(
            "rmsnorm_head_push",
            RmsNormPush {
                hidden_dim: head_dim as u32,
                plus_one_flag: RMS_PLUS_ONE,
                _pad0: 0,
                _pad1: 0,
                eps: self.config.rms_norm_eps,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm_head_bg"),
            layout: &self.pipes.rmsnorm.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scratch.head_norm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push.as_entire_binding() },
            ],
        });
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                p.set_pipeline(&self.pipes.rmsnorm.pipeline);
                p.set_bind_group(0, Some(&bg), &[]);
                p.dispatch_workgroups(n_heads as u32, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }
        // Separate submit: head_norm → buf (avoids STORAGE+COPY usage union).
        self.copy_buffer(&self.scratch.head_norm, buf, total);
    }

    fn dispatch_rmsnorm_weightless_per_head(
        &self,
        buf: &wgpu::Buffer,
        head_dim: usize,
        n_heads: usize,
    ) {
        let total = head_dim * n_heads;
        let push = self.upload_uniform(
            "rmsw_head_push",
            RmsNormWeightlessPush {
                hidden_dim: head_dim as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                eps: self.config.rms_norm_eps,
                _pad3: 0.0,
                _pad4: 0.0,
                _pad5: 0.0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsw_head_bg"),
            layout: &self.pipes.rmsnorm_weightless.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scratch.head_norm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
            ],
        });
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                p.set_pipeline(&self.pipes.rmsnorm_weightless.pipeline);
                p.set_bind_group(0, Some(&bg), &[]);
                p.dispatch_workgroups(n_heads as u32, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }
        self.copy_buffer(&self.scratch.head_norm, buf, total);
    }

    fn dispatch_rope(
        &self,
        buf: &wgpu::Buffer,
        head_dim: usize,
        n_heads: usize,
        pos: u32,
        rope_base: f32,
    ) {
        // rope_v2 is in-place (single read_write binding). That is legal alone.
        // Keep it in its own submit so a batched encoder cannot union this RW
        // usage with an earlier RO bind of the same buffer (scratch.q panic).
        let total = head_dim * n_heads;
        let push = self.upload_uniform(
            "rope_push",
            RopePush {
                head_dim: head_dim as u32,
                pos,
                n_heads: n_heads as u32,
                _pad: 0,
                rope_base,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            },
        );
        // Stage: copy into head_norm, rotate head_norm in-place, copy back.
        // Guarantees `buf` is never RO+RW in one pass even if callers batch.
        self.copy_buffer(buf, &self.scratch.head_norm, total);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope_bg"),
            layout: &self.pipes.rope.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scratch.head_norm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: push.as_entire_binding() },
            ],
        });
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                p.set_pipeline(&self.pipes.rope.pipeline);
                p.set_bind_group(0, Some(&bg), &[]);
                p.dispatch_workgroups(n_heads as u32, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }
        self.copy_buffer(&self.scratch.head_norm, buf, total);
    }

    fn dispatch_kv_write(&self, li: usize, kv_dim: usize, pos: u32) {
        let kv = &self.kv[li];
        let push = self.upload_uniform(
            "kvw_push",
            KvWritePush {
                pos,
                kv_dim: kv_dim as u32,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kvw_bg"),
            layout: &self.pipes.kv_write.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scratch.k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scratch.v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kv.k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kv.v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.kv_write.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(((kv_dim + 255) / 256) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_attention(
        &self,
        li: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        window_start: usize,
        is_swa: bool,
    ) {
        let kv = &self.kv[li];
        // Gemma-4: f_attention_scale = 1.0 (NOT 1/sqrt(head_dim)).
        let scale = self.config.attention_scale;
        let push = self.upload_uniform(
            "attn_push",
            AttnPush {
                head_dim: head_dim as u32,
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                heads_per_kv: (n_heads / n_kv_heads) as u32,
                seq_len: seq_len as u32,
                window_start: window_start as u32,
                use_swa: if is_swa { 1 } else { 0 },
                _pad0: 0,
                scale,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bg"),
            layout: &self.pipes.attn.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scratch.q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: kv.k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kv.v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.scratch.attn_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.attn.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(n_heads as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// dst = a + scale * b  (fused via weighted_accum: dst <- a; dst += scale*b).
    fn dispatch_residual_scaled(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        scale: f32,
    ) {
        // Step 1: dst <- a (memcpy).
        self.copy_buffer(a, dst, self.config.hidden_size);
        // Step 2: dst += scale * b.
        let push = self.upload_uniform(
            "wa_push",
            crate::gemma4_gpu_pipelines::WeightedAccumPush {
                weight: scale,
                len: self.config.hidden_size as u32,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wa_bg"),
            layout: &self.pipes.weighted_accum.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(&self.pipes.weighted_accum.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(((self.config.hidden_size + 255) / 256) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// IQ4 matvec wrapper that picks the right shared pipeline. Dispatches
    /// directly using the source runner's pipelines.
    fn iq4_matvec(
        &self,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        k: usize,
        n: usize,
        is_iq4_nl: bool,
    ) {
        let pipe = if is_iq4_nl {
            self.source.iq4nl_pipe.clone()
        } else {
            self.source.iq4xs_pipe.clone()
        };
        let push = pipe.make_push_buffer(crate::iq4_pipeline::MatvecPush {
            k: k as u32,
            n_rows_total: n as u32,
            row_offset: 0,
            _pad: 0,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        pipe.dispatch(&mut enc, w, x, y, &push, n as u32);
        self.queue.submit(std::iter::once(enc.finish()));
    }

    fn upload_uniform<T: Pod + Zeroable>(&self, label: &'static str, value: T) -> wgpu::Buffer {
        let bytes = bytemuck::bytes_of(&value);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    fn upload_f32_buf(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    fn read_back(&self, src: &wgpu::Buffer, out: &mut [f32]) {
        let bytes = (out.len() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        let sub = self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub));
        let _ = rx.recv();
        let mapped = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        staging.unmap();
    }
}

fn fmt(li: usize, kind: &str) -> String {
    format!("layer_{li}.{kind}")
}
