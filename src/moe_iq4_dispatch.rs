//! MoE FFN expert dispatch for Gemma-4-26B-MoE (IQ4_XS gate||up + IQ4_NL down).
//!
//! Shape facts (per `docs/SHADER_SPEC_FOR_FRIEND.md`):
//!   hidden        = 2816
//!   expert_inner  =  704
//!   n_experts     =  128
//!   top_k         =    8
//!   ffn_gate_up_exps[hidden, 1408, 128]   IQ4_XS    gate||up packed (704+704)
//!   ffn_down_exps   [704,    2816, 128]   IQ4_NL
//!   ffn_gate_inp.weight[hidden, 128]      fp32      router weights
//!   ffn_gate_inp.scale[hidden]            fp32      input scale
//!
//! Pipeline summary, per token:
//!   1. CPU-side (or tiny GPU): gi[i] = x[i] * scale[i]
//!   2. GPU matvec: logits[128] = router_w @ gi
//!   3. CPU readback + stable top-k softmax → (top_k_idx[8], top_k_w[8])
//!   4. For each of top-k experts e, weight w:
//!         gu[1408] = gate_up_exps[e]  @ x        (IQ4_XS, MoE matvec)
//!         h[704]   = gelu(gu[:704]) * gu[704:]
//!         out[2816]= down_exps[e]     @ h        (IQ4_NL, MoE matvec)
//!         y       += w * out
//!
//! Design choice (per requirement #1 vs #2):
//! ------------------------------------------
//! We reuse a single big GPU buffer per (layer, kind) holding all 128
//! experts contiguously and add a `expert_offset_words` (IQ4_XS) /
//! `expert_offset_bytes` (IQ4_NL) push field that biases the matvec into
//! the right slice. No per-expert sub-buffer rebinding, no extra bind
//! groups. Matches the explicit guidance in `MATVEC_HANDOFF_FOR_LLM.md`:
//!   "For MoE, build a per-expert variant by adding `expert_offset_words`
//!    and `expert_stride_words` push fields and offset `bo` accordingly."
//!
//! We DO NOT extend `Iq4MatvecPipeline` because that pipeline's bind
//! group layout uses the 3-field push struct; the MoE shaders use a
//! 4-field push (with the expert offset). Instead, we build sibling
//! pipelines here. They share the LUT layout and dispatch shape.
//!
//! Honest gaps (see `// HONEST GAP:` comments below):
//!   - Whether experts are the OUTERMOST or INNERMOST dim in the GGUF
//!     packed tensor. We assume outermost (e * row_count * row_bytes
//!     stride). TODO: cross-check against llama.cpp `llm_build_moe_ffn`.
//!   - Whether the gate||up split inside `ffn_gate_up_exps` is "gate
//!     rows then up rows" or "interleaved". Per the spec: "First 704
//!     cols = gate, next 704 cols = up" along the 1408 dim. We treat the
//!     1408 as the row dim of the matvec (gu = W @ x with W[1408, 2816]),
//!     so rows 0..704 are gate, rows 704..1408 are up.
//!   - Router weight matrix orientation: we assume row-major
//!     `[n_experts, hidden]` based on llama.cpp convention. If the GGUF
//!     stores it transposed, transpose once at upload.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::iq4_pipeline::{KVALUES_IQ4, QuantKind};
use crate::pamp::{self, MoeDispatchMeta};

// ---------------------------------------------------------------------------
// Push structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MoeIq4xsPush {
    pub k: u32,
    pub n_rows_total: u32,
    pub row_offset: u32,
    pub expert_offset_words: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MoeIq4nlPush {
    pub k: u32,
    pub n_rows_total: u32,
    pub row_offset: u32,
    pub expert_offset_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ScalePush {
    pub hidden: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct RouterPush {
    pub hidden: u32,
    pub n_experts: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SiluSplitPush {
    pub inner: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AccumPush {
    pub weight: f32,
    pub len: u32,
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// Static config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct MoeFfnConfig {
    pub hidden: usize,        // 2816
    pub expert_inner: usize,  // 704
    pub n_experts: usize,     // 128
    pub top_k: usize,         // 8
}

impl MoeFfnConfig {
    pub const GEMMA4: Self = Self {
        hidden: 2816,
        expert_inner: 704,
        n_experts: 128,
        top_k: 8,
    };

    /// gate||up matrix has 2 * inner output rows (gate_rows then up_rows)
    pub fn gate_up_rows(&self) -> usize {
        2 * self.expert_inner
    }

    /// Bytes occupied by ONE expert's gate_up tensor (IQ4_XS).
    /// Shape per expert: [gate_up_rows, hidden]
    pub fn gate_up_bytes_per_expert(&self) -> usize {
        let row_bytes = QuantKind::Iq4Xs.row_bytes(self.hidden);
        self.gate_up_rows() * row_bytes
    }

    /// Bytes per expert for the down tensor (IQ4_NL).
    /// Shape per expert: [hidden, expert_inner]
    pub fn down_bytes_per_expert(&self) -> usize {
        let row_bytes = QuantKind::Iq4Nl.row_bytes(self.expert_inner);
        self.hidden * row_bytes
    }
}

// ---------------------------------------------------------------------------
// Compiled pipeline bundle
// ---------------------------------------------------------------------------

/// Per-quant MoE matvec pipeline. Sibling of `Iq4MatvecPipeline` but with
/// a 4-field push struct (adds `expert_offset_*`).
pub struct MoeMatvecPipeline {
    pub kind: QuantKind,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// LUT buffer — populated once with [`KVALUES_IQ4`].
    pub lut_buffer: wgpu::Buffer,
}

impl MoeMatvecPipeline {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        kind: QuantKind,
        shader_src: &str,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(match kind {
                QuantKind::Iq4Xs => "moe_iq4xs_matvec",
                QuantKind::Iq4Nl => "moe_iq4nl_matvec",
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moe_iq4_matvec_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
                bgl_uniform(4),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_iq4_matvec_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_iq4_matvec_pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let lut_packed: [[f32; 4]; 4] = [
            [KVALUES_IQ4[0], KVALUES_IQ4[1], KVALUES_IQ4[2], KVALUES_IQ4[3]],
            [KVALUES_IQ4[4], KVALUES_IQ4[5], KVALUES_IQ4[6], KVALUES_IQ4[7]],
            [KVALUES_IQ4[8], KVALUES_IQ4[9], KVALUES_IQ4[10], KVALUES_IQ4[11]],
            [KVALUES_IQ4[12], KVALUES_IQ4[13], KVALUES_IQ4[14], KVALUES_IQ4[15]],
        ];
        let lut_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe_iq4_lut"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut_packed));

        Self { kind, device, queue, pipeline, bind_group_layout: bgl, lut_buffer }
    }
}

/// All compiled pipelines needed for one MoE FFN dispatch.
pub struct MoeFfnDispatch {
    pub gate_up_pipe: MoeMatvecPipeline,    // IQ4_XS, 4-field push
    pub down_pipe: MoeMatvecPipeline,        // IQ4_NL, 4-field push
    pub silu_pipe: wgpu::ComputePipeline,
    pub silu_bgl: wgpu::BindGroupLayout,
    pub accum_pipe: wgpu::ComputePipeline,
    pub accum_bgl: wgpu::BindGroupLayout,
    pub scale_pipe: wgpu::ComputePipeline,
    pub scale_bgl: wgpu::BindGroupLayout,
    pub router_pipe: wgpu::ComputePipeline,
    pub router_bgl: wgpu::BindGroupLayout,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    /// Reused GPU scratch — avoids per-layer/per-token buffer allocation.
    scratch: MoeDispatchScratch,
}

/// Persistent buffers for one MoE FFN forward (sized at dispatch construction).
struct MoeDispatchScratch {
    gi_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    logits_readback: wgpu::Buffer,
    gu_buf: wgpu::Buffer,
    h_buf: wgpu::Buffer,
    out_e_buf: wgpu::Buffer,
    silu_push_buf: wgpu::Buffer,
    scale_push_buf: wgpu::Buffer,
    router_push_buf: wgpu::Buffer,
    silu_bg: wgpu::BindGroup,
}

impl MoeDispatchScratch {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, cfg: &MoeFfnConfig, silu_bgl: &wgpu::BindGroupLayout) -> Self {
        let hidden_bytes = (cfg.hidden * 4) as u64;
        let gu_bytes = (cfg.gate_up_rows() * 4) as u64;
        let h_bytes = (cfg.expert_inner * 4) as u64;
        let logits_bytes = (cfg.n_experts * 4) as u64;

        let f32buf = |label: &str, nbytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: nbytes,
                usage,
                mapped_at_creation: false,
            })
        };

        let gi_buf = f32buf("moe_gi", hidden_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let logits_buf = f32buf("moe_router_logits", logits_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let logits_readback = f32buf(
            "moe_router_logits_readback",
            logits_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let gu_buf = f32buf("moe_gu_scratch", gu_bytes, wgpu::BufferUsages::STORAGE);
        let h_buf = f32buf("moe_h_scratch", h_bytes, wgpu::BufferUsages::STORAGE);
        let out_e_buf = f32buf("moe_out_e_scratch", hidden_bytes, wgpu::BufferUsages::STORAGE);

        let silu_push_buf = make_uniform_buf(
            device,
            "moe_silu_push",
            bytemuck::bytes_of(&SiluSplitPush {
                inner: cfg.expert_inner as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        let scale_push_buf = make_uniform_buf(
            device,
            "moe_scale_push",
            bytemuck::bytes_of(&ScalePush {
                hidden: cfg.hidden as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        let router_push_buf = make_uniform_buf(
            device,
            "moe_router_push",
            bytemuck::bytes_of(&RouterPush {
                hidden: cfg.hidden as u32,
                n_experts: cfg.n_experts as u32,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        let silu_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moe_silu_bg"),
            layout: silu_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gu_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: h_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: silu_push_buf.as_entire_binding() },
            ],
        });

        let _ = queue; // reserved for future pre-clear hooks
        Self {
            gi_buf,
            logits_buf,
            logits_readback,
            gu_buf,
            h_buf,
            out_e_buf,
            silu_push_buf,
            scale_push_buf,
            router_push_buf,
            silu_bg,
        }
    }
}

impl MoeFfnDispatch {
    /// Compile every pipeline used by the MoE FFN path. Call once per
    /// device at startup; reuse for every layer.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let gate_up_pipe = MoeMatvecPipeline::new(
            device.clone(),
            queue.clone(),
            QuantKind::Iq4Xs,
            include_str!("../shaders/matvec_iq4xs_moe.wgsl"),
        );
        let down_pipe = MoeMatvecPipeline::new(
            device.clone(),
            queue.clone(),
            QuantKind::Iq4Nl,
            include_str!("../shaders/matvec_iq4nl_moe.wgsl"),
        );

        // GEGLU split (gate||up f32 → h f32) — Gemma-4 MoE uses LLM_FFN_GELU.
        let silu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moe_geglu_mul_split"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/geglu_mul_split_f32.wgsl").into(),
            ),
        });
        let silu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moe_geglu_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_rw(1),
                bgl_uniform(2),
            ],
        });
        let silu_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_silu_pl"),
            bind_group_layouts: &[&silu_bgl],
            push_constant_ranges: &[],
        });
        let silu_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_silu_pipe"),
            layout: Some(&silu_pl),
            module: &silu_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Weighted accumulate (f32)
        let accum_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moe_weighted_accum"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/weighted_accum_f32.wgsl").into(),
            ),
        });
        let accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moe_accum_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_rw(1),
                bgl_uniform(2),
            ],
        });
        let accum_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_accum_pl"),
            bind_group_layouts: &[&accum_bgl],
            push_constant_ranges: &[],
        });
        let accum_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_accum_pipe"),
            layout: Some(&accum_pl),
            module: &accum_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Router input scale
        let scale_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moe_router_input_scale"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/router_input_scale.wgsl").into(),
            ),
        });
        let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moe_scale_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });
        let scale_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_scale_pl"),
            bind_group_layouts: &[&scale_bgl],
            push_constant_ranges: &[],
        });
        let scale_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_scale_pipe"),
            layout: Some(&scale_pl),
            module: &scale_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Router matvec (fp32, [128, hidden] @ [hidden] → [128])
        let router_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moe_router_matvec"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/router_matvec_f32.wgsl").into(),
            ),
        });
        let router_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moe_router_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });
        let router_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moe_router_pl"),
            bind_group_layouts: &[&router_bgl],
            push_constant_ranges: &[],
        });
        let router_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("moe_router_pipe"),
            layout: Some(&router_pl),
            module: &router_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scratch = MoeDispatchScratch::new(&device, &queue, &MoeFfnConfig::GEMMA4, &silu_bgl);

        Self {
            gate_up_pipe,
            down_pipe,
            silu_pipe,
            silu_bgl,
            accum_pipe,
            accum_bgl,
            scale_pipe,
            scale_bgl,
            router_pipe,
            router_bgl,
            device: device.clone(),
            queue: queue.clone(),
            scratch,
        }
    }

    /// One forward pass through the MoE FFN for a single token.
    ///
    /// Buffers required (caller manages lifetime; weights uploaded once at
    /// model load and reused across every token):
    ///   expert_x_gpu : `[hidden]` f32 — ffn_pre_norm_2(attn_out)
    ///   router_x_gpu : `[hidden]` f32 — rms_norm(attn_out) / sqrt(hidden)
    ///   gate_up_exps : raw IQ4_XS bytes for ALL n_experts experts
    ///   down_exps    : raw IQ4_NL bytes for ALL n_experts experts
    ///   router_w     : `[n_experts, hidden]` f32 router weights
    ///   router_scale : `[hidden]` f32 input scale (ffn_gate_inp.scale)
    ///   down_exps_scale : `[n_experts]` f32 — `ffn_down_exps.scale` (llama down_exps_s)
    ///   y_gpu        : `[hidden]` f32 output accumulator (caller pre-zeroed)
    ///
    /// Returns Ok(()) once all dispatches have been submitted AND the
    /// router readback has completed. (Router top-k requires a small CPU
    /// readback of 128 fp32 values — ~512 bytes — which is the only
    /// CPU↔GPU sync in the path. Weights NEVER round-trip.)
    pub fn forward(
        &self,
        cfg: &MoeFfnConfig,
        expert_x_gpu: &wgpu::Buffer,
        router_x_gpu: &wgpu::Buffer,
        gate_up_exps: &wgpu::Buffer,
        down_exps: &wgpu::Buffer,
        down_exps_scale: &[f32],
        router_w: &wgpu::Buffer,
        router_scale: &wgpu::Buffer,
        y_gpu: &wgpu::Buffer,
        meta: Option<&MoeDispatchMeta>,
    ) -> Result<(), String> {
        if cfg.top_k == 0 || cfg.top_k > cfg.n_experts {
            return Err(format!("invalid top_k {} for n_experts {}", cfg.top_k, cfg.n_experts));
        }
        if down_exps_scale.len() != cfg.n_experts {
            return Err(format!(
                "down_exps_scale len {} != n_experts {}",
                down_exps_scale.len(),
                cfg.n_experts
            ));
        }

        let device = &self.device;
        let queue = &self.queue;
        let hidden_u32 = cfg.hidden as u32;
        let inner_u32 = cfg.expert_inner as u32;
        let gate_up_rows_u32 = cfg.gate_up_rows() as u32;
        let scratch = &self.scratch;

        // ------------------------------------------------------------------
        // Stage 1–2: gi = x * scale; router_logits = router_w @ gi  (GPU)
        // ------------------------------------------------------------------
        let scale_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moe_scale_bg"),
            layout: &self.scale_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: router_x_gpu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: router_scale.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.gi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scratch.scale_push_buf.as_entire_binding() },
            ],
        });
        let router_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moe_router_bg"),
            layout: &self.router_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: router_w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.gi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.logits_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scratch.router_push_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_router_encoder"),
        });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("moe_scale_pass"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.scale_pipe);
            p.set_bind_group(0, Some(&scale_bg), &[]);
            p.dispatch_workgroups(((cfg.hidden + 255) / 256) as u32, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("moe_router_pass"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.router_pipe);
            p.set_bind_group(0, Some(&router_bg), &[]);
            p.dispatch_workgroups(cfg.n_experts as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &scratch.logits_buf,
            0,
            &scratch.logits_readback,
            0,
            (cfg.n_experts * 4) as u64,
        );
        queue.submit(Some(enc.finish()));

        // ------------------------------------------------------------------
        // Stage 3: CPU top-k softmax
        // ------------------------------------------------------------------
        let logits = read_back_f32(device, &scratch.logits_readback, cfg.n_experts)?;
        let (top_k_idx, top_k_w) = top_k_softmax(&logits, cfg.top_k);
        pamp::emit_expert_selection(meta, &top_k_idx, &top_k_w);
        if std::env::var("MOE_SLICE").ok().as_deref() == Some("1") {
            let layer = std::env::var("MOE_SLICE_LAYER").unwrap_or_else(|_| "0".into());
            eprintln!(
                "MOE_SLICE layer={layer} router_topk_idx={top_k_idx:?} weights={top_k_w:?} \
                 logits_max={:.4} logits_argmax={}",
                logits.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0),
            );
        }

        // ------------------------------------------------------------------
        // Stage 4: dispatch all top-k experts in ONE submit
        // ------------------------------------------------------------------
        let gu_buf = &scratch.gu_buf;
        let h_buf = &scratch.h_buf;
        let out_e_buf = &scratch.out_e_buf;
        let silu_bg = &scratch.silu_bg;

        let gu_bytes_per_expert = cfg.gate_up_bytes_per_expert();
        let down_bytes_per_expert = cfg.down_bytes_per_expert();
        debug_assert_eq!(gu_bytes_per_expert % 4, 0, "gate_up per-expert stride not u32-aligned");
        debug_assert_eq!(down_bytes_per_expert % 4, 0, "down per-expert stride not u32-aligned");

        let mut expert_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_experts_encoder"),
        });

        for j in 0..cfg.top_k {
            let e = top_k_idx[j];
            let w = top_k_w[j] * down_exps_scale[e];
            let gu_off_words = (e * gu_bytes_per_expert / 4) as u32;
            let down_off_bytes = (e * down_bytes_per_expert) as u32;

            let gu_push_buf = make_uniform_buf(
                device,
                "moe_gu_push",
                bytemuck::bytes_of(&MoeIq4xsPush {
                    k: hidden_u32,
                    n_rows_total: gate_up_rows_u32,
                    row_offset: 0,
                    expert_offset_words: gu_off_words,
                }),
            );
            let gu_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("moe_gu_bg"),
                layout: &self.gate_up_pipe.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gate_up_exps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: expert_x_gpu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gu_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.gate_up_pipe.lut_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gu_push_buf.as_entire_binding() },
                ],
            });

            let down_push_buf = make_uniform_buf(
                device,
                "moe_down_push",
                bytemuck::bytes_of(&MoeIq4nlPush {
                    k: inner_u32,
                    n_rows_total: hidden_u32,
                    row_offset: 0,
                    expert_offset_bytes: down_off_bytes,
                }),
            );
            let down_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("moe_down_bg"),
                layout: &self.down_pipe.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: down_exps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: h_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out_e_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.down_pipe.lut_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: down_push_buf.as_entire_binding() },
                ],
            });

            let accum_push_buf = make_uniform_buf(
                device,
                "moe_accum_push",
                bytemuck::bytes_of(&AccumPush {
                    weight: w,
                    len: hidden_u32,
                    _pad0: 0,
                    _pad1: 0,
                }),
            );
            let accum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("moe_accum_bg"),
                layout: &self.accum_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: out_e_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: y_gpu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: accum_push_buf.as_entire_binding() },
                ],
            });

            {
                let mut p = expert_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("moe_gate_up_pass"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.gate_up_pipe.pipeline);
                p.set_bind_group(0, Some(&gu_bg), &[]);
                p.dispatch_workgroups(gate_up_rows_u32, 1, 1);
            }
            {
                let mut p = expert_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("moe_silu_pass"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.silu_pipe);
                p.set_bind_group(0, Some(silu_bg), &[]);
                p.dispatch_workgroups(((cfg.expert_inner + 255) / 256) as u32, 1, 1);
            }
            {
                let mut p = expert_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("moe_down_pass"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.down_pipe.pipeline);
                p.set_bind_group(0, Some(&down_bg), &[]);
                p.dispatch_workgroups(hidden_u32, 1, 1);
            }
            {
                let mut p = expert_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("moe_accum_pass"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.accum_pipe);
                p.set_bind_group(0, Some(&accum_bg), &[]);
                p.dispatch_workgroups(((cfg.hidden + 255) / 256) as u32, 1, 1);
            }
        }
        queue.submit(Some(expert_enc.finish()));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/// Stable softmax + partial-sort top-k of `logits`.
///
/// Returns `(top_k_indices, top_k_softmax_weights)`. Weights are
/// normalized to sum to 1 across the top-k subset (the standard MoE
/// "renormalize after dropping non-selected" convention). If your model
/// expects softmax-over-all-experts then drop those weights, use
/// `softmax_full_then_pick` instead — but per Gemma-4 + llama.cpp
/// `llm_build_moe_ffn`, renormalization is the right call.
pub fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    let k = k.min(n);

    // partial sort: find top-k indices
    let mut idx: Vec<usize> = (0..n).collect();
    // Use unstable sort by reverse magnitude — `select_nth_unstable_by` would
    // also work but std::sort is fine for n=128.
    idx.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);

    // stable softmax over the selected k logits only
    let mut max_l = f32::NEG_INFINITY;
    for &i in &idx {
        if logits[i] > max_l { max_l = logits[i]; }
    }
    let mut weights: Vec<f32> = idx.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() { *w /= sum; }
    } else {
        // Pathological all-equal case — uniform.
        let u = 1.0 / k as f32;
        for w in weights.iter_mut() { *w = u; }
    }
    (idx, weights)
}

/// Synchronous f32 buffer readback.
fn read_back_f32(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<f32>, String> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel: {e}"))?
        .map_err(|e| format!("readback map: {e:?}"))?;

    let data = slice.get_mapped_range();
    let mut out = vec![0f32; n];
    let bytes = &data[..n * 4];
    out.copy_from_slice(bytemuck::cast_slice(bytes));
    drop(data);
    readback.unmap();
    Ok(out)
}

fn make_uniform_buf(device: &wgpu::Device, label: &'static str, bytes: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

// ---------------------------------------------------------------------------
// BGL helpers
// ---------------------------------------------------------------------------

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-only, exercise the router math)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_softmax_picks_largest() {
        let mut logits = vec![0.0f32; 16];
        logits[3] = 5.0;
        logits[7] = 4.0;
        logits[10] = 4.5;
        let (idx, w) = top_k_softmax(&logits, 3);
        assert_eq!(idx[0], 3);
        assert!(idx.contains(&7));
        assert!(idx.contains(&10));
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_softmax_uniform_weights_when_logits_equal() {
        let logits = vec![1.0f32; 8];
        let (_idx, w) = top_k_softmax(&logits, 4);
        for x in &w {
            assert!((x - 0.25).abs() < 1e-5);
        }
    }

    /// Parse + validate every WGSL shader the dispatch path uses. Catches
    /// shader regressions at `cargo test` time without needing a real GPU.
    #[test]
    fn shaders_parse_under_naga() {
        let sources = [
            ("matvec_iq4xs_moe", include_str!("../shaders/matvec_iq4xs_moe.wgsl")),
            ("matvec_iq4nl_moe", include_str!("../shaders/matvec_iq4nl_moe.wgsl")),
            ("silu_mul_split_f32", include_str!("../shaders/silu_mul_split_f32.wgsl")),
            ("weighted_accum_f32", include_str!("../shaders/weighted_accum_f32.wgsl")),
            ("router_input_scale", include_str!("../shaders/router_input_scale.wgsl")),
            ("router_matvec_f32", include_str!("../shaders/router_matvec_f32.wgsl")),
        ];
        for (name, src) in sources {
            let module = wgpu::naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("naga parse failed for {name}: {e:?}"));
            let mut validator = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("naga validate failed for {name}: {e:?}"));
        }
    }

    #[test]
    fn config_byte_math() {
        let c = MoeFfnConfig::GEMMA4;
        assert_eq!(c.gate_up_rows(), 1408);
        // 2816/256=11 blocks per row × 136 bytes = 1496 bytes per row
        // × 1408 rows = 2,106,368 bytes per expert.
        assert_eq!(c.gate_up_bytes_per_expert(), 1408 * 11 * 136);
        // 704/32=22 blocks per row × 18 bytes = 396 bytes per row
        // × 2816 rows = 1,115,136 bytes per expert.
        assert_eq!(c.down_bytes_per_expert(), 2816 * 22 * 18);
    }
}
