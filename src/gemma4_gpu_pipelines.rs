//! Compiled compute pipelines for the GPU-resident Gemma-4 forward pass.
//!
//! Every shader binding layout, push struct, and pipeline declaration is
//! collected here so the runner can dispatch without inspecting WGSL by
//! hand. All pipelines are Pascal-safe (f32 only, no f16, no subgroup
//! ops).
//!
//! Shaders consumed by this module (every one is GPU-resident):
//!   * embed_lookup_scaled.wgsl  — token-id → hidden state row, sqrt-scaled.
//!   * rmsnorm_f32.wgsl          — Gemma RMSNorm with (1+w) flag.
//!   * rmsnorm_weightless_f32.wgsl — V-projection RMSNorm.
//!   * rope_v2.wgsl              — stride-loop RoPE for any head_dim.
//!   * matvec_iq4xs_correct.wgsl — Q/K/V/O/gate/up matvec.
//!   * matvec_iq4nl_correct.wgsl — down-projection matvec.
//!   * matvec_f32_rowmajor.wgsl  — fp32 LM head matvec.
//!   * silu_mul_split_f32.wgsl   — SwiGLU on packed gate||up (MoE path).
//!   * weighted_accum_f32.wgsl   — y += w * x for residual / MoE accum.
//!   * add.wgsl                  — y = a + b for residual sums.
//!   * router_input_scale.wgsl   — gi = x * router_scale (MoE path).
//!   * router_matvec_f32.wgsl    — fp32 router_w @ gi (MoE path).
//!
//! Note: `Iq4MatvecPipeline`, `MoeFfnDispatch`, and `F32MatvecPipeline`
//! continue to live in their own modules. This file only owns the small
//! activation kernels (norms, rope, residual, embedding lookup) that the
//! existing CPU-bouncing forward pass dropped on the floor.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

// ────────────────────────────────────────────────────────────────────────────
// Push structs (must match the WGSL declarations exactly — same order, same
// padding). Each has tests in `tests::push_struct_byte_layouts` to catch
// drift.
// ────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct EmbedLookupPush {
    pub token_id: u32,
    pub hidden_dim: u32,
    pub embed_scale_flag: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct RmsNormPush {
    pub hidden_dim: u32,
    pub plus_one_flag: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub eps: f32,
    pub _pad2: f32,
    pub _pad3: f32,
    pub _pad4: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct RmsNormWeightlessPush {
    pub hidden_dim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub eps: f32,
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct RopePush {
    pub head_dim: u32,
    pub pos: u32,
    pub n_heads: u32,
    pub _pad: u32,
    pub rope_base: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

// `add.wgsl` already has its own Params struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AddPush {
    pub n_elements: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// `weighted_accum_f32.wgsl` uses (weight: f32, len: u32, pad, pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct WeightedAccumPush {
    pub weight: f32,
    pub len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AttnPush {
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub heads_per_kv: u32,

    pub seq_len: u32,
    pub window_start: u32,
    pub use_swa: u32,
    pub _pad0: u32,

    pub scale: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct KvWritePush {
    pub pos: u32,
    pub kv_dim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SiluMulPush {
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct LogitSoftcapPush {
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub cap: f32,
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
}

// ────────────────────────────────────────────────────────────────────────────
// Pipelines bundle
// ────────────────────────────────────────────────────────────────────────────

/// One compute pipeline plus its bind group layout. Reusable across many
/// dispatches with different bind groups.
pub struct ComputePipe {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

/// All small activation-only pipelines used by the GPU-resident Gemma-4
/// forward pass. Hold one instance per device.
pub struct Gemma4GpuPipelines {
    pub embed_lookup: ComputePipe,
    pub rmsnorm: ComputePipe,
    pub rmsnorm_weightless: ComputePipe,
    pub rope: ComputePipe,
    pub add: ComputePipe,
    pub weighted_accum: ComputePipe,
    pub attn: ComputePipe,
    pub kv_write: ComputePipe,
    pub silu_mul: ComputePipe,
    pub logit_softcap: ComputePipe,
}

impl Gemma4GpuPipelines {
    pub fn new(device: &Arc<wgpu::Device>) -> Self {
        Self {
            embed_lookup: build_pipe(
                device,
                "embed_lookup_scaled",
                include_str!("../shaders/embed_lookup_scaled.wgsl"),
                &[
                    bgl_storage_ro(0),  // E (embedding table)
                    bgl_storage_rw(1),  // H (hidden state out)
                    bgl_uniform(2),     // push
                ],
            ),
            rmsnorm: build_pipe(
                device,
                "rmsnorm_f32",
                include_str!("../shaders/rmsnorm_f32.wgsl"),
                &[
                    bgl_storage_ro(0),  // x
                    bgl_storage_ro(1),  // weight
                    bgl_storage_rw(2),  // y
                    bgl_uniform(3),     // push
                ],
            ),
            rmsnorm_weightless: build_pipe(
                device,
                "rmsnorm_weightless_f32",
                include_str!("../shaders/rmsnorm_weightless_f32.wgsl"),
                &[
                    bgl_storage_ro(0),  // x
                    bgl_storage_rw(1),  // y
                    bgl_uniform(2),     // push
                ],
            ),
            rope: build_pipe(
                device,
                "rope_v2",
                include_str!("../shaders/rope_v2.wgsl"),
                &[
                    bgl_storage_rw(0),  // qk (in-place)
                    bgl_uniform(1),     // push
                ],
            ),
            add: build_pipe(
                device,
                "add",
                include_str!("../shaders/add.wgsl"),
                &[
                    bgl_storage_ro(0),  // a
                    bgl_storage_ro(1),  // b
                    bgl_storage_rw(2),  // output
                    bgl_uniform(3),     // params
                ],
            ),
            weighted_accum: build_pipe(
                device,
                "weighted_accum_f32",
                include_str!("../shaders/weighted_accum_f32.wgsl"),
                &[
                    bgl_storage_ro(0),  // src
                    bgl_storage_rw(1),  // y
                    bgl_uniform(2),     // push
                ],
            ),
            attn: build_pipe(
                device,
                "attn_one_head_gemma",
                include_str!("../shaders/attn_one_head_gemma.wgsl"),
                &[
                    bgl_storage_ro(0),  // q
                    bgl_storage_ro(1),  // k_cache
                    bgl_storage_ro(2),  // v_cache
                    bgl_storage_rw(3),  // out
                    bgl_uniform(4),     // push
                ],
            ),
            kv_write: build_pipe(
                device,
                "kv_cache_write",
                include_str!("../shaders/kv_cache_write.wgsl"),
                &[
                    bgl_storage_ro(0),  // k_in
                    bgl_storage_ro(1),  // v_in
                    bgl_storage_rw(2),  // k_cache
                    bgl_storage_rw(3),  // v_cache
                    bgl_uniform(4),     // push
                ],
            ),
            silu_mul: build_pipe(
                device,
                "silu_mul_separate_f32",
                include_str!("../shaders/silu_mul_separate_f32.wgsl"),
                &[
                    bgl_storage_ro(0),  // gate
                    bgl_storage_ro(1),  // up
                    bgl_storage_rw(2),  // y
                    bgl_uniform(3),     // push
                ],
            ),
            logit_softcap: build_pipe(
                device,
                "logit_softcap_f32",
                include_str!("../shaders/logit_softcap_f32.wgsl"),
                &[
                    bgl_storage_ro(0),  // x
                    bgl_storage_rw(1),  // y
                    bgl_uniform(2),     // push
                ],
            ),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn build_pipe(
    device: &Arc<wgpu::Device>,
    label: &'static str,
    src: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> ComputePipe {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    ComputePipe { pipeline, bgl }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// All push structs must be 16-byte aligned multiples to be safely
    /// usable as WGSL uniforms.
    #[test]
    fn push_struct_byte_layouts() {
        assert_eq!(std::mem::size_of::<EmbedLookupPush>(), 16);
        assert_eq!(std::mem::size_of::<RmsNormPush>(), 32);
        assert_eq!(std::mem::size_of::<RmsNormWeightlessPush>(), 32);
        assert_eq!(std::mem::size_of::<RopePush>(), 32);
        assert_eq!(std::mem::size_of::<AddPush>(), 16);
        assert_eq!(std::mem::size_of::<WeightedAccumPush>(), 16);
        assert_eq!(std::mem::size_of::<AttnPush>(), 48);
        assert_eq!(std::mem::size_of::<KvWritePush>(), 16);
        assert_eq!(std::mem::size_of::<SiluMulPush>(), 16);
        assert_eq!(std::mem::size_of::<LogitSoftcapPush>(), 32);
    }

    /// Parse + validate every WGSL shader the pipeline bundle uses. Catches
    /// shader regressions at `cargo test` time without needing a real GPU.
    #[test]
    fn shaders_parse_under_naga() {
        let sources = [
            ("embed_lookup_scaled", include_str!("../shaders/embed_lookup_scaled.wgsl")),
            ("rmsnorm_f32", include_str!("../shaders/rmsnorm_f32.wgsl")),
            ("rmsnorm_weightless_f32", include_str!("../shaders/rmsnorm_weightless_f32.wgsl")),
            ("rope_v2", include_str!("../shaders/rope_v2.wgsl")),
            ("add", include_str!("../shaders/add.wgsl")),
            ("weighted_accum_f32", include_str!("../shaders/weighted_accum_f32.wgsl")),
            ("attn_one_head_gemma", include_str!("../shaders/attn_one_head_gemma.wgsl")),
            ("kv_cache_write", include_str!("../shaders/kv_cache_write.wgsl")),
            ("silu_mul_separate_f32", include_str!("../shaders/silu_mul_separate_f32.wgsl")),
            ("logit_softcap_f32", include_str!("../shaders/logit_softcap_f32.wgsl")),
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
}
