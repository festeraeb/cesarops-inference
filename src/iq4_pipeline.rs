//! Raw-quantized matvec pipelines for IQ4_XS and IQ4_NL.
//!
//! The WGSL shaders in `shaders/matvec_iq4{xs,nl}_correct.wgsl` operate
//! directly on the GGUF byte stream — no CPU dequantization, no f32
//! intermediate. This module wraps them as reusable wgpu pipelines so the
//! transformer forward pass can call them on raw weight buffers uploaded
//! once at model load.
//!
//! Bindings (matches both shaders):
//!   @binding(0) storage<read>      W    : array<u32>   raw quant bytes as u32
//!   @binding(1) storage<read>      X    : array<f32>   input vector
//!   @binding(2) storage<read_write> Y   : array<f32>   output vector
//!   @binding(3) uniform            lut  : array<vec4<f32>, 4>
//!   @binding(4) uniform            push : (K, N_rows_total, row_offset, _pad)
//!
//! The LUT is a constant 16-value codebook (`KVALUES_IQ4`). It is uploaded
//! once and reused across every IQ4_XS / IQ4_NL dispatch.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

/// Codebook used for both IQ4_XS and IQ4_NL. Mirrors the canonical
/// `kvalues_iq4nl` table from llama.cpp.
pub const KVALUES_IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MatvecPush {
    pub k: u32,
    pub n_rows_total: u32,
    pub row_offset: u32,
    pub _pad: u32,
}

/// Quant kind selector — picks shader source and validates byte budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Iq4Xs,
    Iq4Nl,
}

impl QuantKind {
    /// Bytes per row at this quant + given K. Matches the byte layout used
    /// by `loader::compute_tensor_size`.
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            QuantKind::Iq4Xs => (k + 255) / 256 * 136,
            QuantKind::Iq4Nl => (k + 31) / 32 * 18,
        }
    }
}

/// Compiled pipeline for one quant kind. Reusable across many tensors.
pub struct Iq4MatvecPipeline {
    pub kind: QuantKind,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// LUT buffer — populated once with [`KVALUES_IQ4`], reused forever.
    pub lut_buffer: wgpu::Buffer,
}

impl Iq4MatvecPipeline {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        kind: QuantKind,
        shader_src: &str,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(match kind {
                QuantKind::Iq4Xs => "iq4xs_matvec",
                QuantKind::Iq4Nl => "iq4nl_matvec",
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iq4_matvec_bgl"),
            entries: &[
                bgl_storage_ro(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
                bgl_uniform(4),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iq4_matvec_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("iq4_matvec_pipeline"),
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
            label: Some("iq4_lut"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut_packed));

        Self { kind, device, queue, pipeline, bind_group_layout: bgl, lut_buffer }
    }

    /// Per-call push uniform buffer. Caller owns it.
    pub fn make_push_buffer(&self, push: MatvecPush) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iq4_push"),
            size: std::mem::size_of::<MatvecPush>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&push));
        buf
    }

    /// Build a per-dispatch bind group. The caller already has W (the
    /// raw-quant tensor buffer, uploaded once at model load), X, Y, and
    /// a fresh push buffer.
    pub fn make_bind_group(
        &self,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        push: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iq4_matvec_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.lut_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
            ],
        })
    }

    /// Encode a single matvec dispatch into the supplied encoder.
    /// Caller is responsible for creating `y` and reading it back.
    /// `n_rows` here is how many output rows to launch; the shader still
    /// uses `push.row_offset` to know where they start.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        push: &wgpu::Buffer,
        n_rows: u32,
    ) {
        let bg = self.make_bind_group(w, x, y, push);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("iq4_matvec_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_rows, 1, 1);
    }
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

/// Default WGSL shader sources. Build-time embedded so binaries don't need
/// to find the `shaders/` directory at runtime.
pub fn iq4xs_shader_src() -> &'static str {
    include_str!("../shaders/matvec_iq4xs_correct.wgsl")
}
pub fn iq4nl_shader_src() -> &'static str {
    include_str!("../shaders/matvec_iq4nl_correct.wgsl")
}
