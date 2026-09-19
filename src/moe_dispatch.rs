// src/moe_dispatch.rs
// MoE Expert FFN Dispatch — wgpu compute pipeline
// Gemma-4-26B-MoE: 64 experts, top-2 routing, IQ4_XS quantized weights

use wgpu::util::DeviceExt;

pub const GEMMA_NUM_EXPERTS: usize = 64;
pub const GEMMA_TOPK: usize = 2;

#[derive(Debug)]
pub enum MoeError {
    InvalidTopK,
    InvalidExpertIndex(usize),
    BufferTooSmall,
}

#[derive(Clone)]
pub struct MoeConfig {
    pub hidden_size: u32,       // 2048
    pub intermediate_size: u32, // 16384

    // pipelines
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub silu_mul_pipeline: wgpu::ComputePipeline,
    pub weighted_accum_pipeline: wgpu::ComputePipeline,

    // bindgroup layouts
    pub matmul_bgl: wgpu::BindGroupLayout,
    pub silu_mul_bgl: wgpu::BindGroupLayout,
    pub weighted_accum_bgl: wgpu::BindGroupLayout,

    // quantization mode
    pub weights_are_quantized: bool,

    // IQ4_XS: bytes per 32 weights
    pub iq4_xs_block_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WeightedAccumUniform {
    weight: f32,
    len: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Dispatches: C[M,N] = A[M,K] x B[K,N]
#[allow(clippy::too_many_arguments)]
fn dispatch_matmul(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    m: u32,
    n: u32,
    _k: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("matmul_pass"),
        timestamp_writes: None,
    });

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);

    let wg_x = n.div_ceil(16);
    let wg_y = m.div_ceil(16);

    pass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn create_matmul_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    a_buffer: &wgpu::Buffer,
    a_offset: u64,
    a_size: u64,
    b_buffer: &wgpu::Buffer,
    b_offset: u64,
    b_size: u64,
    out_buffer: &wgpu::Buffer,
    out_offset: u64,
    out_size: u64,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: a_buffer,
                    offset: a_offset,
                    size: Some(std::num::NonZeroU64::new(a_size).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: b_buffer,
                    offset: b_offset,
                    size: Some(std::num::NonZeroU64::new(b_size).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: out_buffer,
                    offset: out_offset,
                    size: Some(std::num::NonZeroU64::new(out_size).unwrap()),
                }),
            },
        ],
    })
}

/// Computes: out[i] = SiLU(gate[i]) * up[i]
fn dispatch_silu_mul(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    len: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("silu_mul"),
        timestamp_writes: None,
    });

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);

    let wg = len.div_ceil(256);
    pass.dispatch_workgroups(wg, 1, 1);
}

/// Computes: accum += weight * src
fn dispatch_weighted_accum(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    len: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("weighted_accum"),
        timestamp_writes: None,
    });

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);

    let wg = len.div_ceil(256);
    pass.dispatch_workgroups(wg, 1, 1);
}

pub fn dispatch_moe_ffn(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    hidden_state: &wgpu::Buffer,
    expert_indices: &[usize],
    expert_weights: &[f64],
    expert_gate_weights: &wgpu::Buffer,
    expert_up_weights: &wgpu::Buffer,
    expert_down_weights: &wgpu::Buffer,
    output: &wgpu::Buffer,
    config: &MoeConfig,
) -> Result<(), MoeError> {
    if expert_indices.len() != expert_weights.len() {
        return Err(MoeError::InvalidTopK);
    }

    let hidden = config.hidden_size;
    let interm = config.intermediate_size;

    let f16_size = 2u64;

    let gate_matrix_elems = (interm as u64) * (hidden as u64);
    let down_matrix_elems = (hidden as u64) * (interm as u64);

    let gate_matrix_bytes = if config.weights_are_quantized {
        (gate_matrix_elems / 32) * config.iq4_xs_block_bytes
    } else {
        gate_matrix_elems * f16_size
    };

    let down_matrix_bytes = if config.weights_are_quantized {
        (down_matrix_elems / 32) * config.iq4_xs_block_bytes
    } else {
        down_matrix_elems * f16_size
    };

    let interm_bytes = (interm as u64) * f16_size;
    let hidden_bytes = (hidden as u64) * f16_size;

    // Scratch buffers
    let gate_proj = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_gate_proj"),
        size: interm_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let up_proj = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_up_proj"),
        size: interm_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let intermediate = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_intermediate"),
        size: interm_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let expert_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_expert_out"),
        size: hidden_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Zero final output accumulator
    {
        let zero_data = vec![0u8; hidden_bytes as usize];
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zero_output"),
            contents: &zero_data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zero_output_encoder"),
        });

        encoder.copy_buffer_to_buffer(&staging, 0, output, 0, hidden_bytes);
        queue.submit(Some(encoder.finish()));
    }

    // Main expert loop
    for (&expert_idx, &routing_weight) in
        expert_indices.iter().zip(expert_weights.iter())
    {
        if expert_idx >= GEMMA_NUM_EXPERTS {
            return Err(MoeError::InvalidExpertIndex(expert_idx));
        }

        let gate_offset = (expert_idx as u64) * gate_matrix_bytes;
        let up_offset = (expert_idx as u64) * gate_matrix_bytes;
        let down_offset = (expert_idx as u64) * down_matrix_bytes;

        // gate_proj = gate @ hidden
        let gate_bg = create_matmul_bind_group(
            device, &config.matmul_bgl,
            expert_gate_weights, gate_offset, gate_matrix_bytes,
            hidden_state, 0, hidden_bytes,
            &gate_proj, 0, interm_bytes,
        );

        // up_proj = up @ hidden
        let up_bg = create_matmul_bind_group(
            device, &config.matmul_bgl,
            expert_up_weights, up_offset, gate_matrix_bytes,
            hidden_state, 0, hidden_bytes,
            &up_proj, 0, interm_bytes,
        );

        // intermediate = SiLU(gate_proj) * up_proj
        let silu_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu_mul_bg"),
            layout: &config.silu_mul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_proj.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up_proj.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: intermediate.as_entire_binding() },
            ],
        });

        // expert_out = down @ intermediate
        let down_bg = create_matmul_bind_group(
            device, &config.matmul_bgl,
            expert_down_weights, down_offset, down_matrix_bytes,
            &intermediate, 0, interm_bytes,
            &expert_out, 0, hidden_bytes,
        );

        // output += routing_weight * expert_out
        let accum_uniform = WeightedAccumUniform {
            weight: routing_weight as f32,
            len: hidden,
            _pad0: 0,
            _pad1: 0,
        };

        let accum_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("accum_uniform"),
            contents: bytemuck::bytes_of(&accum_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let accum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("weighted_accum_bg"),
            layout: &config.weighted_accum_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: expert_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accum_uniform_buf.as_entire_binding() },
            ],
        });

        // Encode all passes for this expert
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_expert_encoder"),
        });

        dispatch_matmul(&mut encoder, &config.matmul_pipeline, &gate_bg, interm, 1, hidden);
        dispatch_matmul(&mut encoder, &config.matmul_pipeline, &up_bg, interm, 1, hidden);
        dispatch_silu_mul(&mut encoder, &config.silu_mul_pipeline, &silu_bg, interm);
        dispatch_matmul(&mut encoder, &config.matmul_pipeline, &down_bg, hidden, 1, interm);
        dispatch_weighted_accum(&mut encoder, &config.weighted_accum_pipeline, &accum_bg, hidden);

        queue.submit(Some(encoder.finish()));
    }

    Ok(())
}
