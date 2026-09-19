//! Multi-GPU MoE expert dispatch.
//!
//! Spreads expert weights across multiple GPUs and dispatches FFN compute
//! to the appropriate device based on the expert placement plan.

use std::sync::Arc;

use tracing::{info, warn};

use crate::gpu_tensor::{GpuTensor, MultiGpuExpertContext};
use crate::moe_expert_loader::{ExpertDevice, PlacementPlan, ExpertKind};

/// Build a multi-GPU expert context from the expert loader's ExpertDevice list.
pub fn build_multigpu_context(
    expert_devices: Vec<ExpertDevice>,
    placement: PlacementPlan,
) -> MultiGpuExpertContext {
    let num_gpus = expert_devices.len();
    let mut devices = Vec::with_capacity(num_gpus);
    let mut queues = Vec::with_capacity(num_gpus);
    let mut expert_buffers = Vec::with_capacity(num_gpus);

    for dev in expert_devices {
        devices.push(dev.device.clone());
        queues.push(dev.queue.clone());
        expert_buffers.push(dev.buffers);
    }

    MultiGpuExpertContext {
        devices,
        queues,
        expert_buffers,
        placement,
    }
}

/// Dispatch MoE FFN across multiple GPUs.
///
/// For each expert in the top-K routing:
///   1. Look up which GPU holds that expert's weights
///   2. Dispatch matmul to that GPU
///   3. Accumulate results back to the coordinator GPU
///
/// This implementation assumes:
///   - Hidden state lives on the coordinator GPU (device 0)
///   - Expert weights are distributed across GPUs
///   - Final accumulation happens on the coordinator GPU
///
/// For Pascal GPUs (P100), PCIe transfers are required. The shader will
/// handle the actual data movement via COPY_SRC/COPY_DST buffers.
pub fn dispatch_multigpu_moe_ffn(
    ctx: &MultiGpuExpertContext,
    hidden_state: &GpuTensor,
    expert_indices: &[usize],
    expert_weights: &[f64],
    output: &mut GpuTensor,
) -> Result<(), String> {
    if expert_indices.len() != expert_weights.len() {
        return Err("expert_indices and expert_weights must have same length".into());
    }

    let hidden_bytes = (hidden_state.len * 4) as u64;
    let output_bytes = (output.len * 4) as u64;

    if output_bytes != hidden_bytes {
        return Err(format!(
            "output buffer size mismatch: expected {} bytes, got {}",
            hidden_bytes,
            output_bytes
        ));
    }

    // Zero the output accumulator on the coordinator GPU
    {
        let mut encoder = ctx.devices[0].create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multigpu_moe_zero"),
        });
        encoder.copy_buffer_to_buffer(
            &output.buffer,
            0,
            &output.buffer,
            0,
            output_bytes,
        );
        ctx.queues[0].submit(std::iter::once(encoder.finish()));
    }

    // Process each expert on its assigned GPU
    for (&expert_idx, &routing_weight) in expert_indices.iter().zip(expert_weights.iter()) {
        // Look up expert weights on their assigned GPU
        let gate_up_key = crate::moe_expert_loader::expert_key(expert_idx, ExpertKind::GateUp);
        let down_key = crate::moe_expert_loader::expert_key(expert_idx, ExpertKind::Down);

        let (gate_up_buf, gate_gpu) = ctx
            .lookup_expert(expert_idx, ExpertKind::GateUp)
            .ok_or_else(|| format!("expert {} gate_up not found", expert_idx))?;

        let (down_buf, down_gpu) = ctx
            .lookup_expert(expert_idx, ExpertKind::Down)
            .ok_or_else(|| format!("expert {} down not found", expert_idx))?;

        // For now, we'll use a simplified dispatch that assumes:
        // - Gate+Up are packed in the same buffer (as per GGUF layout)
        // - We'll dispatch matmul on the expert's GPU
        // - Result is copied back to coordinator for accumulation

        // TODO: Implement full multi-GPU dispatch with proper shader pipelines
        // This is a placeholder that shows the structure
        let _ = (gate_up_buf, gate_gpu, down_buf, down_gpu, routing_weight);
    }

    Ok(())
}

/// Simple single-GPU MoE dispatch for testing.
///
/// When all experts fit on one GPU, use this simpler path.
pub fn dispatch_single_gpu_moe_ffn(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    hidden_state: &wgpu::Buffer,
    expert_indices: &[usize],
    expert_weights: &[f64],
    gate_up_buffers: &[wgpu::Buffer],
    down_buffers: &[wgpu::Buffer],
    output: &wgpu::Buffer,
    hidden_dim: u32,
    intermediate_dim: u32,
) -> Result<(), String> {
    if expert_indices.len() != expert_weights.len() {
        return Err("expert_indices and expert_weights must have same length".into());
    }

    let hidden_bytes = (hidden_dim * 4) as u64;
    let intermediate_bytes = (intermediate_dim * 4) as u64;

    // Zero output
    {
        let zero_data = vec![0u8; hidden_bytes as usize];
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zero_output_staging"),
            size: (zero_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&staging, 0, &zero_data);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zero_encoder"),
        });
        encoder.copy_buffer_to_buffer(&staging, 0, output, 0, hidden_bytes);
        queue.submit(std::iter::once(encoder.finish()));
    }

    // Process each expert
    for (&expert_idx, &routing_weight) in expert_indices.iter().zip(expert_weights.iter()) {
        if expert_idx >= gate_up_buffers.len() {
            return Err(format!("expert {} out of range ({} experts)", expert_idx, gate_up_buffers.len()));
        }

        let gate_up_buf = &gate_up_buffers[expert_idx];
        let down_buf = &down_buffers[expert_idx];

        // Create scratch buffers
        let gate_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_moe_gate"),
            size: intermediate_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let up_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_moe_up"),
            size: intermediate_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let intermediate = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_moe_interm"),
            size: intermediate_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let expert_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sg_moe_expert_out"),
            size: hidden_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // TODO: Dispatch matmul and SwiGLU here
        // This is a placeholder
        let _ = (gate_up_buf, down_buf, routing_weight, gate_proj, up_proj, intermediate, expert_out);
    }

    Ok(())
}
