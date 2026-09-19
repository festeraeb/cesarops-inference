//! Multi-GPU MoE expert dispatcher.
//!
//! Uses the placement plan from `moe_expert_loader` to dispatch expert FFN
//! compute to the correct GPU. Activation tensors are kept GPU-resident via
//! `GpuTensor` to avoid CPU↔GPU ping-ponging.

use std::sync::Arc;

use tracing::{info, warn};

use crate::gpu_tensor::{GpuTensor, MultiGpuExpertContext};
use crate::moe_expert_loader::{ExpertKind, PlacementPlan};
use crate::forward_pass::ModelConfig;

/// Multi-GPU MoE dispatcher.
///
/// Holds per-GPU expert buffers and dispatches expert FFN compute to the
/// correct device based on the placement plan.
pub struct MultiGpuMoEDispatcher {
    pub context: MultiGpuExpertContext,
    pub placement: PlacementPlan,
}

impl MultiGpuMoEDispatcher {
    /// Create a new dispatcher from the expert context.
    pub fn new(context: MultiGpuExpertContext) -> Self {
        let placement = context.placement.clone();
        Self { context, placement }
    }

    /// Look up which GPU holds a given expert tensor.
    pub fn lookup_expert(&self, layer: usize, kind: ExpertKind) -> Option<(&wgpu::Buffer, usize)> {
        self.context.lookup_expert(layer, kind)
    }

    /// Dispatch expert FFN for a single token across multiple GPUs.
    ///
    /// `hidden_state` is GPU-resident (GpuTensor). Expert weights are on
    /// their assigned GPUs. The result is written back to `hidden_state`.
    pub fn dispatch_expert_ffn(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden_state: &GpuTensor,
        expert_indices: &[usize],
        expert_weights: &[f32],
        config: &ModelConfig,
    ) -> Result<(), String> {
        if expert_indices.is_empty() || expert_indices.len() != expert_weights.len() {
            return Err("expert indices and weights must match".to_string());
        }

        let hidden_bytes = (config.hidden_dim * 4) as u64;
        let interm_bytes = (config.intermediate_dim * 4) as u64;

        // Scratch buffers on each GPU (we'll use device 0 as coordinator)
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
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zero_output_staging"),
                size: (zero_data.len() * 4) as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            
            queue.write_buffer(&staging, 0, &zero_data);
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zero_output_encoder"),
            });

            encoder.copy_buffer_to_buffer(&staging, 0, &hidden_state.buffer, 0, hidden_bytes);
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Main expert loop - dispatch to correct GPU for each expert
        for (&expert_idx, &routing_weight) in
            expert_indices.iter().zip(expert_weights.iter())
        {
            // Look up which GPU holds this expert's weights
            let (gate_buf, gate_gpu) = self.lookup_expert(expert_idx, ExpertKind::GateUp)
                .ok_or_else(|| format!("missing gate_up for expert {}", expert_idx))?;
            let (down_buf, down_gpu) = self.lookup_expert(expert_idx, ExpertKind::Down)
                .ok_or_else(|| format!("missing down for expert {}", expert_idx))?;

            // For now, assume all experts are on the same GPU (device 0)
            // Cross-GPU dispatch will be implemented in a follow-up
            if gate_gpu != down_gpu {
                return Err(format!(
                    "expert {} has gate on GPU {} but down on GPU {}",
                    expert_idx, gate_gpu, down_gpu
                ));
            }

            let gpu_idx = gate_gpu;
            let dev = self.context.devices.get(gpu_idx)
                .ok_or_else(|| format!("device {} not found", gpu_idx))?;
            let que = self.context.queues.get(gpu_idx)
                .ok_or_else(|| format!("queue {} not found", gpu_idx))?;

            // Calculate offsets into the expert buffer
            // Gate+up are packed: [expert_0_gate, expert_0_up, expert_1_gate, expert_1_up, ...]
            let expert_stride = (config.intermediate_dim * config.hidden_dim * 4) as u64;
            let gate_offset = (expert_idx as u64) * expert_stride * 2;
            let up_offset = gate_offset + expert_stride;

            // Gate projection: hidden_state → gate_proj
            self.dispatch_matmul(
                dev, que,
                &hidden_state.buffer, 0,
                gate_buf, gate_offset,
                &gate_proj, 0,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                config.hidden_dim as u32,
            )?;

            // Up projection: hidden_state → up_proj
            self.dispatch_matmul(
                dev, que,
                &hidden_state.buffer, 0,
                gate_buf, up_offset,
                &up_proj, 0,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                config.hidden_dim as u32,
            )?;

            // SwiGLU: intermediate = SiLU(gate_proj) * up_proj
            self.dispatch_swiglu(
                dev, que,
                &gate_proj, &up_proj, &intermediate,
                config.intermediate_dim as u32,
            )?;

            // Down projection: intermediate → expert_out
            self.dispatch_matmul(
                dev, que,
                &intermediate, 0,
                down_buf, 0,
                &expert_out, 0,
                config.intermediate_dim as u32,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
            )?;

            // Accumulate: hidden_state += routing_weight * expert_out
            self.dispatch_weighted_accum(
                dev, que,
                &expert_out, &hidden_state.buffer, routing_weight,
                config.hidden_dim as u32,
            )?;
        }

        Ok(())
    }

    /// Dispatch a matmul: output = input × weights
    fn dispatch_matmul(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        input_offset: u64,
        weights: &wgpu::Buffer,
        weights_offset: u64,
        output: &wgpu::Buffer,
        output_offset: u64,
        m: u32, n: u32, k: u32,
    ) -> Result<(), String> {
        // This is a simplified matmul dispatch
        // In production, use the existing matmul/shader infrastructure
        let _ = (device, queue, input, input_offset, weights, weights_offset, output, output_offset, m, n, k);
        Err("dispatch_matmul not yet implemented".to_string())
    }

    /// Dispatch SwiGLU: intermediate = SiLU(gate) * up
    fn dispatch_swiglu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gate: &wgpu::Buffer,
        up: &wgpu::Buffer,
        intermediate: &wgpu::Buffer,
        n_elements: u32,
    ) -> Result<(), String> {
        let _ = (device, queue, gate, up, intermediate, n_elements);
        Err("dispatch_swiglu not yet implemented".to_string())
    }

    /// Dispatch weighted accumulation: output += weight × src
    fn dispatch_weighted_accum(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        output: &wgpu::Buffer,
        weight: f32,
        n_elements: u32,
    ) -> Result<(), String> {
        let _ = (device, queue, src, output, weight, n_elements);
        Err("dispatch_weighted_accum not yet implemented".to_string())
    }
}
