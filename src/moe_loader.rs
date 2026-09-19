// src/moe_loader.rs
// GGUF MoE Expert Tensor Loader
// Loads packed expert weights from memory-mapped GGUF into GPU buffers

use std::collections::HashMap;
use crate::tensor_loader_safe::{TensorInfo, LoadError};

#[derive(Debug)]
pub struct MoeLayerBuffers {
    pub router_gate: wgpu::Buffer,      // [num_experts, hidden_size]
    pub expert_gates: wgpu::Buffer,     // packed [num_experts * intermediate_size, hidden_size]
    pub expert_ups: wgpu::Buffer,       // packed [num_experts * intermediate_size, hidden_size]
    pub expert_downs: wgpu::Buffer,     // packed [num_experts * hidden_size, intermediate_size]
}

pub fn load_moe_layer(
    mmap: &[u8],
    tensor_registry: &HashMap<String, TensorInfo>,
    layer_idx: usize,
    num_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<MoeLayerBuffers, LoadError> {
    let layer_prefix = format!("blk.{}.", layer_idx);

    // Helper to get tensor by common name variants
    let get_tensor = |base: &str| -> Result<&TensorInfo, LoadError> {
        let candidates = [
            format!("{}{}{}.weight", layer_prefix, base, "_exps"),
            format!("{}{}{}.weight", layer_prefix, base, "_exp"),
            format!("{}{}.weight", layer_prefix, base),
        ];

        for name in &candidates {
            if let Some(info) = tensor_registry.get(name) {
                return Ok(info);
            }
        }
        Err(LoadError::MissingTensor(format!("MoE tensor {}{}*", layer_prefix, base)))
    };

    // 1. Router (gate_inp)
    let router_info = tensor_registry
        .get(&format!("{}ffn_gate_inp.weight", layer_prefix))
        .or_else(|| tensor_registry.get(&format!("{}ffn_gate_inp_shexp.weight", layer_prefix)))
        .ok_or_else(|| LoadError::MissingTensor(format!("{}ffn_gate_inp.weight", layer_prefix)))?;

    // 2. Expert projections
    let gate_info = get_tensor("ffn_gate")?;
    let up_info = get_tensor("ffn_up")?;
    let down_info = get_tensor("ffn_down")?;

    // Validate shapes roughly
    if gate_info.shape[0] != num_experts * intermediate_size || gate_info.shape[1] != hidden_size {
        tracing::warn!("Unexpected shape for expert gates: {:?}", gate_info.shape);
    }

    // Create buffers + upload (raw quantized data for IQ4_XS — dequant happens in shader)
    let router_gate = create_and_upload_tensor(mmap, router_info, device, queue)?;
    let expert_gates = create_and_upload_tensor(mmap, gate_info, device, queue)?;
    let expert_ups = create_and_upload_tensor(mmap, up_info, device, queue)?;
    let expert_downs = create_and_upload_tensor(mmap, down_info, device, queue)?;

    Ok(MoeLayerBuffers {
        router_gate,
        expert_gates,
        expert_ups,
        expert_downs,
    })
}

fn create_and_upload_tensor(
    mmap: &[u8],
    info: &TensorInfo,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Buffer, LoadError> {
    let size = info.n_bytes() as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&info.name),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy raw quantized bytes from mmap
    let offset = info.offset as usize;
    let data = &mmap[offset..offset + info.n_bytes()];

    queue.write_buffer(&buffer, 0, data);
    queue.submit([]);

    Ok(buffer)
}
