//! GPU-resident tensor wrapper for zero-copy activation passing between layers.
//!
//! Instead of CPU↔GPU ping-ponging per layer, we keep the hidden state
//! as a wgpu::Buffer throughout the forward pass. Only the final logits
//! and sampling inputs are read back to CPU.

use std::sync::Arc;

/// A tensor that lives on the GPU. Tracks the buffer, size, and element count.
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub len: usize,   // number of elements
    pub _device: Arc<wgpu::Device>,
    pub _queue: Arc<wgpu::Queue>,
}

impl GpuTensor {
    /// Create a new GPU tensor from CPU data.
    pub fn from_cpu(data: &[f32], device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> Self {
        let len = data.len();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_tensor"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        Self {
            buffer: buf,
            len,
            _device: device.clone(),
            _queue: queue.clone(),
        }
    }

    /// Read the tensor back to CPU.
    pub fn to_cpu(&self) -> Vec<f32> {
        let len = self.len;
        let staging = self._device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_tensor_staging"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self._device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_tensor_readback"),
        });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, (len * 4) as u64);
        let sub_idx = self._queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self._device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
        let _ = rx.recv();
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out
    }

    /// Create a GPU tensor of zeros with the given length.
    pub fn zeros(len: usize, device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_tensor_zeros"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Pre-zero via a small dispatch or just write zeros directly.
        let zeros = vec![0u8; len * 4];
        queue.write_buffer(&buf, 0, &zeros);
        Self {
            buffer: buf,
            len,
            _device: device.clone(),
            _queue: queue.clone(),
        }
    }
}

/// GPU-resident weight cache for transformer layers.
///
/// Stores all projection weights (Q, K, V, O, gate, up, down) as GPU buffers
/// so the forward pass never re-uploads them. Each layer has its own cache.
pub struct GpuWeightCache {
    pub q: wgpu::Buffer,
    pub k: wgpu::Buffer,
    pub v: wgpu::Buffer,
    pub o: wgpu::Buffer,
    pub gate: wgpu::Buffer,
    pub up: wgpu::Buffer,
    pub down: wgpu::Buffer,
}

impl GpuWeightCache {
    pub fn new(
        q: wgpu::Buffer,
        k: wgpu::Buffer,
        v: wgpu::Buffer,
        o: wgpu::Buffer,
        gate: wgpu::Buffer,
        up: wgpu::Buffer,
        down: wgpu::Buffer,
    ) -> Self {
        Self { q, k, v, o, gate, up, down }
    }
}

/// Multi-GPU expert dispatch context.
///
/// Holds per-GPU expert buffers and the placement plan that maps
/// (layer, expert) → (gpu_index, buffer_key).
pub struct MultiGpuExpertContext {
    pub devices: Vec<Arc<wgpu::Device>>,
    pub queues: Vec<Arc<wgpu::Queue>>,
    pub expert_buffers: Vec<std::collections::HashMap<String, wgpu::Buffer>>,
    pub placement: crate::moe_expert_loader::PlacementPlan,
}

impl MultiGpuExpertContext {
    pub fn lookup_expert(&self, layer: usize, kind: crate::moe_expert_loader::ExpertKind) -> Option<(&wgpu::Buffer, usize)> {
        let entry = self.placement.lookup(layer, kind)?;
        let &(gpu_idx, ref key) = entry;
        let buf = self.expert_buffers[gpu_idx].get(key)?;
        Some((buf, gpu_idx))
    }
}
