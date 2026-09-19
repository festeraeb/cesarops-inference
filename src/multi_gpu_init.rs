//! Multi-GPU initialization for MoE expert loading.
//!
//! Discovers all Vulkan adapters, allocates per-GPU contexts, and builds
//! the expert placement plan for distributed MoE inference.

use std::sync::Arc;

use tracing::{info, warn};

use crate::hardware::{audit_system, GpuNodeInfo};
use crate::moe_expert_loader::{build_devices, ExpertDevice, PlacementPlan, enumerate_experts, plan_placement, upload_experts};
use crate::loader::ModelWeights;

/// Multi-GPU context for MoE expert loading.
///
/// Contains:
///   - Per-GPU Vulkan devices and queues
///   - Expert weight buffers distributed across GPUs
///   - Placement plan mapping (layer, expert) → (gpu_index, buffer_key)
pub struct MultiGpuContext {
    pub expert_devices: Vec<ExpertDevice>,
    pub placement: PlacementPlan,
    pub experts: Vec<crate::moe_expert_loader::ExpertTensor>,
}

impl MultiGpuContext {
    /// Get the number of GPUs in this context.
    pub fn num_gpus(&self) -> usize {
        self.expert_devices.len()
    }

    /// Get the GPU info for debugging.
    pub fn gpu_info(&self) -> Vec<String> {
        self.expert_devices
            .iter()
            .map(|d| format!("{}: {} MB free", d.name, d.free_bytes() / (1024 * 1024)))
            .collect()
    }
}

/// Initialize multi-GPU context for MoE expert loading.
///
/// 1. Audit system hardware via nvidia-smi
/// 2. Enumerate Vulkan adapters
/// 3. Compute expert tensors in the model
/// 4. Build placement plan
/// 5. Upload expert weights to their assigned GPUs
///
/// Returns `None` if no Vulkan adapters are found or if the model has no MoE experts.
pub async fn init_multigpu_context(
    model: &ModelWeights,
    budget_per_gpu_mb: u64,
) -> Option<MultiGpuContext> {
    // Step 1: Audit hardware
    let profile = audit_system();
    info!("Hardware profile: {} GPUs detected", profile.gpu_nodes.len());

    if profile.gpu_nodes.is_empty() {
        warn!("No GPUs detected — MoE multi-GPU disabled");
        return None;
    }

    // Step 2: Enumerate Vulkan adapters
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let mut adapter_list = Vec::new();
    for gpu in &profile.gpu_nodes {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await?;

        let info = adapter.get_info();
        if info.name == gpu.name {
            adapter_list.push((adapter, gpu));
        }
    }

    if adapter_list.is_empty() {
        warn!("No Vulkan adapters match nvidia-smi GPUs — MoE multi-GPU disabled");
        return None;
    }

    info!("Vulkan adapters: {}", adapter_list.len());

    // Step 3: Create wgpu devices and queues
    let mut device_inputs = Vec::new();
    for (adapter, gpu_info) in adapter_list {
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some(&format!("multigpu_device_{}", gpu_info.index)),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024,
                    max_buffer_size: 1024 * 1024 * 1024,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ).await.ok()?;

        let budget = budget_per_gpu_mb * 1024 * 1024;
        device_inputs.push((Arc::new(device), Arc::new(queue), gpu_info.name.clone(), budget));
    }

    // Step 4: Build ExpertDevice list
    let expert_devices = build_devices(device_inputs);
    info!("Expert devices: {}", expert_devices.len());

    // Step 5: Enumerate MoE experts in the model
    let experts = enumerate_experts(model);
    info!("Found {} MoE experts across {} layers", experts.len(), experts.iter().map(|e| e.layer).max().unwrap_or(0) + 1);

    if experts.is_empty() {
        warn!("No MoE experts found in model — multi-GPU not applicable");
        return None;
    }

    // Step 6: Build placement plan
    let placement = plan_placement(&expert_devices, &experts);

    // Step 7: Upload expert weights
    let mut expert_devices_mut: Vec<ExpertDevice> = expert_devices;
    if let Err(e) = upload_experts(&mut expert_devices_mut, &placement, &experts, model) {
        warn!("Failed to upload experts: {} — falling back to single-GPU", e);
        return None;
    }

    Some(MultiGpuContext {
        expert_devices: expert_devices_mut,
        placement,
        experts,
    })
}

/// Build a multi-GPU expert context from existing ExpertDevice list.
///
/// This is useful when you already have a wgpu context and want to add
/// MoE expert loading to it.
pub fn build_multigpu_context_from_devices(
    devices: Vec<ExpertDevice>,
    model: &ModelWeights,
) -> Option<MultiGpuContext> {
    let experts = enumerate_experts(model);
    if experts.is_empty() {
        return None;
    }
    let placement = plan_placement(&devices, &experts);

    let mut devices_mut = devices;
    if let Err(e) = upload_experts(&mut devices_mut, &placement, &experts, model) {
        warn!("Failed to upload experts: {}", e);
        return None;
    }

    Some(MultiGpuContext {
        expert_devices: devices_mut,
        placement,
        experts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multigpu_context_creation() {
        // This test verifies the structure compiles and the types align
        // Actual Vulkan initialization requires runtime GPU hardware
        let _ = crate::hardware::audit_system();
    }
}
