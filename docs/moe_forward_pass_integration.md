# MoE Forward Pass Integration — Architecture Reference

## Layer Loop Pattern

For each layer:
1. Run attention (GPU)
2. If MoE:
   - Async readback hidden → CPU staging
   - CPU router (f64 precision, cached gate weights)
   - Upload (expert_indices, expert_weights) → GPU
   - dispatch_moe_ffn on GPU
3. Else: dense FFN

## Key Optimizations
- Gate weights cached on CPU after first load (NOT per forward pass)
- Hidden readback uses async staging buffer
- Router runs while GPU may already be executing next ops
- Only small tensors cross CPU boundary (top-k indices + weights)

## Sync Cost
Unavoidable: GPU hidden → CPU router → GPU dispatch

Mitigations:
- Cache router weights
- Async staging buffer
- Keep router input small
- Future: move router to GPU compute shader (eliminates CPU roundtrip)

## GGUF Expert Loading

Detection:
```rust
fn is_expert_tensor(name: &str) -> bool {
    name.contains("ffn_gate_exps.weight")
        || name.contains("ffn_gate.") && name.contains(".weight")
}
```

Strategy: Pack all experts into one contiguous GPU buffer per layer:
```
[ expert0 | expert1 | ... | expert63 ]
```

```rust
pub struct MoeLayerBuffers {
    pub experts_packed: GpuBuffer,
    pub expert_offsets: Vec<u32>,
    pub router_gate: GpuBuffer,
}
```

## Future: GPU Router Fast Path
Move router off CPU entirely — run as tiny Vulkan compute shader.
Eliminates GPU→CPU copy, f64 bottleneck, and sync stall.
