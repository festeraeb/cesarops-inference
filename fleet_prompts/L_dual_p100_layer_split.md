You are an expert in multi-GPU inference + Rust + wgpu + PCIe pipeline parallelism. Design layer-split execution across two P100s on a single host so we can run 32B-class Q4_K_M models that don't fit in a single 16 GB VRAM.

## Hardware context

- Host: Lenovo T440, dual Tesla P100-PCIE-16GB
- Both GPUs same NUMA node, same PCIe Host Bridge (NODE topology per nvidia-smi)
- PCIe Gen3 x16 = 15.75 GB/s unidirectional per slot
- Inter-GPU activation transfer cost per layer per token: ~6 KB (hidden_dim=5120 fp16 for 32B class) — negligible vs bandwidth
- Combined VRAM: 32 GB

## Target model class

`/codebase/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf` (19 GB)
or `Qwen3.6-35B-A3B-Q4_K_M.gguf` (20 GB) — note Qwen3.6 is MoE which
is a separate problem, defer that case.

For the 32B dense Q4_K_M: 19 GB weights + ~2 GB KV cache at 4K context
+ activations + scratch ≈ 24 GB total. Doesn't fit on one P100 (16 GB).
Splits 12 GB / 12 GB across two P100s, comfortable on both.

## Existing engine state

- `src/main.rs` `--gpu N` flag selects ONE adapter at startup
- `src/gpu_context.rs` holds device + queue, single-GPU only
- `TransformerDecoder.forward()` runs all 64 layers serially on one device
- KV cache lives entirely on one GPU (per-layer)
- No cross-GPU activation transfer code

## Architecture decision: pipeline parallel, NOT tensor parallel

Pipeline parallel splits LAYERS across GPUs:
- GPU 0: layers 0..31 (first half)
- GPU 1: layers 32..63 (second half)

Tensor parallel splits each layer's weight matrices across GPUs.
Tensor parallel needs all-reduce per layer = bandwidth death on
PCIe Gen3. Pipeline parallel needs only one activation transfer per
layer-boundary crossing = single 6 KB hop between halves.

For our hardware, pipeline parallel wins.

## Deliverables

### 1. Multi-GPU context

```rust
// src/multi_gpu.rs
pub struct MultiGpuContext {
    pub gpus: Vec<Arc<GpuContext>>,    // one per device
    pub layer_assignment: Vec<usize>,  // layer_idx -> gpu_idx
    pub xfer_buffers: Vec<TransferBuffer>,  // staging buffers for crossings
}

pub struct TransferBuffer {
    pub src_gpu: usize,
    pub dst_gpu: usize,
    pub host_staging: Vec<u8>,         // CPU-side staging via host-visible buffer
    pub size_bytes: usize,
}
```

The transfer mechanism on Pascal/Vulkan: GPU A writes to a HOST_VISIBLE
+ HOST_COHERENT staging buffer, CPU readback (or just memory barrier),
GPU B reads from its own HOST_VISIBLE buffer that mirrors the data.
Pascal does NOT support direct GPU-to-GPU DMA in Vulkan without
specific extensions (VK_KHR_external_memory not commonly supported on
Tesla/datacenter Linux drivers).

So the cross-GPU transfer protocol is:
1. GPU 0 finishes layer 31, writes activation to staging
2. CPU memcpy staging[GPU0] → staging[GPU1] (memory bandwidth ~25 GB/s)
3. GPU 1 reads from its staging, runs layers 32..63

Activation size at hidden_dim=5120 fp16 = 10 KB per token. Memcpy
cost: ~0.4 µs. Negligible vs forward-pass cost per layer.

### 2. Layer assignment strategy

```rust
pub fn assign_layers(n_layers: usize, n_gpus: usize) -> Vec<usize> {
    // Even split for now. Smarter: balance by per-layer compute,
    // for transformer all layers are uniform so just split evenly.
    (0..n_layers).map(|i| i * n_gpus / n_layers).collect()
}
```

For 64-layer 32B + 2 GPUs: layers 0..31 → GPU 0, layers 32..63 → GPU 1.

### 3. Forward pass split

`TransformerDecoder.forward()` modification:

```rust
impl TransformerDecoder {
    pub fn forward_multi_gpu(
        &self,
        token: u32,
        position: usize,
        ctx: &MultiGpuContext,
        kv_caches: &mut [KvCache],   // one per GPU
    ) -> Vec<f32> {
        let mut hidden = self.embed(token);  // on GPU 0
        let mut current_gpu = 0usize;

        for layer_idx in 0..self.config.num_layers {
            let target_gpu = ctx.layer_assignment[layer_idx];

            if target_gpu != current_gpu {
                hidden = ctx.transfer(current_gpu, target_gpu, &hidden);
                current_gpu = target_gpu;
            }

            hidden = self.execute_layer(
                layer_idx,
                hidden,
                position,
                &ctx.gpus[target_gpu],
                &mut kv_caches[target_gpu],
            );
        }

        // Final norm + lm_head on whichever GPU layer N-1 ran on
        self.lm_head(hidden, &ctx.gpus[current_gpu])
    }
}
```

### 4. Weight loading split

`tensor_loader_safe.rs` extension: load tensors per-GPU based on
layer assignment. Tensors with name pattern `blk.{i}.*` go to
`ctx.gpus[layer_assignment[i]]`. Embedding + final norm + lm_head go
to GPU 0 (or wherever first/last layers sit).

```rust
pub fn load_model_split(
    weights: &ModelWeights,
    ctx: &MultiGpuContext,
) -> Result<()>;
```

### 5. KV cache per-GPU

Each GPU owns its layer-range KV cache. KV state for layer i lives
only on `ctx.gpus[layer_assignment[i]]`. Never transferred between
GPUs (KV is layer-local, only activations cross).

### 6. CLI / config

Extend `--gpu` to accept `--gpu 0,1` for multi-GPU split. Or new flag
`--multi-gpu` that auto-detects and splits across all available
adapters. Show concrete CLI examples:

```
cesarops-inference serve --model deepseek-32b.gguf --multi-gpu
cesarops-inference serve --model qwen-7b.gguf --gpu 1   # single-GPU explicit
```

### 7. Smoke test

`scripts/smoke_dual_p100.sh`:
1. Load DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf with --multi-gpu
2. Confirm both GPUs report ~10-12 GB used via nvidia-smi
3. Run `--prompt "What is 2+2?" --max-tokens 15`
4. Confirm output coherent
5. Report tokens/sec

Expected: dual-P100 should run a 32B Q4_K_M at roughly half the
single-GPU 7B rate. So if 7B Q4_K_M runs ~1.5 t/s on one P100, 32B
on dual P100 runs ~0.5-0.8 t/s. Slow but functional.

## Constraints
- Pascal sm_60 specific limitations (no GPU-to-GPU DMA in Vulkan)
- Single-host only (no networking, that's a separate project)
- Conservative barriers — same hazard model as single-GPU path
- ~600-900 LOC total across new module + existing diff
- Both single-GPU paths MUST keep working (smoke + regression tests
  on 1.5B Q6_K and 1070 cross-card both pass post-integration)

## Output
Five files:
1. `src/multi_gpu.rs` — MultiGpuContext + transfer mechanism (~250 LOC)
2. `src/transformer.rs` diff — forward_multi_gpu method (~100 LOC)
3. `src/tensor_loader_safe.rs` diff — load_model_split (~100 LOC)
4. `src/main.rs` diff — CLI extension (~30 LOC)
5. `scripts/smoke_dual_p100.sh` — new test

Plus a memo on the cross-GPU transfer cost measurement methodology
(how to verify the 0.4 µs memcpy claim, what to log).
