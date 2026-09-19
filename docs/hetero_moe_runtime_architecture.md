# Heterogeneous Distributed Inference Core (HDIC)

## Target Architecture — from specialist

### Hardware Topology
```
RTX 2060 Super (Coordinator)
  → Routing brain, KV cache, attention hub
  → Speculative draft generation (fast)

P100 × 2 (Heavy compute)
  → Large GEMM, FFN expert execution
  → Bulk throughput

GTX 1070 (Mid compute)
  → Medium tiled FFN, overflow experts

GTX 1060 / P106-100 (Fallback)
  → Quantized ops, low-priority experts

P1000 (Always-on orchestrator)
  → Intake routing, health, n8n decisions
```

### Three Critical Systems

#### 1. Distributed KV Cache Engine (vLLM-style)
- KV segments + token deltas (not full recomputation)
- Chunk-based sync (not per-token network)
- Adaptive compression: FP16 → INT8 → INT4 based on network/GPU pressure
- KV reuse amortizes network cost

#### 2. CUDA/Vulkan Kernel Pack (GPU-specialized)
- P100: large GEMM (FP16/FP32 throughput, register-heavy)
- 1070: medium tiled FFN (cache-tuned)
- 1060: fallback quantized ops
- RTX 2060: tensor-core attention + routing

#### 3. RTX 2060 Attention Offload Hub
- Attention is memory-bandwidth-bound + latency-sensitive
- CANNOT scatter across machines
- RTX 2060 = attention + KV + merge
- Workers only return: QKV projections + FFN outputs

### Inference Pipeline
```
TOKEN
  ↓
RTX 2060 → compute QKV
  ↓
dispatch FFN to P100 / 1070
  ↓
return FFN output
  ↓
RTX 2060 → attention + merge
  ↓
next token
```

### Advanced Systems

#### RL-Based Router
- State: expert_id, layer_id, gpu_utilization[], kv_pressure, network_latency
- Action: assign to P100/1070/1060/2060/defer
- Reward: (throughput / latency) - divergence * 0.5
- Replaces static heuristics with learned placement

#### Cross-Node Speculative Decoding
- RTX 2060 drafts tokens quickly (small model or early exit)
- P100/1070/1060 verify in parallel
- Network latency hidden behind batching

#### Pipeline Parallel Transformer
- Layers split across GPUs by compute profile
- Layers 0-3: RTX 2060 (attention + embedding)
- Layers 3-12: P100 (heavy FFN)
- Layers 12-20: GTX 1070 (mid FFN)
- Layers 20+: GTX 1060 (fallback)
- Overlapped execution stages

#### Dynamic Expert Migration
- Experts move between GPUs based on load
- Trigger: utilization > 85% OR memory_pressure > 80%
- Target: least-loaded device
- Coarse-grained (not per-token)

#### KV Cache Compression
- FP16 (default) → INT8 (GPU pressure > 70%) → INT4 (network pressure > 80%)
- 4-16× bandwidth reduction
- Makes multi-node inference viable on Ethernet

### Current Bugs to Fix (Single-Node First)

1. **Layer loop stuck at Layer 0** — likely HTTP timeout causing retry
2. **12s per layer** — re-uploading weights from CPU every matmul call
3. **CPU↔GPU round-trip per op** — should keep tensors on GPU between layers

### Fix Priority
1. Keep weights on GPU (bind persistent buffers, don't re-upload)
2. Keep hidden state on GPU between layers (no readback until final logits)
3. Then: multi-node dispatch
