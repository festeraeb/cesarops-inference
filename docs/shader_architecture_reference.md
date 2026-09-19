# Vulkan/WGSL Shader Architecture Reference

## From specialist — production inference shader stack

### Core shader families

**Dense GEMM / MatVec:**
- FP16/BF16 tensor-core GEMM
- Q4_0 dequant matvec
- Q4_K blockwise quant GEMM
- IQ4_XS importance-aware 4-bit GEMM
- Q6_K mixed precision GEMM
- Q8_0 fast fallback GEMM

**MoE-specific kernels:**
- Router top-k (softmax/topk)
- Token sorting (histogram/prefix/scatter)
- Grouped GEMM (expert-major matmul)
- Gather/scatter (output restoration)

### Critical insight: Fused dequant + MMA

The biggest performance gain is NOT better arithmetic. It is:
**fusing dequantization with MMA**

DO NOT: decode entire matrix to fp16 then multiply
INSTEAD: decode directly into registers/shared memory inside GEMM tiles

This is how llama.cpp, TensorRT-LLM, vLLM, ExLlama, and MLC get high throughput.

### IQ4_XS reality

IQ4_XS is NOT a simple nibble quant. It uses:
- Packed 4-bit weights
- Block scales
- Importance-aware scaling
- Nonlinear reconstruction

The exact packing layout differs between implementations.
What people actually do: decode block → fp16 registers/shared memory → then MMA/GEMM

### Production Vulkan compute architecture

```
global memory:
    packed quant blocks

shader:
    decode quant block
    unpack into fp16
    cooperative matrix multiply
    subgroup reduction
```

### Tier progression

**Tier 1 (we have this):**
- FP16 GEMM ✅
- Q4/Q6_K matvec ✅
- RMSNorm ✅
- RoPE ✅
- Attention ✅

**Tier 2 (next):**
- Cooperative matrices / subgroup MMA
- Tiled shared-memory GEMMs
- Fused dequant+MMA

**Tier 3 (MoE performance):**
- IQ4_XS fused decode+MMA
- Grouped MoE GEMMs
- Expert routing kernels

### Key code patterns

**Q4 block decode:**
```glsl
uint packed = weights[idx >> 1];
uint nibble = (idx & 1) == 0 ? (packed & 0xF) : ((packed >> 4) & 0xF);
float w = (float(nibble) - 8.0) * scale;
acc += w * activation;
```

**FP16 tiled GEMM (16x16 tiles, shared memory):**
```glsl
shared float16_t Asub[16][16];
shared float16_t Bsub[16][16];

for (uint tile = 0; tile < K; tile += 16) {
    // Load tile to shared
    Asub[ly][lx] = A[row * K + tile + lx];
    Bsub[ly][lx] = B[(tile + ly) * N + col];
    barrier();
    // Accumulate
    for (uint k = 0; k < 16; ++k)
        acc += float(Asub[ly][k]) * float(Bsub[k][lx]);
    barrier();
}
```

**Cooperative matrix (tensor cores via Vulkan):**
```glsl
#extension GL_KHR_cooperative_matrix : enable
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, 16, gl_MatrixUseAccumulator> acc;
```

### What we need from specialist next

1. Real Vulkan IQ4_XS fused decode+MMA compute shader
2. Cooperative matrix GEMM for Pascal (if supported) or subgroup fallback
3. Flash-attention Vulkan kernel
4. Full MoE grouped GEMM with expert-major layout
