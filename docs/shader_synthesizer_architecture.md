# Shader Synthesizer — Self-Improving GPU Kernel Pipeline

## Architecture

```
Rust Orchestrator (Host)
    │
    ├── GPU Probe (Pascal/Turing/Ampere detection)
    ├── Shader Scanner (find existing kernels)
    ├── Benchmark DB (performance history)
    │
    ▼
Prompt Builder (Gemma 4 LLM)
    ▼
Generated GLSL/WGSL
    ▼
SPIR-V Compiler (glslc)
    ▼
Vulkan/wgpu Execution
    ▼
Performance Profiler
    ▼
Evolution Engine (iterates until convergence)
```

## Pipeline: detect → scan → prompt → generate → compile → benchmark → evolve

## Pascal constraints (always inject):
- ❌ No subgroup-heavy reliance
- ❌ No tensor cores
- ❌ No heavy shared memory tiling
- ❌ No INT8 dot pipelines
- ✅ Register-resident loops
- ✅ FP16 where possible
- ✅ Q4 packing with manual unpack
- ✅ Warp-independent execution
- ✅ Minimal branching

## Turing (2060) advantages:
- ✅ Good INT8 throughput
- ✅ Tensor cores (FP16/INT8 mixed)
- ✅ Subgroup shuffle
- ✅ Better shared memory config

## Evolution loop:
1. Generate shader → run → benchmark
2. Feed result back into prompt ("Current best: X ms/token. Improve Y.")
3. Regenerate → benchmark → compare
4. Keep best, iterate

## Key insight:
Biggest real performance gains come from **better memory coalescing**,
not exotic quantization tricks.
