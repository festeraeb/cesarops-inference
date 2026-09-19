# Warp-Optimized IQ4_XS Decode + Turing (2060) Tuning Guide

## From specialist — warp-level decode, no shared memory, exact ggml mapping

### Key correction: shared memory removal

Removing shared memory is NOT universally faster. What matters:
- Register pressure
- Instruction divergence
- Memory coalescing
- Reuse distance
- Tensor core vs scalar ALU balance

On RTX 2060 (Turing SM75):
- Warp-shuffle reductions (`__shfl_sync` / `subgroupAdd`)
- Register-resident block metadata
- Minimal LDS unless tiling > 64-128 bytes per warp

### Warp pattern: "one block per warp"

32 threads = 1 warp = decode 1-2 quant blocks
Each thread handles 1 lane of dequant + dot accumulation

```
int lane = threadIdx.x & 31;
int warp_id = threadIdx.x >> 5;

BlockQ4 b = load_block(global_ptr + warp_id);

uint8_t v = b.qs[lane >> 1];
int lo = v & 0xF;
int hi = v >> 4;

float scale = __half2float(b.d);
float a = (lo - 8) * scale;
float c = (hi - 8) * scale;

acc += a * vecA[idx] + c * vecA[idx+1];
```

### Warp reduction (no shared memory)

```
for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
}
if (lane == 0) write_out(acc);
```

### Vulkan/WGSL equivalent

```glsl
uint lane = gl_SubgroupInvocationID;
float acc = decode_iq4(block, lane);
acc = subgroupAdd(acc);
```

### Exact ggml IQ4_XS layout mapping (zero guesswork)

DO NOT hardcode layout. Derive from:
1. `ggml-quants.c` → `dequantize_row_iq4_xs()`
2. `typedef struct { ... } block_iq4_xs;`
3. `#define QK_IQ4_XS`

Safe Rust representation:
```rust
#[repr(C)]
struct Iq4XsBlock {
    raw: [u8; BLOCK_SIZE], // fill after confirming ggml constant
}

impl Iq4XsBlock {
    fn scale(&self) -> f16 {
        // only after confirming offset in ggml source
    }
    fn weight_nibble(&self, i: usize) -> i8 {
        // decode via verified bit ops from ggml.c
    }
}
```

### RTX 2060 (Turing SM75) optimizations

- 8 GB VRAM → bandwidth matters more than ALU
- Good shuffle performance
- ~64KB configurable shared memory per SM
- No async copy (Ampere feature)

**Best split:**
- Registers: per-thread decode
- Shared memory: optional activation tile staging (if reused > warp scope)

**Sweet spot:**
- 1 warp = 1 output tile (1x128 or 1x64)
- 2 warps per SM resident
- Heavy `subgroupAdd` / shuffle usage
- Avoid divergent branching in decode
- 64-96 registers/thread max (beyond → occupancy collapse)

**Memory coalescing (critical):**
Store GGML blocks warp-contiguous:
```
[warp0 block0][warp0 block1]...[warp1 block0]...
```
NOT column-major by neuron.

### Production Vulkan compute structure

```
SSBO = GGML blocks (IQ4_XS)
SSBO = activations
push constants = scales / offsets
subgroup ops = warp shuffle equivalent
```

### Next steps for our engine

1. Verify exact `block_iq4_xs` layout from our llama.cpp version
2. Write Rust accessor with verified offsets
3. Port warp-shuffle reduction to WGSL (`subgroupAdd`)
4. Benchmark: shared-memory version vs subgroup version on P100 + 2060
