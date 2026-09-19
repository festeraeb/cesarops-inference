# Task: IQ4_XS Quantized Matrix-Vector Multiply Shader (WGSL)

Write a WGSL compute shader that performs matrix-vector multiplication directly on IQ4_XS quantized weight matrices WITHOUT full dequantization.

## Context

For Gemma-4-26B-MoE with 64 experts, fully dequantizing all expert weights to f16 would consume ~50 GB VRAM. Instead, we keep weights in IQ4_XS format and dequantize on-the-fly during the matmul.

## What it does

Computes: output[row] = dot(weight_row[row, :], input[:])

Where:
- weight_row is stored as IQ4_XS (24 bytes per 32 elements)
- input is f16 [K]
- output is f16 [M]
- M = number of rows (e.g. 16384 for gate/up, 2048 for down)
- K = number of columns (e.g. 2048 for gate/up, 16384 for down)

## IQ4_XS block layout (24 bytes per 32 weights):
- bytes 0..1: d (f16 global scale)
- bytes 2..3: scales_h (high bits of 4 sub-block scales)
- bytes 4..7: scales_l (4 bytes, one per sub-block)
- bytes 8..23: qs (16 bytes = 32 nibbles packed)

## Nonlinear codebook:
```
const IQ4_XS_TABLE : array<f16, 16> = array<f16, 16>(
    f16(-1.0000), f16(-0.6962), f16(-0.5251), f16(-0.3949),
    f16(-0.2844), f16(-0.1848), f16(-0.0911), f16( 0.0000),
    f16( 0.0796), f16( 0.1609), f16( 0.2461), f16( 0.3379),
    f16( 0.4407), f16( 0.5626), f16( 0.7230), f16( 1.0000)
);
```

## Shader interface:
```wgsl
@group(0) @binding(0) var<storage, read> weights : array<u32>;  // IQ4_XS packed
@group(0) @binding(1) var<storage, read> input : array<f16>;    // [K]
@group(0) @binding(2) var<storage, read_write> output : array<f16>; // [M]
@group(0) @binding(3) var<uniform> params : MatvecParams;
```

Where MatvecParams:
```wgsl
struct MatvecParams {
    M : u32,           // output rows
    K : u32,           // input columns (= weight columns)
    weight_offset : u32, // byte offset into weights buffer for this expert
};
```

## Algorithm per output row:
1. Each workgroup computes one output row (or a few rows)
2. For each block of 32 elements in the row:
   - Load the IQ4_XS block header (d, scales_h, scales_l)
   - For each of 32 elements: dequant on-the-fly and multiply by input[col]
   - Accumulate dot product in f32 for precision
3. Reduce within workgroup
4. Write final f16 result to output[row]

## Performance considerations:
- Workgroup size: 256 (process 8 blocks of 32 per workgroup = 256 elements per pass)
- Use shared memory for input tile caching
- Accumulate in f32, cast to f16 only at final write
- Each workgroup handles one row; dispatch M workgroups

## Output:
Complete WGSL shader. Under 120 lines.
