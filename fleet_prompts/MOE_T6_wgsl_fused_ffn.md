# Task: Port Fused MoE FFN Shader from Vulkan GLSL to WGSL

You are a GPU compute shader expert. Port the following Vulkan GLSL compute shader to WGSL for use with wgpu.

## Source (Vulkan GLSL):

```glsl
#version 460
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x = 64) in;

layout(push_constant) uniform PushConsts {
    uint hidden_size;      // 2048
    uint intermediate_size; // 16384
    uint num_tokens;
    uint top_k;            // 2
    uint num_experts;      // 64
} pc;

layout(set = 0, binding = 0, scalar) readonly buffer HiddenIn { float hidden_in[]; };
layout(set = 0, binding = 1, scalar) readonly buffer ExpertIndices { uint expert_indices[]; };
layout(set = 0, binding = 2, scalar) readonly buffer ExpertWeights { float expert_weights[]; };
layout(set = 0, binding = 3, scalar) readonly buffer ExpertWeightsPacked { float weights[]; };
layout(set = 0, binding = 4, scalar) readonly buffer GateOffsets { uint gate_offsets[]; };
layout(set = 0, binding = 5, scalar) readonly buffer UpOffsets { uint up_offsets[]; };
layout(set = 0, binding = 6, scalar) readonly buffer DownOffsets { uint down_offsets[]; };
layout(set = 0, binding = 7, scalar) writeonly buffer HiddenOut { float hidden_out[]; };

shared float gate_vec[8192];

float silu(float x) { return x / (1.0 + exp(-x)); }

void main() {
    uint token_id = gl_GlobalInvocationID.x;
    if (token_id >= pc.num_tokens) return;
    uint hidden_base = token_id * pc.hidden_size;

    for (uint h = 0; h < pc.hidden_size; ++h) {
        hidden_out[hidden_base + h] = 0.0;
    }

    for (uint k = 0; k < pc.top_k; ++k) {
        uint routing_idx = token_id * pc.top_k + k;
        uint expert_id = expert_indices[routing_idx];
        float route_weight = expert_weights[routing_idx];

        uint gate_offset = gate_offsets[expert_id];
        uint up_offset = up_offsets[expert_id];
        uint down_offset = down_offsets[expert_id];

        barrier();
        for (uint i = gl_LocalInvocationID.x; i < pc.intermediate_size; i += gl_WorkGroupSize.x) {
            float gate_val = 0.0;
            float up_val = 0.0;
            uint gate_row = gate_offset + i * pc.hidden_size;
            uint up_row = up_offset + i * pc.hidden_size;
            for (uint h = 0; h < pc.hidden_size; ++h) {
                float x = hidden_in[hidden_base + h];
                gate_val += weights[gate_row + h] * x;
                up_val += weights[up_row + h] * x;
            }
            gate_vec[i] = silu(gate_val) * up_val;
        }
        barrier();

        for (uint h = gl_LocalInvocationID.x; h < pc.hidden_size; h += gl_WorkGroupSize.x) {
            float acc = 0.0;
            uint down_row = down_offset + h * pc.intermediate_size;
            for (uint i = 0; i < pc.intermediate_size; ++i) {
                acc += weights[down_row + i] * gate_vec[i];
            }
            hidden_out[hidden_base + h] += acc * route_weight;
        }
        barrier();
    }
}
```

## WGSL Requirements:

1. Replace `push_constant` with a uniform buffer struct
2. Replace `gl_GlobalInvocationID` with `@builtin(global_invocation_id)`
3. Replace `gl_LocalInvocationID` with `@builtin(local_invocation_id)`
4. Replace `gl_WorkGroupSize` with the literal workgroup size (64)
5. Replace `barrier()` with `workgroupBarrier()`
6. Replace `shared` with `var<workgroup>`
7. Use `@group(0) @binding(N)` for all buffers
8. Use `var<storage, read>` and `var<storage, read_write>`
9. WGSL uses `fn` not return-type-first syntax
10. WGSL exp() is available directly

## Uniform struct (replaces push constants):
```wgsl
struct MoeParams {
    hidden_size : u32,
    intermediate_size : u32,
    num_tokens : u32,
    top_k : u32,
    num_experts : u32,
};
```

## Output:
Complete WGSL shader. Must be a direct 1:1 port of the GLSL above. Do NOT simplify or restructure — keep the same algorithm, same loop structure, same shared memory usage. Just translate the syntax.

Under 100 lines.
