// moe_ffn_fused.comp
//
// Vulkan GLSL compute shader for fused MoE FFN execution.
// Reference implementation from specialist — to be ported to WGSL for wgpu.
//
// Fuses: gate_proj, up_proj, SiLU, elementwise mul, down_proj, weighted accumulation
// Assumes: top_k = 2 (Gemma4), experts packed into one giant buffer
// One invocation = one token

#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64) in;

layout(push_constant) uniform PushConsts {
    uint hidden_size;
    uint intermediate_size;
    uint num_tokens;
    uint top_k;
    uint num_experts;
} pc;

layout(set = 0, binding = 0, scalar) readonly buffer HiddenIn { float hidden_in[]; };
layout(set = 0, binding = 1, scalar) readonly buffer ExpertIndices { uint expert_indices[]; };
layout(set = 0, binding = 2, scalar) readonly buffer ExpertWeights { float expert_weights[]; };
layout(set = 0, binding = 3, scalar) readonly buffer ExpertWeightsPacked { float weights[]; };
layout(set = 0, binding = 4, scalar) readonly buffer ExpertOffsets { uint gate_offsets[]; };
layout(set = 0, binding = 5, scalar) readonly buffer UpOffsets { uint up_offsets[]; };
layout(set = 0, binding = 6, scalar) readonly buffer DownOffsets { uint down_offsets[]; };
layout(set = 0, binding = 7, scalar) writeonly buffer HiddenOut { float hidden_out[]; };

shared float gate_vec[8192];
shared float up_vec[8192];

float silu(float x) {
    return x / (1.0 + exp(-x));
}

void main() {
    uint token_id = gl_GlobalInvocationID.x;
    if (token_id >= pc.num_tokens) return;

    uint hidden_base = token_id * pc.hidden_size;

    // Zero output
    for (uint h = 0; h < pc.hidden_size; ++h) {
        hidden_out[hidden_base + h] = 0.0;
    }

    // Top-k experts
    for (uint k = 0; k < pc.top_k; ++k) {
        uint routing_idx = token_id * pc.top_k + k;
        uint expert_id = expert_indices[routing_idx];
        float route_weight = expert_weights[routing_idx];

        uint gate_offset = gate_offsets[expert_id];
        uint up_offset   = up_offsets[expert_id];
        uint down_offset = down_offsets[expert_id];

        // gate_proj + up_proj → SwiGLU
        barrier();
        for (uint i = gl_LocalInvocationID.x; i < pc.intermediate_size; i += gl_WorkGroupSize.x) {
            float gate_val = 0.0;
            float up_val   = 0.0;
            uint gate_row = gate_offset + i * pc.hidden_size;
            uint up_row   = up_offset   + i * pc.hidden_size;

            for (uint h = 0; h < pc.hidden_size; ++h) {
                float x = hidden_in[hidden_base + h];
                gate_val += weights[gate_row + h] * x;
                up_val   += weights[up_row   + h] * x;
            }

            gate_vec[i] = silu(gate_val) * up_val;
        }
        barrier();

        // down_proj + weighted accumulation
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
