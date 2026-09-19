// shaders/matvec_iq4xs_coopmat.glsl
//
// IQ4_XS matvec, Turing/Ampere fast path using GL_KHR_cooperative_matrix.
// Pascal devices MUST NOT load this — fall back to matvec_iq4xs_correct.wgsl.
//
// Compile:
//   glslc --target-env=vulkan1.3 -fshader-stage=compute \
//         shaders/matvec_iq4xs_coopmat.glsl \
//         -o shaders/matvec_iq4xs_coopmat.spv
//
// Load via wgpu::ShaderSource::SpirV (naga in wgpu 24 cannot lower
// OpTypeCooperativeMatrixKHR from front-end code).
//
// Numeric correctness oracle: matvec_iq4xs_correct.wgsl plus
// `bridge::dequant_iq4_xs`. Run the validator binary across the same
// tensor with both kernels to confirm no drift.

#version 460
#extension GL_KHR_cooperative_matrix             : require
#extension GL_KHR_memory_scope_semantics         : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_16bit_storage           : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

const uint M = 16;
const uint N = 16;
const uint K = 16;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer W { uint   w[]; };
layout(set = 0, binding = 1, std430) readonly buffer X { float  x[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Y { float y[]; };
layout(set = 0, binding = 3, std140) uniform Lut { vec4 lut[4]; };
layout(push_constant) uniform Push {
    uint K_total;
    uint N_total;
    uint row_offset;
} pc;

shared float16_t tileA[M][K];
shared float16_t tileB[K][N];
shared float     Cshared[M * N];

float lut_lookup(uint idx) {
    uint g = idx >> 2u;
    uint l = idx & 3u;
    return (l == 0u) ? lut[g].x
         : (l == 1u) ? lut[g].y
         : (l == 2u) ? lut[g].z
         :             lut[g].w;
}

float dequant_iq4xs_element(uint row, uint k) {
    const uint BLOCK_SIZE  = 256u;
    const uint BLOCK_WORDS = 34u;
    uint blocks_per_row = (pc.K_total + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    uint b  = k / BLOCK_SIZE;
    uint i  = k % BLOCK_SIZE;
    uint bo = row * blocks_per_row * BLOCK_WORDS + b * BLOCK_WORDS;

    uint w0 = w[bo];
    float16_t d = float16_t(unpackHalf2x16(w0).x);
    uint scales_h = (w0 >> 16u) & 0xFFFFu;
    uint w1 = w[bo + 1u];

    uint ib = i >> 5u;
    uint sl_byte = (w1 >> ((ib >> 1u) * 8u)) & 0xFFu;
    uint scale_low = ((ib & 1u) == 0u) ? (sl_byte & 0xFu) : ((sl_byte >> 4u) & 0xFu);
    uint scale_high = (scales_h >> (ib * 2u)) & 0x3u;
    int  scale_6bit = int((scale_high << 4u) | scale_low) - 32;

    uint byte_in_qs = i >> 1u;
    uint qs_word    = bo + 2u + (byte_in_qs >> 2u);
    uint qs_shift   = (byte_in_qs & 3u) * 8u;
    uint qs_byte    = (w[qs_word] >> qs_shift) & 0xFFu;
    uint q_nib      = ((i & 1u) == 0u) ? (qs_byte & 0xFu) : ((qs_byte >> 4u) & 0xFu);

    return float(d) * float(scale_6bit) * lut_lookup(q_nib);
}

void main() {
    uint row_tile = gl_WorkGroupID.x * M;
    uint k_tile_count = (pc.K_total + K - 1u) / K;

    coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> Cmat =
        coopmat<float, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(0.0);

    for (uint kt = 0; kt < k_tile_count; ++kt) {
        for (uint a = gl_LocalInvocationID.x; a < M * K; a += gl_WorkGroupSize.x) {
            uint mr = a / K;
            uint kc = a % K;
            float v = dequant_iq4xs_element(row_tile + mr, kt * K + kc);
            tileA[mr][kc] = float16_t(v);
        }
        for (uint a = gl_LocalInvocationID.x; a < K * N; a += gl_WorkGroupSize.x) {
            uint kr = a / N;
            uint nc = a % N;
            float v = (nc == 0u) ? x[kt * K + kr] : 0.0;
            tileB[kr][nc] = float16_t(v);
        }
        barrier();

        coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> A;
        coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> B;
        coopMatLoadKHR(A, tileA[0], 0, K, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoadKHR(B, tileB[0], 0, N, gl_CooperativeMatrixLayoutRowMajor);
        Cmat = coopMatMulAddKHR(A, B, Cmat);
        barrier();
    }

    coopMatStoreKHR(Cmat, Cshared, 0, N, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    if (gl_LocalInvocationID.x < M) {
        y[row_tile + gl_LocalInvocationID.x] =
            Cshared[gl_LocalInvocationID.x * N];
    }
}
