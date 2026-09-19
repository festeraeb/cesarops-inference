#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer W {
    uint q[];          // IQ4_XS packed blocks (u8 accessed via u32)
};

layout(set = 0, binding = 1) readonly buffer X {
    float x[];            // input vector
};

layout(set = 0, binding = 2) writeonly buffer Y {
    float y[];
};

layout(push_constant) uniform PC {
    int row_stride;       // number of blocks per row
    int input_stride;
} pc;

layout(set = 0, binding = 3) uniform Table {
    float kvalues_iq4nl[16];
};

shared float partial[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x;

    float acc = 0.0;

    for (int b = int(tid); b < pc.row_stride; b += 256) {

        uint base = (row * pc.row_stride + b) * 16u;

        for (int i = 0; i < 16; i++) {
            uint byte_idx = base + i;
            uint word = q[byte_idx >> 2];
            uint shift = (byte_idx & 3u) * 8u;
            uint byte_val = (word >> shift) & 0xFFu;

            uint lo = byte_val & 0xFu;
            uint hi = byte_val >> 4u;

            uint xi0 = b * 32 + i * 2;
            uint xi1 = xi0 + 1;

            acc += kvalues_iq4nl[lo] * x[xi0];
            acc += kvalues_iq4nl[hi] * x[xi1];
        }
    }

    partial[tid] = acc;
    barrier();

    for (uint s = 128u; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        barrier();
    }

    if (tid == 0) y[row] = partial[0];
}
