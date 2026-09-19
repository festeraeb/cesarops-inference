```wgsl
struct MoeParams {
    hidden_size: u32,
    intermediate_size: u32,
    num_tokens: u32,
    top_k: u32,
    num_experts: u32,
};

@group(0) @binding(0) var<uniform> pc: MoeParams;
@group(0) @binding(1) var<storage, read> hidden_in: array<f32>;
@group(0) @binding(2) var<storage, read> expert_indices: array<u32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read> gate_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> up_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> down_offsets: array<u32>;
@group(0) @binding(8) var<storage, read_write> hidden_out: array<f32>;

var<workgroup> gate_vec: array<f32, 8192>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let token_id = global_id.x;
    if (token_id >= pc.num_tokens) { return; }
    let hidden_base = token_id * pc.hidden_size;

    for (var h: u32 = 0u; h < pc.hidden_size; h++) {
        hidden_out[hidden_base + h] = 0.0;
    }

    for (var k: u32 = 0u; k < pc.top_k; k++) {
        let routing_idx = token_id * pc.top_k + k;
        let expert_id = expert_indices[routing_idx];
        let route_weight = expert_weights[routing_idx];

        let gate_offset = gate_offsets[expert_id];
        let up_offset = up_offsets[expert_id];
        let down_offset = down_offsets[expert_id];

        workgroupBarrier();
        for (var i = local_id.x; i < pc.intermediate_size; i += 64u) {
            var gate_val: f32 = 0.0;
            var up_val: f32 = 0.0;
            let gate_row = gate_offset + i * pc.hidden_size;
            let up_row = up_offset + i * pc.hidden_size;
            for (var h: u32 = 0u; h < pc.hidden_size; h++) {
                let x = hidden_in[hidden_base + h];
                gate_val += weights[gate_row + h] * x;
                up_val += weights[up_row + h] * x;
            }
            gate_vec[i] = silu(gate_val) * up_val;
        }
        workgroupBarrier();

        for (var h = local_id.x; h < pc.hidden_size; h += 64u) {
            var acc: f32 = 0.0;
            let down_row = down_offset + h * pc.intermediate_size;
            for (var i: u32 = 0u; i < pc.intermediate_size; i++) {
                acc += weights[down_row + i] * gate_vec[i];
            }
            hidden_out[hidden_base + h] += acc * route_weight;
        }
        workgroupBarrier();
    }
}
```
