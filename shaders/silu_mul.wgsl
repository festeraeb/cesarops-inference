requires f16;

@group(0) @binding(0)
var<storage, read> gate : array<f16>;

@group(0) @binding(1)
var<storage, read> up : array<f16>;

@group(0) @binding(2)
var<storage, read_write> out : array<f16>;

fn silu(x : f16) -> f16 {
    return x / (f16(1.0) + exp(-x));
}

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id)
    gid : vec3<u32>
) {
    let i = gid.x;
    out[i] = silu(gate[i]) * up[i];
}
