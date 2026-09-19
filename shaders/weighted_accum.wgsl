requires f16;

struct Params {
    weight : f32,
    len : u32,
};

@group(0) @binding(0)
var<storage, read> src : array<f16>;

@group(0) @binding(1)
var<storage, read_write> dst : array<f16>;

@group(0) @binding(2)
var<uniform> params : Params;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id)
    gid : vec3<u32>
) {
    let i = gid.x;

    if (i >= params.len) {
        return;
    }

    dst[i] =
        dst[i] +
        f16(params.weight) * src[i];
}
