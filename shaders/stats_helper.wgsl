// Drop-in WGSL helper: GPU-side min/max/NaN tracking via atomic-on-u32
// with sign-flip-bit-cast trick. Bind at @group(0) @binding(5).
//
// Layout MUST match Rust LogitStats (gpu_stats.rs):
//   3 atomic u32 + 1 pad u32 = 16 bytes.

struct Stats {
    min_val: atomic<u32>,
    max_val: atomic<u32>,
    has_nan: atomic<u32>,
    _pad: u32,
};

// Caller's bind group must place this at binding 5 of group 0.
// Existing kernels (matvec_pc, attention_pc) currently use bindings
// 0..4 — slot 5 is free.
@group(0) @binding(5) var<storage, read_write> stats: Stats;

// Transform f32 -> u32 such that bitwise atomicMin/atomicMax on the u32
// preserve float ordering. See gpu_stats.rs for the inverse.
fn float_to_ordered_u32(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    if ((bits & 0x80000000u) != 0u) {
        // Negative: flip all bits
        return ~bits;
    } else {
        // Non-negative: flip just sign bit
        return bits | 0x80000000u;
    }
}

// Update the stats buffer with a single value.
// Call after computing each output element in your kernel.
fn update_stats(x: f32) {
    if (x != x) {  // NaN check (NaN != NaN is true)
        atomicStore(&stats.has_nan, 1u);
        return;
    }

    // Inf: also flag as nan for our purposes (it's a debug signal)
    if (x > 3.4028235e38 || x < -3.4028235e38) {
        atomicStore(&stats.has_nan, 1u);
        return;
    }

    let v = float_to_ordered_u32(x);

    atomicMin(&stats.min_val, v);
    atomicMax(&stats.max_val, v);
}
