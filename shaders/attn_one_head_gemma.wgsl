// shaders/attn_one_head_gemma.wgsl
//
// Single-head decode-step attention with optional sliding-window masking.
// One workgroup processes one query head. Each thread cooperates on the
// kv_len axis, then reduces.
//
// Inputs:
//   Q          : array<f32>          shape [n_heads * head_dim]
//   K_cache    : array<f32>          shape [seq_len, n_kv_heads, head_dim]
//                                    layout flat: [pos * kv_stride + kv_off + d]
//   V_cache    : array<f32>          shape [seq_len, n_kv_heads, head_dim]
//   out        : array<f32>          shape [n_heads * head_dim]
//
// Push:
//   head_dim, n_heads, n_kv_heads, heads_per_kv  (uint)
//   seq_len, window_start, _pad0, _pad1          (uint)
//   scale (f32) — Gemma-4 uses 1.0, not 1/sqrt(d)
//
// Pascal-safe (f32 only). Workgroup size 256.
//
// IMPORTANT: this shader does NOT clear `out` first; it writes the head
// slot directly. The caller pre-zeros the `out` buffer if needed.
// Causal mask is implicit because seq_len is the kv_len at decode time
// (the current token's K/V is the last entry, so no positions > current).

struct Push {
    head_dim:      u32,
    n_heads:       u32,
    n_kv_heads:    u32,
    heads_per_kv:  u32,

    seq_len:       u32,
    window_start:  u32,   // ignored when use_swa = 0
    use_swa:       u32,   // 1 → mask positions < window_start
    _pad0:         u32,

    scale:         f32,   // Gemma-4: 1.0 (not 1/sqrt(head_dim))
    _pad1:         f32,
    _pad2:         f32,
    _pad3:         f32,
};

@group(0) @binding(0) var<storage, read>       q       : array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache : array<f32>;
@group(0) @binding(2) var<storage, read>       v_cache : array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf : array<f32>;
@group(0) @binding(4) var<uniform>             push    : Push;

const TG: u32 = 256u;

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_sum: array<f32, 256>;
// shared per-position scratch large enough for a few-thousand-token window.
// For Gemma 4 SWA this is ≤1024, so 4096 is plenty headroom. Tokens past
// 4096 in the global-attention case fall back to no-shared-mem scratch
// (we re-read the score from K each pass — slower but correct).
var<workgroup> sh_scores: array<f32, 4096>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let head = wid.x;
    if (head >= push.n_heads) { return; }

    let kv_head = head / push.heads_per_kv;
    let q_base  = head * push.head_dim;
    let kv_off  = kv_head * push.head_dim;
    let kv_stride = push.n_kv_heads * push.head_dim;

    let use_window = push.use_swa != 0u;
    let win_start  = push.window_start;
    let use_shared = push.seq_len <= 4096u;

    // ── Phase 1: per-position scaled dot product, find local max ─────────
    var local_max: f32 = -3.4e38;
    var p: u32 = tid;
    while (p < push.seq_len) {
        var score: f32;
        if (use_window && p < win_start) {
            score = -3.4e38;
        } else {
            // dot(q, k_cache[p, kv_head, :])
            var dot: f32 = 0.0;
            let k_base = p * kv_stride + kv_off;
            var d: u32 = 0u;
            loop {
                if (d >= push.head_dim) { break; }
                dot = dot + q[q_base + d] * k_cache[k_base + d];
                d = d + 1u;
            }
            score = dot * push.scale;
        }
        if (use_shared) {
            sh_scores[p] = score;
        }
        if (score > local_max) { local_max = score; }
        p = p + TG;
    }
    sh_max[tid] = local_max;
    workgroupBarrier();

    // Reduce max across workgroup.
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_max[tid] = max(sh_max[tid], sh_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = sh_max[0];

    // ── Phase 2: exp(score - max), row sum ───────────────────────────────
    var local_sum: f32 = 0.0;
    p = tid;
    while (p < push.seq_len) {
        var score: f32;
        if (use_shared) {
            score = sh_scores[p];
        } else {
            // Recompute (slow path).
            if (use_window && p < win_start) {
                score = -3.4e38;
            } else {
                var dot: f32 = 0.0;
                let k_base = p * kv_stride + kv_off;
                var d: u32 = 0u;
                loop {
                    if (d >= push.head_dim) { break; }
                    dot = dot + q[q_base + d] * k_cache[k_base + d];
                    d = d + 1u;
                }
                score = dot * push.scale;
            }
        }
        let shifted = score - row_max;
        let e = select(0.0, exp(shifted), shifted > -70.0);
        if (use_shared) {
            sh_scores[p] = e;   // store exp for phase 3
        }
        local_sum = local_sum + e;
        p = p + TG;
    }
    sh_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce sum.
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv_sum = select(0.0, 1.0 / sh_sum[0], sh_sum[0] > 1e-12);

    // ── Phase 3: weighted sum of V ────────────────────────────────────────
    // Each thread owns a slice of the head_dim axis.
    var dim: u32 = tid;
    while (dim < push.head_dim) {
        var acc: f32 = 0.0;
        var pp: u32 = 0u;
        while (pp < push.seq_len) {
            var w: f32;
            if (use_shared) {
                w = sh_scores[pp] * inv_sum;
            } else {
                // Recompute again. Slow.
                if (use_window && pp < win_start) {
                    w = 0.0;
                } else {
                    var dot: f32 = 0.0;
                    let k_base = pp * kv_stride + kv_off;
                    var d: u32 = 0u;
                    loop {
                        if (d >= push.head_dim) { break; }
                        dot = dot + q[q_base + d] * k_cache[k_base + d];
                        d = d + 1u;
                    }
                    let shifted = dot * push.scale - row_max;
                    let e = select(0.0, exp(shifted), shifted > -70.0);
                    w = e * inv_sum;
                }
            }
            if (w != 0.0) {
                acc = acc + w * v_cache[pp * kv_stride + kv_off + dim];
            }
            pp = pp + 1u;
        }
        out_buf[q_base + dim] = acc;
        dim = dim + TG;
    }
}
