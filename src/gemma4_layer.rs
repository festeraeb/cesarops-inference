//! Single-token transformer layer for Gemma-4-26B-MoE (dense FFN bypass).
//!
//! This file implements ONE Gemma-4 transformer block for the hot path:
//! single-token decode against a growing KV cache. The MoE expert branch
//! (gate_inp + gate_up_exps + down_exps + top-k routing) is intentionally
//! **not** built here — that lives in a sibling module.
//!
//! Quirks vs. the existing Qwen transformer (`transformer.rs`):
//!   * Per-layer head_dim — full-attention layers use 512, SWA layers use 256.
//!   * Dual RoPE base — 1e6 for full attention, 1e4 for sliding window.
//!   * Per-head RMSNorm on Q and K BEFORE rope (`attn_q_norm`, `attn_k_norm`).
//!   * Weightless RMSNorm on V (no learned scale; mirrors candle's
//!     `gemma4::text::v_norm`).
//!   * `post_attention_norm` is applied AFTER O projection, BEFORE residual add.
//!   * Sliding-window mask: token at position p attends to `[max(0, p-1024), p]`.
//!   * `layer_output_scale`: per-layer fp32 scalar multiplied into each
//!     residual delta before the add.
//!   * Gemma-style RMSNorm: `(x / rms(x)) * (weight + 1.0)`.
//!
//! Weight quantization:
//!   * Q/K/V/O, gate, up — IQ4_XS (qtype 23).
//!   * down — IQ4_NL (qtype 20).
//!
//! GPU strategy:
//!   * All seven big projections are dispatched through `Iq4MatvecPipeline`
//!     against raw quant byte buffers uploaded once at model load.
//!   * Norms are tiny fp32 vectors (2k–8k floats); we keep them CPU-side and
//!     dequant on load via `bridge::dequantize_tensor`.
//!   * Attention dot/mask/softmax/value runs on CPU in this dense bypass —
//!     same shape as `transformer.rs::TransformerDecoder::forward`. A GPU
//!     attention shader can replace it later without touching this file's
//!     external interface.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::bridge;
use crate::iq4_pipeline::{Iq4MatvecPipeline, MatvecPush, QuantKind};
use crate::kv_cache::KvCache;
use crate::loader::{ModelWeights, TensorRegion};

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Per-model Gemma-4 hyperparameters that don't vary per layer.
///
/// Values are pulled from GGUF metadata at load time. The dense FFN
/// `intermediate_size_dense` (= 2112 for Gemma-4-26B-MoE) is the bypass
/// width used by every layer's `ffn_gate / ffn_up / ffn_down` triplet.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim_full: usize,            // 512 — full-attention head width
    pub head_dim_swa: usize,             // 256 — sliding-window head width
    pub intermediate_size_dense: usize,  // 2112 — dense FFN bypass width
    pub rope_theta_full: f32,            // 1e6
    pub rope_theta_swa: f32,             // 1e4
    /// Global (full-attention) layers use proportional RoPE: only the first
    /// `partial_rotary_factor * head_dim / 2` pair indices rotate (candle default 0.25).
    pub partial_rotary_factor: f32,
    pub sliding_window: usize,           // 1024
    pub rms_norm_eps: f32,
    /// Attention logit scale before softmax. Gemma 4 sets this to 1.0 in
    /// llama.cpp (`f_attention_scale`); it does **not** use `1/sqrt(head_dim)`.
    pub attention_scale: f32,
}

impl Gemma4Config {
    /// Resolve `(head_dim, rope_theta)` for a layer based on its SWA flag.
    #[inline]
    pub fn layer_geometry(&self, is_swa: bool) -> (usize, f32) {
        if is_swa {
            (self.head_dim_swa, self.rope_theta_swa)
        } else {
            (self.head_dim_full, self.rope_theta_full)
        }
    }
}

/// One Gemma-4 transformer block, with all projection weights pre-uploaded
/// to the GPU as raw IQ4_XS / IQ4_NL byte streams. Norms live on the CPU.
///
/// The struct is constructed once at model load via [`Gemma4Layer::load`]
/// and consumed many times via [`Gemma4Layer::forward`].
pub struct Gemma4Layer {
    pub layer_idx: usize,
    pub is_swa: bool,
    /// Per-layer KV head count — Gemma-4's `attention.head_count_kv` array
    /// can vary per layer (currently the same 8 across the 30 layers in the
    /// 26B-MoE checkpoint, but we honor it as variable).
    pub n_kv_heads: usize,

    // ── CPU-side norm weights (tiny fp32 tensors, dequantized once) ────────
    /// `[hidden_size]` — pre-attention RMSNorm.
    pub attn_norm: Vec<f32>,
    /// `[head_dim]` — applied per-head to Q after projection, before rope.
    pub attn_q_norm: Vec<f32>,
    /// `[head_dim]` — applied per-head to K after projection, before rope.
    pub attn_k_norm: Vec<f32>,
    /// `[hidden_size]` — applied to attention output BEFORE the residual add.
    pub post_attention_norm: Vec<f32>,
    /// `[hidden_size]` — pre-FFN RMSNorm (the `pre_feedforward_layernorm` in
    /// HF / candle naming; GGUF stores this as `ffn_norm.weight`).
    pub ffn_norm: Vec<f32>,
    /// `[hidden_size]` — post-FFN RMSNorm, applied to FFN output BEFORE
    /// the second residual add.
    pub post_ffw_norm: Vec<f32>,
    /// Per-layer scalar that multiplies the residual delta on every add.
    pub layer_output_scale: f32,

    // ── GPU-resident projection weights (raw quant bytes) ─────────────────
    /// `attn_q.weight` — IQ4_XS, shape `[hidden, q_dim]` in row-major.
    pub q_w: wgpu::Buffer,
    /// `attn_k.weight` — IQ4_XS, shape `[hidden, kv_dim]`.
    pub k_w: wgpu::Buffer,
    /// `attn_v.weight` — IQ4_XS, shape `[hidden, kv_dim]`.
    pub v_w: wgpu::Buffer,
    /// True for shared-KV layers (every 6th layer in Gemma-4-26B-MoE).
    /// When true, V is computed from the K projection instead of a separate V weight.
    pub uses_shared_kv: bool,
    /// `attn_output.weight` — IQ4_XS, shape `[q_dim, hidden]`.
    pub o_w: wgpu::Buffer,
    /// `ffn_gate.weight` — IQ4_XS, shape `[hidden, intermediate_size_dense]`.
    pub gate_w: wgpu::Buffer,
    /// `ffn_up.weight` — IQ4_XS, shape `[hidden, intermediate_size_dense]`.
    pub up_w: wgpu::Buffer,
    /// `ffn_down.weight` — IQ4_NL, shape `[intermediate_size_dense, hidden]`.
    pub down_w: wgpu::Buffer,

    /// Cached projection geometry — derived from `cfg` and `is_swa` once.
    q_dim: usize,
    kv_dim: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction (model load path)
// ─────────────────────────────────────────────────────────────────────────────

impl Gemma4Layer {
    /// Load one Gemma-4 layer from a memory-mapped GGUF model.
    ///
    /// All projection tensors are uploaded to GPU buffers as **raw quant
    /// bytes** (no fp32 dequant intermediate). Norm weights are dequantized
    /// to fp32 once and kept on the CPU heap.
    ///
    /// Fails loudly (`Err`) if any required tensor is missing — we never
    /// silently substitute zeros for a layer weight.
    pub fn load(
        layer_idx: usize,
        is_swa: bool,
        n_kv_heads: usize,
        cfg: &Gemma4Config,
        weights: &ModelWeights,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        let (head_dim, _) = cfg.layer_geometry(is_swa);
        let q_dim = cfg.num_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // ── Norms (CPU fp32) ───────────────────────────────────────────────
        let attn_norm = load_norm(weights, &name(layer_idx, "attn_norm"), cfg.hidden_size)?;
        let attn_q_norm = load_norm(weights, &name(layer_idx, "attn_q_norm"), head_dim)?;
        let attn_k_norm = load_norm(weights, &name(layer_idx, "attn_k_norm"), head_dim)?;
        let post_attention_norm =
            load_norm(weights, &name(layer_idx, "post_attention_norm"), cfg.hidden_size)?;
        let ffn_norm = load_norm(weights, &name(layer_idx, "ffn_norm"), cfg.hidden_size)?;
        let post_ffw_norm =
            load_norm(weights, &name(layer_idx, "post_ffw_norm"), cfg.hidden_size)?;

        // HONEST GAP: the Gemma-4-26B-MoE checkpoint also exposes
        //   blk.{i}.post_ffw_norm_1.weight
        //   blk.{i}.post_ffw_norm_2.weight
        //   blk.{i}.pre_ffw_norm_2.weight
        // These do NOT appear in the candle reference (gemma4/text.rs uses
        // only pre_feedforward_layernorm + post_feedforward_layernorm) and
        // the spec doc does not pin their order. They almost certainly bracket
        // the MoE expert path (gate_up_exps → down_exps), which is built by
        // the sibling MoE agent. Keeping them out of this dense-bypass layer
        // is the conservative choice; if a future spec update places them
        // around the dense FFN too, wire them in here at the same call sites
        // as ffn_norm / post_ffw_norm. Flagged for human review.

        let layer_output_scale =
            load_scalar(weights, &name(layer_idx, "layer_output_scale"))?;

        // ── Projection weights (GPU, raw quant bytes) ─────────────────────
        let q_w = upload_quant(
            weights,
            &name(layer_idx, "attn_q"),
            QuantKind::Iq4Xs,
            cfg.hidden_size,
            q_dim,
            device,
            queue,
            "gemma4.attn_q",
        )?;
        let k_w = upload_quant(
            weights,
            &name(layer_idx, "attn_k"),
            QuantKind::Iq4Xs,
            cfg.hidden_size,
            kv_dim,
            device,
            queue,
            "gemma4.attn_k",
        )?;
        // Gemma-4 shared-KV layers (every 6th starting at 5) have no
        // separate attn_v.weight — they reuse the K buffer as V.
        // We detect this by checking tensor presence and set a flag.
        let has_v = weights.tensors.contains_key(&name(layer_idx, "attn_v"));
        let v_w = if has_v {
            upload_quant(
                weights,
                &name(layer_idx, "attn_v"),
                QuantKind::Iq4Xs,
                cfg.hidden_size,
                kv_dim,
                device,
                queue,
                "gemma4.attn_v",
            )?
        } else {
            // Shared-KV layer: create a dummy 4-byte buffer as placeholder.
            // The forward pass will use the K buffer for V when uses_shared_kv=true.
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gemma4.attn_v_shared_placeholder"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let o_w = upload_quant(
            weights,
            &name(layer_idx, "attn_output"),
            QuantKind::Iq4Xs,
            q_dim,
            cfg.hidden_size,
            device,
            queue,
            "gemma4.attn_output",
        )?;
        let gate_w = upload_quant(
            weights,
            &name(layer_idx, "ffn_gate"),
            QuantKind::Iq4Xs,
            cfg.hidden_size,
            cfg.intermediate_size_dense,
            device,
            queue,
            "gemma4.ffn_gate",
        )?;
        let up_w = upload_quant(
            weights,
            &name(layer_idx, "ffn_up"),
            QuantKind::Iq4Xs,
            cfg.hidden_size,
            cfg.intermediate_size_dense,
            device,
            queue,
            "gemma4.ffn_up",
        )?;
        let down_w = upload_quant(
            weights,
            &name(layer_idx, "ffn_down"),
            QuantKind::Iq4Nl,
            cfg.intermediate_size_dense,
            cfg.hidden_size,
            device,
            queue,
            "gemma4.ffn_down",
        )?;

        Ok(Self {
            layer_idx,
            is_swa,
            n_kv_heads,
            attn_norm,
            attn_q_norm,
            attn_k_norm,
            post_attention_norm,
            ffn_norm,
            post_ffw_norm,
            layer_output_scale,
            q_w,
            k_w,
            v_w,
            uses_shared_kv: !has_v,
            o_w,
            gate_w,
            up_w,
            down_w,
            q_dim,
            kv_dim,
        })
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    /// Attention sub-block only (steps 1–8). MoE models run a separate expert
    /// FFN path after this; do not call [`Self::forward_dense_ffn`] on 26B-MoE.
    pub fn forward_attention(
        &self,
        cfg: &Gemma4Config,
        hidden_state: &mut Vec<f32>,
        position: usize,
        kv_cache: &mut KvCache,
        iq4xs_pipe: &Iq4MatvecPipeline,
        layer_idx: usize,
        scratch: &Iq4MatvecScratch,
        io: &mut MatvecIo,
    ) {
        debug_assert_eq!(iq4xs_pipe.kind, QuantKind::Iq4Xs);
        debug_assert_eq!(hidden_state.len(), cfg.hidden_size);

        let (head_dim, rope_theta) = cfg.layer_geometry(self.is_swa);
        let h = cfg.hidden_size;
        let n_heads = cfg.num_heads;
        let n_kv = self.n_kv_heads;
        let heads_per_kv = n_heads / n_kv;
        let eps = cfg.rms_norm_eps;

        // Snapshot residual for the attention block.
        let mut residual = hidden_state.clone();
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "pre_attn", &residual);

        // ── 1. attn_norm — llama.cpp build_norm(RMS) = rms * raw weight
        // (NOT weight+1; that is Gemma-1/2/3 HF convention). Toggle via
        // GEMMA4_NORM_PLUS_ONE=1 only for experiments.
        rmsnorm_gemma_inplace(hidden_state, &self.attn_norm, eps);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_attn_norm", hidden_state);

        // ── 2. Q / K / V projections (GPU, raw IQ4_XS) ────────────────────
        let v_w = if self.uses_shared_kv {
            None
        } else {
            Some(&self.v_w)
        };
        matvec_iq4_qkv(
            iq4xs_pipe,
            &self.q_w,
            &self.k_w,
            v_w,
            hidden_state,
            h,
            self.q_dim,
            self.kv_dim,
            scratch,
            &mut io.q,
            &mut io.k,
            &mut io.v,
        );
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_q_proj", &io.q);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_k_proj", &io.k);

        // ── 3. Per-head RMSNorm on Q and K (Gemma-style), weightless on V ─
        for head in 0..n_heads {
            let off = head * head_dim;
            rmsnorm_gemma_inplace(&mut io.q[off..off + head_dim], &self.attn_q_norm, eps);
        }
        for kvh in 0..n_kv {
            let off = kvh * head_dim;
            rmsnorm_gemma_inplace(&mut io.k[off..off + head_dim], &self.attn_k_norm, eps);
            rmsnorm_weightless_inplace(&mut io.v[off..off + head_dim], eps);
        }
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_q_norm", &io.q);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_k_norm", &io.k);

        // ── 4. RoPE (per-layer base + type) ───────────────────────────────
        // SWA layers: standard full RoPE. Global layers: proportional RoPE
        // (rotate only the first partial_rotary_factor slice — see candle
        // `ProportionalRotaryEmbedding` in gemma4/text.rs).
        if self.is_swa {
            apply_rope_full(&mut io.q, n_heads, head_dim, position, rope_theta);
            apply_rope_full(&mut io.k, n_kv, head_dim, position, rope_theta);
        } else {
            apply_rope_proportional(
                &mut io.q,
                n_heads,
                head_dim,
                position,
                rope_theta,
                cfg.partial_rotary_factor,
            );
            apply_rope_proportional(
                &mut io.k,
                n_kv,
                head_dim,
                position,
                rope_theta,
                cfg.partial_rotary_factor,
            );
        }
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_q_rope", &io.q);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_k_rope", &io.k);
        if !io.v.is_empty() {
            crate::gemma4_runner::moe_slice_log(
                layer_idx,
                position,
                "post_v_h0",
                &io.v[..head_dim.min(io.v.len())],
            );
        }

        // ── 5. Push K/V into the cache and run causal (+SWA) attention ────
        kv_cache.push(self.layer_idx, io.k.clone(), io.v.clone());

        let cached_keys = kv_cache.get_keys(self.layer_idx);
        let cached_values = kv_cache.get_values(self.layer_idx);
        let seq_len = cached_keys.len(); // includes the token we just pushed

        // SWA window: token at position p attends only to positions
        // [max(0, p - sliding_window), p]. seq_len-1 is the new token's pos
        // within this cache.
        let window_start = if self.is_swa {
            (seq_len - 1).saturating_sub(cfg.sliding_window)
        } else {
            0
        };

        let mut attn_out = vec![0.0f32; self.q_dim];
        let scale = cfg.attention_scale;

        for head in 0..n_heads {
            let kv_head = head / heads_per_kv;
            let q_off = head * head_dim;
            let kv_off = kv_head * head_dim;

            // Attention scores against all in-window past positions.
            let mut scores = Vec::with_capacity(seq_len);
            let mut max_score = f32::NEG_INFINITY;
            for pos in 0..seq_len {
                if pos < window_start {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let k_vec = &cached_keys[pos];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += io.q[q_off + d] * k_vec[kv_off + d];
                }
                let s = dot * scale;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }

            // Softmax with numerical stabilization.
            let slice_attn = head == 0
                && crate::gemma4_runner::moe_slice_active(layer_idx, position);
            let raw_scores = if slice_attn {
                Some(scores.clone())
            } else {
                None
            };
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                if s.is_finite() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                } else {
                    *s = 0.0;
                }
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

            if slice_attn {
                let probs: Vec<f32> = scores.iter().map(|s| s * inv_sum).collect();
                crate::gemma4_runner::moe_slice_log_attn_vec(
                    layer_idx,
                    position,
                    &format!("attn_scores_h0_seq{seq_len}"),
                    raw_scores.as_ref().unwrap(),
                );
                crate::gemma4_runner::moe_slice_log_attn_vec(
                    layer_idx,
                    position,
                    &format!("attn_probs_h0_seq{seq_len}"),
                    &probs,
                );
            }

            // Weighted sum of cached V vectors.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for pos in 0..seq_len {
                    let w = scores[pos] * inv_sum;
                    if w != 0.0 {
                        acc += w * cached_values[pos][kv_off + d];
                    }
                }
                attn_out[q_off + d] = acc;
            }
        }

        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_attn_core", &attn_out);

        // ── 6. O projection (GPU, IQ4_XS) ─────────────────────────────────
        matvec_iq4(iq4xs_pipe, &self.o_w, &attn_out, self.q_dim, h, scratch, &mut io.y);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_o_raw", &io.y);

        // ── 7. post_attention_norm BEFORE residual add ────────────────────
        rmsnorm_gemma_inplace(&mut io.y, &self.post_attention_norm, eps);
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_o_norm", &io.y);

        // ── 8. Residual add → attn_out (layer_output_scale applied once at
        // end of the full layer block in llama.cpp, not here).
        for i in 0..h {
            hidden_state[i] = residual[i] + io.y[i];
        }
        crate::gemma4_runner::moe_slice_log(layer_idx, position, "post_attn", hidden_state);
    }

    /// Dense GEGLU bypass FFN (shared exp MLP for Gemma-4 MoE, or non-MoE
    /// variants).
    pub fn forward_dense_ffn(
        &self,
        cfg: &Gemma4Config,
        hidden_state: &mut Vec<f32>,
        iq4xs_pipe: &Iq4MatvecPipeline,
        iq4nl_pipe: &Iq4MatvecPipeline,
        scratch: &Iq4MatvecScratch,
        io: &mut MatvecIo,
    ) {
        debug_assert_eq!(iq4xs_pipe.kind, QuantKind::Iq4Xs);
        debug_assert_eq!(iq4nl_pipe.kind, QuantKind::Iq4Nl);
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let mut residual = hidden_state.clone();

        rmsnorm_gemma_inplace(hidden_state, &self.ffn_norm, eps);

        let inter = cfg.intermediate_size_dense;
        matvec_iq4_gate_up(
            iq4xs_pipe,
            &self.gate_w,
            &self.up_w,
            hidden_state,
            h,
            inter,
            scratch,
            &mut io.gate,
            &mut io.up,
        );

        let mut ffn_hidden = vec![0.0f32; inter];
        for i in 0..inter {
            let g = io.gate[i];
            let gelu = 0.5 * g * (1.0 + (0.7978845608028654 * (g + 0.044715 * g * g * g)).tanh());
            ffn_hidden[i] = gelu * io.up[i];
        }

        matvec_iq4(iq4nl_pipe, &self.down_w, &ffn_hidden, inter, h, scratch, &mut io.y);
        rmsnorm_gemma_inplace(&mut io.y, &self.post_ffw_norm, eps);

        for i in 0..h {
            hidden_state[i] = residual[i] + self.layer_output_scale * io.y[i];
        }
    }

    /// Compute the shared exp MLP "down" projection only (no residual add,
    /// no RMSNorm, no layer_output_scale), for Gemma-4 MoE where we need
    /// `cur_mlp + cur_moe` before the post-FFN norms.
    ///
    /// `x` must be already normalized (e.g. by `ffn_norm` in llama.cpp).
    pub fn forward_shared_mlp_down_only(
        &self,
        cfg: &Gemma4Config,
        x: &[f32],
        iq4xs_pipe: &Iq4MatvecPipeline,
        iq4nl_pipe: &Iq4MatvecPipeline,
        scratch: &Iq4MatvecScratch,
        io: &mut MatvecIo,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), cfg.hidden_size);
        debug_assert_eq!(iq4xs_pipe.kind, QuantKind::Iq4Xs);
        debug_assert_eq!(iq4nl_pipe.kind, QuantKind::Iq4Nl);

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size_dense;
        let gelu_approx = |g: f32| 0.5 * g
            * (1.0
                + (0.7978845608028654 * (g + 0.044715 * g * g * g)).tanh());

        // gate/up projections on x
        matvec_iq4_gate_up(
            iq4xs_pipe,
            &self.gate_w,
            &self.up_w,
            x,
            h,
            inter,
            scratch,
            &mut io.gate,
            &mut io.up,
        );

        // GEGLU hidden: gelu(gate) * up
        let mut ffn_hidden = vec![0.0f32; inter];
        for i in 0..inter {
            ffn_hidden[i] = gelu_approx(io.gate[i]) * io.up[i];
        }

        matvec_iq4(iq4nl_pipe, &self.down_w, &ffn_hidden, inter, h, scratch, &mut io.y);
        io.y.clone()
    }

    /// Full dense block: attention + dense FFN bypass (non-MoE variants).
    pub fn forward(
        &self,
        cfg: &Gemma4Config,
        hidden_state: &mut Vec<f32>,
        position: usize,
        kv_cache: &mut KvCache,
        iq4xs_pipe: &Iq4MatvecPipeline,
        iq4nl_pipe: &Iq4MatvecPipeline,
        scratch: &Iq4MatvecScratch,
        io: &mut MatvecIo,
    ) {
        self.forward_attention(
            cfg,
            hidden_state,
            position,
            kv_cache,
            iq4xs_pipe,
            self.layer_idx,
            scratch,
            io,
        );
        self.forward_dense_ffn(cfg, hidden_state, iq4xs_pipe, iq4nl_pipe, scratch, io);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — tensor loading
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn name(layer_idx: usize, suffix: &str) -> String {
    format!("blk.{}.{}.weight", layer_idx, suffix)
}

/// Load an fp32 norm weight from the GGUF file. Fails loudly if the tensor
/// is missing or the wrong size — never substitutes zeros.
fn load_norm(
    weights: &ModelWeights,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, String> {
    let region = weights
        .tensors
        .get(name)
        .ok_or_else(|| format!("Missing tensor `{}`", name))?;
    let bytes = weights
        .tensor_bytes(name)
        .ok_or_else(|| format!("Tensor `{}` has no mapped bytes", name))?;
    let n_elements: usize = region.shape.iter().product();
    let data = bridge::dequantize_tensor(bytes, region.quant_type, n_elements);
    if data.len() < expected_elements {
        return Err(format!(
            "Tensor `{}` produced {} elements but layer expects {}",
            name,
            data.len(),
            expected_elements
        ));
    }
    let mut out = data;
    out.truncate(expected_elements);
    Ok(out)
}

/// Load a single fp32 scalar. `layer_output_scale` is shape `[1]`.
fn load_scalar(weights: &ModelWeights, name: &str) -> Result<f32, String> {
    let v = load_norm(weights, name, 1)?;
    Ok(v[0])
}

/// Upload raw quant bytes for a projection tensor to a GPU storage buffer.
///
/// The buffer layout is exactly the GGUF on-disk byte stream — the IQ4_XS /
/// IQ4_NL shaders read it as `array<u32>` and synthesize byte-level fields
/// via shift+mask. We pad the upload up to a multiple of 4 bytes so the
/// `array<u32>` view never reads past the buffer end.
fn upload_quant(
    weights: &ModelWeights,
    name: &str,
    expected_kind: QuantKind,
    k: usize,
    n: usize,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    label: &'static str,
) -> Result<wgpu::Buffer, String> {
    let region: &TensorRegion = weights
        .tensors
        .get(name)
        .ok_or_else(|| format!("Missing tensor `{}`", name))?;

    let expected_qtype = match expected_kind {
        QuantKind::Iq4Xs => 23u32,
        QuantKind::Iq4Nl => 20u32,
    };
    if region.quant_type != expected_qtype {
        return Err(format!(
            "Tensor `{}`: expected quant type {} ({:?}), got {} — use a uniform IQ4_XS GGUF (not bartowski mixed-quant)",
            name, expected_qtype, expected_kind, region.quant_type
        ));
    }

    // Sanity-check shape against the (k, n) the layer is wiring up. GGUF
    // stores tensors as N rows of K elements (shape `[K, N]`).
    if region.shape.len() != 2 || region.shape[0] != k || region.shape[1] != n {
        return Err(format!(
            "Tensor `{}`: shape {:?} mismatches layer geometry [{}, {}]",
            name, region.shape, k, n
        ));
    }

    let expected_bytes = expected_kind.row_bytes(k) * n;
    if region.size != expected_bytes {
        return Err(format!(
            "Tensor `{}`: declared size {} bytes mismatches expected {} \
             bytes for {:?} at K={}, N={}",
            name, region.size, expected_bytes, expected_kind, k, n
        ));
    }

    let bytes = weights
        .tensor_bytes(name)
        .ok_or_else(|| format!("Tensor `{}` has no mapped bytes", name))?;

    // wgpu storage buffers viewed as `array<u32>` need a 4-byte aligned
    // length. Pad with zeros if the raw byte stream falls short.
    let padded_len = (bytes.len() + 3) & !3;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: padded_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    if padded_len > bytes.len() {
        let pad = vec![0u8; padded_len - bytes.len()];
        queue.write_buffer(&buffer, bytes.len() as u64, &pad);
    }
    Ok(buffer)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — GPU matvec (one projection)
// ─────────────────────────────────────────────────────────────────────────────

/// Reusable CPU output buffers for batched IQ4 matvec calls.
pub struct MatvecIo {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub y: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl MatvecIo {
    pub fn new() -> Self {
        Self {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            y: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
        }
    }
}

/// Reusable GPU buffers for IQ4 matvec — avoids per-projection allocation.
pub struct Iq4MatvecScratch {
    x_buf: wgpu::Buffer,
    y_bufs: [wgpu::Buffer; 3],
    staging: wgpu::Buffer,
    staging_bytes: u64,
    max_rows: usize,
}

impl Iq4MatvecScratch {
    pub fn new(device: &wgpu::Device, max_k: usize, max_rows: usize) -> Self {
        let x_bytes = (max_k * 4) as u64;
        let y_bytes = (max_rows * 4) as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            x_buf: mk("iq4_x", x_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            y_bufs: [
                mk("iq4_y0", y_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
                mk("iq4_y1", y_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
                mk("iq4_y2", y_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            ],
            staging: mk("iq4_staging", y_bytes, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            staging_bytes: y_bytes,
            max_rows,
        }
    }
}

fn matvec_iq4_dispatch(
    pipe: &Iq4MatvecPipeline,
    encoder: &mut wgpu::CommandEncoder,
    w: &wgpu::Buffer,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    k: usize,
    n: usize,
) -> u64 {
    let push_buf = pipe.make_push_buffer(MatvecPush {
        k: k as u32,
        n_rows_total: n as u32,
        row_offset: 0,
        _pad: 0,
    });
    pipe.dispatch(encoder, w, x_buf, y_buf, &push_buf, n as u32);
    (n * std::mem::size_of::<f32>()) as u64
}

fn read_back_y(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    y_buf: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    mut encoder: wgpu::CommandEncoder,
    y_bytes: u64,
    out: &mut [f32],
) {
    encoder.copy_buffer_to_buffer(y_buf, 0, staging, 0, y_bytes);
    let sub_idx = queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
    let _ = rx.recv();
    let mapped = slice.get_mapped_range();
    out.copy_from_slice(bytemuck::cast_slice(&mapped[..y_bytes as usize]));
    drop(mapped);
    staging.unmap();
}

/// Single IQ4 matvec with scratch reuse.
pub fn matvec_iq4(
    pipe: &Iq4MatvecPipeline,
    w: &wgpu::Buffer,
    input: &[f32],
    k: usize,
    n: usize,
    scratch: &Iq4MatvecScratch,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(input.len(), k);
    debug_assert!(n <= scratch.max_rows);
    out.resize(n, 0.0);

    let device = &pipe.device;
    let queue = &pipe.queue;
    queue.write_buffer(&scratch.x_buf, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_matvec_encoder"),
    });
    let y_bytes = matvec_iq4_dispatch(pipe, &mut encoder, w, &scratch.x_buf, &scratch.y_bufs[0], k, n);
    read_back_y(
        device,
        queue,
        &scratch.y_bufs[0],
        &scratch.staging,
        encoder,
        y_bytes,
        out,
    );
}

/// Q/K/V projections sharing one input upload and one queue submit.
pub fn matvec_iq4_qkv(
    pipe: &Iq4MatvecPipeline,
    q_w: &wgpu::Buffer,
    k_w: &wgpu::Buffer,
    v_w: Option<&wgpu::Buffer>,
    input: &[f32],
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    scratch: &Iq4MatvecScratch,
    q: &mut Vec<f32>,
    k: &mut Vec<f32>,
    v: &mut Vec<f32>,
) {
    q.resize(q_dim, 0.0);
    k.resize(kv_dim, 0.0);
    v.resize(kv_dim, 0.0);

    let device = &pipe.device;
    let queue = &pipe.queue;
    queue.write_buffer(&scratch.x_buf, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_qkv_encoder"),
    });
    let q_bytes = matvec_iq4_dispatch(pipe, &mut encoder, q_w, &scratch.x_buf, &scratch.y_bufs[0], h, q_dim);
    let k_bytes = matvec_iq4_dispatch(pipe, &mut encoder, k_w, &scratch.x_buf, &scratch.y_bufs[1], h, kv_dim);
    let v_bytes = if let Some(vw) = v_w {
        Some(matvec_iq4_dispatch(pipe, &mut encoder, vw, &scratch.x_buf, &scratch.y_bufs[2], h, kv_dim))
    } else {
        None
    };
    queue.submit(std::iter::once(encoder.finish()));

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_qkv_readback_q"),
    });
    read_back_y(device, queue, &scratch.y_bufs[0], &scratch.staging, encoder, q_bytes, q);

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_qkv_readback_k"),
    });
    read_back_y(device, queue, &scratch.y_bufs[1], &scratch.staging, encoder, k_bytes, k);

    if let Some(v_bytes) = v_bytes {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("iq4_qkv_readback_v"),
        });
        read_back_y(device, queue, &scratch.y_bufs[2], &scratch.staging, encoder, v_bytes, v);
    } else {
        v.copy_from_slice(k);
    }
}

/// Gate + up projections (same input) in one GPU submit.
pub fn matvec_iq4_gate_up(
    pipe: &Iq4MatvecPipeline,
    gate_w: &wgpu::Buffer,
    up_w: &wgpu::Buffer,
    input: &[f32],
    h: usize,
    inter: usize,
    scratch: &Iq4MatvecScratch,
    gate: &mut Vec<f32>,
    up: &mut Vec<f32>,
) {
    gate.resize(inter, 0.0);
    up.resize(inter, 0.0);

    let device = &pipe.device;
    let queue = &pipe.queue;
    queue.write_buffer(&scratch.x_buf, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_gate_up_encoder"),
    });
    let gate_bytes = matvec_iq4_dispatch(pipe, &mut encoder, gate_w, &scratch.x_buf, &scratch.y_bufs[0], h, inter);
    let up_bytes = matvec_iq4_dispatch(pipe, &mut encoder, up_w, &scratch.x_buf, &scratch.y_bufs[1], h, inter);
    queue.submit(std::iter::once(encoder.finish()));

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_gate_up_readback_gate"),
    });
    read_back_y(
        device,
        queue,
        &scratch.y_bufs[0],
        &scratch.staging,
        encoder,
        gate_bytes,
        gate,
    );

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iq4_gate_up_readback_up"),
    });
    read_back_y(
        device,
        queue,
        &scratch.y_bufs[1],
        &scratch.staging,
        encoder,
        up_bytes,
        up,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — CPU math (norms, rope)
// ─────────────────────────────────────────────────────────────────────────────

/// Gemma-style RMSNorm: `out = (x / sqrt(mean(x^2) + eps)) * (weight + 1.0)`.
/// In-place mutate.
///
/// IMPORTANT: Gemma 1/2/3 use `(weight + 1.0)` because their weights are
/// stored centered around zero (so `0.0` means identity gain). Some Gemma
/// variants and most other models store weights centered around 1.0
/// directly (so `1.0` means identity). The Gemma-4 26B MoE GGUF stores
/// raw weights with magnitudes ~3-9 — using the `+1.0` convention would
/// produce gains of ~5-10×, which corrupts numerics. We default to the
/// raw-weight path for this checkpoint. Toggle via `GEMMA4_NORM_PLUS_ONE=1`
/// env var if a future Gemma-4 stores weights the other way.
fn rmsnorm_gemma_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    let mut sum_sq = 0.0f32;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    let plus_one = std::env::var("GEMMA4_NORM_PLUS_ONE").map(|v| v == "1").unwrap_or(false);
    if plus_one {
        for i in 0..n {
            x[i] = x[i] * inv_rms * (weight[i] + 1.0);
        }
    } else {
        for i in 0..n {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }
}

/// Weightless RMSNorm — used on V and on the MoE router input per llama.cpp
/// `gemma4.cpp` (`ggml_rms_norm(attn_out)` before router matmul).
pub(crate) fn rmsnorm_weightless_inplace(x: &mut [f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
    for xi in x.iter_mut() {
        *xi *= inv_rms;
    }
}

/// Standard RoPE on every adjacent pair `(x[2p], x[2p+1])` per head.
fn apply_rope_full(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let n_rot_pairs = head_dim / 2;
    for head in 0..n_heads {
        let off = head * head_dim;
        rotate_rope_pairs(x, off, n_rot_pairs, head_dim, pos, theta);
    }
}

/// Proportional RoPE for global-attention layers (Gemma 4).
/// Only the first `partial_rotary_factor * head_dim / 2` pair indices rotate;
/// remaining dimensions are left untouched (identity), matching candle's
/// zero-padded inv_freq tail in `ProportionalRotaryEmbedding`.
fn apply_rope_proportional(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    partial_rotary_factor: f32,
) {
    let n_rot_pairs = ((partial_rotary_factor * head_dim as f32) / 2.0).floor() as usize;
    let n_rot_pairs = n_rot_pairs.min(head_dim / 2);
    for head in 0..n_heads {
        let off = head * head_dim;
        rotate_rope_pairs(x, off, n_rot_pairs, head_dim, pos, theta);
    }
}

#[inline]
fn rotate_rope_pairs(
    x: &mut [f32],
    off: usize,
    n_rot_pairs: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    for pair in 0..n_rot_pairs {
        let i = pair * 2;
        if i + 1 >= head_dim {
            break;
        }
        let freq = 1.0f32 / theta.powf((2 * pair) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin_v, cos_v) = (angle.sin(), angle.cos());
        let x0 = x[off + i];
        let x1 = x[off + i + 1];
        x[off + i] = x0 * cos_v - x1 * sin_v;
        x[off + i + 1] = x0 * sin_v + x1 * cos_v;
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;

    #[test]
    fn proportional_rope_leaves_tail_identity() {
        let head_dim = 512;
        let factor = 0.25;
        let n_rot = ((factor * head_dim as f32) / 2.0).floor() as usize;
        assert_eq!(n_rot, 64);

        let mut v: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.01).collect();
        let tail: Vec<f32> = v[n_rot * 2..].to_vec();

        apply_rope_proportional(&mut v, 1, head_dim, 3, 1_000_000.0, factor);

        assert_eq!(&v[n_rot * 2..], tail.as_slice());
        assert_ne!(v[0], 0.0);
        assert_ne!(v[1], 0.01);
    }
}

// `Pod`/`Zeroable` are imported but unused at module scope; suppress with a
// no-op static use so callers writing their own push structs can still
// pull them via `gemma4_layer::*` if desired.
#[allow(dead_code)]
const fn _bytemuck_smoke<T: Pod + Zeroable>() {}
