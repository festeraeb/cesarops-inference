# Gemma-4 layer-0 bisect checklist ↔ MOE_SLICE phases

Prompt: token ids `2,236800,236781,236800,236784` (`<bos> 3 x 3 =`)
Dump: `regression/baselines/gemma4_l0_slice_checklist_2026-08-27.log`
Env: `MOE_SLICE=1 MOE_SLICE_LAYER=0 MOE_SLICE_ALL=1 MOE_SLICE_DIMS=16`

| # | Checklist stage | MOE_SLICE phase(s) |
|---|-----------------|--------------------|
| 1 | embedding lookup | `post_embed_raw` |
| 2 | embedding scale (`×√h`) | `post_embed` |
| 3 | pre-attention residual | `pre_attn` |
| 4 | attention RMSNorm | `post_attn_norm` |
| 5 | Q/K/V projections | `post_q_proj`, `post_k_proj`, `post_v_h0` |
| 6 | Q/K normalization | `post_q_norm`, `post_k_norm` |
| 7 | RoPE | `post_q_rope`, `post_k_rope` |
| 8 | attention output | `post_attn_core` (+ `attn_scores_h0_*` / `attn_probs_h0_*`) |
| 9 | O projection | `post_o_raw` → `post_o_norm` |
| 10 | attention residual | `post_attn` / `post_attn_residual` |
| 11 | pre-FFN/MoE norm | `pre_mlp_norm`, `pre_moe_norm`, `pre_router` |
| 12 | router logits + experts + weights | `router_topk_idx=… weights=…` |
| 13 | expert output | `post_moe_expert_raw` → `post_moe_expert` |
| 14 | post-FFN/MoE norm/scaling | `post_mlp`, `post_ffn_sum`, `post_ffn_norm`, `post_layer_residual`, `post_layer` (`× layer_output_scale`) |
| 15 | final layer-0 residual | `post_layer` |

## Snapshot (first-16 rms @ pos=0 / pos=1 / pos=4)

| phase | pos0 rms | pos1 rms | pos4 rms |
|-------|----------|----------|----------|
| post_embed | 1.55 | 1.47 | 1.32 |
| post_attn_norm | 3.20 | **6.78** | 4.29 |
| post_q_proj | 18.6 | **150.6** | 63.1 |
| post_q_norm | 1.02 | 2.66 | 1.67 |
| post_attn | 1.61 | 1.93 | 1.85 |
| post_ffn_norm | 2.32 | **12.26** | 4.45 |
| post_layer | 0.20 | 0.75 | 0.36 |

## Result still wrong
- Generated token id: **14186** (want ≈ **236800** / `"3"`)

## Notes
- Celeste GGUF: `embedding_length_per_layer_input=0` (no per-layer emb path).
- Layer 0 is SWA (`pattern[0]=true`), head_dim=256, n_rot_swa=256, θ=1e4.
- `post_k_norm` ≪ `post_q_norm` (~10×) — likely weight scale; confirm vs llama.
- Hotspot for llama compare: **pos=1 `post_q_proj` → `post_ffn_norm`**.
