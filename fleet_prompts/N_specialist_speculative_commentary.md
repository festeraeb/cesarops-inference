Specialist's commentary on the speculative rewrite (NOT the code itself).
Useful as integration context. Speculative.rs FILE STILL PENDING.

---

# Architecture summary (specialist)

Your speculative system is 3 coupled subsystems, not one file:

1. **Draft generator** (cheap model)
   - Located in: TransformerDecoder::forward() + SpeculativeDecoder::draft_propose()
   - Role: produces K-step rollout, never verified yet, drives entropy reduction

2. **Main verifier** (expensive model)
   - Located in: SpeculativeDecoder::main_verify_batch() + TransformerDecoder::forward()
   - Role: re-evaluates every proposed draft token in-context, produces K dependent logits sequences, defines correctness boundary

3. **Rejection + repair sampler** (critical missing piece)
   - Located in: sample_repair_distribution()
   - Role: defines correction distribution: max(0, p_main - p_draft) — this is what makes speculative decoding not just sampling, but probability correction under model disagreement

# What most people miss (and your engine now fixes)

NOT: "draft model generates tokens, main model accepts them"
IS: "draft model proposes a trajectory, main model evaluates probability drift across that trajectory, and you correct only at divergence points using residual probability mass"

# Where this connects into the GPU stack

This file is CPU orchestration only. Depends on:
- GPU-side execution (matvec_*, attention_pc.wgsl, softmax.wgsl, stats_helper.wgsl)
- KV cache interaction (rollback on rejection, commit on acceptance)

Your speculative file now correctly enforces:
  draft_kv.snapshot → rollback on rejection
  main_kv.snapshot → rollback on rejection

Prevents: drift desync, corrupted context, silent divergence bugs.

# What this file expects from the engine

1. ModelWeights exists and is shared as Arc<ModelWeights>
2. TransformerDecoder is stateless except KV
3. KV cache rollback is semantically valid (rollback_to + commit_through are logical, not physical deletes)

# Performance note (important)

Currently:
- ❌ correctness-first
- ❌ not speed-optimized

Because main_verify_batch() is still sequential K forwards. No batched KV reuse. No attention reuse between drafts.

Real next upgrade (the t/s jump):
  for (token, _) in drafts {
      self.main.forward(...)
  }
Replace with KV-augmented prefix reuse:
  - shared prefix KV
  - only recompute suffix
  - avoid recomputing full attention per step
  → 2-4× decode speed improvement

# Status

Specialist says the rewritten file is:
  ✔ mathematically correct
  ✔ deterministic (seeded RNG)
  ✔ rollback-safe KV consistent
  ✔ parity-testable at temp=0
  ✔ compatible with CPU + GPU hybrid runtime
  ✔ ready for batched forward upgrade

But the actual `src/speculative.rs` source code was NOT pasted in this
delivery. Need to ask specialist to paste the file.

---

# Action when source arrives

Apply our locked overrides per the cluster-drift lesson:
- Verify softmax uses online max-subtract (4th-strike pattern)
- Verify GQA-aware KV indexing if it touches K/V layout
- Verify SubmissionIndex sync on any reset paths
- Cross-check ModelWeights field is Arc<ModelWeights> not Arc<Box<ModelWeights>>
- Confirm seed is exposed as u64 in SpecConfig

Then:
1. Drop into src/speculative.rs (replace existing 148-LOC stub)
2. Save the parity test from specialist (already stashed at
   N_specialist_speculative_parity_test.md) into tests/speculative_parity.rs
3. Adapt the parity test's KvCache scope per polish note 1 (move KvCache
   outside the for-loop, share across both decoder + spec)
4. cargo build, cargo test --test speculative_parity
5. Smoke test gate
