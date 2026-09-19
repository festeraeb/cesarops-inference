Your speculative.rs from the prior batch has 4 blocking issues that prevent both compilation and correct decoding. Need a rewrite of just that one file. Everything else from your batch was clean.

Issues that must be fixed:

1. `todo!()` in the weights argument to `decoder.forward()` calls. Won't compile. The forward signature is:
   ```
   fn forward(&self, hidden_state: &mut Vec<f32>, position: usize,
              weights: &ModelWeights, kv_cache: &mut KvCache) -> Vec<f32>
   ```
   You need a `weights: &ModelWeights` reference plumbed through SpeculativeDecoder. Add it as a `weights: Arc<ModelWeights>` field on SpeculativeDecoder OR pass through the generate() signature.

2. Loop processes ONE draft token per iteration. That's not speculative decoding — it's serial decoding with extra steps. The whole point is: propose K tokens with the draft model, verify all K in a single main pass, accept the longest valid prefix, sample one repair token from the rejection-corrected distribution at the first rejection point.

   Required loop shape:
   ```
   while not done:
       drafts = draft_propose(ctx, K=draft_window)   // K tokens with logits
       main_logits_per_pos = main_verify_batch(ctx, drafts)  // K logit vecs
       accepted = []
       for i in 0..K:
           p_main_dist = softmax(main_logits_per_pos[i])
           p_draft_dist = softmax(drafts[i].logits)
           alpha = min(1, p_main_dist[drafts[i].token] /
                          p_draft_dist[drafts[i].token].max(1e-20))
           u = rng.gen()
           if u <= alpha:
               accepted.push(drafts[i].token)
               ctx.push(drafts[i].token)
           else:
               // Sample repair from max(0, p_main - p_draft) renormalized
               repair = sample_repair_distribution(p_main_dist, p_draft_dist, &mut rng)
               ctx.push(repair)
               break
       if accepted.is_empty() and no repair:
           // edge case: K=0 fallback
           single_main_step(ctx)
   ```

3. `rand()` using `nanoseconds % 1000` is not random. Use the `rand` crate (rand = "0.8") with `rand::rngs::StdRng` and a seed parameter on SpecConfig so tests are reproducible. Specifically:
   ```
   pub struct SpecConfig {
       pub draft_window: usize,
       pub temperature: f32,
       pub max_tokens: usize,
       pub seed: u64,
   }
   ```
   And in SpeculativeDecoder add a `rng: rand::rngs::StdRng` field seeded from config.seed.

4. `sample_repair` is a stub returning 0. The correct implementation:
   ```
   fn sample_repair_distribution(
       p_main: &[f32],
       p_draft: &[f32],
       rng: &mut impl Rng,
   ) -> u32 {
       // Compute diff = max(0, p_main - p_draft) elementwise
       let diff: Vec<f32> = p_main.iter().zip(p_draft.iter())
           .map(|(m, d)| (m - d).max(0.0))
           .collect();
       // Renormalize
       let sum: f32 = diff.iter().sum();
       if sum < 1e-20 {
           // Fall back to argmax of p_main
           return argmax(p_main);
       }
       let normalized: Vec<f32> = diff.iter().map(|x| x / sum).collect();
       // Categorical sample
       let r: f32 = rng.gen();
       let mut cum = 0.0;
       for (i, &p) in normalized.iter().enumerate() {
           cum += p;
           if r <= cum { return i as u32; }
       }
       (normalized.len() - 1) as u32
   }
   ```

Additional requirements for the rewrite:

- For the temp=0 (greedy) special case, alpha collapses to 1 if the draft argmax matches the main argmax, else 0. Include that as a fast-path so the parity test (greedy spec output == greedy non-spec output) is exact.

- For `main_verify_batch`: since our existing forward() takes one token at a time, a CORRECT but slow implementation loops K times calling forward() with successive ctx-extensions, capturing each step's logits. Mark this with a `// PERF: replace with batched forward when TransformerDecoder.forward_batch lands` comment. Don't try to implement a real batched forward — that's a separate kernel project.

- Online softmax max-subtract is mandatory in `softmax()` — your prior version had it, keep it. Add a sanity assert that max_val is finite before exp().

- The KvCache rollback on rejection: when verify fails at index i, call self.main_kv.rollback_to(snapshot_pos) and self.draft_kv.rollback_to (their respective pre-batch positions). Both KVs need rollback because the draft proposed beyond the accept point. Use the snapshot_pos / rollback_to / commit_through methods you added to KvCache in the prior file.

- Output ONLY the rewritten src/speculative.rs file. The other 4 files from your prior batch are good as-is. Don't re-emit them.

- Keep the parity test in tests/speculative_parity.rs. Make it actually exercise the SpeculativeDecoder.generate() path with temp=0 against a hand-rolled greedy non-spec reference. Both should produce identical output for a 32-token sequence.

Output: full src/speculative.rs (no truncation, no placeholders, no todo!()) + the corrected tests/speculative_parity.rs. That's it.
