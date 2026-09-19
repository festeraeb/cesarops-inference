# Task: Polish moe_sort.rs — Token Sorting by Expert

You are a senior Rust developer. Polish the following MoE token sorting implementation based on specialist review feedback.

## Current code (working, needs polish):

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct ExpertRange {
    pub start: u32,
    pub count: u32,
}

pub struct SortedExperts {
    pub sorted_token_ids: Vec<u32>,
    pub sorted_expert_ids: Vec<u32>,
    pub sorted_route_weights: Vec<f32>,
    pub expert_ranges: Vec<ExpertRange>,
}

pub fn sort_tokens_by_expert(
    expert_indices: &[u32],
    expert_weights: &[f32],
    num_tokens: usize,
    num_experts: usize,
    top_k: usize,
) -> SortedExperts {
    let mut counts = vec![0u32; num_experts];
    for token in 0..num_tokens {
        for k in 0..top_k {
            let idx = token * top_k + k;
            let expert = expert_indices[idx] as usize;
            counts[expert] += 1;
        }
    }

    let mut expert_ranges = vec![ExpertRange::default(); num_experts];
    let mut running = 0u32;
    for expert in 0..num_experts {
        let count = counts[expert];
        expert_ranges[expert] = ExpertRange { start: running, count };
        running += count;
    }

    let total_assignments = num_tokens * top_k;
    let mut sorted_token_ids = vec![0u32; total_assignments];
    let mut sorted_expert_ids = vec![0u32; total_assignments];
    let mut sorted_route_weights = vec![0f32; total_assignments];

    let mut offsets: Vec<u32> = expert_ranges.iter().map(|r| r.start).collect();

    for token in 0..num_tokens {
        for k in 0..top_k {
            let idx = token * top_k + k;
            let expert = expert_indices[idx] as usize;
            let dst = offsets[expert] as usize;
            sorted_token_ids[dst] = token as u32;
            sorted_expert_ids[dst] = expert as u32;
            sorted_route_weights[dst] = expert_weights[idx];
            offsets[expert] += 1;
        }
    }

    SortedExperts {
        sorted_token_ids,
        sorted_expert_ids,
        sorted_route_weights,
        expert_ranges,
    }
}
```

## Specialist feedback to apply:

1. **Remove `sorted_expert_ids`** — it's redundant. The expert is implied by the range owner when iterating `expert_ranges[e]`. Saves bandwidth/cache.

2. **Add bounds checking** — validate `expert < num_experts` to prevent OOB panics on invalid routing.

3. **Add `#[inline]`** on the main function (it's called per-layer per-token-batch).

4. **Add a `num_assignments()` method** on `SortedExperts` for convenience.

5. **Add doc comments** explaining the CSR-like layout and why stable ordering matters.

6. **Add a simple unit test** that verifies the sort with known inputs (3 tokens, top_k=2, 4 experts).

7. **Keep the algorithm identical** — it's correct. Only polish the API surface and safety.

## Output:
Complete polished `moe_sort.rs` file with all changes applied. Include the unit test at the bottom with `#[cfg(test)]`.
