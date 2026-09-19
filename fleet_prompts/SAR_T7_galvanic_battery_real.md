You are a Rust + numerical-physics specialist. Replace the stub `galvanic_battery.rs` in cesarops-inference with a real galvanic ion plume + temporal grid diff implementation.

## What "galvanic battery" actually is

When dissimilar metals (steel hull + bronze fittings + copper wiring) sit submerged in salt or fresh water, they form a galvanic cell. The hull becomes the anode and slowly dissolves, releasing iron ions into the surrounding water. This produces a **detectable plume** in spectral imaging — specifically:

- Iron ion plume causes localized increase in turbidity at specific SWIR bands
- The plume has a temporal signature: persistent near the wreck, dispersed downstream
- Cross-day satellite stack reveals the plume by SUBTRACTING the baseline mean of N days from each daily frame

This is the third leg of the triple-lock: vision says "anomaly here", physics says "and there's a galvanic plume above it that doesn't move with currents".

## Existing stub (replace this)

`/home/cesarops/wreckhunter2000-1/cesarops-inference/src/galvanic_battery.rs`:

```rust
use std::sync::Arc;
use crate::arena::InferenceArena;

pub struct GalvanicBattery {
    pub arena: Arc<InferenceArena>,
}

impl GalvanicBattery {
    pub fn new(arena: Arc<InferenceArena>) -> Self { Self { arena } }

    pub fn evaluate_galvanic_ion_plume(
        &self,
        em_signal_data: &[f64],
        before_layer: &[f64],
        after_layer: &[f64],
    ) -> f64 {
        let _ = (em_signal_data, before_layer, after_layer);
        0.000321f64  // PLACEHOLDER
    }

    pub fn execute_temporal_grid_diff(
        &self,
        config: &TemporalConfig,
        sar_temporal_stack: &[f64],
    ) -> Vec<f64> {
        let _ = (config, sar_temporal_stack);
        vec![]  // PLACEHOLDER
    }
}
```

## Real implementation requirements

### Function 1: `execute_temporal_grid_diff`

Input:
- `temporal_stack: &[f64]` — flattened 3D SWIR stack, layout `[day][row][col]` row-major
- `n_days: usize`, `rows: usize`, `cols: usize`
- `baseline_method: BaselineMethod` — `Mean` or `Median` (use Median for robustness — single-day cloud cover doesn't poison the result)

Algorithm:
1. **Validate**: temporal_stack.len() == n_days * rows * cols
2. **For each (row, col)**: compute the temporal baseline across all n_days at that pixel (mean or median)
3. **Compute persistence map**: for each (row, col), count how many days the pixel exceeded baseline + threshold (e.g., baseline + 1.5 * stddev)
4. **Output**: 2D persistence map, same (rows, cols) shape, values = days_above_baseline / n_days (normalized 0..1)

The persistence map: a pixel where the SWIR signal is consistently elevated across days = stationary plume. Random noise = low persistence. Currents move the plume = decreasing persistence over distance from source.

```rust
pub enum BaselineMethod { Mean, Median }

pub fn execute_temporal_grid_diff(
    &self,
    temporal_stack: &[f64],
    n_days: usize,
    rows: usize,
    cols: usize,
    baseline_method: BaselineMethod,
    threshold_stddevs: f64,  // default 1.5
) -> Result<Vec<f64>, &'static str>;
```

Returns a Vec<f64> of size `rows * cols` where each value is in [0.0, 1.0].

### Function 2: `evaluate_galvanic_ion_plume`

Input: persistence map (output from function 1) + a candidate location (row, col) + a search radius

Returns: a "plume strength" score in [0.0, 1.0] = mean persistence within the search radius around the candidate.

```rust
pub fn evaluate_galvanic_ion_plume(
    &self,
    persistence_map: &[f64],
    rows: usize,
    cols: usize,
    candidate_row: usize,
    candidate_col: usize,
    search_radius: usize,  // default 8 pixels
) -> Result<f64, &'static str>;
```

### Function 3 (NEW): `classify_plume_signature`

Returns a tier classification:
```rust
pub fn classify_plume_signature(&self, plume_strength: f64) -> &'static str {
    match plume_strength {
        s if s > 0.8 => "strong_galvanic_plume",      // very persistent, likely active corrosion
        s if s > 0.5 => "moderate_plume",              // present, possible target
        s if s > 0.25 => "weak_plume",                 // marginal — could be sediment or biological
        _ => "no_plume_signal",
    }
}
```

### Constraints

- NO new dependencies — pure Rust math
- NO `unwrap()` on user input
- Return Result<_, &'static str> for all dimension/length checks
- For median: simple sort approach is fine at our grid sizes (no need for histogram median)
- `#[cfg(test)]` tests:
  - `test_dimension_mismatch_returns_err`
  - `test_uniform_stack_zero_persistence` — all days identical → 0% persistence above baseline
  - `test_synthetic_plume_high_persistence` — 7 days × 64×64 grid, inject a 5-pixel "hot spot" at (32,32) on every day, verify persistence at that pixel is ~1.0 and persistence elsewhere is near 0.0
  - `test_evaluate_finds_plume_in_radius` — given the synthetic plume map from above, querying at (32,32) with radius=4 should return a high score

### Expected output shape

Provide:
1. Full replacement for `src/galvanic_battery.rs` (~200 LOC)
2. Unit tests inline
3. Comment block at top explaining the physics + cite the SAR mission notes

Output format: just the `=== FILE: src/galvanic_battery.rs ===` block. No preamble.

Existing call sites: check via `grep -rn 'galvanic_battery' /home/cesarops/wreckhunter2000-1/cesarops-inference/src/`. If any caller uses old signature, keep a compat wrapper with default values.

Begin now.
