You are a Rust + numerical-physics specialist. Replace the stub `magnetic_eraser.rs` in cesarops-inference with a real sub-nT anomaly extraction implementation.

## What "magnetic eraser" actually is

The aeromagnetic worker (cesarops-aeromagnetic-worker, already running in real wgpu compute on Pascal P100) detects gross magnetic dipoles. The "eraser" stage is a CPU-side pre-processing pass that ERASES the broad-spectrum geomagnetic baseline so only sub-nanoTesla residual anomalies remain. These residuals are what reveal small ferrous masses (an aircraft fuselage, a lost engine block, a cannon).

The physics:
- Earth's geomagnetic field at any point varies by ~25-50 nT over a flight line due to natural geological gradients
- A 10-ton ferrous wreck at 200ft depth produces a ~0.5-2 nT signature
- The signal is buried under the noise unless you subtract the regional baseline first
- Standard technique: high-pass filter via subtraction of a long-window moving median (NOT mean — median rejects the dipole signal that we want to preserve)

## Existing stub (replace this)

`/home/cesarops/wreckhunter2000-1/cesarops-inference/src/magnetic_eraser.rs`:

```rust
use std::sync::Arc;
use crate::arena::InferenceArena;

pub struct MagneticEraser {
    pub arena: Arc<InferenceArena>,
}

impl MagneticEraser {
    pub fn new(arena: Arc<InferenceArena>) -> Self { Self { arena } }

    pub fn erase_standard_baseline(
        &self,
        magnetic_grid: &[f64],
        rows: usize,
        cols: usize,
    ) -> Vec<f64> {
        let _ = (rows, cols);
        magnetic_grid.iter().map(|&v| v * 0.001).collect()  // PLACEHOLDER — pretends to subtract
    }

    pub fn classify_target_signature(&self, residual_pull: f64) -> &'static str {
        let _residual = 0.000321f64;  // PLACEHOLDER
        match residual_pull.abs() {
            r if r > 1.0 => "large_ferrous_mass",
            r if r > 0.1 => "medium_target",
            _ => "background",
        }
    }
}
```

## Real implementation requirements

### Function 1: `erase_standard_baseline`

Input:
- `magnetic_grid: &[f64]` — flattened 2D mag grid, row-major
- `rows: usize`, `cols: usize` — dimensions
- `window_size: usize` — moving-median window (default 31, must be odd)

Algorithm:
1. **Validate**: rows * cols == grid.len(). If not, return Err.
2. **For each row**: compute a 1D moving-median with `window_size` and subtract it from the row. Use a histogram or sorted-window approach (NOT a full sort per pixel — that's O(n²)).
3. **For each column** (transposed pass): compute the same 1D moving-median and subtract.
4. The result is a band-pass filtered grid: regional gradient removed (low freq), pixel noise still present (high freq), dipole-scale signals (mid freq, ~window_size/4 wavelength) preserved.
5. Return Vec<f64> same shape as input.

For the median, use this efficient sliding-window approach:
```
for each row:
    sorted_window = sorted(grid[row][0..window_size])
    for j in window_size/2 .. cols - window_size/2:
        median = sorted_window[window_size/2]
        out[row][j] = grid[row][j] - median
        # slide window: remove grid[row][j - window_size/2], insert grid[row][j + window_size/2 + 1]
```

A simple correct implementation can use `Vec::sort_unstable()` per window position; that's O(rows * cols * window_size * log(window_size)) which is fast enough at our grid sizes. Optimize later.

### Function 2: `classify_target_signature`

Returns a tier classification based on residual nT amplitude:

```rust
pub fn classify_target_signature(&self, residual_nt: f64) -> &'static str {
    match residual_nt.abs() {
        r if r > 5.0 => "large_ferrous_anomaly",       // >5 nT — large wreck or geological
        r if r > 2.0 => "medium_ferrous_target",       // 2-5 nT — vessel-class
        r if r > 0.5 => "small_ferrous_target",        // 0.5-2 nT — aircraft / engine block
        r if r > 0.1 => "marginal_signal",             // 0.1-0.5 nT — possible target, low confidence
        _ => "background_noise",                        // <0.1 nT — below detection threshold
    }
}
```

### Function 3 (NEW): `extract_anomaly_centers`

After erasing the baseline, find local extrema (peak positives + peak negatives within `local_radius`) above a `min_amplitude` threshold. These are candidate dipole centers.

```rust
pub struct AnomalyCenter {
    pub row: usize,
    pub col: usize,
    pub amplitude_nt: f64,
    pub polarity: Polarity,  // Positive | Negative
}

pub enum Polarity {
    Positive,
    Negative,
}

pub fn extract_anomaly_centers(
    &self,
    erased_grid: &[f64],
    rows: usize,
    cols: usize,
    local_radius: usize,        // typical: 5
    min_amplitude_nt: f64,      // typical: 0.5
) -> Vec<AnomalyCenter>;
```

Scan all pixels; a pixel is a local maximum/minimum if it's strictly greater/less than every other pixel within local_radius (Chebyshev distance). Return all such pixels with their amplitude and polarity.

### Constraints

- NO new dependencies — pure Rust math
- NO `unwrap()` on user input
- Return Result<Vec<f64>, &'static str> from erase_standard_baseline so dimension errors are typed
- Handle window_size > min(rows, cols): clamp window or return Err
- Add `#[cfg(test)]` tests:
  - `test_uniform_field_zeros_after_erasure` — flat input should produce all-zero output
  - `test_synthetic_dipole_survives_erasure` — inject a 1.5 nT dipole, verify it's preserved after baseline subtraction
  - `test_extract_anomaly_centers_finds_dipole` — synthetic dipole should appear in extract_anomaly_centers result
  - `test_classify_target_buckets` — boundary cases

### Expected output shape

Provide:
1. Full replacement for `src/magnetic_eraser.rs` (~200 LOC)
2. Unit tests inline
3. Comment block at top explaining the physics + cite the SAR mission notes

Output format: just the `=== FILE: src/magnetic_eraser.rs ===` block. No preamble, no scaffolding.

Existing call sites: check via `grep -rn 'magnetic_eraser' /home/cesarops/wreckhunter2000-1/cesarops-inference/src/`. If any caller uses old signature, keep a compat wrapper.

Begin now.
