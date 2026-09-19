You are a Rust + numerical-physics specialist + curvelet expert. Replace the stub `satellite_stitch.rs` in cesarops-inference with a real curvelet-based sub-pixel drift correction implementation.

## What "satellite stitch" actually is

Satellite tiles from different days drift by sub-pixel amounts due to:
- Orbital position variance (Sentinel-2 ground track repeats at 10-day cycle, not exactly identical)
- Atmospheric refraction differences
- Sensor pointing accuracy limits
- DEM-driven parallax in mountainous areas

For wreck detection, we stack 20+ daily tiles to extract weak persistent signals. Sub-pixel drift smears the stack — a 0.3-pixel drift across 20 days makes a 5-pixel-wide wreck signature look like a 11-pixel blur.

The fix: **curvelet-based sub-pixel alignment**. Each tile gets transformed via curvelet decomposition (which captures linear/curved features at multiple scales and angles), the master tile's curvelet coefficients are cross-correlated with each daily tile's coefficients, and the peak-correlation offset (in sub-pixel resolution) is the drift. Apply the inverse drift to align all tiles to the master.

We have nauticuvs-full (the f64-precision curvelet library) available now in the workspace. This is the consumer.

## Existing stub (replace this)

`/home/cesarops/wreckhunter2000-1/cesarops-inference/src/satellite_stitch.rs`:

```rust
use std::sync::Arc;
use crate::arena::InferenceArena;

pub struct SatelliteStitcher {
    pub arena: Arc<InferenceArena>,
}

impl SatelliteStitcher {
    pub fn new(arena: Arc<InferenceArena>) -> Self { Self { arena } }

    pub fn generate_custom_structural_grid(
        &self,
        master_tile: &[f64],
        rows: usize,
        cols: usize,
    ) -> Vec<f64> {
        // PLACEHOLDER — returns input unchanged
        master_tile.to_vec()
    }

    pub fn calculate_true_coordinates_from_master(
        &self,
        candidate_tile: &[f64],
        master_grid: &[f64],
        rows: usize,
        cols: usize,
    ) -> (f64, f64) {
        let _ = (candidate_tile, master_grid, rows, cols);
        (0.0, 0.0)  // PLACEHOLDER drift offsets
    }
}
```

## Real implementation requirements

We have `nauticuvs-full` (renamed to import as `nauticuvs` via path-rename in Cargo.toml). It exposes:

```rust
pub fn curvelet_forward(input: ndarray::Array2<f64>) -> Result<CurveletCoeffs, CurveletError>;
pub fn curvelet_inverse(coeffs: CurveletCoeffs) -> Result<ndarray::Array2<f64>, CurveletError>;
pub struct CurveletConfig { pub scales: usize, pub directions: usize }
pub struct CurveletCoeffs;
```

Add to cesarops-inference Cargo.toml:
```toml
nauticuvs = { path = "../nauticuvs", package = "nauticuvs-full" }
ndarray = "0.16"
```

(if these are already there, skip; if not, add them)

### Function 1: `compute_curvelet_signature`

Convert a tile into its curvelet coefficients (used as a "structural fingerprint" for cross-correlation).

```rust
pub fn compute_curvelet_signature(
    &self,
    tile: &[f64],
    rows: usize,
    cols: usize,
) -> Result<CurveletSignature, &'static str>;

pub struct CurveletSignature {
    // Flattened representation of the curvelet decomposition
    // (just the high-frequency detail bands — drop the low-pass scale)
    pub high_freq_bands: Vec<f64>,
    pub width: usize,
    pub height: usize,
}
```

### Function 2: `estimate_drift_offset`

Compute sub-pixel drift between candidate and master via phase correlation of their curvelet signatures.

```rust
pub fn estimate_drift_offset(
    &self,
    master_sig: &CurveletSignature,
    candidate_sig: &CurveletSignature,
    max_search_pixels: usize,  // default 5
) -> Result<DriftOffset, &'static str>;

pub struct DriftOffset {
    pub dx_pixels: f64,  // sub-pixel resolution
    pub dy_pixels: f64,
    pub correlation_peak: f64,  // 0..1 confidence
}
```

Algorithm:
1. Compute 2D phase correlation between master and candidate high-freq bands
2. Find peak in correlation surface
3. Sub-pixel refinement via parabolic fit on 3×3 neighborhood around the peak
4. Convert peak position back to (dx, dy) in original pixel space
5. Return sub-pixel offsets + correlation peak strength

For the FFT step: use `rustfft = "6"` (already a dep of nauticuvs-full). Don't add a new FFT crate.

Phase correlation formula:
```
P(u, v) = (F(master) * conj(F(candidate))) / |F(master) * conj(F(candidate))|
shift = argmax(IFFT(P))
```

### Function 3: `align_tile_to_master`

Apply a known drift to a candidate tile via bilinear interpolation, producing an aligned tile.

```rust
pub fn align_tile_to_master(
    &self,
    candidate_tile: &[f64],
    rows: usize,
    cols: usize,
    drift: &DriftOffset,
) -> Vec<f64>;
```

Bilinear interpolation: each output pixel (i, j) samples from the candidate at (i + dy, j + dx). Sub-pixel sampling uses the standard bilinear formula. Out-of-bounds samples = 0.0.

### Function 4 (NEW): `stack_aligned_tiles`

Top-level pipeline: given N tiles + a master, align all and produce mean+stddev maps.

```rust
pub fn stack_aligned_tiles(
    &self,
    tiles: &[&[f64]],  // N tiles, each rows*cols
    rows: usize,
    cols: usize,
    master_idx: usize,  // which tile is the alignment master
) -> Result<StackResult, &'static str>;

pub struct StackResult {
    pub mean_map: Vec<f64>,
    pub stddev_map: Vec<f64>,
    pub drift_offsets: Vec<DriftOffset>,  // length N (master = (0,0))
}
```

This is what the SAR mission's "20+ day temporal stack" actually invokes.

### Constraints

- nauticuvs-full path-rename trick: `nauticuvs = { path = "../nauticuvs", package = "nauticuvs-full" }` — verify this is in cesarops-inference Cargo.toml; add if missing
- `ndarray = "0.16"` in cesarops-inference (it's a dep of nauticuvs already)
- NO `unwrap()` on user input
- Return `Result<_, &'static str>` for dimension errors
- Tests:
  - `test_zero_drift_self_correlation` — master vs master should produce drift (0, 0) and correlation_peak ≈ 1.0
  - `test_known_shift_recovers` — shift master by exactly (2, 0) integer pixels, verify recovered drift matches within 0.1 pixel
  - `test_curvelet_signature_dimension_match` — output bands sum to expected feature count
- Comment block at top explaining the physics + cite the SAR mission notes

### Expected output shape

Provide:
1. Full replacement for `src/satellite_stitch.rs` (~250 LOC)
2. Unit tests inline
3. Cargo.toml additions (if any) at top of response in a `=== DIFF: cesarops-inference/Cargo.toml ===` block

Output format:
```
=== DIFF: cesarops-inference/Cargo.toml ===
// only the new dep lines

=== FILE: src/satellite_stitch.rs ===
// full file
```

No preamble. Code only.

If the curvelet API in nauticuvs-full doesn't quite match what's described above (because real curvelet libs vary), use the closest-matching public function from the crate's lib.rs and document the mapping in a comment.

Begin now.
