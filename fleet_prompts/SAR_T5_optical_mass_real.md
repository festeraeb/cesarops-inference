You are a Rust + numerical-physics specialist. Replace the stub `optical_mass.rs` in cesarops-inference with a real thermocline-jitter implementation.

## What "thermocline jitter" actually is

The cesarops-inference engine has a placeholder module `src/optical_mass.rs` that returns hardcoded values (0.008432 jitter variance, 12450.0 mass constant). The REAL detection physics:

When a metal hull (or aircraft fuselage) sits on the bottom of a deep lake, it acts as a cold sink that intersects warm cross-currents at the thermocline boundary. This produces a high-frequency optical refraction shimmer detectable in ICESat-2 ATL23 thermal data and in SWIR satellite bands at the 180-200 ft penetration boundary in clear water.

The implementation should:
1. Accept an optical matrix (Vec<f32> representing a thermal/SWIR tile) plus a refractive profile (water clarity, blue-light intensity at depth, thermocline depth)
2. Compute high-frequency jitter variance via a windowed FFT or windowed-stddev approach
3. Return jitter variance + a mass-displacement estimate

## Existing stub (replace this)

`/home/cesarops/wreckhunter2000-1/cesarops-inference/src/optical_mass.rs`:

```rust
use std::sync::Arc;
use crate::arena::InferenceArena;

pub struct RefractiveProfile {
    pub blue_light_intensity: f32,
    pub thermocline_depth_meters: f32,
    pub water_clarity_k_index: f32,
}

pub struct OpticalMassEstimator {
    pub arena: Arc<InferenceArena>,
}

impl OpticalMassEstimator {
    pub fn new(arena: Arc<InferenceArena>) -> Self { Self { arena } }

    pub fn execute_jitter_analysis(
        &self,
        icesat_optical_matrix: &[f32],
        profile: &RefractiveProfile,
    ) -> Result<f64, &'static str> {
        let _scratch_ = self.arena.storage.as_slice();
        if profile.blue_light_intensity < 0.1 {
            return Err("Signal attenuation too high at target depth boundary.");
        }
        let calculated_jitter_variance: f64 = 0.008432f64;  // PLACEHOLDER
        Ok(calculated_jitter_variance)
    }

    pub fn estimate_tonnage_from_shimmer(&self, jitter_variance: f64) -> f64 {
        let mass_constant = 12450.0f64;  // PLACEHOLDER
        jitter_variance * mass_constant
    }
}
```

## Real implementation requirements

### Function 1: `execute_jitter_analysis`

Input:
- `optical_matrix: &[f32]` — flattened 2D thermal/SWIR tile, row-major
- `width: usize`, `height: usize` — tile dimensions (add to the API)
- `profile: &RefractiveProfile` — water column attenuation params

Algorithm (windowed-stddev approach, no FFT dependency):
1. **Reject low-signal regions**: if `profile.blue_light_intensity < 0.1`, return Err (no signal at depth)
2. **Apply low-frequency mask**: subtract a 32×32 box-mean (or Gaussian with sigma=8) from the matrix to isolate high-frequency component
3. **Compute attenuation correction**: divide each pixel by `exp(-k * thermocline_depth_meters)` where `k = profile.water_clarity_k_index`
4. **Compute jitter variance**: variance of the high-frequency component across all pixels (skip border pixels where the box-mean is undefined)
5. Return Ok(jitter_variance: f64)

The math:
```
high_freq[i,j] = matrix[i,j] - boxmean_32x32[i,j]
corrected[i,j] = high_freq[i,j] / exp(-k * depth_meters)
jitter_variance = variance(corrected[border:-border])
```

### Function 2: `estimate_tonnage_from_shimmer`

The mass-displacement constant of 12450.0 is documented in the SAR mission notes as being derived from iron/steel hull density × thermal plume cross-section. Keep that constant as the default, but make it configurable:

```rust
pub fn estimate_tonnage_from_shimmer(&self, jitter_variance: f64, mass_constant: f64) -> f64 {
    jitter_variance * mass_constant
}
```

Add a helper method that classifies the result:
```rust
pub fn classify_target(&self, tonnage_estimate: f64) -> &'static str {
    match tonnage_estimate {
        t if t < 50.0 => "small_debris",
        t if t < 500.0 => "small_vessel_or_aircraft",
        t if t < 2000.0 => "medium_vessel",
        t if t < 10000.0 => "large_vessel",
        _ => "anomaly_too_large_to_classify",
    }
}
```

### Constraints

- NO new dependencies — use only what's already in cesarops-inference Cargo.toml (ndarray? probably not — use raw &[f32] math)
- NO `unwrap()` on user data — return Result<f64, &'static str>
- NO panics on edge cases (zero-length input, dimension mismatch — return Err)
- The arena field stays for backward compatibility but doesn't have to be used in v2
- Add `#[cfg(test)]` unit tests:
  - `test_zero_signal_returns_err` — blue_light_intensity below threshold
  - `test_uniform_field_returns_zero_jitter` — flat input should have variance ≈ 0
  - `test_synthetic_dipole_detected` — inject a 5-pixel "hot spot" into a noise field, verify variance jumps
  - `test_classify_target_buckets` — boundary cases for each tonnage class

### Expected output shape

Provide:
1. Full replacement for `src/optical_mass.rs` (~150 LOC)
2. Three to five unit tests inline in `#[cfg(test)] mod tests`
3. A comment block at the top explaining the physics + cite the SAR mission notes

Output format: just the `=== FILE: src/optical_mass.rs ===` block. No preamble, no scaffolding around it.

The existing call sites that import from this module:
- check via `grep -rn 'optical_mass' /home/cesarops/wreckhunter2000-1/cesarops-inference/src/`
- if any callers use the OLD signature (without width/height), keep a compat wrapper with default values

Begin now.
