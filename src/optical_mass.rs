// src/optical_mass.rs
//
// THERMOCLINE JITTER DETECTION MODULE
//
// Physics:
//   A submerged ferrous mass (steel hull, aircraft fuselage) acts as a cold
//   sink at the thermocline boundary. Warm cross-currents intersecting the
//   sink produce high-frequency optical refraction shimmer detectable in
//   ICESat-2 ATL23 thermal data and in SWIR satellite bands at the 180-200 ft
//   penetration boundary in clear water.
//
//   This module isolates that high-frequency shimmer from the broad thermal
//   gradient via a windowed-mean subtraction, applies water-column attenuation
//   correction, and reports jitter variance + mass-displacement estimate.
//
// Algorithm (windowed-stddev approach, no FFT dependency):
//   1. Reject low-signal regions: blue_light_intensity below 0.1 = no signal at depth
//   2. High-pass via 32x32 box-mean subtraction
//   3. Attenuation correction: divide by exp(-k * thermocline_depth)
//   4. Variance over interior pixels (skip border where box-mean is undefined)
//
// Mass constant 12450.0 derived from SAR mission notes:
//   (iron/steel hull density) * (thermal plume cross-section coefficient).
//   Configurable in v2 — the constant is what tonnage classification keys off.

use std::sync::Arc;
use crate::arena::InferenceArena;

pub struct RefractiveProfile {
    pub blue_light_intensity: f32,    // 0..1, attenuated signal at target depth
    pub thermocline_depth_meters: f32,
    pub water_clarity_k_index: f32,   // K-index, higher = clearer water (less attenuation)
}

pub struct OpticalMassEstimator {
    pub arena: Arc<InferenceArena>,
}

impl OpticalMassEstimator {
    pub fn new(arena: Arc<InferenceArena>) -> Self {
        Self { arena }
    }

    /// Legacy compatibility wrapper for existing call sites.
    /// Assumes a square tile if dimensions are not provided.
    pub fn execute_jitter_analysis_legacy(
        &self,
        icesat_optical_matrix: &[f32],
        profile: &RefractiveProfile,
    ) -> Result<f64, &'static str> {
        let side = (icesat_optical_matrix.len() as f64).sqrt() as usize;
        if side * side != icesat_optical_matrix.len() {
            return Err("Legacy call: matrix must be square (cannot infer width/height).");
        }
        self.execute_jitter_analysis(icesat_optical_matrix, side, side, profile)
    }

    /// Core jitter analysis. Returns variance of the high-frequency component
    /// after baseline subtraction + attenuation correction.
    pub fn execute_jitter_analysis(
        &self,
        optical_matrix: &[f32],
        width: usize,
        height: usize,
        profile: &RefractiveProfile,
    ) -> Result<f64, &'static str> {
        if profile.blue_light_intensity < 0.1 {
            return Err("Signal attenuation too high at target depth boundary.");
        }
        if optical_matrix.len() != width * height {
            return Err("Dimension mismatch between matrix length and width*height.");
        }
        if width < 33 || height < 33 {
            return Err("Tile dimensions too small for 32x32 box-mean window.");
        }

        // 1. Compute 32x32 box-mean (low-pass), subtract from each interior pixel.
        let window_size: usize = 32;
        let half_win: usize = window_size / 2;
        let win_area = (window_size * window_size) as f32;

        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut count = 0usize;

        // Attenuation factor — corrected[i,j] = high_freq[i,j] / exp(-k * depth)
        let exp_arg = -profile.water_clarity_k_index * profile.thermocline_depth_meters;
        let attenuation_factor = exp_arg.exp();
        if !attenuation_factor.is_finite() || attenuation_factor <= 0.0 {
            return Err("Invalid attenuation factor (water_clarity_k_index or thermocline_depth out of range).");
        }
        let attenuation_factor_f64 = attenuation_factor as f64;

        // Interior region only; border = half_win on each side.
        for y in half_win..(height - half_win) {
            for x in half_win..(width - half_win) {
                // Compute 32x32 box mean centered at (y, x).
                // Naive but bounded; for production swap to integral image.
                let mut sum = 0.0f32;
                for wy in 0..window_size {
                    let py = y + wy - half_win;
                    let row_start = py * width;
                    for wx in 0..window_size {
                        let px = x + wx - half_win;
                        sum += optical_matrix[row_start + px];
                    }
                }
                let mean = sum / win_area;
                let center = optical_matrix[y * width + x];
                let high_freq = center - mean;
                let corrected = (high_freq as f64) / attenuation_factor_f64;

                sum_x += corrected;
                sum_x2 += corrected * corrected;
                count += 1;
            }
        }

        if count == 0 {
            return Err("No interior pixels available for variance computation.");
        }

        // Var(X) = E[X^2] - (E[X])^2. Clamp to >= 0 to absorb fp roundoff.
        let n = count as f64;
        let mean = sum_x / n;
        let variance = (sum_x2 / n) - (mean * mean);
        Ok(variance.max(0.0))
    }

    /// Convert jitter variance to displacement mass (tons) via the empirical
    /// constant. Default mass_constant = 12450.0 from SAR mission notes.
    pub fn estimate_tonnage_from_shimmer(
        &self,
        jitter_variance: f64,
        mass_constant: f64,
    ) -> f64 {
        jitter_variance * mass_constant
    }

    /// Tier-classify a tonnage estimate into a target category.
    pub fn classify_target(&self, tonnage_estimate: f64) -> &'static str {
        match tonnage_estimate {
            t if t < 50.0 => "small_debris",
            t if t < 500.0 => "small_vessel_or_aircraft",
            t if t < 2000.0 => "medium_vessel",
            t if t < 10000.0 => "large_vessel",
            _ => "anomaly_too_large_to_classify",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn mock_arena() -> Arc<InferenceArena> {
        // InferenceArena::new requires libc on Linux for NUMA pinning;
        // a 1-byte test arena is fine — the optical_mass code doesn't
        // actually use the storage buffer.
        InferenceArena::new(1, 0)
    }

    fn setup() -> OpticalMassEstimator {
        OpticalMassEstimator::new(mock_arena())
    }

    fn good_profile() -> RefractiveProfile {
        RefractiveProfile {
            blue_light_intensity: 1.0,
            thermocline_depth_meters: 20.0,
            water_clarity_k_index: 0.05,
        }
    }

    #[test]
    fn zero_signal_returns_err() {
        let estimator = setup();
        let profile = RefractiveProfile {
            blue_light_intensity: 0.05, // below threshold
            thermocline_depth_meters: 50.0,
            water_clarity_k_index: 0.1,
        };
        let matrix = vec![1.0f32; 64 * 64];
        let result = estimator.execute_jitter_analysis(&matrix, 64, 64, &profile);
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_returns_err() {
        let estimator = setup();
        let matrix = vec![1.0f32; 100]; // 10x10 worth of data, but...
        let result = estimator.execute_jitter_analysis(&matrix, 50, 50, &good_profile());
        assert!(result.is_err());
    }

    #[test]
    fn small_tile_returns_err() {
        let estimator = setup();
        let matrix = vec![1.0f32; 16 * 16]; // smaller than window
        let result = estimator.execute_jitter_analysis(&matrix, 16, 16, &good_profile());
        assert!(result.is_err());
    }

    #[test]
    fn uniform_field_returns_zero_jitter() {
        let estimator = setup();
        let matrix = vec![10.0f32; 50 * 50];
        let result = estimator
            .execute_jitter_analysis(&matrix, 50, 50, &good_profile())
            .unwrap();
        assert!(result < 1e-6, "uniform field should have ~zero jitter, got {}", result);
    }

    #[test]
    fn synthetic_dipole_detected() {
        let estimator = setup();
        let mut matrix = vec![10.0f32; 64 * 64];
        // Inject high-frequency spikes
        matrix[32 * 64 + 32] = 50.0;
        matrix[32 * 64 + 33] = -30.0;

        let var_clean = estimator
            .execute_jitter_analysis(&vec![10.0f32; 64 * 64], 64, 64, &good_profile())
            .unwrap();
        let var_noisy = estimator
            .execute_jitter_analysis(&matrix, 64, 64, &good_profile())
            .unwrap();

        assert!(var_noisy > var_clean, "synthetic spikes should raise jitter variance");
    }

    #[test]
    fn classify_target_buckets() {
        let estimator = setup();
        assert_eq!(estimator.classify_target(10.0), "small_debris");
        assert_eq!(estimator.classify_target(300.0), "small_vessel_or_aircraft");
        assert_eq!(estimator.classify_target(1500.0), "medium_vessel");
        assert_eq!(estimator.classify_target(5000.0), "large_vessel");
        assert_eq!(estimator.classify_target(20000.0), "anomaly_too_large_to_classify");
    }

    #[test]
    fn tonnage_estimate_proportional() {
        let estimator = setup();
        let jitter = 0.01;
        let mass_const = 12450.0;
        let tons = estimator.estimate_tonnage_from_shimmer(jitter, mass_const);
        assert!((tons - 124.5).abs() < 1e-6);
    }
}
