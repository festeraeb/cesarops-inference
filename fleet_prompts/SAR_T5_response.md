=== FILE: src/optical_mass.rs ===
/*
 * THERMOCLINE JITTER DETECTION MODULE
 * 
 * Physics Overview:
 * A metal hull/fuselage acting as a cold sink at the thermocline boundary creates 
 * high-frequency refractive shimmer in SWIR/Thermal bands. This module isolates 
 * high-frequency optical jitter from low-frequency thermal gradients.
 * 
 * Math:
 * 1. High-pass filter: high_freq[i,j] = matrix[i,j] - boxmean_32x32[i,j]
 * 2. Attenuation correction: corrected[i,j] = high_freq[i,j] / exp(-k * depth)
 * 3. Variance: Var(corrected) across the non-border region.
 * 
 * Constant 12450.0 derived from SAR mission notes: (Iron/Steel density * Thermal plume cross-section).
 */

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
            return Err("Legacy call: Matrix must be square.");
        }
        self.execute_jitter_analysis(icesat_optical_matrix, side, side, profile)
    }

    /// Core jitter analysis using a windowed-stddev approach.
    pub fn execute_jitter_analysis(
        &self,
        optical_matrix: &[f32],
        width: usize,
        height: usize,
        profile: &RefractiveProfile,
    ) -> Result<f64, &'static str> {
        // 1. Signal validation
        if profile.blue_light_intensity < 0.1 {
            return Err("Signal attenuation too high at target depth boundary.");
        }
        if optical_matrix.len() != width * height {
            return Err("Dimension mismatch between matrix and width/height.");
        }
        if width < 33 || height < 33 {
            return Err("Tile dimensions too small for 32x32 box-mean window.");
        }

        // 2. Compute 32x32 Box-Mean (Low-pass filter)
        // We use a sliding window approach to isolate high-frequency components.
        let window_size = 32;
        let half_win = window_size / 2;
        let mut high_freq = vec![0.0f32; width * height];

        for y in half_win..(height - half_win) {
            for x in half_win..(width - half_win) {
                let mut sum = 0.0f32;
                for wy in 0..window_size {
                    for wx in 0..window_size {
                        let py = y + wy - half_win;
                        let px = x + wx - half_win;
                        sum += optical_matrix[py * width + px];
                    }
                }
                let mean = sum / (window_size * window_size) as f32;
                let idx = y * width + x;
                high_freq[idx] = optical_matrix[idx] - mean;
            }
        }

        // 3. Attenuation Correction & Variance Calculation
        // corrected[i,j] = high_freq[i,j] / exp(-k * depth)
        let attenuation_factor = (-profile.water_clarity_k_index * profile.thermocline_depth_meters).exp();
        if attenuation_factor <= 0.0 {
            return Err("Invalid attenuation calculation.");
        }

        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut count = 0usize;

        // Only process the interior to avoid border artifacts from the box-mean
        for y in half_win..(height - half_win) {
            for x in half_win..(width - half_win) {
                let idx = y * width + x;
                let corrected = (high_freq[idx] as f64) / (attenuation_factor as f64);
                
                sum_x += corrected;
                sum_x2 += corrected * corrected;
                count += 1;
            }
        }

        if count == 0 {
            return Err("No valid pixels for variance calculation.");
        }

        // Variance formula: E[X^2] - (E[X])^2
        let mean = sum_x / count as f64;
        let variance = (sum_x2 / count as f64) - (mean * mean);

        // Ensure floating point precision doesn't return negative near zero
        Ok(variance.max(0.0))
    }

    pub fn estimate_tonnage_from_shimmer(&self, jitter_variance: f64, mass_constant: f64) -> f64 {
        jitter_variance * mass_constant
    }

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

    fn setup() -> OpticalMassEstimator {
        let arena = Arc::new(InferenceArena { storage: vec![] }); // Mock arena
        OpticalMassEstimator::new(arena)
    }

    #[test]
    fn test_zero_signal_returns_err() {
        let estimator = setup();
        let profile = RefractiveProfile {
            blue_light_intensity: 0.05, // Below 0.1 threshold
            thermocline_depth_meters: 50.0,
            water_clarity_k_index: 0.1,
        };
        let matrix = vec![1.0; 1024 * 1024];
        let result = estimator.execute_jitter_analysis(&matrix, 1024, 1024, &profile);
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_field_returns_zero_jitter() {
        let estimator = setup();
        let profile = RefractiveProfile {
            blue_light_intensity: 1.0,
            thermocline_depth_meters: 20.0,
            water_clarity_k_index: 0.1,
        };
        let matrix = vec![10.0; 40 * 40]; // 40x40 to satisfy > 33 requirement
        let result = estimator.execute_jitter_analysis(&matrix, 40, 40, &profile).unwrap();
        assert!(result < 1e-6);
    }

    #[test]
    fn test_synthetic_dipole_detected() {
        let estimator = setup();
        let profile = RefractiveProfile {
            blue_light_intensity: 1.0,
            thermocline_depth_meters: 20.0,
            water_clarity_k_index: 0.1,
        };
        let mut matrix = vec![10.0; 50 * 50];
        // Inject a high-frequency spike (the "shimmer")
        matrix[25 * 50 + 25] = 50.0; 
        
        let variance_clean = estimator.execute_jitter_analysis(&vec![10.0; 50*50], 50, 50, &profile).unwrap();
        let variance_noisy = estimator.execute_jitter_analysis(&matrix, 50, 50, &profile).unwrap();
        
        assert!(variance_noisy > variance_clean);
    }

    #[test]
    fn test_classify_target_buckets() {
        let estimator = setup();
        assert_eq!(estimator.classify_target(10.0), "small_debris");
        assert_eq!(estimator.classify_target(300.0), "small_vessel_or_aircraft");
        assert_eq!(estimator.classify_target(1500.0), "medium_vessel");
        assert_eq!(estimator.classify_target(5000.0), "large_vessel");
        assert_eq!(estimator.classify_target(20000.0), "anomaly_too_large_to_classify");
    }
}
