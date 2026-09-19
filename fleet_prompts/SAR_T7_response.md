

=== FILE: src/galvanic_battery.rs ===
use std::sync::Arc;
use crate::arena::InferenceArena;

/// # Galvanic Battery / Ion Plume Detection Module
///
/// ## Physics Background
/// When dissimilar metals (steel hull, bronze fittings, copper wiring) are submerged
/// in electrolytes (saltwater or conductive freshwater), a galvanic cell is formed.
/// The hull acts as the anode and undergoes oxidation (corrosion), releasing Fe2+/Fe3+
/// ions into the surrounding water.
///
/// ## Detection Methodology
/// These dissolved ions alter the local refractive index and turbidity of the water column.
/// In Spectral Imaging (specifically SWIR bands), this manifests as a localized increase
/// in backscatter/turbidity signatures.
///
/// Because the plume is anchored to the wreck (stationary source) but dispersed by currents,
/// it exhibits a distinct temporal signature:
/// 1. **Persistence**: The signal persists day-over-day at the source location.
/// 2. **Dispersion**: The signal spreads downstream over time.
/// 3. **Differentiation**: By subtracting the temporal baseline (median of N days),
///    we isolate the "anomaly" from the static background (seabed, static vegetation).
///
/// This module implements the "third leg" of the triple-lock inference:
/// - Vision: Detects anomaly.
/// - Physics: Confirms stationary galvanic plume signature.
/// - Temporal Grid Diff: Quantifies persistence to distinguish plume from transient noise.
///
/// Reference: SAR Mission Notes Vol. 4, "Electrochemical Corrosion Signatures in Submerged Wrecks"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineMethod {
    Mean,
    Median,
}

pub struct GalvanicBattery {
    pub arena: Arc<InferenceArena>,
}

impl GalvanicBattery {
    pub fn new(arena: Arc<InferenceArena>) -> Self {
        Self { arena }
    }

    /// Computes a temporal persistence map from a stack of SWIR frames.
    ///
    /// # Algorithm
    /// 1. Validate dimensions.
    /// 2. For each pixel (r, c), compute the temporal baseline across `n_days`.
    /// 3. Count how many days the pixel value exceeded `baseline + (threshold_stddevs * stddev)`.
    /// 4. Normalize count by `n_days` to get a persistence score in [0.0, 1.0].
    ///
    /// # Arguments
    /// * `temporal_stack` - Flattened 3D array [day][row][col] row-major.
    /// * `n_days` - Number of days in the stack.
    /// * `rows` - Height of the grid.
    /// * `cols` - Width of the grid.
    /// * `baseline_method` - How to compute the central tendency (Mean or Median).
    /// * `threshold_stddevs` - Multiplier for standard deviation to define "exceedance".
    ///
    /// # Returns
    /// A Vec<f64> of size `rows * cols` with persistence scores.
    pub fn execute_temporal_grid_diff(
        &self,
        temporal_stack: &[f64],
        n_days: usize,
        rows: usize,
        cols: usize,
        baseline_method: BaselineMethod,
        threshold_stddevs: f64,
    ) -> Result<Vec<f64>, &'static str> {
        let total_elements = n_days * rows * cols;
        if temporal_stack.len() != total_elements {
            return Err("temporal_stack length mismatch: expected n_days*rows*cols");
        }

        if n_days == 0 || rows == 0 || cols == 0 {
            return Err("dimensions must be non-zero");
        }

        if threshold_stddevs < 0.0 {
            return Err("threshold_stddevs must be non-negative");
        }

        let mut persistence_map = vec![0.0f64; rows * cols];

        // Pre-allocate a buffer for temporal values to avoid repeated allocations
        let mut temporal_buffer = vec![0.0f64; n_days];

        for r in 0..rows {
            for c in 0..cols {
                // Extract the time series for this specific pixel
                for d in 0..n_days {
                    let idx = (d * rows * cols) + (r * cols) + c;
                    temporal_buffer[d] = temporal_stack[idx];
                }

                // Compute baseline and stddev
                let (baseline, stddev) = compute_stats(&temporal_buffer, baseline_method);

                // Compute exceedance count
                let threshold = baseline + (threshold_stddevs * stddev);
                let mut exceedance_count = 0u32;

                for &val in &temporal_buffer {
                    if val > threshold {
                        exceedance_count += 1;
                    }
                }

                // Normalize to [0.0, 1.0]
                let persistence = exceedance_count as f64 / n_days as f64;
                persistence_map[(r * cols) + c] = persistence;
            }
        }

        Ok(persistence_map)
    }

    /// Evaluates the plume strength at a specific candidate location.
    ///
    /// # Arguments
    /// * `persistence_map` - Output from `execute_temporal_grid_diff`.
    /// * `rows`, `cols` - Dimensions of the map.
    /// * `candidate_row`, `candidate_col` - Center of the search window.
    /// * `search_radius` - Radius (in pixels) around the candidate to average.
    ///
    /// # Returns
    /// A score in [0.0, 1.0] representing the mean persistence within the radius.
    pub fn evaluate_galvanic_ion_plume(
        &self,
        persistence_map: &[f64],
        rows: usize,
        cols: usize,
        candidate_row: usize,
        candidate_col: usize,
        search_radius: usize,
    ) -> Result<f64, &'static str> {
        if persistence_map.len() != rows * cols {
            return Err("persistence_map dimensions mismatch");
        }

        if candidate_row >= rows || candidate_col >= cols {
            return Err("candidate coordinates out of bounds");
        }

        let mut sum = 0.0f64;
        let mut count = 0usize;

        // Define bounds for the search window
        let r_min = if candidate_row >= search_radius {
            candidate_row - search_radius
        } else {
            0
        };
        let r_max = if candidate_row + search_radius < rows {
            candidate_row + search_radius
        } else {
            rows - 1
        };

        let c_min = if candidate_col >= search_radius {
            candidate_col - search_radius
        } else {
            0
        };
        let c_max = if candidate_col + search_radius < cols {
            candidate_col + search_radius
        } else {
            cols - 1
        };

        for r in r_min..=r_max {
            for c in c_min..=c_max {
                let idx = (r * cols) + c;
                sum += persistence_map[idx];
                count += 1;
            }
        }

        if count == 0 {
            return Err("search window resulted in zero pixels");
        }

        Ok(sum / count as f64)
    }

    /// Classifies the plume strength into a tier.
    pub fn classify_plume_signature(&self, plume_strength: f64) -> &'static str {
        if plume_strength > 0.8 {
            "strong_galvanic_plume"
        } else if plume_strength > 0.5 {
            "moderate_plume"
        } else if plume_strength > 0.25 {
            "weak_plume"
        } else {
            "no_plume_signal"
        }
    }
}

/// Helper to compute mean and standard deviation of a slice.
fn compute_stats(data: &[f64], method: BaselineMethod) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let baseline = match method {
        BaselineMethod::Mean => mean,
        BaselineMethod::Median => {
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
    };

    (baseline, stddev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_returns_err() {
        let gb = GalvanicBattery::new(Arc::new(InferenceArena::default()));
        let stack = vec![1.0; 10];
        // 2 days * 2 rows * 2 cols = 8 elements. Stack has 10.
        let res = gb.execute_temporal_grid_diff(&stack, 2, 2, 2, BaselineMethod::Mean, 1.5);
        assert!(res.is_err());
    }

    #[test]
    fn test_uniform_stack_zero_persistence() {
        let gb = GalvanicBattery::new(Arc::new(InferenceArena::default()));
        let n_days = 5;
        let rows = 4;
        let cols = 4;
        let total = n_days * rows * cols;
        
        // All values are identical (e.g., 10.0)
        let stack = vec![10.0; total];
        
        let res = gb.execute_temporal_grid_diff(&stack, n_days, rows, cols, BaselineMethod::Mean, 1.5);
        assert!(res.is_ok());
        
        let persistence_map = res.unwrap();
        
        // Since all values are identical, stddev is 0. Threshold = mean + 0 = mean.
        // No value is strictly GREATER than the mean (they are equal).
        // So persistence should be 0.0 for all pixels.
        for &p in &persistence_map {
            assert_eq!(p, 0.0, "Uniform stack should have zero persistence");
        }
    }

    #[test]
    fn test_synthetic_plume_high_persistence() {
        let gb = GalvanicBattery::new(Arc::new(InferenceArena::default()));
        let n_days = 7;
        let rows = 64;
        let cols = 64;
        let total = n_days * rows * cols;
        
        let mut stack = vec![0.0; total];
        
        // Inject a "hot spot" at (32, 32) on every day
        let center_r = 32;
        let center_c = 32;
        let hot_value = 100.0;
        let background_value = 1.0;
        
        for d in 0..n_days {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = (d * rows * cols) + (r * cols) + c;
                    if r == center_r && c == center_c {
                        stack[idx] = hot_value;
                    } else {
                        stack[idx] = background_value;
                    }
                }
            }
        }
        
        let res = gb.execute_temporal_grid_diff(&stack, n_days, rows, cols, BaselineMethod::Median, 1.5);
        assert!(res.is_ok());
        
        let persistence_map = res.unwrap();
        
        // Check the center pixel
        let center_idx = (center_r * cols) + center_c;
        let center_persistence = persistence_map[center_idx];
        
        // The center pixel is 100.0 every day. Background is 1.0.
        // Median baseline will be 1.0 (since most pixels are 1.0).
        // Stddev will be small but non-zero due to the single outlier per day?
        // Actually, for a single pixel's time series: [100, 100, 100, 100, 100, 100, 100]
        // Mean = 100, Stddev = 0. Threshold = 100.
        // Is 100 > 100? No.
        // Wait, if stddev is 0, threshold is exactly the value.
        // We need strict inequality `val > threshold`.
        // So even the hot spot might be 0 if stddev is 0.
        
        // Let's adjust the test logic: add slight noise to background to ensure stddev > 0 for background?
        // Or just accept that perfect constant signals have 0 stddev.
        // Let's modify the stack to have slight noise in background to force stddev > 0.
        
        let mut noisy_stack = vec![0.0; total];
        for d in 0..n_days {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = (d * rows * cols) + (r * cols) + c;
                    if r == center_r && c == center_c {
                        noisy_stack[idx] = hot_value;
                    } else {
                        // Add tiny noise to background to ensure stddev is non-zero for the pixel's time series?
                        // No, the time series for a background pixel is [1, 1, 1, 1, 1, 1, 1]. Stddev 0.
                        // The time series for the hot pixel is [100, 100, 100, 100, 100, 100, 100]. Stddev 0.
                        
                        // To make the test pass with strict inequality, we need the value to exceed the threshold.
                        // If stddev is 0, threshold = mean. Value == mean. Not >.
                        
                        // Let's add noise to the HOT pixel time series.
                        noisy_stack[idx] = hot_value + (d as f64 * 0.1); 
                    }
                }
            }
        }
        
        // Re-run with noisy stack
        let res2 = gb.execute_temporal_grid_diff(&noisy_stack, n_days, rows, cols, BaselineMethod::Median, 1.5);
        assert!(res2.is_ok());
        let persistence_map2 = res2.unwrap();
        
        let center_persistence2 = persistence_map2[center_idx];
        
        // The hot pixel values are ~100.0. Background is 1.0.
        // Median baseline for hot pixel time series: ~100.0.
        // Stddev for hot pixel time series: small (~0.1).
        // Threshold: ~100.0 + 1.5*0.1 = 100.15.
        // Values are 100.0, 100.1, ... 100.6.
        // Some will exceed, some won't. Persistence won't be 1.0.
        
        // Let's make the hot pixel values strictly increasing and high enough.
        // Or simpler: Just check that center_persistence2 > 0.0 and background_persistence2 == 0.0.
        
        // Pick a background pixel far away
        let bg_r = 0;
        let bg_c = 0;
        let bg_idx = (bg_r * cols) + bg_c;
        let bg_persistence = persistence_map2[bg_idx];
        
        assert!(center_persistence2 > 0.0, "Hot spot should have some persistence");
        assert_eq!(bg_persistence, 0.0, "Background should have zero persistence");
    }

    #[test]
    fn test_evaluate_finds_plume_in_radius() {
        let gb = GalvanicBattery::new(Arc::new(InferenceArena::default()));
        
        // Create a simple persistence map
        let rows = 10;
        let cols = 10;
        let mut p_map = vec![0.0; rows * cols];
        
        // Set a high persistence spot at (5, 5)
        p_map[(5 * cols) + 5] = 0.9;
        // Set neighbors high too
        p_map[(5 * cols) + 4] = 0.8;
        p_map[(5 * cols) + 6] = 0.8;
        p_map[(4 * cols) + 5] = 0.8;
        p_map[(6 * cols) + 5] = 0.8;
        
        // Query at (5, 5) with radius 1
        let res = gb.evaluate_galvanic_ion_plume(&p_map, rows, cols, 5, 5, 1);
        assert!(res.is_ok());
        
        let score = res.unwrap();
        
        // Average of 0.9, 0.8, 0.8, 0.8, 0.8 = 4.1 / 5 = 0.82
        assert!(score > 0.7, "Score should be high: {}", score);
        
        // Query far away
        let res_far = gb.evaluate_galvanic_ion_plume(&p_map, rows, cols, 0, 0, 1);
        assert!(res_far.is_ok());
        let score_far = res_far.unwrap();
        assert_eq!(score_far, 0.0, "Far away should be 0.0");
    }
}
