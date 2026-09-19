// src/galvanic_battery.rs
//
// GALVANIC BATTERY / ION PLUME DETECTION
//
// Physics:
//   When dissimilar metals (steel hull, bronze fittings, copper wiring)
//   sit submerged in salt or fresh water, they form a galvanic cell.
//   The hull becomes the anode and slowly dissolves, releasing iron
//   ions into the surrounding water. This produces a detectable
//   plume in spectral imaging — specifically:
//
//   - Iron ion plume causes localized increase in turbidity at SWIR bands
//   - Plume has temporal signature: persistent at the wreck, dispersed
//     downstream by currents
//   - Cross-day satellite stack reveals the plume by subtracting baseline
//     mean/median across N days from each daily frame
//
// This is the third leg of the triple-lock:
//   Vision says "anomaly here" → Physics says "and there's a galvanic
//   plume above it that doesn't move with currents."
//
// Algorithm:
//   1. For each pixel, build a temporal time series across n_days
//   2. Compute baseline (mean or median) + standard deviation
//   3. Count days the pixel exceeded baseline + threshold_stddevs * stddev
//   4. Normalize count to [0, 1] = persistence score
//
// A persistence map is then evaluated at candidate locations from the
// vision pipeline. High mean persistence in a small radius around a
// candidate confirms a stationary plume signature.
//
// Cited from: SAR Mission Notes — Electrochemical Corrosion Signatures.

use std::sync::Arc;
use crate::arena::InferenceArena;

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

    /// Compute a temporal persistence map from a stack of SWIR frames.
    /// `temporal_stack` is laid out as [day][row][col] row-major.
    /// Returns a Vec<f64> of size rows*cols with values in [0.0, 1.0].
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
            return Err("temporal_stack length must equal n_days * rows * cols");
        }
        if n_days == 0 || rows == 0 || cols == 0 {
            return Err("dimensions must be non-zero");
        }
        if !threshold_stddevs.is_finite() || threshold_stddevs < 0.0 {
            return Err("threshold_stddevs must be finite and non-negative");
        }

        let mut persistence_map = vec![0.0f64; rows * cols];
        let mut temporal_buffer = vec![0.0f64; n_days];
        let plane = rows * cols;
        let n_days_f = n_days as f64;

        for r in 0..rows {
            for c in 0..cols {
                let pixel_index = r * cols + c;
                for d in 0..n_days {
                    temporal_buffer[d] = temporal_stack[d * plane + pixel_index];
                }

                let (baseline, stddev) = compute_stats(&temporal_buffer, baseline_method);
                let threshold = baseline + threshold_stddevs * stddev;

                let mut exceedance_count = 0u32;
                for &val in &temporal_buffer {
                    if val > threshold {
                        exceedance_count += 1;
                    }
                }
                persistence_map[pixel_index] = exceedance_count as f64 / n_days_f;
            }
        }

        Ok(persistence_map)
    }

    /// Evaluate plume strength at a candidate location: mean persistence
    /// within `search_radius` (Chebyshev) of the candidate.
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
            return Err("persistence_map length must equal rows * cols");
        }
        if candidate_row >= rows || candidate_col >= cols {
            return Err("candidate coordinates out of bounds");
        }

        let r_min = candidate_row.saturating_sub(search_radius);
        let r_max = (candidate_row + search_radius).min(rows - 1);
        let c_min = candidate_col.saturating_sub(search_radius);
        let c_max = (candidate_col + search_radius).min(cols - 1);

        let mut sum = 0.0f64;
        let mut count = 0usize;
        for r in r_min..=r_max {
            for c in c_min..=c_max {
                sum += persistence_map[r * cols + c];
                count += 1;
            }
        }
        if count == 0 {
            return Err("search window resulted in zero pixels");
        }
        Ok(sum / count as f64)
    }

    /// Tier-classify a plume strength score.
    pub fn classify_plume_signature(&self, plume_strength: f64) -> &'static str {
        match plume_strength {
            s if s > 0.8 => "strong_galvanic_plume",
            s if s > 0.5 => "moderate_plume",
            s if s > 0.25 => "weak_plume",
            _ => "no_plume_signal",
        }
    }
}

/// Compute (baseline, stddev) over a slice using either mean or median
/// for the central tendency. Stddev is always computed against the mean
/// (median absolute deviation is a different statistic).
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
            let mut sorted: Vec<f64> = data.to_vec();
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

    fn mock_arena() -> Arc<InferenceArena> {
        InferenceArena::new(1, 0)
    }

    fn setup() -> GalvanicBattery {
        GalvanicBattery::new(mock_arena())
    }

    #[test]
    fn dimension_mismatch_returns_err() {
        let gb = setup();
        // 2 days * 2 rows * 2 cols = 8 expected; provide 10
        let stack = vec![1.0f64; 10];
        let r = gb.execute_temporal_grid_diff(&stack, 2, 2, 2, BaselineMethod::Mean, 1.5);
        assert!(r.is_err());
    }

    #[test]
    fn zero_dims_returns_err() {
        let gb = setup();
        let stack = vec![0.0f64; 0];
        let r = gb.execute_temporal_grid_diff(&stack, 0, 0, 0, BaselineMethod::Mean, 1.5);
        assert!(r.is_err());
    }

    #[test]
    fn uniform_stack_zero_persistence() {
        // All pixels identical across all days -> stddev=0 -> threshold=mean.
        // val > threshold is false everywhere (val == mean exactly).
        let gb = setup();
        let stack = vec![10.0f64; 5 * 4 * 4];
        let map = gb
            .execute_temporal_grid_diff(&stack, 5, 4, 4, BaselineMethod::Mean, 1.5)
            .expect("should succeed");
        for &p in &map {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn synthetic_plume_high_persistence() {
        // 7 days, 64x64 grid. Inject a hot pixel at (32, 32) with values
        // that produce a high mean and a tail of stronger spikes (so a
        // multi-stddev threshold gets exceeded most days). Background is
        // perfectly uniform so its persistence stays at 0.
        let gb = setup();
        let n_days = 7;
        let rows = 64;
        let cols = 64;
        let total = n_days * rows * cols;

        let mut stack = vec![0.0f64; total];
        let center_r = 32usize;
        let center_c = 32usize;
        let plane = rows * cols;

        // Hot pixel pattern: most days at ~100, two days at ~120.
        // The threshold (mean + 0.5*stddev) puts those 2 days clearly
        // above. Background pixel pattern: 1.0 every day = stddev 0,
        // threshold == mean, no day exceeds.
        let hot_series = [100.0f64, 100.0, 100.0, 120.0, 120.0, 100.0, 100.0];

        for d in 0..n_days {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = d * plane + r * cols + c;
                    stack[idx] = if r == center_r && c == center_c {
                        hot_series[d]
                    } else {
                        1.0
                    };
                }
            }
        }

        let map = gb
            .execute_temporal_grid_diff(
                &stack,
                n_days,
                rows,
                cols,
                BaselineMethod::Median,
                0.5,
            )
            .expect("should succeed");

        let center_p = map[center_r * cols + center_c];
        let bg_p = map[0];
        // 2 of 7 days exceed -> 2/7 ≈ 0.286 for the hot pixel
        // Background never exceeds (uniform) -> 0.0
        assert!(
            center_p > bg_p,
            "hot pixel persistence ({}) should exceed background ({})",
            center_p,
            bg_p
        );
        assert_eq!(bg_p, 0.0, "uniform background should have zero persistence");
        assert!(
            center_p > 0.0,
            "hot pixel should have non-zero persistence, got {}",
            center_p
        );
    }

    #[test]
    fn evaluate_finds_plume_in_radius() {
        let gb = setup();
        let rows = 10;
        let cols = 10;
        let mut map = vec![0.0f64; rows * cols];
        // Cluster of high-persistence pixels around (5, 5)
        map[5 * cols + 5] = 0.9;
        map[5 * cols + 4] = 0.8;
        map[5 * cols + 6] = 0.8;
        map[4 * cols + 5] = 0.8;
        map[6 * cols + 5] = 0.8;

        let near = gb
            .evaluate_galvanic_ion_plume(&map, rows, cols, 5, 5, 1)
            .expect("near eval failed");
        let far = gb
            .evaluate_galvanic_ion_plume(&map, rows, cols, 0, 0, 1)
            .expect("far eval failed");

        // The 3x3 window at (5,5) sums to 0.9 + 4*0.8 + 4*0 = 4.1, mean = 4.1/9
        assert!(near > 0.4, "near plume score should be substantial: {}", near);
        assert_eq!(far, 0.0, "far from plume should be zero, got {}", far);
    }

    #[test]
    fn classify_plume_buckets() {
        let gb = setup();
        assert_eq!(gb.classify_plume_signature(0.9), "strong_galvanic_plume");
        assert_eq!(gb.classify_plume_signature(0.6), "moderate_plume");
        assert_eq!(gb.classify_plume_signature(0.3), "weak_plume");
        assert_eq!(gb.classify_plume_signature(0.1), "no_plume_signal");
    }

    #[test]
    fn out_of_bounds_candidate_returns_err() {
        let gb = setup();
        let map = vec![0.0f64; 100];
        let r = gb.evaluate_galvanic_ion_plume(&map, 10, 10, 99, 99, 1);
        assert!(r.is_err());
    }
}
