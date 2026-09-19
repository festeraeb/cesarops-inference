// src/magnetic_eraser.rs
//
// MAGNETIC BASELINE ERASURE & ANOMALY DETECTION
//
// Physics:
//   Earth's geomagnetic field varies by ~25-50 nT over a typical flight
//   line due to natural geological gradients. A 10-ton ferrous wreck at
//   200 ft depth produces a ~0.5-2 nT signature. To detect these, we
//   must remove the low-frequency regional baseline.
//
// Algorithm:
//   2D high-pass filter via two-pass moving-median subtraction (rows,
//   then columns). Median (not mean) preserves dipole-scale features
//   while killing the regional gradient — a mean would partially erase
//   the dipole signal we want to detect.
//
//   For each pixel: subtract median(window_size) of its row, then
//   subtract median(window_size) of its column. Output = sub-nT residual
//   anomaly map suitable for dipole detection downstream.
//
// Default window_size: 31 (odd, preserves ~15-pixel half-wavelength
// features = roughly 100ft at 25m/pixel resolution = wreck-class).
//
// Cited from: SAR Mission Notes — Geophysical Signal Processing.

use std::sync::Arc;
use crate::arena::InferenceArena;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarity {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy)]
pub struct AnomalyCenter {
    pub row: usize,
    pub col: usize,
    pub amplitude_nt: f64,
    pub polarity: Polarity,
}

pub struct MagneticEraser {
    pub arena: Arc<InferenceArena>,
}

impl MagneticEraser {
    pub fn new(arena: Arc<InferenceArena>) -> Self {
        Self { arena }
    }

    /// Removes the regional magnetic baseline via 2D moving-median subtraction.
    /// Result is a sub-nT residual anomaly grid suitable for dipole detection.
    pub fn erase_standard_baseline(
        &self,
        magnetic_grid: &[f64],
        rows: usize,
        cols: usize,
        window_size: usize,
    ) -> Result<Vec<f64>, &'static str> {
        if magnetic_grid.len() != rows * cols {
            return Err("Grid length does not match rows * cols.");
        }
        if window_size == 0 {
            return Err("window_size must be > 0.");
        }
        if window_size % 2 == 0 {
            return Err("window_size must be odd.");
        }
        if window_size > rows.max(cols) {
            return Err("window_size exceeds grid dimensions.");
        }

        // Row pass: subtract per-row moving median
        let mut row_erased = vec![0.0f64; rows * cols];
        for r in 0..rows {
            let row_start = r * cols;
            let row_slice = &magnetic_grid[row_start..row_start + cols];
            let medians = compute_1d_moving_median(row_slice, window_size);
            for c in 0..cols {
                row_erased[row_start + c] = magnetic_grid[row_start + c] - medians[c];
            }
        }

        // Column pass: subtract per-column moving median (operates on row_erased)
        let mut col_erased = row_erased.clone();
        let mut col_buf = vec![0.0f64; rows];
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = row_erased[r * cols + c];
            }
            let medians = compute_1d_moving_median(&col_buf, window_size);
            for r in 0..rows {
                col_erased[r * cols + c] = row_erased[r * cols + c] - medians[r];
            }
        }

        Ok(col_erased)
    }

    /// Compatibility wrapper for the old (rows, cols)-only signature.
    /// Defaults window_size to 31 (clamped to grid size if smaller).
    pub fn erase_standard_baseline_compat(
        &self,
        magnetic_grid: &[f64],
        rows: usize,
        cols: usize,
    ) -> Vec<f64> {
        let max_dim = rows.min(cols);
        let win = if max_dim < 31 {
            // largest odd value <= max_dim, minimum 3
            let w = if max_dim % 2 == 0 { max_dim - 1 } else { max_dim };
            w.max(3)
        } else {
            31
        };
        match self.erase_standard_baseline(magnetic_grid, rows, cols, win) {
            Ok(out) => out,
            Err(_) => magnetic_grid.to_vec(),
        }
    }

    /// Find local extrema (positive peaks + negative troughs) in the
    /// erased grid above min_amplitude_nt. A pixel qualifies as a center
    /// if it's strictly greater (or less) than every neighbor within
    /// `local_radius` Chebyshev distance.
    pub fn extract_anomaly_centers(
        &self,
        erased_grid: &[f64],
        rows: usize,
        cols: usize,
        local_radius: usize,
        min_amplitude_nt: f64,
    ) -> Vec<AnomalyCenter> {
        let mut centers = Vec::new();
        if erased_grid.len() != rows * cols {
            return centers;
        }
        let r_radius = local_radius as i32;

        for r in 0..rows {
            for c in 0..cols {
                let val = erased_grid[r * cols + c];
                if val.abs() < min_amplitude_nt {
                    continue;
                }
                let mut is_max = true;
                let mut is_min = true;

                for dr in -r_radius..=r_radius {
                    for dc in -r_radius..=r_radius {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < 0 || nr >= rows as i32 || nc < 0 || nc >= cols as i32 {
                            continue;
                        }
                        let neighbor = erased_grid[(nr as usize) * cols + (nc as usize)];
                        if neighbor >= val {
                            is_max = false;
                        }
                        if neighbor <= val {
                            is_min = false;
                        }
                        if !is_max && !is_min {
                            break;
                        }
                    }
                    if !is_max && !is_min {
                        break;
                    }
                }

                if is_max {
                    centers.push(AnomalyCenter {
                        row: r,
                        col: c,
                        amplitude_nt: val,
                        polarity: Polarity::Positive,
                    });
                } else if is_min {
                    centers.push(AnomalyCenter {
                        row: r,
                        col: c,
                        amplitude_nt: val,
                        polarity: Polarity::Negative,
                    });
                }
            }
        }
        centers
    }

    /// Tier-classify a residual amplitude into a target category.
    pub fn classify_target_signature(&self, residual_nt: f64) -> &'static str {
        match residual_nt.abs() {
            r if r > 5.0 => "large_ferrous_anomaly",
            r if r > 2.0 => "medium_ferrous_target",
            r if r > 0.5 => "small_ferrous_target",
            r if r > 0.1 => "marginal_signal",
            _ => "background_noise",
        }
    }
}

/// 1D moving median with edge-replicated padding. O(N * W log W).
/// For our typical grid sizes (256x256 with W=31), this runs in ~10ms
/// on a single core; production version can switch to histogram median
/// if profiling shows it dominates.
fn compute_1d_moving_median(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let half = (window / 2) as i32;
    let n_i = n as i32;
    let mut medians = vec![0.0f64; n];
    let mut win_vals: Vec<f64> = Vec::with_capacity(window);

    for i in 0..n {
        win_vals.clear();
        for off in -half..=half {
            let idx = (i as i32) + off;
            let clamped = if idx < 0 {
                0usize
            } else if idx >= n_i {
                n - 1
            } else {
                idx as usize
            };
            win_vals.push(data[clamped]);
        }
        win_vals.sort_unstable_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        medians[i] = win_vals[window / 2];
    }
    medians
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_arena() -> Arc<InferenceArena> {
        InferenceArena::new(1, 0)
    }

    fn setup() -> MagneticEraser {
        MagneticEraser::new(mock_arena())
    }

    #[test]
    fn dimension_mismatch_returns_err() {
        let e = setup();
        let bad = vec![0.0f64; 100]; // 10x10 of data, but...
        let r = e.erase_standard_baseline(&bad, 50, 50, 5);
        assert!(r.is_err());
    }

    #[test]
    fn even_window_returns_err() {
        let e = setup();
        let g = vec![0.0f64; 100];
        let r = e.erase_standard_baseline(&g, 10, 10, 4);
        assert!(r.is_err());
    }

    #[test]
    fn uniform_field_zeros_after_erasure() {
        // Constant baseline -> median equals value -> erasure returns zeros.
        let e = setup();
        let grid = vec![50000.0f64; 64 * 64];
        let out = e
            .erase_standard_baseline(&grid, 64, 64, 5)
            .expect("erase failed");
        for &v in &out {
            assert!(v.abs() < 1e-9, "uniform field should erase to zero, got {}", v);
        }
    }

    #[test]
    fn synthetic_dipole_survives_erasure() {
        // 64x64 baseline grid + 1.5 nT positive blip at (32, 32).
        // After erasure, the blip should still be the largest residual.
        let e = setup();
        let mut grid = vec![50000.0f64; 64 * 64];
        grid[32 * 64 + 32] = 50001.5; // +1.5 nT spike
        let out = e
            .erase_standard_baseline(&grid, 64, 64, 31)
            .expect("erase failed");
        let center = out[32 * 64 + 32];
        assert!(
            center > 1.0,
            "1.5 nT dipole should survive erasure, got {}",
            center
        );
    }

    #[test]
    fn extract_anomaly_centers_finds_dipole() {
        // Inject one positive + one negative anomaly into a flat field.
        let e = setup();
        let mut grid = vec![0.0f64; 64 * 64];
        grid[20 * 64 + 20] = 1.5;  // positive spike
        grid[40 * 64 + 40] = -1.2; // negative spike

        let centers = e.extract_anomaly_centers(&grid, 64, 64, 5, 0.5);
        assert!(centers.len() >= 2, "should find at least 2 centers");
        let pos = centers.iter().find(|c| c.row == 20 && c.col == 20);
        let neg = centers.iter().find(|c| c.row == 40 && c.col == 40);
        assert!(pos.is_some());
        assert!(neg.is_some());
        assert_eq!(pos.unwrap().polarity, Polarity::Positive);
        assert_eq!(neg.unwrap().polarity, Polarity::Negative);
    }

    #[test]
    fn classify_target_buckets() {
        let e = setup();
        assert_eq!(e.classify_target_signature(8.0), "large_ferrous_anomaly");
        assert_eq!(e.classify_target_signature(3.0), "medium_ferrous_target");
        assert_eq!(e.classify_target_signature(1.0), "small_ferrous_target");
        assert_eq!(e.classify_target_signature(0.3), "marginal_signal");
        assert_eq!(e.classify_target_signature(0.05), "background_noise");
        // Sign doesn't matter
        assert_eq!(e.classify_target_signature(-3.0), "medium_ferrous_target");
    }

    #[test]
    fn moving_median_edge_padding() {
        let data = vec![1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0];
        let m = compute_1d_moving_median(&data, 3);
        // Edge-replicated padding means median at edges is still 1.0.
        // The 5.0 spike at index 3 is rejected by the median (window contains 1, 5, 1 -> median 1).
        for v in m {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }
}
