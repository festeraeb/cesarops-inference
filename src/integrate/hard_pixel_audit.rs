//! Hard-pixel audit helpers (z-score + region anomalies + depth correction).

use serde::{Deserialize, Serialize};

pub const ZION_CONSTANT: f64 = 1.47;
pub const DEPTH_THRESHOLD_FT: f64 = 400.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionAnomaly {
    pub center_row: usize,
    pub center_col: usize,
    pub pixel_count: usize,
    pub max_abs_z: f64,
}

pub fn zscores(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = var.sqrt().max(1e-9);
    data.iter().map(|v| (v - mean) / std).collect()
}

pub fn estimate_length_ft(pixel_span: usize, pixel_size_m: f64) -> f64 {
    (pixel_span as f64 * pixel_size_m) * 3.28084
}

pub fn apply_zion_constant(detected_length_ft: f64, depth_ft: f64) -> f64 {
    if depth_ft > DEPTH_THRESHOLD_FT {
        detected_length_ft / ZION_CONSTANT
    } else {
        detected_length_ft
    }
}

pub fn pixel_distance_m(p1: (usize, usize), p2: (usize, usize), pixel_size_m: f64) -> f64 {
    let dr = p1.0 as f64 - p2.0 as f64;
    let dc = p1.1 as f64 - p2.1 as f64;
    (dr * dr + dc * dc).sqrt() * pixel_size_m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zion_correction_only_for_deep_sites() {
        assert!((apply_zion_constant(147.0, 500.0) - 100.0).abs() < 1e-6);
        assert!((apply_zion_constant(147.0, 180.0) - 147.0).abs() < 1e-6);
    }
}
