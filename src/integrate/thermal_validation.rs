//! Thermal detection validation — port of `wreckhunter/tests/validate_detection.py`.

use serde::{Deserialize, Serialize};

pub const TARGET_LAT: f64 = 42.948873;
pub const TARGET_LON: f64 = -86.976619;
pub const ORIGINAL_ZSCORE: f64 = 5.46;
pub const ZSCORE_THRESHOLD: f64 = 2.5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermalValidationResult {
    pub mean: f32,
    pub std: f32,
    pub zscore_max: f32,
    pub anomaly_count: u32,
    pub target_found: bool,
}

pub fn thermal_zscore_stats(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let max_z = values
        .iter()
        .map(|v| ((v - mean) / (std + 1e-6)).abs())
        .fold(0.0f32, f32::max);
    (mean, std, max_z)
}

pub fn count_anomalies(values: &[f32], threshold: f64) -> u32 {
    let (mean, std, _) = thermal_zscore_stats(values);
    values
        .iter()
        .filter(|v| ((**v - mean) / (std + 1e-6)).abs() > threshold as f32)
        .count() as u32
}

pub fn pixel_from_geotransform(lon: f64, lat: f64, origin_x: f64, px_x: f64, origin_y: f64, px_y: f64) -> (i32, i32) {
    let col = ((lon - origin_x) / px_x) as i32;
    let row = ((lat - origin_y) / px_y) as i32;
    (row, col)
}
