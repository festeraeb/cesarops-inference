//! Full lake Michigan + Superior run — port of `wreckhunter/full_lake_michigan_run.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermalRunStats {
    pub mean: f32,
    pub std: f32,
    pub max_zscore: f32,
    pub anomaly_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakingBoatResult {
    pub oil_detected: bool,
    pub oil_pixel_count: u32,
    pub max_zscore: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileRunResult {
    pub filename: String,
    pub thermal: Option<ThermalRunStats>,
    pub leaking_boat: Option<LeakingBoatResult>,
}

pub fn process_thermal_pair(b10: &[f32], b11: &[f32]) -> ThermalRunStats {
    let n = b10.len().min(b11.len());
    if n == 0 {
        return ThermalRunStats {
            mean: 0.0,
            std: 0.0,
            max_zscore: 0.0,
            anomaly_count: 0,
        };
    }
    let thermal: Vec<f32> = b10
        .iter()
        .zip(b11.iter())
        .take(n)
        .map(|(a, b)| (a + b) / 2.0)
        .collect();
    let mean = thermal.iter().sum::<f32>() / n as f32;
    let std = (thermal.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    let mut max_z = 0.0f32;
    let mut count = 0u32;
    for v in &thermal {
        let z = ((v - mean) / (std + 1e-6)).abs();
        if z > max_z {
            max_z = z;
        }
        if z > 2.5 {
            count += 1;
        }
    }
    ThermalRunStats {
        mean,
        std,
        max_zscore: max_z,
        anomaly_count: count,
    }
}

pub fn default_search_dirs() -> Vec<&'static str> {
    vec![
        "wreckhunter2000/data/cache/census_raw/2021_low_water",
        "wreckhunter2000/data/cache/census_raw/2025_rossa",
    ]
}
