//! Small batch tile test — port of `wreckhunter/small_batch_anomaly_test.py`.

use serde::{Deserialize, Serialize};

pub const MAX_TILES: u32 = 5;
pub const ZSCORE_THRESHOLD: f32 = 2.5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileBatchResult {
    pub filename: String,
    pub mean: f32,
    pub std: f32,
    pub max_zscore: f32,
    pub anomaly_count: u32,
    pub error: Option<String>,
}

pub fn analyze_tile_stats(values: &[f32]) -> TileBatchResult {
    if values.is_empty() {
        return TileBatchResult {
            filename: String::new(),
            mean: 0.0,
            std: 0.0,
            max_zscore: 0.0,
            anomaly_count: 0,
            error: Some("empty tile".into()),
        };
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let mut max_z = 0.0f32;
    let mut anomalies = 0u32;
    for v in values {
        let z = ((v - mean) / (std + 1e-6)).abs();
        if z > max_z {
            max_z = z;
        }
        if z > ZSCORE_THRESHOLD {
            anomalies += 1;
        }
    }
    TileBatchResult {
        filename: String::new(),
        mean,
        std,
        max_zscore: max_z,
        anomaly_count: anomalies,
        error: None,
    }
}

pub fn take_first_n(paths: &[String], n: u32) -> Vec<String> {
    paths.iter().take(n as usize).cloned().collect()
}
