//! Per-tile processing — port of `wreckhunter/process_tiles.py`.

use serde::{Deserialize, Serialize};

pub const ZSCORE_THRESHOLD: f32 = 2.5;
pub const TOP_ANOMALY_LIMIT: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileAnomaly {
    pub row: u32,
    pub col: u32,
    pub zscore: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileProcessResult {
    pub tile: String,
    pub filename: String,
    pub mean: f32,
    pub std: f32,
    pub anomaly_count: u32,
    pub top_anomalies: Vec<TileAnomaly>,
    pub error: Option<String>,
}

pub fn process_tile_values(values: &[f32]) -> TileProcessResult {
    if values.is_empty() {
        return TileProcessResult {
            tile: String::new(),
            filename: String::new(),
            mean: 0.0,
            std: 0.0,
            anomaly_count: 0,
            top_anomalies: vec![],
            error: Some("empty".into()),
        };
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    let mut scored: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i, ((v - mean) / (std + 1e-6)).abs()))
        .filter(|(_, z)| *z > ZSCORE_THRESHOLD)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<TileAnomaly> = scored
        .into_iter()
        .take(TOP_ANOMALY_LIMIT)
        .map(|(idx, z)| TileAnomaly {
            row: (idx / 100) as u32,
            col: (idx % 100) as u32,
            zscore: z,
        })
        .collect();
    TileProcessResult {
        tile: String::new(),
        filename: String::new(),
        mean,
        std,
        anomaly_count: top.len() as u32,
        top_anomalies: top,
        error: None,
    }
}
