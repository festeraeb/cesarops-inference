//! Full-lake scan helpers (anchor calibration + anomaly selection).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    pub source: String,
    pub row: usize,
    pub col: usize,
}

pub fn anchor_points() -> Vec<GeoPoint> {
    vec![
        GeoPoint { lat: 41.8900, lon: -87.6044 },
        GeoPoint { lat: 42.3638, lon: -87.8034 },
        GeoPoint { lat: 41.7258, lon: -86.9047 },
        GeoPoint { lat: 42.8000, lon: -87.8178 },
        GeoPoint { lat: 42.7784, lon: -86.2066 },
        GeoPoint { lat: 43.2314, lon: -86.3481 },
        GeoPoint { lat: 44.7947, lon: -87.3142 },
        GeoPoint { lat: 44.6919, lon: -86.2544 },
    ]
}

pub fn apply_anchor_calibration(lat: f64, lon: f64) -> GeoPoint {
    let anchors = anchor_points();
    let mut total_weight = 0.0f64;
    let mut corr_lat = 0.0f64;
    let mut corr_lon = 0.0f64;
    for a in anchors {
        let dist = ((lat - a.lat).powi(2) + (lon - a.lon).powi(2)).sqrt();
        if dist < 1e-4 {
            return GeoPoint { lat, lon };
        }
        let w = 1.0 / (dist + 0.01);
        total_weight += w;
        corr_lat += a.lat * w;
        corr_lon += a.lon * w;
    }
    let blend = 0.9;
    GeoPoint {
        lat: lat * blend + (corr_lat / total_weight) * (1.0 - blend),
        lon: lon * blend + (corr_lon / total_weight) * (1.0 - blend),
    }
}

pub fn top_abs_zscores(grid: &[f32], width: usize, source: &str, top_n: usize) -> Vec<AnomalyDetection> {
    if grid.is_empty() || width == 0 {
        return vec![];
    }
    let n = grid.len() as f64;
    let mean = grid.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = grid.iter().map(|v| ((*v as f64) - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-9);
    let mut rows: Vec<(usize, f64)> = grid
        .iter()
        .enumerate()
        .map(|(i, v)| (i, ((*v as f64) - mean) / std))
        .collect();
    rows.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter()
        .take(top_n)
        .map(|(idx, z)| AnomalyDetection {
            lat: 0.0,
            lon: 0.0,
            zscore: z,
            source: source.to_string(),
            row: idx / width,
            col: idx % width,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_stays_near_input() {
        let p = apply_anchor_calibration(42.4, -87.2);
        assert!((p.lat - 42.4).abs() < 0.2);
        assert!((p.lon + 87.2).abs() < 0.2);
    }
}
