//! Three-tile offset analysis — port of `wreckhunter/tools/three_tile_offset_analysis.py`.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileSpec {
    pub base: String,
    pub date: String,
    pub satellite: String,
    pub resolution_m: u32,
    pub detection_lat: f64,
    pub detection_lon: f64,
    pub detection_zscore: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PixelAnomaly {
    pub pixel_y: u32,
    pub pixel_x: u32,
    pub zscore: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermalStats {
    pub mean: f32,
    pub std: f32,
    pub anomaly_count: u64,
    pub top_anomalies: Vec<PixelAnomaly>,
}

pub fn default_triple_lock_tile() -> TileSpec {
    TileSpec {
        base: "HLS.S30.T16TDN.2025244T163839.v2.0".into(),
        date: "2025-09-01".into(),
        satellite: "Sentinel-2".into(),
        resolution_m: 10,
        detection_lat: 42.948873,
        detection_lon: -86.976619,
        detection_zscore: 5.46,
    }
}

pub fn thermal_stats(b10: &Array2<f32>, b11: &Array2<f32>, z_threshold: f32, top_k: usize) -> ThermalStats {
    let thermal = (b10 + b11) / 2.0;
    let mean = thermal.mean().unwrap_or(0.0);
    let std = {
        let v = thermal.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        v.sqrt().max(1e-6)
    };
    let z = (&thermal - mean) / std;
    let anomaly_count = z.iter().filter(|&&v| v.abs() > z_threshold).count() as u64;
    let mut scored: Vec<(u32, u32, f32)> = z
        .indexed_iter()
        .map(|((y, x), &v)| (y as u32, x as u32, v.abs()))
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let top_anomalies = scored
        .into_iter()
        .take(top_k)
        .map(|(pixel_y, pixel_x, zscore)| PixelAnomaly {
            pixel_y,
            pixel_x,
            zscore,
        })
        .collect();
    ThermalStats {
        mean,
        std,
        anomaly_count,
        top_anomalies,
    }
}

pub fn offset_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    6371000.0 * 2.0 * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn thermal_stats_finds_anomalies() {
        let b10 = arr2(&[[1.0, 2.0], [3.0, 100.0]]);
        let b11 = arr2(&[[1.0, 2.0], [3.0, 100.0]]);
        let stats = thermal_stats(&b10, &b11, 1.5, 2);
        assert!(stats.anomaly_count >= 1);
        assert!(!stats.top_anomalies.is_empty());
    }

    #[test]
    fn offset_is_zero_for_same_point() {
        assert!(offset_meters(42.0, -87.0, 42.0, -87.0).abs() < 1.0);
    }
}
