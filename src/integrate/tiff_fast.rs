//! Fast TIFF z-score scan — port of `fast_scan.py` (CPU/ndarray path).

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RasterShape {
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnomalyPoint {
    pub row: u32,
    pub col: u32,
    pub zscore: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastScanResult {
    pub processor: String,
    pub shape: RasterShape,
    pub anomalies: Vec<AnomalyPoint>,
    pub total_anomalies: u64,
    pub threshold: f32,
}

pub fn process_tiff_fast(tile: &Array2<f32>, threshold: f32, max_points: usize) -> FastScanResult {
    let mean = tile.mean().unwrap_or(0.0);
    let std = {
        let v = tile.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        v.sqrt().max(1e-6)
    };
    let z = (tile - mean) / std;
    let total_anomalies = z.iter().filter(|&&v| v.abs() > threshold).count() as u64;
    let mut scored: Vec<(u32, u32, f32)> = z
        .indexed_iter()
        .filter(|(_, &v)| v.abs() > threshold)
        .map(|((y, x), v)| (y as u32, x as u32, *v))
        .collect();
    scored.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let anomalies = scored
        .into_iter()
        .take(max_points)
        .map(|(row, col, zscore)| AnomalyPoint { row, col, zscore })
        .collect();
    FastScanResult {
        processor: "rust-ndarray".into(),
        shape: RasterShape {
            width: tile.ncols(),
            height: tile.nrows(),
        },
        anomalies,
        total_anomalies,
        threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn finds_hot_pixel() {
        let tile = arr2(&[[1.0, 2.0], [3.0, 50.0]]);
        let r = process_tiff_fast(&tile, 1.5, 10);
        assert!(r.total_anomalies >= 1);
    }
}
