//! Tile Z-score anomaly stats — Rust port of `cesarops_engine.gpu_process_tile` (CPU path).

use ndarray::Array2;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TileZscoreResult {
    pub tile_id: String,
    pub anomaly_count: u64,
    pub mean: f32,
    pub std: f32,
    pub max_zscore: f32,
    pub gpu_used: bool,
    pub backend: &'static str,
}

/// Grayscale `tile` values (row-major), Z-score threshold default 2.5.
pub fn process_tile_gray(tile: &Array2<f32>, tile_id: &str, z_thresh: f32) -> TileZscoreResult {
    let mean = tile.mean().unwrap_or(0.0);
    let var = tile.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
    let std = var.sqrt().max(1e-6);

    let z = (tile - mean) / std;
    let abs_z = z.mapv(f32::abs);
    let anomaly_count = abs_z.iter().filter(|&&v| v > z_thresh).count() as u64;
    let max_zscore = abs_z.iter().copied().fold(0.0_f32, f32::max);

    TileZscoreResult {
        tile_id: tile_id.to_string(),
        anomaly_count,
        mean,
        std,
        max_zscore,
        gpu_used: false,
        backend: "rust_ndarray",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn finds_spike() {
        let mut tile = Array2::<f32>::zeros((8, 8));
        tile[[4, 4]] = 1000.0;
        let r = process_tile_gray(&tile, "t0", 3.0);
        assert!(r.anomaly_count >= 1, "count={} max_z={}", r.anomaly_count, r.max_zscore);
    }
}
