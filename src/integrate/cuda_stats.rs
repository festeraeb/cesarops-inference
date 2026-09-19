//! CUDA-style tile stats (CPU/Rust) — port of `cuda_direct.process_tiff_cuda` core math.

use ndarray::Array2;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaTileStats {
    pub mean: f32,
    pub std: f32,
    pub anomaly_count: u64,
    pub threshold: f32,
}

/// Z-score anomaly count on a float raster (replaces CuPy path for fleet CPU nodes).
pub fn tile_anomaly_stats(tile: &Array2<f32>, threshold: f32, passes: u32) -> CudaTileStats {
    let mut mean = tile.mean().unwrap_or(0.0);
    let mut std = {
        let v = tile.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        v.sqrt().max(1e-6)
    };
    for _ in 0..passes.saturating_sub(1) {
        let z = (tile - mean) / std;
        mean = z.mean().unwrap_or(mean);
        let v = z.mapv(|x| x.powi(2)).mean().unwrap_or(0.0);
        std = v.sqrt().max(1e-6);
    }
    let z = (tile - mean) / std;
    let anomaly_count = z.iter().filter(|&&v| v.abs() > threshold).count() as u64;
    CudaTileStats {
        mean,
        std,
        anomaly_count,
        threshold,
    }
}
