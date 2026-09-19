//! CUDA benchmark + KMZ export — port of `tests/benchmarks/cuda_test_kmz.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaDeviceInfo {
    pub available: bool,
    pub gpu_name: Option<String>,
    pub compute_capability: Option<String>,
    pub total_memory_mb: Option<u32>,
    pub multiprocessor_count: Option<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaBenchmarkResults {
    pub matmul_1024_ms: Option<f32>,
    pub zscore_2048_ms: Option<f32>,
    pub anomaly_count: Option<u32>,
    pub memory_bandwidth_mbps: Option<f32>,
    pub filter_512_ms: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KmzPlacemark {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub z_score: f32,
}

pub fn census_log_payload(run_name: &str, tile_count: u32, detection_count: u32) -> serde_json::Value {
    serde_json::json!({
        "run_name": run_name,
        "tile_count": tile_count,
        "detection_count": detection_count,
        "classification": "cesarops_cuda_test",
    })
}

pub fn parse_gpu_line(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        if line.contains("GPU") || line.contains("NVIDIA") || line.contains("Quadro") {
            return Some(line.trim().to_string());
        }
    }
    None
}
