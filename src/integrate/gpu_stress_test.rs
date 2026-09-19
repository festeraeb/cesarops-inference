//! GPU stress test plan — port of `benchmarks/gpu_stress_test.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_PIXELS: u64 = 30_131_886;
pub const TILE_SIDE: u32 = 5490;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StressTestJob {
    pub tiff_path: PathBuf,
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StressTestResult {
    pub job: StressTestJob,
    pub elapsed_secs: f32,
    pub success: bool,
    pub pixels_per_second: Option<f64>,
    pub gpu_confirmed: bool,
}

impl StressTestResult {
    pub fn from_timing(job: StressTestJob, elapsed_secs: f32, success: bool, stdout: &str) -> Self {
        let pps = if success && elapsed_secs > 0.0 {
            Some(DEFAULT_PIXELS as f64 / elapsed_secs as f64)
        } else {
            None
        };
        Self {
            job,
            elapsed_secs,
            success,
            pixels_per_second: pps,
            gpu_confirmed: stdout.contains("Quadro M2200") && stdout.contains("is active"),
        }
    }
}

pub fn likely_gpu_used(elapsed_secs: f32) -> bool {
    (0.5..=30.0).contains(&elapsed_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_when_fast_enough() {
        let r = StressTestResult::from_timing(
            StressTestJob {
                tiff_path: PathBuf::from("a.tif"),
                threshold: 2.0,
            },
            2.0,
            true,
            "GPU processing complete Quadro M2200 is active",
        );
        assert!(r.pixels_per_second.unwrap() > 1_000_000.0);
    }
}
