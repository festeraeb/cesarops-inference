//! End-to-end GPU pipeline test — port of `tests/test_end_to_end_gpu.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineStep {
    pub index: u32,
    pub total: u32,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineTestReport {
    pub build_ok: bool,
    pub gpu_detect_ok: bool,
    pub tiff_process_ok: bool,
    pub m2200_confirmed: bool,
}

pub const PIPELINE_STEPS: [&str; 4] = [
    "Building Rust GPU Engine",
    "Testing GPU Detection",
    "Processing Test TIFF",
    "Summary",
];

pub fn evaluate_gpu_detect(stdout: &str) -> (bool, bool) {
    let m2200 = stdout.contains("Quadro M2200");
    let nvidia = stdout.contains("NVIDIA");
    (m2200 || nvidia, m2200)
}

pub fn evaluate_tiff_process(stdout: &str) -> bool {
    stdout.contains("GPU processing complete") || stdout.contains("Detected")
}
