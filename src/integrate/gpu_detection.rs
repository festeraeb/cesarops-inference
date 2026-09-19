//! GPU validation — port of `tests/hardware/test_gpu_validation.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuValidationResult {
    pub rust_engine_ok: bool,
    pub m2200_detected: bool,
    pub nvidia_detected: bool,
    pub message: String,
}

pub fn evaluate_rust_gpu_stdout(stdout: &str) -> GpuValidationResult {
    let m2200 = stdout.contains("Quadro M2200");
    let nvidia = stdout.contains("NVIDIA");
    GpuValidationResult {
        rust_engine_ok: m2200 || nvidia,
        m2200_detected: m2200,
        nvidia_detected: nvidia,
        message: if m2200 {
            "Quadro M2200 detected and active".into()
        } else if nvidia {
            "NVIDIA GPU detected but not M2200".into()
        } else {
            "No NVIDIA GPU detected".into()
        },
    }
}

pub fn evaluate_wgpu_adapter(info: &str) -> bool {
    info.contains("NVIDIA")
}
