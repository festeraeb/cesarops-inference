//! Minimal GPU health check — port of `tests/hardware/test_cuda_minimal.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaMinimalTestResult {
    pub device: GpuDeviceInfo,
    pub upload_ok: bool,
    pub download_ok: bool,
    pub data_match: bool,
}

pub fn evaluate_transfer(cpu: &[f32], gpu_roundtrip: &[f32]) -> CudaMinimalTestResult {
    let data_match = cpu.len() == gpu_roundtrip.len()
        && cpu.iter().zip(gpu_roundtrip).all(|(a, b)| (a - b).abs() < 1e-5);
    CudaMinimalTestResult {
        device: GpuDeviceInfo {
            name: "unknown".into(),
            compute_major: 0,
            compute_minor: 0,
        },
        upload_ok: !gpu_roundtrip.is_empty(),
        download_ok: !gpu_roundtrip.is_empty(),
        data_match,
    }
}

pub fn is_success(r: &CudaMinimalTestResult) -> bool {
    r.upload_ok && r.download_ok && r.data_match
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_when_equal() {
        let cpu = vec![1.0, 2.0, 3.0];
        let r = evaluate_transfer(&cpu, &cpu);
        assert!(is_success(&r));
    }
}
