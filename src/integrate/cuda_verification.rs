//! CUDA toolkit verification — port of `wreckhunter/utils/verify_cuda.py` (nvidia-smi path).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaVerificationStep {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaVerificationReport {
    pub steps: Vec<CudaVerificationStep>,
    pub gpu_name: Option<String>,
    pub compute_capability: Option<String>,
    pub memory_gb: Option<f32>,
    pub all_passed: bool,
}

pub fn parse_nvidia_smi_query(stdout: &str) -> CudaVerificationReport {
    let mut gpu_name = None;
    let mut compute = None;
    let mut memory_gb = None;
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if gpu_name.is_none() && !line.contains("utilization") {
            gpu_name = Some(line.to_string());
        }
        if let Some(cap) = line.strip_prefix("Compute Capability:") {
            compute = Some(cap.trim().to_string());
        }
        if let Some(mem) = line.strip_prefix("Total Memory:") {
            if let Some(gb) = mem.trim().strip_suffix("GB") {
                memory_gb = gb.trim().parse().ok();
            }
        }
    }
    let mut steps = vec![CudaVerificationStep {
        name: "gpu_detection".into(),
        passed: gpu_name.is_some(),
        detail: gpu_name.clone().unwrap_or_else(|| "no GPU line".into()),
    }];
    if let Some(ref name) = gpu_name {
        steps.push(CudaVerificationStep {
            name: "m2200_expected".into(),
            passed: name.contains("M2200") || name.contains("Quadro"),
            detail: name.clone(),
        });
    }
    let all_passed = steps.iter().all(|s| s.passed);
    CudaVerificationReport {
        steps,
        gpu_name,
        compute_capability: compute,
        memory_gb,
        all_passed,
    }
}

pub fn summarize_verification(report: &CudaVerificationReport) -> String {
    let status = if report.all_passed {
        "[SUCCESS] CUDA TOOLKIT VERIFIED"
    } else {
        "[FAIL] CUDA verification incomplete"
    };
    format!(
        "{status}\nGPU: {}\nCompute: {}\nMemory GB: {:?}",
        report.gpu_name.as_deref().unwrap_or("unknown"),
        report.compute_capability.as_deref().unwrap_or("unknown"),
        report.memory_gb
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_smi_block() {
        let out = "Quadro M2200\nCompute Capability: 5.2\nTotal Memory: 4.0 GB\n";
        let r = parse_nvidia_smi_query(out);
        assert!(r.all_passed);
        assert_eq!(r.gpu_name.as_deref(), Some("Quadro M2200"));
    }
}
