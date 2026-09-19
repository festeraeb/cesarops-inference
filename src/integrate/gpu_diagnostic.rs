//! GPU diagnostic helpers — port of `diagnose_gpu.py`.

use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuCheckResult {
    pub passed: bool,
    pub message: String,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OverallStatus {
    Ready,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuDiagnosticResult {
    pub nvidia_driver: GpuCheckResult,
    pub vulkan_runtime: GpuCheckResult,
    pub rust_gpu_engine: GpuCheckResult,
    pub overall_status: OverallStatus,
}

pub type DiagnosticSummary = GpuDiagnosticResult;

impl GpuCheckResult {
    pub fn new(passed: bool, message: &str, details: Option<String>) -> Self {
        Self {
            passed,
            message: message.to_string(),
            details,
        }
    }
}

impl GpuDiagnosticResult {
    pub fn new(
        nvidia_driver: GpuCheckResult,
        vulkan_runtime: GpuCheckResult,
        rust_gpu_engine: GpuCheckResult,
    ) -> Self {
        let passed = [nvidia_driver.passed, vulkan_runtime.passed, rust_gpu_engine.passed]
            .iter()
            .filter(|&&p| p)
            .count();
        let overall_status = match passed {
            3 => OverallStatus::Ready,
            2 => OverallStatus::Partial,
            _ => OverallStatus::Failed,
        };
        Self {
            nvidia_driver,
            vulkan_runtime,
            rust_gpu_engine,
            overall_status,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.overall_status == OverallStatus::Ready
    }

    pub fn report(&self) -> String {
        format!(
            "NVIDIA: {}\nVulkan: {}\nRust GPU: {}\nOverall: {:?}",
            self.nvidia_driver.message, self.vulkan_runtime.message, self.rust_gpu_engine.message,
            self.overall_status
        )
    }
}

pub fn nvidia_status_from_output(stdout: &str) -> GpuCheckResult {
    if stdout.contains("Quadro M2200") {
        GpuCheckResult::new(true, "Quadro M2200 detected", Some(stdout.to_string()))
    } else if stdout.contains("NVIDIA") {
        GpuCheckResult::new(false, "NVIDIA GPU but not M2200", Some(stdout.to_string()))
    } else {
        GpuCheckResult::new(false, "No NVIDIA GPU in output", Some(stdout.to_string()))
    }
}

pub fn vulkan_status_from_output(stdout: &str) -> GpuCheckResult {
    if stdout.contains("NVIDIA") {
        GpuCheckResult::new(true, "Vulkan NVIDIA present", Some(stdout.to_string()))
    } else {
        GpuCheckResult::new(false, "Vulkan missing NVIDIA", Some(stdout.to_string()))
    }
}

pub fn rust_gpu_status_from_output(stdout: &str) -> GpuCheckResult {
    if stdout.contains("Quadro M2200") {
        GpuCheckResult::new(true, "Rust engine sees M2200", Some(stdout.to_string()))
    } else {
        GpuCheckResult::new(false, "Rust engine GPU check inconclusive", Some(stdout.to_string()))
    }
}

pub fn summarize_diagnostic(result: &GpuDiagnosticResult) -> String {
    result.report()
}

pub fn check_nvidia_smi() -> GpuCheckResult {
    match Command::new("nvidia-smi").output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            if out.status.success() {
                nvidia_status_from_output(&stdout)
            } else {
                GpuCheckResult::new(false, "nvidia-smi failed", Some(stdout))
            }
        }
        Err(e) => GpuCheckResult::new(false, "nvidia-smi missing", Some(e.to_string())),
    }
}

pub fn check_vulkan() -> GpuCheckResult {
    match Command::new("vulkaninfo").arg("--summary").output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            if out.status.success() {
                vulkan_status_from_output(&stdout)
            } else {
                GpuCheckResult::new(false, "vulkaninfo failed", Some(stdout))
            }
        }
        Err(e) => GpuCheckResult::new(false, "vulkaninfo missing", Some(e.to_string())),
    }
}

pub fn check_rust_gpu_engine() -> GpuCheckResult {
    GpuCheckResult::new(
        false,
        "Rust GPU engine check skipped in library build",
        None,
    )
}

pub fn run_diagnostic() -> GpuDiagnosticResult {
    GpuDiagnosticResult::new(check_nvidia_smi(), check_vulkan(), check_rust_gpu_engine())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overall_ready_when_all_pass() {
        let r = GpuDiagnosticResult::new(
            GpuCheckResult::new(true, "a", None),
            GpuCheckResult::new(true, "b", None),
            GpuCheckResult::new(true, "c", None),
        );
        assert!(r.is_ready());
    }

    #[test]
    fn parses_nvidia_output() {
        let r = nvidia_status_from_output("NVIDIA Quadro M2200");
        assert!(r.passed);
    }
}
