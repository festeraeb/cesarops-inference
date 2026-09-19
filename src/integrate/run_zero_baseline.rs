//! Run-zero baseline metadata — port of `benchmarks/run_zero.py` system block.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemInfo {
    pub gpu_name: String,
    pub gpu_vendor: String,
    pub gpu_type: String,
    pub cpu_cores: u32,
    pub system_ram_gb: f32,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            gpu_name: "Unknown".into(),
            gpu_vendor: "Unknown".into(),
            gpu_type: "Unknown".into(),
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(0),
            system_ram_gb: 48.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunZeroRecord {
    pub run_name: String,
    pub tile_count: u32,
    pub total_detections: u32,
    pub gpu_name: String,
    pub duration_seconds: f32,
}

pub fn parse_gpu_from_scanner_line(line: &str) -> Option<SystemInfo> {
    let lower = line.to_lowercase();
    if lower.contains("nvidia") {
        return Some(SystemInfo {
            gpu_name: line.trim().to_string(),
            gpu_vendor: "NVIDIA".into(),
            gpu_type: "Discrete".into(),
            ..Default::default()
        });
    }
    if lower.contains("intel") {
        return Some(SystemInfo {
            gpu_name: line.trim().to_string(),
            gpu_vendor: "Intel".into(),
            gpu_type: "Integrated".into(),
            ..Default::default()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_line() {
        let info = parse_gpu_from_scanner_line("NVIDIA Quadro M2200").unwrap();
        assert_eq!(info.gpu_vendor, "NVIDIA");
    }
}
