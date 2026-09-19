//! Xenon CUDA checker — port of `check_xenon_cuda.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommandProbe {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct XenonCudaStatus {
    pub reachable: bool,
    pub cuda_device_count: u32,
    pub summary: String,
}

pub fn parse_cuda_device_count(text: &str) -> u32 {
    for line in text.lines() {
        let lower = line.to_lowercase();
        if lower.contains("cuda device") || lower.contains("device count") {
            if let Some(n) = line.split_whitespace().find_map(|t| t.parse::<u32>().ok()) {
                return n;
            }
        }
        if line.trim().parse::<u32>().is_ok() && lower.contains("gpu") {
            if let Ok(n) = line.trim().parse() {
                return n;
            }
        }
    }
    text.lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .next()
        .unwrap_or(0)
}

pub fn summarize_xenon_status(probe: &CommandProbe) -> XenonCudaStatus {
    let count = parse_cuda_device_count(&probe.stdout);
    let reachable = probe.exit_code == 0;
    let summary = if reachable && count > 0 {
        format!("Xenon CUDA OK ({count} device(s))")
    } else if reachable {
        "Xenon reachable but no CUDA devices parsed".into()
    } else {
        format!("Xenon probe failed: {}", probe.stderr.trim())
    };
    XenonCudaStatus {
        reachable,
        cuda_device_count: count,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_count() {
        assert_eq!(parse_cuda_device_count("CUDA devices: 2"), 2);
    }

    #[test]
    fn summarizes_ok() {
        let probe = CommandProbe {
            command: "nvidia-smi -L".into(),
            exit_code: 0,
            stdout: "GPU 0\nGPU 1".into(),
            stderr: String::new(),
        };
        let s = summarize_xenon_status(&probe);
        assert!(s.reachable);
    }
}
