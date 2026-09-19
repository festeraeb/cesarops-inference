//! Config-driven GPU pipeline — port of `wreckhunter/run_configured_pipeline.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineConfig {
    pub min_confidence: f32,
    pub threshold: f32,
    pub overlap: f32,
    pub data_dir: PathBuf,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            threshold: 2.5,
            overlap: 0.1,
            data_dir: PathBuf::from("wreckhunter2000/data"),
        }
    }
}

pub const THERMAL_GLOBS: &[&str] = &["**/*B10.tif", "**/*B11.tif"];

pub fn gpu_command(tiff: &PathBuf, config: &PipelineConfig) -> Vec<String> {
    vec![
        "cesarops-gpu".into(),
        tiff.display().to_string(),
        "--threshold".into(),
        config.threshold.to_string(),
    ]
}

pub fn load_config_json(value: &serde_json::Value) -> PipelineConfig {
    PipelineConfig {
        min_confidence: value
            .get("min_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32,
        threshold: value
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(2.5) as f32,
        overlap: value
            .get("overlap")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1) as f32,
        data_dir: value
            .get("data_dir")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("wreckhunter2000/data")),
    }
}
