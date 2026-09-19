//! Straits + Fox Island runner — port of `wreckhunter/straits_fox_runner.py`.

use serde::{Deserialize, Serialize};

pub const REQUIRED_PACKAGES: &[&str] = &["h5py", "numpy", "rasterio", "requests"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineScript {
    pub name: String,
    pub relative_path: String,
}

pub fn straits_fox_scripts() -> Vec<PipelineScript> {
    vec![
        PipelineScript {
            name: "VIIRS thermal ingest".into(),
            relative_path: "scripts/viirs_ingest.py".into(),
        },
        PipelineScript {
            name: "GPU anomaly pass".into(),
            relative_path: "scripts/gpu_scan.py".into(),
        },
    ]
}

pub fn missing_packages(installed: &[&str]) -> Vec<&'static str> {
    REQUIRED_PACKAGES
        .iter()
        .copied()
        .filter(|p| !installed.iter().any(|i| i == p))
        .collect()
}
