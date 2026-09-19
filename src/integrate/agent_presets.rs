//! Agent GUI presets — port of `cesarops_agent_gui.py` PRESETS (headless config).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessingPreset {
    pub bands: Vec<String>,
    pub zscore_threshold: f64,
    pub min_anomaly_count: u32,
    pub satellite: String,
}

pub fn default_presets() -> HashMap<&'static str, ProcessingPreset> {
    HashMap::from([
        (
            "Default Thermal",
            ProcessingPreset {
                bands: vec!["B10".into(), "B11".into()],
                zscore_threshold: 2.5,
                min_anomaly_count: 10,
                satellite: "Any".into(),
            },
        ),
        (
            "High Confidence",
            ProcessingPreset {
                bands: vec![
                    "B10".into(),
                    "B11".into(),
                    "B04".into(),
                    "B05".into(),
                ],
                zscore_threshold: 3.0,
                min_anomaly_count: 50,
                satellite: "Sentinel-2".into(),
            },
        ),
        (
            "Deep Wreck Hunt",
            ProcessingPreset {
                bands: vec!["B10".into(), "B11".into()],
                zscore_threshold: 2.0,
                min_anomaly_count: 100,
                satellite: "Landsat-8".into(),
            },
        ),
    ])
}
