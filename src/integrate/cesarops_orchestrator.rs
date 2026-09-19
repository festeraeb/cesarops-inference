//! Sensor probe orchestrator planning primitives.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaConfig {
    pub bbox: [f64; 4],
    pub label: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    pub thermal_zscore: f64,
    pub sar_coherence: f64,
    pub glint_ratio_b08_b04: f64,
    pub swot_ssh_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorPlan {
    pub area_key: String,
    pub sensors: Vec<String>,
    pub thresholds: Thresholds,
}

pub fn known_areas() -> HashMap<&'static str, AreaConfig> {
    HashMap::from([
        ("lake_michigan_south", AreaConfig { bbox: [-88.0, 42.0, -87.0, 43.0], label: "Lake MI South (Zion Trench/Andaste)" }),
        ("lake_superior", AreaConfig { bbox: [-91.0, 46.5, -84.5, 48.0], label: "Lake Superior (Deep Basin)" }),
    ])
}

pub fn default_thresholds() -> Thresholds {
    Thresholds {
        thermal_zscore: 2.5,
        sar_coherence: 0.6,
        glint_ratio_b08_b04: 1.5,
        swot_ssh_m: 0.015,
    }
}

pub fn build_plan(area_key: &str, sensors: &[&str], thresholds: Option<Thresholds>) -> Option<OrchestratorPlan> {
    if !known_areas().contains_key(area_key) {
        return None;
    }
    let allowed = ["thermal", "nir_swir", "sar", "swot"];
    let mut sensor_vec = Vec::new();
    for s in sensors {
        if allowed.contains(s) {
            sensor_vec.push((*s).to_string());
        }
    }
    if sensor_vec.is_empty() {
        sensor_vec.push("thermal".to_string());
    }
    Some(OrchestratorPlan {
        area_key: area_key.to_string(),
        sensors: sensor_vec,
        thresholds: thresholds.unwrap_or_else(default_thresholds),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_default_plan() {
        let p = build_plan("lake_michigan_south", &["thermal", "sar"], None).unwrap();
        assert_eq!(p.sensors.len(), 2);
        assert!((p.thresholds.thermal_zscore - 2.5).abs() < 1e-9);
    }
}
