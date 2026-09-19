//! M2200 minimal GPU test — port of `cesarops-gpu/tests/test_m2200_minimal.py`.

use serde::{Deserialize, Serialize};

pub const TEST_WIDTH: u32 = 100;
pub const TEST_HEIGHT: u32 = 100;
pub const COLD_DELTA_K: f32 = 50.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticAnomaly {
    pub row: u32,
    pub col: u32,
}

pub fn default_anomaly_centers() -> Vec<SyntheticAnomaly> {
    vec![
        SyntheticAnomaly { row: 25, col: 25 },
        SyntheticAnomaly { row: 25, col: 75 },
        SyntheticAnomaly { row: 50, col: 50 },
        SyntheticAnomaly { row: 75, col: 25 },
        SyntheticAnomaly { row: 75, col: 75 },
    ]
}

pub fn thermal_k_at_ambient(ambient_k: f32) -> f32 {
    ambient_k - COLD_DELTA_K
}

pub fn parse_detected_count(stdout: &str) -> Option<u32> {
    for line in stdout.lines() {
        if line.contains("Detected") && line.contains("anomalies") {
            if let Some(n) = line.split_whitespace().nth(1) {
                return n.parse().ok();
            }
        }
    }
    None
}

pub fn gpu_confirmed(stdout: &str) -> bool {
    stdout.contains("Quadro M2200") && stdout.contains("is active")
}
