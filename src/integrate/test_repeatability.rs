//! Repeatability test criteria — port of `tests/test_repeatability.py`.

use serde::{Deserialize, Serialize};

pub const MAX_POSITION_DRIFT_M: f64 = 5.0;
pub const MAX_SCORE_DRIFT: f64 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepeatabilityCriteria {
    pub max_position_drift_m: f64,
    pub max_score_drift: f64,
    pub require_same_detection_count: bool,
}

impl Default for RepeatabilityCriteria {
    fn default() -> Self {
        Self {
            max_position_drift_m: MAX_POSITION_DRIFT_M,
            max_score_drift: MAX_SCORE_DRIFT,
            require_same_detection_count: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectionSnapshot {
    pub utm_easting: f64,
    pub utm_northing: f64,
    pub score: f64,
}

pub fn position_drift_m(a: &DetectionSnapshot, b: &DetectionSnapshot) -> f64 {
    let de = a.utm_easting - b.utm_easting;
    let dn = a.utm_northing - b.utm_northing;
    (de * de + dn * dn).sqrt()
}

pub fn passes_repeatability(
    a: &DetectionSnapshot,
    b: &DetectionSnapshot,
    criteria: &RepeatabilityCriteria,
) -> bool {
    position_drift_m(a, b) <= criteria.max_position_drift_m
        && (a.score - b.score).abs() <= criteria.max_score_drift
}

pub fn sha256_hex(data: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    format!("{:016x}", h.finish())
}
