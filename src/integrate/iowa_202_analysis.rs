//! Iowa-202 inverse projection — port of `wreckhunter/iowa_202_analysis.py`.

use serde::{Deserialize, Serialize};

pub const ZION_CONSTANT: f64 = 1.47;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetProfile {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub detected_length_ft: f32,
    pub drift_corrected_length_ft: f32,
    pub estimated_mass_tons: u32,
    pub depth_ft: u32,
}

pub fn un_squeeze_length(detected_ft: f32) -> f32 {
    detected_ft / ZION_CONSTANT as f32
}

pub fn cargo_type_from_mass(length_ft: f32, mass_tons: u32) -> &'static str {
    let ratio = mass_tons as f32 / length_ft.max(1.0);
    if ratio > 40.0 {
        "bulk_cargo"
    } else if ratio > 15.0 {
        "general_merchandise"
    } else {
        "unknown"
    }
}

pub fn default_target_b() -> TargetProfile {
    TargetProfile {
        name: "Monster (Target B)".into(),
        lat: 42.4180,
        lon: -87.2350,
        detected_length_ft: 338.0,
        drift_corrected_length_ft: 342.8,
        estimated_mass_tons: 14_474,
        depth_ft: 180,
    }
}
