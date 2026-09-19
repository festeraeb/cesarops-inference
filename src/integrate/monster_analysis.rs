//! Monster site analysis — port of `analysis/monster_site_analysis.py`.

use serde::{Deserialize, Serialize};

pub const MONSTER_LAT: f64 = 42.4180;
pub const MONSTER_LON: f64 = -87.2350;
pub const ANCASTE_LAT: f64 = 42.4125;
pub const ANCASTE_LON: f64 = -87.2500;
pub const ZION_CONSTANT: f64 = 1.47;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HullTarget {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub detected_length_ft: f32,
    pub drift_correction: f32,
}

pub fn corrected_length_ft(detected: f32, drift: f32) -> f32 {
    detected * drift
}

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos()
            * lat2.to_radians().cos()
            * (dlon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().asin()
}

pub fn estimate_mass_tons(length_ft: f32) -> f32 {
    length_ft * 42.5
}

pub fn inverse_projection_length(detected_ft: f32) -> f32 {
    detected_ft / ZION_CONSTANT as f32
}
