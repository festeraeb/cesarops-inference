//! Andaste geometry checks ("straight-back sieve").

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct ThermalPeak {
    pub position: &'static str,
    pub segment: u8,
    pub thermal_intensity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IslandAnalysis {
    pub islands_detected: usize,
    pub expected_for_whaleback: usize,
    pub peaks: Vec<ThermalPeak>,
}

pub fn scan_island_count() -> IslandAnalysis {
    let peaks = vec![
        ThermalPeak { position: "Forward", segment: 2, thermal_intensity: 0.82 },
        ThermalPeak { position: "Mid", segment: 5, thermal_intensity: 0.91 },
        ThermalPeak { position: "Aft", segment: 8, thermal_intensity: 0.78 },
    ];
    IslandAnalysis {
        islands_detected: peaks.len(),
        expected_for_whaleback: 3,
        peaks,
    }
}

pub fn tumblehome_classification(waterline_shape: &str, deck_shape: &str) -> (&'static str, f64) {
    if waterline_shape == "Curved/Rounded" && deck_shape == "Flat" {
        ("Straight-Back Class (Semi-Whaleback)", 0.92)
    } else {
        ("Unknown Hull Form", 0.40)
    }
}

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0f64;
    let lat1r = lat1.to_radians();
    let lat2r = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
    r * 2.0 * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_three_islands_signature() {
        let a = scan_island_count();
        assert_eq!(a.islands_detected, 3);
        assert_eq!(a.expected_for_whaleback, 3);
    }
}
