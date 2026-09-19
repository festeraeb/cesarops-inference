//! Mackinac bridge coordinate calibration — port of `wreckhunter/bridge_calibrate.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GeoRef {
    pub label: &'static str,
    pub lat: f64,
    pub lon: f64,
}

pub const MACKINAC_REFS: [GeoRef; 3] = [
    GeoRef {
        label: "North tower anchor",
        lat: 45.81656,
        lon: -84.72769,
    },
    GeoRef {
        label: "South tower center",
        lat: 45.78633,
        lon: -84.72705,
    },
    GeoRef {
        label: "Bridge midpoint",
        lat: 45.80140,
        lon: -84.72740,
    },
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoundTripError {
    pub label: String,
    pub row: i32,
    pub col: i32,
    pub error_m: f64,
    pub in_bounds: bool,
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

/// Pixel error from lat/lon offset (rough metres-per-degree at ~45°N).
pub fn lat_lon_offset_to_metres(dlat_deg: f64, dlon_deg: f64, ref_lat: f64) -> (f64, f64) {
    let m_per_deg_lat = 111_320.0;
    let m_per_deg_lon = 111_320.0 * ref_lat.to_radians().cos();
    (dlat_deg * m_per_deg_lat, dlon_deg * m_per_deg_lon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_span_bridge() {
        assert!(MACKINAC_REFS[0].lat > MACKINAC_REFS[1].lat);
    }

    #[test]
    fn haversine_small_offset() {
        let d = haversine_m(45.80, -84.72, 45.801, -84.721);
        assert!(d > 0.0 && d < 5000.0);
    }
}
