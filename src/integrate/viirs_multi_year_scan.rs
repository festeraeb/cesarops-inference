//! VIIRS multi-year fusion — port of `forensics/viirs_multi_year_scan.py`.

use serde::{Deserialize, Serialize};

pub const ZSCORE_TOLERANCE: f32 = 0.5;
pub const LOCATION_TOLERANCE_M: f32 = 100.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViirsFileMeta {
    pub product: String,
    pub year: Option<i32>,
    pub filename: String,
}

pub fn parse_viirs_filename(name: &str) -> Option<ViirsFileMeta> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let product = parts[0].to_string();
    let date_str = parts[1];
    if !date_str.starts_with('A') || date_str.len() < 8 {
        return None;
    }
    let year = date_str[1..5].parse().ok();
    Some(ViirsFileMeta {
        product,
        year,
        filename: name.to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViirsDetection {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f32,
    pub year: i32,
}

pub fn same_target(a: &ViirsDetection, b: &ViirsDetection) -> bool {
    let dz = (a.zscore - b.zscore).abs();
    let dlat = (a.lat - b.lat).abs() * 111_320.0;
    let dlon = (a.lon - b.lon).abs() * 85_000.0;
    let dist = (dlat * dlat + dlon * dlon).sqrt() as f32;
    dz <= ZSCORE_TOLERANCE && dist <= LOCATION_TOLERANCE_M
}
