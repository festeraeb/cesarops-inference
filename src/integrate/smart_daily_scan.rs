//! Smart daily temporal scan — port of `wreckhunter/smart_daily_scan.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakeScanBbox {
    pub name: String,
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
    pub priority: u32,
}

pub fn lake_scan_order() -> Vec<LakeScanBbox> {
    vec![
        LakeScanBbox {
            name: "Lake Michigan".into(),
            min_lat: 42.4,
            min_lon: -87.5,
            max_lat: 45.5,
            max_lon: -85.5,
            priority: 1,
        },
        LakeScanBbox {
            name: "Lake Erie".into(),
            min_lat: 41.5,
            min_lon: -83.5,
            max_lat: 42.5,
            max_lon: -80.5,
            priority: 2,
        },
    ]
}

pub fn point_in_bbox(lat: f64, lon: f64, b: &LakeScanBbox) -> bool {
    lat >= b.min_lat && lat <= b.max_lat && lon >= b.min_lon && lon <= b.max_lon
}

pub fn sweep_years(start: i32, end: i32) -> Vec<i32> {
    (start..=end).collect()
}
