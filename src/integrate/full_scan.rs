//! Full lake scan with anchor-lock — port of `wreckhunter/full_scan.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnchorPoint {
    pub lat: f64,
    pub lon: f64,
    pub utm_e: f64,
    pub utm_n: f64,
}

pub fn anchor_points() -> HashMap<&'static str, AnchorPoint> {
    let mut m = HashMap::new();
    m.insert(
        "Wind Point Light",
        AnchorPoint {
            lat: 42.8000,
            lon: -87.8178,
            utm_e: 428_500.0,
            utm_n: 4_740_000.0,
        },
    );
    m.insert(
        "Holland Harbor Light",
        AnchorPoint {
            lat: 42.7784,
            lon: -86.2066,
            utm_e: 555_000.0,
            utm_n: 4_738_000.0,
        },
    );
    m.insert(
        "Chicago Harbor Light",
        AnchorPoint {
            lat: 41.8900,
            lon: -87.6044,
            utm_e: 447_000.0,
            utm_n: 4_638_000.0,
        },
    );
    m.insert(
        "Waukegan Harbor Light",
        AnchorPoint {
            lat: 42.3638,
            lon: -87.8034,
            utm_e: 429_000.0,
            utm_n: 4_690_000.0,
        },
    );
    m
}

pub const THERMAL_BAND_GLOBS: &[&str] = &[
    "*B04.tif", "*B08.tif", "*B11.tif", "*B12.tif", "*red.tif", "*nir.tif",
];

pub fn is_thermal_band(path: &str) -> bool {
    THERMAL_BAND_GLOBS.iter().any(|g| {
        let suffix = g.trim_start_matches('*');
        path.ends_with(suffix)
    })
}

pub fn default_gpu_threshold() -> f32 {
    2.5
}
