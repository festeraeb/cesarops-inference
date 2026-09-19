//! Multi-sensor scan config + anchor-lock — Rust port of `multi_sensor_scan.py` / `lake_michigan_full_scan.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorPoint {
    pub lat: f64,
    pub lon: f64,
    pub utm_e: f64,
    pub utm_n: f64,
}

pub fn anchor_points() -> HashMap<&'static str, AnchorPoint> {
    HashMap::from([
        (
            "Wind Point Light",
            AnchorPoint {
                lat: 42.8000,
                lon: -87.8178,
                utm_e: 428_500.0,
                utm_n: 4_740_000.0,
            },
        ),
        (
            "Holland Harbor Light",
            AnchorPoint {
                lat: 42.7784,
                lon: -86.2066,
                utm_e: 555_000.0,
                utm_n: 4_738_000.0,
            },
        ),
        (
            "Chicago Harbor Light",
            AnchorPoint {
                lat: 41.8900,
                lon: -87.6044,
                utm_e: 447_000.0,
                utm_n: 4_638_000.0,
            },
        ),
        (
            "Waukegan Harbor Light",
            AnchorPoint {
                lat: 42.3638,
                lon: -87.8034,
                utm_e: 429_000.0,
                utm_n: 4_690_000.0,
            },
        ),
    ])
}

#[derive(Debug, Clone, Serialize)]
pub struct SensorConfig {
    pub name: &'static str,
    pub patterns: &'static [&'static str],
    pub threshold: f32,
    pub description: &'static str,
}

pub fn sensor_configs() -> HashMap<&'static str, SensorConfig> {
    HashMap::from([
        (
            "thermal",
            SensorConfig {
                name: "Thermal (SWIR B11/B12)",
                patterns: &["*B11.tif", "*B12.tif"],
                threshold: 2.5,
                description: "Thermal anomaly detection",
            },
        ),
        (
            "optical",
            SensorConfig {
                name: "Optical (NIR/Red B08/B04)",
                patterns: &["*B04.tif", "*B08.tif", "*red.tif", "*nir.tif"],
                threshold: 2.0,
                description: "Aluminum signature via NIR/Red",
            },
        ),
        (
            "sar",
            SensorConfig {
                name: "SAR (Sentinel-1 VV)",
                patterns: &["*vv.tif", "*s1a*.tiff", "*s1b*.tiff"],
                threshold: 3.0,
                description: "SAR surface roughness",
            },
        ),
        (
            "swot",
            SensorConfig {
                name: "SWOT (Surface Water)",
                patterns: &["*swot*.tif", "*ssh*.tif"],
                threshold: 2.5,
                description: "SSH displacement anomalies",
            },
        ),
        (
            "magnetic",
            SensorConfig {
                name: "Magnetic Anomaly",
                patterns: &["*emag*.tif", "*magnetic*.tif", "*mag*.tif"],
                threshold: 3.0,
                description: "Ferrous metal detection",
            },
        ),
    ])
}

/// Distance-weighted blend toward nearest anchor (simplified anchor-lock).
pub fn anchor_calibration_offset(utm_e: f64, utm_n: f64, anchors: &HashMap<&str, AnchorPoint>) -> (f64, f64) {
    let mut best_d = f64::MAX;
    let mut de = 0.0;
    let mut dn = 0.0;
    for ap in anchors.values() {
        let d = ((utm_e - ap.utm_e).powi(2) + (utm_n - ap.utm_n).powi(2)).sqrt();
        if d < best_d {
            best_d = d;
            de = ap.utm_e - utm_e;
            dn = ap.utm_n - utm_n;
        }
    }
    if best_d > 50_000.0 {
        return (0.0, 0.0);
    }
    let w = (1.0 - (best_d / 50_000.0)).clamp(0.0, 1.0);
    (de * w, dn * w)
}
