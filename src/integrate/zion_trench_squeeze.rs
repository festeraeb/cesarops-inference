//! Zion trench high-res squeeze — port of `missions/zion_trench_squeeze.py`.

use serde::{Deserialize, Serialize};

pub const SQUEEZE_RADIUS_KM: f64 = 5.0;
pub const GRID_CELL_SIZE_M: f64 = 100.0;
pub const THRESHOLD_AGGRESSIVE: f32 = 1.5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AndasteCenter {
    pub easting: f64,
    pub northing: f64,
    pub lat: f64,
    pub lon: f64,
    pub depth_ft: u32,
    pub name: String,
}

pub fn andaste_center() -> AndasteCenter {
    AndasteCenter {
        easting: 457_990.7,
        northing: 4_702_720.4,
        lat: 42.4757,
        lon: -87.5111,
        depth_ft: 180,
        name: "SS Andaste (Whaleback)".into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrenchGridCell {
    pub easting: f64,
    pub northing: f64,
    pub lat: f64,
    pub lon: f64,
    pub distance_m: f64,
}

pub fn generate_trench_grid(center: &AndasteCenter) -> Vec<TrenchGridCell> {
    let radius_m = SQUEEZE_RADIUS_KM * 1000.0;
    let e_min = ((center.easting - radius_m) / GRID_CELL_SIZE_M).floor() as i32;
    let e_max = ((center.easting + radius_m) / GRID_CELL_SIZE_M).ceil() as i32;
    let n_min = ((center.northing - radius_m) / GRID_CELL_SIZE_M).floor() as i32;
    let n_max = ((center.northing + radius_m) / GRID_CELL_SIZE_M).ceil() as i32;
    let mut cells = Vec::new();
    for e_idx in e_min..=e_max {
        for n_idx in n_min..=n_max {
            let cell_e = (e_idx as f64 + 0.5) * GRID_CELL_SIZE_M;
            let cell_n = (n_idx as f64 + 0.5) * GRID_CELL_SIZE_M;
            let dist = ((cell_e - center.easting).powi(2) + (cell_n - center.northing).powi(2)).sqrt();
            if dist <= radius_m {
                let lat = center.lat + (cell_n - center.northing) / 111_320.0;
                let lon = center.lon + (cell_e - center.easting) / (111_320.0 * 0.7);
                cells.push(TrenchGridCell {
                    easting: cell_e,
                    northing: cell_n,
                    lat,
                    lon,
                    distance_m: dist,
                });
            }
        }
    }
    cells
}

pub const THERMAL_PATTERNS: &[&str] = &["*B11.tif", "*B12.tif"];
pub const OPTICAL_PATTERNS: &[&str] = &["*B04.tif", "*B08.tif"];
pub const SAR_PATTERNS: &[&str] = &["*vv.tif", "*vh.tif"];
