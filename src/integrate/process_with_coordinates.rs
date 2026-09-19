//! GPU tile processing with coordinates — port of `processing/anomaly_extractor.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PixelAnomalyOut {
    pub row: i32,
    pub col: i32,
    pub z_score: f32,
    pub utm_easting: f64,
    pub utm_northing: f64,
    pub wgs84_lat: f64,
    pub wgs84_lon: f64,
}

pub fn parse_gpu_anomaly_line(line: &str) -> Option<(i32, i32, f32)> {
    let line = line.trim();
    if !line.contains("Pixel") || !line.contains("Z-Score") {
        return None;
    }
    let nums: Vec<f32> = line
        .split(|c: char| !c.is_ascii_digit() && c != '.')
        .filter_map(|s| s.parse().ok())
        .collect();
    if nums.len() >= 3 {
        Some((nums[0] as i32, nums[1] as i32, nums[2]))
    } else {
        None
    }
}

pub fn pixel_to_approx_utm(row: i32, col: i32, height: i32, pixel_m: f64) -> (f64, f64) {
    let easting = 450_000.0 + (col as f64) * pixel_m;
    let northing = 4_700_000.0 + ((height - row) as f64) * pixel_m;
    (easting, northing)
}

pub fn approx_utm_to_wgs84(easting: f64, northing: f64) -> (f64, f64) {
    let lat = northing / 111_320.0;
    let lon = -87.5 + (easting - 500_000.0) / 111_320.0;
    (lat, lon)
}

pub fn build_anomaly(row: i32, col: i32, z: f32, height: i32) -> PixelAnomalyOut {
    let (e, n) = pixel_to_approx_utm(row, col, height, 30.0);
    let (lat, lon) = approx_utm_to_wgs84(e, n);
    PixelAnomalyOut {
        row,
        col,
        z_score: z,
        utm_easting: e,
        utm_northing: n,
        wgs84_lat: lat,
        wgs84_lon: lon,
    }
}
