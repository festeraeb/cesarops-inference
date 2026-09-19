//! Wreck coordinate pixel probe — port of `wreck_pixel_probe.py`.

use serde::{Deserialize, Serialize};

pub const STRAITS_WRECK_TARGETS: [(&str, f64, f64, u32); 6] = [
    ("Minneapolis", 45.80852, -84.73173, 124),
    ("William Young", 45.81295, -84.69872, 120),
    ("M. Stalker", 45.79367, -84.68437, 85),
    ("Cedarville", 45.78725, -84.67080, 40),
    ("Eber Ward", 45.81272, -84.81888, 128),
    ("Sandusky", 45.79932, -84.83748, 77),
];

pub const BAND_PRIORITY: [&str; 6] = ["blue", "green", "red", "nir", "lwir11", "swir16"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PixelProbeResult {
    pub wreck_name: String,
    pub band: String,
    pub raw_dn: f64,
    pub scene_mean: f64,
    pub scene_std: f64,
    pub zscore: f64,
}

pub fn pixel_zscore(raw: f64, mean: f64, std: f64) -> Option<f64> {
    if std <= 0.0 {
        return None;
    }
    Some((raw - mean) / std)
}

pub fn parse_sensor_band(filename_stem: &str) -> Option<(String, String)> {
    let mut parts = filename_stem.split('.');
    let sensor = parts.next()?.to_string();
    let band = parts.next()?.to_string();
    Some((sensor, band))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zscore_works() {
        assert_eq!(pixel_zscore(110.0, 100.0, 5.0), Some(2.0));
    }
}
