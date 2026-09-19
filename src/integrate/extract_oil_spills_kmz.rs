//! Oil spill KMZ extraction — port of `leaking_boat/extract_oil_spills_kmz.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakingBoatSensor {
    pub oil_detected: bool,
    pub oil_pixel_count: u32,
    pub max_zscore: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OilSpillFeature {
    pub feature_type: String,
    pub pixels: u32,
    pub area_km2: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PixelSizeM {
    pub landsat_l30: u32,
    pub sentinel_s30: u32,
}

impl Default for PixelSizeM {
    fn default() -> Self {
        Self {
            landsat_l30: 30,
            sentinel_s30: 10,
        }
    }
}

pub fn extract_oil_pixels(lb: &LeakingBoatSensor, min_pixels: u32) -> Vec<OilSpillFeature> {
    if !lb.oil_detected || lb.oil_pixel_count < min_pixels {
        return Vec::new();
    }
    let area_km2 = (lb.oil_pixel_count as f32 * 30.0 * 30.0) / 1_000_000.0;
    vec![OilSpillFeature {
        feature_type: "oil".into(),
        pixels: lb.oil_pixel_count,
        area_km2,
        confidence: lb.max_zscore,
    }]
}

pub fn parse_leaking_boat_from_tile(tile: &serde_json::Value) -> LeakingBoatSensor {
    let lb = tile
        .get("sensors")
        .and_then(|s| s.get("leaking_boat"))
        .cloned()
        .unwrap_or_default();
    LeakingBoatSensor {
        oil_detected: lb.get("oil_detected").and_then(|v| v.as_bool()).unwrap_or(false),
        oil_pixel_count: lb
            .get("oil_pixel_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        max_zscore: lb
            .get("max_zscore")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
    }
}

impl Default for LeakingBoatSensor {
    fn default() -> Self {
        Self {
            oil_detected: false,
            oil_pixel_count: 0,
            max_zscore: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_small_spills() {
        let lb = LeakingBoatSensor {
            oil_detected: true,
            oil_pixel_count: 10,
            max_zscore: 3.0,
        };
        assert!(extract_oil_pixels(&lb, 50).is_empty());
    }
}
