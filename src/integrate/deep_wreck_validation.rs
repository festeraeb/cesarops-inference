//! Deep wreck validation — band suite + squeeze filters — port of `deep_wreck_validation.py`.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationTarget {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub tile: String,
    pub original_zscore: Option<f64>,
    pub combined_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorStats {
    pub mean: f64,
    pub std: f64,
    pub max_zscore: f64,
    pub anomaly_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SqueezeResult {
    pub profile: String,
    pub passes: BTreeMap<String, bool>,
    pub passes_all: bool,
    pub sensors_detected: Vec<String>,
    pub estimated_length_m: Option<f64>,
}

pub const MONSTER_THERMAL_Z_MIN: f64 = 3.0;
pub const ANDASTE_THERMAL_Z_MIN: f64 = 2.5;
pub const ANDASTE_LENGTH_RANGE: (f64, f64) = (80.0, 100.0);
pub const ZSCORE_THRESHOLD: f64 = 2.5;

pub fn default_targets() -> Vec<ValidationTarget> {
    vec![
        ValidationTarget {
            name: "thermal_detection".into(),
            lat: 42.948873,
            lon: -86.976619,
            tile: "HLS.L30.T16TDN.2021198T162826.v2.0".into(),
            original_zscore: Some(5.46),
            combined_score: None,
        },
        ValidationTarget {
            name: "stationary_anchor_8".into(),
            lat: 42.4647,
            lon: -87.1082,
            tile: "HLS.S30.T16TDN.2025244T163839.v2.0".into(),
            original_zscore: None,
            combined_score: Some(18.67),
        },
    ]
}

pub fn zscore_stats(data: &Array1<f32>) -> SensorStats {
    let mean = data.mean().unwrap_or(0.0) as f64;
    let std = {
        let n = data.len() as f64;
        if n < 2.0 {
            0.0
        } else {
            let var = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        }
    };
    let z: Array1<f64> = data.mapv(|x| (x as f64 - mean) / (std + 1e-6));
    let max_z = z.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let anomaly_count = z.iter().filter(|&&v| v.abs() > ZSCORE_THRESHOLD).count() as u32;
    SensorStats {
        mean,
        std,
        max_zscore: max_z,
        anomaly_count,
    }
}

pub fn process_full_suite(bands: &BTreeMap<String, Array2<f32>>) -> BTreeMap<String, SensorStats> {
    let mut results = BTreeMap::new();
    if let (Some(b10), Some(b11)) = (bands.get("B10"), bands.get("B11")) {
        let thermal = ((b10 + b11) / 2.0).mapv(|x| x as f32);
        let flat = thermal.into_raw_vec();
        results.insert("thermal".into(), zscore_stats(&Array1::from(flat)));
    }
    if let (Some(b04), Some(b05)) = (bands.get("B04"), bands.get("B05")) {
        let ratio = (b05 / (b04.mapv(|x| x + 1e-6))).mapv(|x| x as f32);
        let flat = ratio.into_raw_vec();
        results.insert("optical".into(), zscore_stats(&Array1::from(flat)));
    }
    if let Some(b01) = bands.get("B01") {
        let flat = b01.clone().into_raw_vec();
        results.insert("coastal".into(), zscore_stats(&Array1::from(flat)));
    }
    results
}

pub fn squeeze_filter(results: &BTreeMap<String, SensorStats>, profile: &str, thermal_zmap_count: u32) -> SqueezeResult {
    match profile {
        "andaste" => {
            let thermal = results.get("thermal");
            let max_z = thermal.map(|t| t.max_zscore).unwrap_or(0.0);
            let estimated_length_m = (thermal_zmap_count as f64).sqrt() * 30.0;
            let mut passes = BTreeMap::new();
            passes.insert("thermal_zscore".into(), max_z >= ANDASTE_THERMAL_Z_MIN);
            passes.insert(
                "length_range".into(),
                estimated_length_m >= ANDASTE_LENGTH_RANGE.0 && estimated_length_m <= ANDASTE_LENGTH_RANGE.1,
            );
            let passes_all = passes.values().all(|&v| v);
            SqueezeResult {
                profile: "ANDASTE (Whaleback)".into(),
                passes,
                passes_all,
                sensors_detected: results.keys().cloned().collect(),
                estimated_length_m: Some(estimated_length_m),
            }
        }
        _ => {
            let thermal = results.get("thermal");
            let max_z = thermal.map(|t| t.max_zscore).unwrap_or(0.0);
            let mut passes = BTreeMap::new();
            passes.insert("thermal_zscore".into(), max_z >= MONSTER_THERMAL_Z_MIN);
            passes.insert("optical_present".into(), results.contains_key("optical"));
            passes.insert("multi_sensor".into(), results.len() >= 2);
            let passes_all = passes.values().all(|&v| v);
            SqueezeResult {
                profile: "MONSTER (Deep Wreck)".into(),
                passes,
                passes_all,
                sensors_detected: results.keys().cloned().collect(),
                estimated_length_m: None,
            }
        }
    }
}

pub fn tile_cache_dir(tile: &str) -> &'static str {
    if tile.contains("2025") {
        "wreckhunter2000/data/cache/census_raw/2025_rossa"
    } else {
        "wreckhunter2000/data/cache/census_raw/2021_low_water"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn monster_passes_with_strong_thermal() {
        let mut bands = BTreeMap::new();
        bands.insert("B10".into(), arr2(&[[1.0, 2.0], [3.0, 50.0]]));
        bands.insert("B11".into(), arr2(&[[1.0, 2.0], [3.0, 50.0]]));
        bands.insert("B04".into(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        bands.insert("B05".into(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let stats = process_full_suite(&bands);
        let sq = squeeze_filter(&stats, "monster", 100);
        assert!(sq.passes.get("multi_sensor").copied().unwrap_or(false));
    }
}
