//! Triple-lock fusion — thermal + SAR + optical agreement (`triple_lock_fusion.py`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorHit {
    pub lat: f64,
    pub lon: f64,
    pub sensor: &'static str,
    pub zscore: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleLockTarget {
    pub lat: f64,
    pub lon: f64,
    pub thermal_z: Option<f64>,
    pub sar_z: Option<f64>,
    pub optical_z: Option<f64>,
    pub locks: u8,
    pub fused_confidence: f64,
}

/// Thermal cold-sink Z-score for one band sample (Kelvin).
pub fn thermal_zscore(value_k: f64, mean_k: f64, std_k: f64) -> f64 {
    if std_k.abs() < f64::EPSILON {
        return 0.0;
    }
    (value_k - mean_k) / std_k
}

/// Cold-sink: significantly colder than surroundings (negative Z).
#[inline]
pub fn is_cold_sink(z: f64, threshold: f64) -> bool {
    z < -threshold
}

/// Grid cell key for ~100 m fusion (approx at mid-latitudes).
pub fn fusion_cell(lat: f64, lon: f64) -> (i32, i32) {
    ((lat * 1000.0) as i32, (lon * 1000.0) as i32)
}

/// Fuse hits that share a cell; require at least `min_locks` sensor types.
pub fn fuse_hits(hits: &[SensorHit], min_locks: u8) -> Vec<TripleLockTarget> {
    use std::collections::HashMap;

    let mut cells: HashMap<(i32, i32), TripleLockTarget> = HashMap::new();

    for h in hits {
        let key = fusion_cell(h.lat, h.lon);
        let entry = cells.entry(key).or_insert_with(|| TripleLockTarget {
            lat: h.lat,
            lon: h.lon,
            thermal_z: None,
            sar_z: None,
            optical_z: None,
            locks: 0,
            fused_confidence: 0.0,
        });
        match h.sensor {
            "thermal" => {
                if entry.thermal_z.is_none() {
                    entry.thermal_z = Some(h.zscore);
                    entry.locks += 1;
                }
            }
            "sar" => {
                if entry.sar_z.is_none() {
                    entry.sar_z = Some(h.zscore);
                    entry.locks += 1;
                }
            }
            "optical" => {
                if entry.optical_z.is_none() {
                    entry.optical_z = Some(h.zscore);
                    entry.locks += 1;
                }
            }
            _ => {}
        }
        entry.fused_confidence = entry.fused_confidence.max(h.confidence);
    }

    cells
        .into_values()
        .filter(|t| t.locks >= min_locks)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triple_agreement_fuses() {
        let hits = vec![
            SensorHit {
                lat: 42.5,
                lon: -87.0,
                sensor: "thermal",
                zscore: -3.0,
                confidence: 0.9,
            },
            SensorHit {
                lat: 42.5,
                lon: -87.0,
                sensor: "sar",
                zscore: 2.5,
                confidence: 0.85,
            },
            SensorHit {
                lat: 42.5,
                lon: -87.0,
                sensor: "optical",
                zscore: 2.0,
                confidence: 0.8,
            },
        ];
        let fused = fuse_hits(&hits, 3);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].locks, 3);
    }
}
