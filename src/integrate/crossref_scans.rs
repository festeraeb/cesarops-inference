//! Multi-epoch scan cross-reference — port of `crossref_scans.py`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MATCH_RADIUS_DEG: f64 = 0.004;
pub const HI_CONF_THRESHOLD: u32 = 2;
pub const STRAITS_BBOX: (f64, f64, f64, f64) = (45.70, -84.80, 46.05, -84.10);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanDetection {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    #[serde(rename = "type")]
    pub det_type: String,
    pub known_wreck_name: Option<String>,
    pub line5_candidate: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectionCluster {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    pub det_type: String,
    pub types: BTreeSet<String>,
    pub count: u32,
}

pub fn in_straits_bbox(lat: f64, lon: f64) -> bool {
    let (lat_min, lon_min, lat_max, lon_max) = STRAITS_BBOX;
    lat >= lat_min && lat <= lat_max && lon >= lon_min && lon <= lon_max
}

pub fn filter_scan_detections(dets: &[ScanDetection]) -> Vec<ScanDetection> {
    dets.iter()
        .filter(|d| in_straits_bbox(d.lat, d.lon) && d.zscore.abs() <= 8.0)
        .cloned()
        .collect()
}

pub fn cluster_detections(dets: &[ScanDetection], radius_deg: f64) -> Vec<DetectionCluster> {
    let mut clusters: Vec<DetectionCluster> = Vec::new();
    let mut sorted: Vec<_> = dets.iter().collect();
    sorted.sort_by(|a, b| b.zscore.abs().partial_cmp(&a.zscore.abs()).unwrap_or(std::cmp::Ordering::Equal));
    for d in sorted {
        let mut merged = false;
        for c in &mut clusters {
            if (d.lat - c.lat).abs() < radius_deg && (d.lon - c.lon).abs() < radius_deg {
                let n = c.count as f64;
                c.lat = (c.lat * n + d.lat) / (n + 1.0);
                c.lon = (c.lon * n + d.lon) / (n + 1.0);
                c.count += 1;
                if d.zscore.abs() > c.zscore.abs() {
                    c.zscore = d.zscore;
                    c.det_type = d.det_type.clone();
                }
                c.types.insert(d.det_type.clone());
                merged = true;
                break;
            }
        }
        if !merged {
            clusters.push(DetectionCluster {
                lat: d.lat,
                lon: d.lon,
                zscore: d.zscore,
                det_type: d.det_type.clone(),
                types: BTreeSet::from([d.det_type.clone()]),
                count: 1,
            });
        }
    }
    clusters
}

pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_bbox() {
        let d = ScanDetection {
            lat: 45.8,
            lon: -84.7,
            zscore: 3.0,
            det_type: "thermal".into(),
            known_wreck_name: None,
            line5_candidate: None,
        };
        assert_eq!(filter_scan_detections(&[d]).len(), 1);
    }
}
