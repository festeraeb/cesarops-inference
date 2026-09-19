//! MB2 cluster crosscheck — port of `mb2_crosscheck.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mb2Detection {
    pub lat: f64,
    pub lon: f64,
    #[serde(rename = "type")]
    pub det_type: String,
    pub zscore: Option<f64>,
    pub mb2_zone: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mb2CrosscheckReport {
    pub cluster_lat: f64,
    pub cluster_lon: f64,
    pub radius_km: f64,
    pub nearby_count: usize,
    pub type_counts: HashMap<String, usize>,
    pub peak_zscore_by_type: HashMap<String, f64>,
    pub hc_in_mb2_zone: usize,
}

pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

pub fn analyze_mb2_cluster(
    detections: &[Mb2Detection],
    cluster_lat: f64,
    cluster_lon: f64,
    radius_km: f64,
) -> Mb2CrosscheckReport {
    let nearby: Vec<_> = detections
        .iter()
        .filter(|d| haversine_km(cluster_lat, cluster_lon, d.lat, d.lon) <= radius_km)
        .collect();
    let mut type_counts = HashMap::new();
    let mut peak_zscore_by_type = HashMap::new();
    for d in &nearby {
        *type_counts.entry(d.det_type.clone()).or_insert(0) += 1;
        let z = d.zscore.unwrap_or(0.0);
        peak_zscore_by_type
            .entry(d.det_type.clone())
            .and_modify(|peak: &mut f64| *peak = (*peak).max(z))
            .or_insert(z);
    }
    let hc_in_mb2_zone = detections
        .iter()
        .filter(|d| d.det_type == "hydrocarbon" && d.mb2_zone.unwrap_or(false))
        .count();
    Mb2CrosscheckReport {
        cluster_lat,
        cluster_lon,
        radius_km,
        nearby_count: nearby.len(),
        type_counts,
        peak_zscore_by_type,
        hc_in_mb2_zone,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_nearby() {
        let dets = vec![
            Mb2Detection {
                lat: 42.42,
                lon: -82.34,
                det_type: "thermal".into(),
                zscore: Some(3.0),
                mb2_zone: None,
            },
            Mb2Detection {
                lat: 50.0,
                lon: -80.0,
                det_type: "thermal".into(),
                zscore: Some(1.0),
                mb2_zone: None,
            },
        ];
        let r = analyze_mb2_cluster(&dets, 42.42065, -82.34270, 5.0);
        assert_eq!(r.nearby_count, 1);
    }
}
