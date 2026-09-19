//! Crossref anomaly ↔ known wreck matching — port of `match_wrecks.py`.

use serde::{Deserialize, Serialize};

pub const MATCH_M: f64 = 457.0;
pub const NEAR_M: f64 = 1500.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WreckRecord {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub depth_ft: String,
    pub year_lost: String,
    pub wreck_type: String,
    pub lat_min: Option<f64>,
    pub lat_max: Option<f64>,
    pub lon_min: Option<f64>,
    pub lon_max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrossrefHit {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    pub types: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchStatus {
    InsideBbox,
    PossibleMatch,
    Nearby,
    NoCloseWreck,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WreckMatchResult {
    pub hit: CrossrefHit,
    pub nearest_name: String,
    pub distance_m: f64,
    pub status: MatchStatus,
}

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    r * 2.0 * a.sqrt().asin()
}

pub fn inside_bbox(hit: &CrossrefHit, wreck: &WreckRecord) -> bool {
    match (wreck.lat_min, wreck.lat_max, wreck.lon_min, wreck.lon_max) {
        (Some(lat_min), Some(lat_max), Some(lon_min), Some(lon_max)) => {
            hit.lat >= lat_min && hit.lat <= lat_max && hit.lon >= lon_min && hit.lon <= lon_max
        }
        _ => false,
    }
}

pub fn classify_match(distance_m: f64, in_bbox: bool) -> MatchStatus {
    if in_bbox {
        MatchStatus::InsideBbox
    } else if distance_m < MATCH_M {
        MatchStatus::PossibleMatch
    } else if distance_m < NEAR_M {
        MatchStatus::Nearby
    } else {
        MatchStatus::NoCloseWreck
    }
}

pub fn match_hit_to_wrecks(hit: &CrossrefHit, wrecks: &[WreckRecord]) -> WreckMatchResult {
    let mut best = wrecks
        .iter()
        .map(|w| (haversine_m(hit.lat, hit.lon, w.lat, w.lon), w))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .expect("wrecks non-empty");
    let in_bbox = wrecks.iter().any(|w| inside_bbox(hit, w));
    let status = classify_match(best.0, in_bbox);
    WreckMatchResult {
        hit: hit.clone(),
        nearest_name: best.1.name.clone(),
        distance_m: best.0,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_tight_match() {
        assert!(matches!(
            classify_match(100.0, false),
            MatchStatus::PossibleMatch
        ));
    }
}
