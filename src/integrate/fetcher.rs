//! Satellite fetcher config — port of `wreckhunter/fetcher.py` (bounds + API payloads).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const USGS_API_BASE: &str = "https://earthexplorer.usgs.gov/api/v1";
pub const SENTINEL_HUB_BASE: &str = "https://services.sentinel-hub.com/api/v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakeBounds {
    pub north: f64,
    pub south: f64,
    pub west: f64,
    pub east: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DateWindow {
    pub start: String,
    pub end: String,
}

pub fn lake_michigan_bounds() -> LakeBounds {
    LakeBounds {
        north: 46.10,
        south: 41.60,
        west: -88.10,
        east: -84.70,
    }
}

pub fn lake_superior_bounds() -> LakeBounds {
    LakeBounds {
        north: 49.00,
        south: 46.00,
        west: -92.50,
        east: -84.00,
    }
}

pub fn all_great_lakes_bounds() -> LakeBounds {
    LakeBounds {
        north: 49.00,
        south: 41.00,
        west: -92.50,
        east: -76.00,
    }
}

pub fn ice_break_windows() -> Vec<DateWindow> {
    vec![
        DateWindow {
            start: "2024-03-15".into(),
            end: "2024-04-30".into(),
        },
        DateWindow {
            start: "2025-03-15".into(),
            end: "2025-04-30".into(),
        },
    ]
}

pub fn low_silt_windows() -> Vec<DateWindow> {
    vec![
        DateWindow {
            start: "2023-05-20".into(),
            end: "2023-06-15".into(),
        },
        DateWindow {
            start: "2024-05-20".into(),
            end: "2024-06-15".into(),
        },
    ]
}

/// USGS scene-search JSON body (Landsat OT C2 L2).
pub fn usgs_scene_search_payload(
    bbox: &LakeBounds,
    date_range: &DateWindow,
    max_results: u32,
) -> serde_json::Value {
    serde_json::json!({
        "datasetName": "landsat_ot_c2_l2",
        "maxResults": max_results,
        "startingNumber": 1,
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": { "latitude": bbox.south, "longitude": bbox.west },
            "upperRight": { "latitude": bbox.north, "longitude": bbox.east }
        },
        "temporalFilter": {
            "start": date_range.start,
            "end": date_range.end
        },
        "acquisitionType": "L1GT"
    })
}

pub fn credentials_from_env() -> HashMap<&'static str, Option<String>> {
    let mut m = HashMap::new();
    m.insert(
        "USGS_USERNAME",
        std::env::var("USGS_USERNAME").ok(),
    );
    m.insert(
        "USGS_PASSWORD",
        std::env::var("USGS_PASSWORD").ok(),
    );
    m.insert(
        "SENTINEL_CLIENT_ID",
        std::env::var("SENTINEL_CLIENT_ID").ok(),
    );
    m.insert(
        "SENTINEL_CLIENT_SECRET",
        std::env::var("SENTINEL_CLIENT_SECRET").ok(),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usgs_payload_has_bbox() {
        let p = usgs_scene_search_payload(
            &lake_michigan_bounds(),
            &DateWindow {
                start: "2025-01-01".into(),
                end: "2025-12-31".into(),
            },
            50,
        );
        assert!(p.get("spatialFilter").is_some());
    }
}
