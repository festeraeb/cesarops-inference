//! Great Lakes bboxes + pull priorities — Rust port of `prioritized_pull_v2.py` config.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bbox {
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LakeConfig {
    pub name: &'static str,
    pub bbox: Bbox,
    pub priority: u8,
}

pub fn great_lakes_catalog() -> HashMap<&'static str, LakeConfig> {
    HashMap::from([
        (
            "MICHIGAN",
            LakeConfig {
                name: "Lake Michigan",
                bbox: Bbox {
                    lon_min: -87.9,
                    lat_min: 41.5,
                    lon_max: -85.5,
                    lat_max: 46.0,
                },
                priority: 1,
            },
        ),
        (
            "ERIE",
            LakeConfig {
                name: "Lake Erie",
                bbox: Bbox {
                    lon_min: -83.5,
                    lat_min: 41.5,
                    lon_max: -80.5,
                    lat_max: 42.5,
                },
                priority: 2,
            },
        ),
        (
            "HURON",
            LakeConfig {
                name: "Lake Huron",
                bbox: Bbox {
                    lon_min: -83.5,
                    lat_min: 43.5,
                    lon_max: -81.5,
                    lat_max: 45.5,
                },
                priority: 3,
            },
        ),
        (
            "SUPERIOR",
            LakeConfig {
                name: "Lake Superior",
                bbox: Bbox {
                    lon_min: -92.0,
                    lat_min: 46.5,
                    lon_max: -84.0,
                    lat_max: 48.0,
                },
                priority: 4,
            },
        ),
        (
            "ONTARIO",
            LakeConfig {
                name: "Lake Ontario",
                bbox: Bbox {
                    lon_min: -77.5,
                    lat_min: 43.5,
                    lon_max: -76.0,
                    lat_max: 44.5,
                },
                priority: 5,
            },
        ),
    ])
}

/// Lakes sorted by fleet priority (Michigan first).
pub fn lakes_by_priority() -> Vec<(&'static str, LakeConfig)> {
    let mut v: Vec<_> = great_lakes_catalog().into_iter().collect();
    v.sort_by_key(|(_, c)| c.priority);
    v
}
