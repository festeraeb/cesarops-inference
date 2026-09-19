//! Prioritized multi-lake satellite pull — port of `wreckhunter/prioritized_satellite_pull.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakePullConfig {
    pub name: String,
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
    pub priority: u32,
}

pub fn great_lakes_pull_config() -> HashMap<&'static str, LakePullConfig> {
    let mut m = HashMap::new();
    m.insert(
        "MICHIGAN",
        LakePullConfig {
            name: "Lake Michigan".into(),
            lon_min: -87.9,
            lat_min: 41.5,
            lon_max: -85.5,
            lat_max: 46.0,
            priority: 1,
        },
    );
    m.insert(
        "ERIE",
        LakePullConfig {
            name: "Lake Erie".into(),
            lon_min: -83.5,
            lat_min: 41.5,
            lon_max: -80.5,
            lat_max: 42.5,
            priority: 2,
        },
    );
    m
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PullMode {
    SwotOnly,
    AllDates,
    Prioritized,
}

pub fn sort_lakes_by_priority(lakes: &[String], catalog: &HashMap<&str, LakePullConfig>) -> Vec<String> {
    let mut out = lakes.to_vec();
    out.sort_by_key(|id| {
        catalog
            .get(id.as_str())
            .map(|c| c.priority)
            .unwrap_or(99)
    });
    out
}
