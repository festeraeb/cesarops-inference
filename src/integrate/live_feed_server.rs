//! Live feed / sorter API — port of `cesarops/live_feed_server.py`.

use serde::{Deserialize, Serialize};

pub const DEFAULT_PORT: u16 = 8080;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveFeedRoutes {
    pub sorter_ui: String,
    pub kmz_feed: String,
    pub api_sorter: String,
}

impl Default for LiveFeedRoutes {
    fn default() -> Self {
        Self {
            sorter_ui: "/sorter".into(),
            kmz_feed: "/feed.kmz".into(),
            api_sorter: "/api/sorter".into(),
        }
    }
}

pub const CONFIDENCE_LEVELS: &[&str] = &["VERY_HIGH", "HIGH", "MEDIUM", "LOW"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveFeedSite {
    pub lat: f64,
    pub lon: f64,
    pub confidence_level: String,
    pub concept: String,
}

pub fn group_sites_by_confidence(sites: &[LiveFeedSite]) -> std::collections::HashMap<String, Vec<&LiveFeedSite>> {
    let mut m = std::collections::HashMap::new();
    for site in sites {
        m.entry(site.confidence_level.clone())
            .or_insert_with(Vec::new)
            .push(site);
    }
    m
}
