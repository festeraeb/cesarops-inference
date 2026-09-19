//! Anchor-lock network and Zion trench target metadata.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchor {
    pub name: &'static str,
    pub lat: f64,
    pub lon: f64,
    pub anchor_type: &'static str,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZionTarget {
    pub name: &'static str,
    pub lat: f64,
    pub lon: f64,
    pub length_ft: u32,
    pub target_type: &'static str,
}

pub fn anchor_network() -> Vec<Anchor> {
    vec![
        Anchor { name: "North Point Light", lat: 43.0642, lon: -87.8728, anchor_type: "Steel Tower", notes: "Milwaukee entrance" },
        Anchor { name: "Waukegan Harbor", lat: 42.3636, lon: -87.8036, anchor_type: "Steel Tower", notes: "ZION REFERENCE" },
        Anchor { name: "Grand Haven Pierhead", lat: 43.0636, lon: -86.2544, anchor_type: "Steel Tower", notes: "Coast Guard City" },
        Anchor { name: "Chicago Harbor Light", lat: 41.8897, lon: -87.6047, anchor_type: "Steel Caisson", notes: "Breakwater" },
        Anchor { name: "Michigan City East Pier", lat: 41.7136, lon: -86.8864, anchor_type: "Steel Tower", notes: "Active harbor" },
    ]
}

pub fn zion_targets() -> Vec<ZionTarget> {
    vec![
        ZionTarget { name: "Andaste (SS)", lat: 42.4125, lon: -87.2500, length_ft: 266, target_type: "Whaleback" },
        ZionTarget { name: "Monster (Unknown)", lat: 42.4180, lon: -87.2350, length_ft: 343, target_type: "Steel Freighter" },
        ZionTarget { name: "Loading Boom (1925)", lat: 42.4137, lon: -87.2488, length_ft: 117, target_type: "Steel Structure" },
    ]
}

pub fn candidate_length_match(pixel_count: u32, min_px: u32, max_px: u32) -> bool {
    pixel_count >= min_px && pixel_count <= max_px
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_zion_reference_anchor() {
        assert!(anchor_network().iter().any(|a| a.notes.contains("ZION")));
    }

    #[test]
    fn candidate_filter_works() {
        assert!(candidate_length_match(66, 50, 100));
        assert!(!candidate_length_match(234, 50, 100));
    }
}
