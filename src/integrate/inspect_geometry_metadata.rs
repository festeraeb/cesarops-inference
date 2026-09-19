//! Geometry metadata tag filter — port of `inspect_geometry_metadata.py`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const GEO_KEY_HINTS: &[&str] = &[
    "sun", "solar", "azimuth", "zenith", "elevation", "view", "incidence", "illumin", "angle",
    "satellite", "sensor", "earth_sun", "distance", "mean_angle", "off_nadir", "cloud",
    "scene_center", "acquisition", "date", "time", "spacecraft", "platform", "processing_level",
    "product_id", "station_id", "wrs", "mgrs", "orbit",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeometryTagSet {
    pub file: String,
    pub geo_keys: BTreeMap<String, String>,
}

pub fn is_geometry_tag(key: &str) -> bool {
    let kl = key.to_lowercase();
    GEO_KEY_HINTS.iter().any(|hint| kl.contains(hint))
}

pub fn extract_geometry_tags(tags: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    tags.iter()
        .filter(|(k, _)| is_geometry_tag(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

pub fn effective_water_depth_factor(solar_zenith_deg: f64) -> f64 {
    solar_zenith_deg.to_radians().cos()
}

pub fn shadow_height_m(shadow_length_m: f64, solar_elevation_deg: f64) -> f64 {
    shadow_length_m * solar_elevation_deg.to_radians().tan()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sun_tags() {
        let mut tags = BTreeMap::new();
        tags.insert("SUN_AZIMUTH".into(), "145".into());
        tags.insert("PRODUCT".into(), "HLS".into());
        let geo = extract_geometry_tags(&tags);
        assert!(geo.contains_key("SUN_AZIMUTH"));
        assert!(!geo.contains_key("PRODUCT"));
    }
}
