//! STAC geometry fetch params and bathymetric physics — port of `fetch_geometry_metadata.py`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const STAC_SEARCH_URL: &str = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search";
pub const STRAITS_SEARCH_BBOX: [f64; 4] = [-84.85, 45.70, -84.10, 46.10];

pub const BATHY_PROPERTY_KEYS: &[&str] = &[
    "view:sun_azimuth",
    "view:sun_elevation",
    "view:off_nadir",
    "eo:cloud_cover",
    "platform",
    "datetime",
    "MEAN_SUN_AZIMUTH_ANGLE",
    "MEAN_SUN_ZENITH_ANGLE",
    "MEAN_VIEW_AZIMUTH_ANGLE",
    "MEAN_VIEW_ZENITH_ANGLE",
    "SPATIAL_COVERAGE",
    "CLOUD_COVERAGE",
    "SUN_AZIMUTH",
    "SUN_ELEVATION",
    "EARTH_SUN_DISTANCE",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GranuleRef {
    pub collection: String,
    pub granule_id: String,
    pub short_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BathyPhysicsSummary {
    pub sun_elevation_deg: f64,
    pub solar_zenith_deg: f64,
    pub cos_solar_zenith: f64,
    pub blue_1e_depth_m: f64,
    pub depth_correction_factor: f64,
    pub bridge_shadow_length_m: f64,
    pub stumpf_error_pct: f64,
}

pub fn default_sep2024_granules() -> Vec<GranuleRef> {
    vec![
        GranuleRef {
            collection: "HLSL30.v2.0".into(),
            granule_id: "HLS.L30.T16TFR.2024247T163117.v2.0".into(),
            short_name: "Landsat HLS 16TFR Sep3".into(),
        },
        GranuleRef {
            collection: "HLSS30.v2.0".into(),
            granule_id: "HLS.S30.T16TFR.2024247T163117.v2.0".into(),
            short_name: "Sentinel HLS 16TFR Sep3".into(),
        },
        GranuleRef {
            collection: "HLSS30.v2.0".into(),
            granule_id: "HLS.S30.T16TGR.2024247T163117.v2.0".into(),
            short_name: "Sentinel HLS 16TGR Sep3".into(),
        },
        GranuleRef {
            collection: "HLSS30.v2.0".into(),
            granule_id: "HLS.S30.T16TGS.2024247T163117.v2.0".into(),
            short_name: "Sentinel HLS 16TGS Sep3".into(),
        },
    ]
}

pub fn stac_search_body(date: &str) -> Value {
    json!({
        "collections": ["HLSL30.v2.0", "HLSS30.v2.0"],
        "datetime": format!("{date}T00:00:00Z/{date}T23:59:59Z"),
        "bbox": STRAITS_SEARCH_BBOX,
        "limit": 20
    })
}

pub fn is_bathy_property_key(key: &str) -> bool {
    let kl = key.to_lowercase();
    kl.contains("sun")
        || kl.contains("solar")
        || kl.contains("azimuth")
        || kl.contains("zenith")
        || kl.contains("elevation")
        || kl.contains("view")
        || kl.contains("nadir")
        || kl.contains("cloud")
        || kl.contains("angle")
        || kl.contains("platform")
        || kl.contains("coverage")
        || kl.contains("earth_sun")
        || kl.contains("incidence")
}

pub fn extract_bathy_properties(props: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    props
        .iter()
        .filter(|(k, _)| is_bathy_property_key(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

pub fn parse_sun_elevation_from_props(props: &BTreeMap<String, Value>) -> Option<f64> {
    for (k, v) in props {
        let kl = k.to_lowercase();
        if kl.contains("sun_elevation") || kl.contains("sun_el") {
            if let Some(n) = v.as_f64() {
                return Some(n);
            }
        }
    }
    None
}

pub fn approximate_solar_elevation(lat_deg: f64, day_of_year: i32, hour_angle_deg: f64) -> f64 {
    let lat_rad = lat_deg.to_radians();
    let decl = (-23.45 * ((360.0 / 365.0) * (day_of_year as f64 + 10.0)).to_radians().cos())
        .to_radians();
    let ha = hour_angle_deg.to_radians();
    let sin_el = lat_rad.sin() * decl.sin() + lat_rad.cos() * decl.cos() * ha.cos();
    sin_el.asin().to_degrees()
}

pub fn compute_bathy_physics(sun_elevation_deg: f64) -> BathyPhysicsSummary {
    let zen = 90.0 - sun_elevation_deg;
    let cos_zen = zen.to_radians().cos();
    let depth_corr = if cos_zen > 0.01 { 1.0 / cos_zen } else { 99.0 };
    let blue_depth = 25.0 * cos_zen;
    let bridge_height_m = 167.0;
    let shadow_len = if sun_elevation_deg > 0.5 {
        bridge_height_m / sun_elevation_deg.to_radians().tan()
    } else {
        0.0
    };
    BathyPhysicsSummary {
        sun_elevation_deg,
        solar_zenith_deg: zen,
        cos_solar_zenith: cos_zen,
        blue_1e_depth_m: blue_depth,
        depth_correction_factor: depth_corr,
        bridge_shadow_length_m: shadow_len,
        stumpf_error_pct: (depth_corr - 1.0) * 100.0,
    }
}

pub fn bathy_summary_from_stac_or_fallback(props: &BTreeMap<String, Value>, lat: f64, doy: i32) -> BathyPhysicsSummary {
    let el = parse_sun_elevation_from_props(props).unwrap_or_else(|| approximate_solar_elevation(lat, doy, -22.5));
    compute_bathy_physics(el)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stac_body_has_collections() {
        let body = stac_search_body("2024-09-03");
        assert_eq!(body["collections"][0], "HLSL30.v2.0");
    }

    #[test]
    fn bathy_physics_positive_depth() {
        let s = compute_bathy_physics(46.0);
        assert!(s.blue_1e_depth_m > 0.0);
        assert!(s.depth_correction_factor >= 1.0);
    }
}
