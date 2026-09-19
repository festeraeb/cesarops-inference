//! Anonymous altimetry FTP pull — port of `wreckhunter/pull_altimetry_anonymous.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const AVISO_FTP_HOST: &str = "avisoftp.cnes.fr";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakeAltimetryBbox {
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
}

pub fn great_lakes_altimetry_bboxes() -> HashMap<&'static str, LakeAltimetryBbox> {
    let mut m = HashMap::new();
    m.insert(
        "MICHIGAN",
        LakeAltimetryBbox {
            lon_min: -87.9,
            lat_min: 41.5,
            lon_max: -85.5,
            lat_max: 46.0,
        },
    );
    m.insert(
        "ERIE",
        LakeAltimetryBbox {
            lon_min: -83.5,
            lat_min: 41.5,
            lon_max: -80.5,
            lat_max: 42.5,
        },
    );
    m
}

pub fn satellite_ftp_path(satellite: &str) -> Option<&'static str> {
    match satellite.to_uppercase().as_str() {
        "SARAL" => Some("/AVISO/pub/saral"),
        "HY2D" => Some("/AVISO/pub/hy2d"),
        "JASON3" => Some("/AVISO/pub/jason3"),
        "SENTINEL6" => Some("/AVISO/pub/sentinel-6"),
        _ => None,
    }
}

pub fn podaac_url(satellite: &str) -> Option<&'static str> {
    match satellite.to_uppercase().as_str() {
        "JASON3" => Some("https://podaac-opendap.jpl.nasa.gov/opendap/allData/jason3/"),
        "SENTINEL6" => Some("https://podaac-opendap.jpl.nasa.gov/opendap/allData/sentinel-6/"),
        _ => None,
    }
}
