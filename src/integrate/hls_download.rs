//! HLS/CMR granule selection — ports `hls_dl.py`, `hls_download.py`, etc.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CMR_GRANULES: &str = "https://cmr.earthdata.nasa.gov/search/granules.json";
pub const STRAITS_BBOX: &str = "-87.9,45.78,-84.6,46.0";
pub const STRAITS_TEMPORAL: &str = "2015-01-01T00:00:00Z/2016-12-31T23:59:59Z";
pub const KEY_BANDS: &[&str] = &["B03.tif", "B04.tif", "B08.tif", "B10.tif", "B11.tif", "Fmask.tif"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CmrGranuleQuery {
    pub short_name: String,
    pub bounding_box: String,
    pub temporal: String,
    pub page_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GranuleBandLink {
    pub band: String,
    pub href: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedGranule {
    pub title: String,
    pub date: String,
    pub tile: String,
    pub bands: Vec<GranuleBandLink>,
}

pub fn cmr_query_hls30(product: &str) -> CmrGranuleQuery {
    CmrGranuleQuery {
        short_name: product.into(),
        bounding_box: STRAITS_BBOX.into(),
        temporal: STRAITS_TEMPORAL.into(),
        page_size: 2000,
    }
}

pub fn parse_tile_from_title(title: &str) -> String {
    title.split('.').nth(2).unwrap_or("?").to_string()
}

pub fn extract_band_links(links: &[serde_json::Value]) -> Vec<GranuleBandLink> {
    let mut out = Vec::new();
    for link in links {
        let href = link.get("href").and_then(|v| v.as_str()).unwrap_or("");
        if !href.ends_with(".tif") {
            continue;
        }
        let band = href.rsplit('/').next().unwrap_or("").to_string();
        if KEY_BANDS.contains(&band.as_str()) {
            out.push(GranuleBandLink { band, href: href.into() });
        }
    }
    out
}

pub fn select_latest_per_month(granules: &[SelectedGranule]) -> Vec<SelectedGranule> {
    let mut by_month: BTreeMap<String, Vec<&SelectedGranule>> = BTreeMap::new();
    for g in granules {
        if g.date.len() >= 7 {
            by_month.entry(g.date[..7].to_string()).or_default().push(g);
        }
    }
    let mut selected = Vec::new();
    for year in ["2015", "2016"] {
        for month in 3..=12 {
            let ym = format!("{year}-{month:02}");
            if let Some(rows) = by_month.get(&ym) {
                let latest = rows.iter().map(|g| g.date.as_str()).max().unwrap_or("");
                for g in rows.iter().filter(|g| g.date == latest) {
                    selected.push((*g).clone());
                }
            }
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tile_token() {
        assert_eq!(
            parse_tile_from_title("HLS.S30.T16TFR.20240903.v2.0"),
            "T16TFR"
        );
    }
}
