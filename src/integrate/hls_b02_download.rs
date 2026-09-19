//! HLS B02 band download — port of `satellite/b02_download.py`.

use serde::{Deserialize, Serialize};

pub const CMR_GRANULES_URL: &str = "https://cmr.earthdata.nasa.gov/search/granules.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct B02DownloadJob {
    pub granule_title: String,
    pub product_short_name: String,
    pub dest_path: String,
}

pub fn product_for_title(title: &str) -> &'static str {
    if title.contains("L30") {
        "HLSL30"
    } else {
        "HLSS30"
    }
}

pub fn b02_dest_name(title: &str) -> String {
    format!("{title}.B02.tif")
}

pub fn granule_query_params(title: &str, product: &str) -> Vec<(&'static str, String)> {
    vec![
        ("short_name", product.to_string()),
        ("page_size", "1".into()),
        ("granule_ur", title.to_string()),
    ]
}

pub fn pick_b02_href(links: &[serde_json::Value]) -> Option<String> {
    for link in links {
        let href = link.get("href").and_then(|v| v.as_str()).unwrap_or("");
        let rel = link.get("rel").and_then(|v| v.as_str()).unwrap_or("");
        if href.contains("B02.tif") && rel.contains("data#") {
            return Some(href.to_string());
        }
    }
    None
}

pub fn should_skip_existing(size_bytes: u64) -> bool {
    size_bytes > 0
}
