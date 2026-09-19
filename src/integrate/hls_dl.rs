//! Alias module for `hls_dl.py` — uses shared HLS granule helpers.

pub use super::hls_download::{
    cmr_query_hls30, extract_band_links, parse_tile_from_title, select_latest_per_month,
    CmrGranuleQuery, GranuleBandLink, SelectedGranule, CMR_GRANULES, KEY_BANDS, STRAITS_BBOX,
    STRAITS_TEMPORAL,
};
