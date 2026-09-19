//! GeoTIFF inventory — port of `wreckhunter/inventory_geotiffs.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MIN_TIFF_BYTES: u64 = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedTiffName {
    pub filename: String,
    pub satellite: String,
    pub acquisition_date: String,
    pub band: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeotiffInventory {
    pub timestamp: String,
    pub total_tiles: u32,
    pub by_location: std::collections::HashMap<String, u32>,
    pub by_satellite: std::collections::HashMap<String, u32>,
    pub by_band: std::collections::HashMap<String, u32>,
    pub tiles: Vec<ParsedTiffName>,
}

pub fn parse_hls_filename(name: &str, size_bytes: u64) -> Option<ParsedTiffName> {
    if size_bytes < MIN_TIFF_BYTES {
        return None;
    }
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 6 {
        return None;
    }
    let satellite_code = parts[1];
    let satellite = match satellite_code {
        "L30" => "Landsat-8",
        "S30" => "Sentinel-2",
        other => other,
    };
    let date_code = parts.get(3).map(|s| &s[..s.len().min(8)]).unwrap_or("Unknown");
    let band = parts
        .iter()
        .find(|p| p.starts_with('B') && p.len() <= 4)
        .unwrap_or(&"Unknown")
        .to_string();
    Some(ParsedTiffName {
        filename: name.to_string(),
        satellite: satellite.to_string(),
        acquisition_date: date_code.to_string(),
        band,
        size_bytes,
    })
}

pub fn record_tile(inv: &mut GeotiffInventory, location: &str, tile: ParsedTiffName) {
    inv.total_tiles += 1;
    *inv.by_location.entry(location.to_string()).or_insert(0) += 1;
    *inv
        .by_satellite
        .entry(tile.satellite.clone())
        .or_insert(0) += 1;
    *inv.by_band.entry(tile.band.clone()).or_insert(0) += 1;
    inv.tiles.push(tile);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hls_name() {
        let p = parse_hls_filename(
            "HLS.L30.T16TDN.2021182T162824.v2.0.B10.tif",
            500_000,
        )
        .unwrap();
        assert_eq!(p.satellite, "Landsat-8");
        assert_eq!(p.band, "B10");
    }
}
