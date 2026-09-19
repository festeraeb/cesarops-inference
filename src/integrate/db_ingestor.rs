//! Probe JSON → SQLite ingestor — port of `wreckhunter/db_ingestor.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_DB: &str = "cesarops_master.db";
pub const WATCH_DIR: &str = "outputs/probes";
pub const PROCESSED_MARKER_DIR: &str = "outputs/probes/.processed";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeFeature {
    pub lat: f64,
    pub lon: f64,
    pub sensor: Option<String>,
    pub confidence: f32,
    pub concept: String,
    pub tile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeGeoJson {
    pub tile_id: Option<String>,
    pub features: Vec<ProbeGeoJsonFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeGeoJsonFeature {
    pub geometry: ProbeGeometry,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeGeometry {
    pub coordinates: Vec<f64>,
}

pub const ANOMALY_HITS_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS anomaly_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL, lon REAL,
    sensor TEXT,
    confidence REAL,
    concept TEXT,
    tile_id TEXT,
    ingested_at TEXT
);
"#;

pub fn parse_probe_json(value: &serde_json::Value) -> Vec<ProbeFeature> {
    let tile_id = value
        .get("tile_id")
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    let mut out = Vec::new();
    let Some(features) = value.get("features").and_then(|v| v.as_array()) else {
        return out;
    };
    for feat in features {
        let coords = feat
            .get("geometry")
            .and_then(|g| g.get("coordinates"))
            .and_then(|c| c.as_array());
        let Some(coords) = coords else { continue };
        if coords.len() < 2 {
            continue;
        }
        let lon = coords[0].as_f64().unwrap_or(0.0);
        let lat = coords[1].as_f64().unwrap_or(0.0);
        let props = feat.get("properties").cloned().unwrap_or_default();
        out.push(ProbeFeature {
            lat,
            lon,
            sensor: props.get("sensor").and_then(|v| v.as_str()).map(String::from),
            confidence: props
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32,
            concept: props
                .get("concept")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            tile_id: tile_id.clone(),
        });
    }
    out
}

pub fn marker_path_for(probe: &PathBuf) -> PathBuf {
    PathBuf::from(PROCESSED_MARKER_DIR).join(
        probe
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown.json".into()),
    )
}
