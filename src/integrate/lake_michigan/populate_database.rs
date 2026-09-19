//! Populate census DB from run JSON — port of `wreckhunter/populate_database.py`.

use serde::{Deserialize, Serialize};

pub const CENSUS_DB: &str = "wreckhunter2000/LAKE_MICHIGAN_CENSUS_2026.db";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunTileRecord {
    pub filename: String,
    pub sensor_type: String,
    pub zscore: f32,
    pub anomaly_count: u32,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

pub const ANOMALY_HITS_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS anomaly_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_path TEXT,
    sensor_type TEXT,
    zscore REAL,
    anomaly_count INTEGER,
    lat REAL,
    lon REAL,
    detected_at TEXT
);
"#;

pub fn parse_results_json(value: &serde_json::Value) -> Vec<RunTileRecord> {
    let mut out = Vec::new();
    let Some(results) = value.get("results").and_then(|v| v.as_array()) else {
        return out;
    };
    for tile in results {
        let sensors = tile.get("sensors").and_then(|v| v.as_object());
        let thermal_z = sensors
            .and_then(|s| s.get("thermal"))
            .and_then(|t| t.get("max_zscore"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let count = sensors
            .and_then(|s| s.get("thermal"))
            .and_then(|t| t.get("anomaly_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        out.push(RunTileRecord {
            filename: tile
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            sensor_type: "thermal".into(),
            zscore: thermal_z,
            anomaly_count: count,
            lat: None,
            lon: None,
        });
    }
    out
}

pub fn sql_insert_hit(rec: &RunTileRecord, detected_at: &str) -> (String, Vec<serde_json::Value>) {
    (
        r#"INSERT INTO anomaly_hits (tile_path, sensor_type, zscore, anomaly_count, lat, lon, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)"#
            .into(),
        vec![
            serde_json::json!(rec.filename),
            serde_json::json!(rec.sensor_type),
            serde_json::json!(rec.zscore),
            serde_json::json!(rec.anomaly_count),
            serde_json::json!(rec.lat),
            serde_json::json!(rec.lon),
            serde_json::json!(detected_at),
        ],
    )
}
