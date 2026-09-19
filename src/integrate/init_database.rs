//! SQLite schema bootstrap — port of `cesarops/db/init_database.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_DB: &str = "outputs/cesarops.db";
pub const SCHEMA_FILE: &str = "cesarops_comprehensive_schema.sql";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InitDatabaseConfig {
    pub db_path: PathBuf,
    pub schema_path: PathBuf,
    pub reset_existing: bool,
}

impl Default for InitDatabaseConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from(DEFAULT_DB),
            schema_path: PathBuf::from(SCHEMA_FILE),
            reset_existing: true,
        }
    }
}

/// Minimal schema when comprehensive SQL file is missing.
pub const MINIMAL_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS scan_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT,
    run_type TEXT,
    input_directory TEXT,
    output_directory TEXT,
    tile_count INTEGER,
    min_confidence REAL,
    gpu_name TEXT,
    total_detections INTEGER DEFAULT 0,
    start_time DATETIME,
    end_time DATETIME,
    duration_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tiles_processed (
    tile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    tile_path TEXT,
    tile_prefix TEXT,
    satellite_type TEXT,
    acquisition_date TEXT,
    bands_processed TEXT,
    width_pixels INTEGER,
    height_pixels INTEGER,
    gpu_compute_time_seconds REAL,
    thermal_mean REAL,
    thermal_stddev REAL,
    raw_anomaly_count INTEGER,
    FOREIGN KEY (run_id) REFERENCES scan_runs(run_id)
);
CREATE TABLE IF NOT EXISTS raw_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    pixel_row INTEGER,
    pixel_col INTEGER,
    wgs84_lat REAL,
    wgs84_lon REAL,
    z_score REAL,
    confidence_score REAL,
    FOREIGN KEY (tile_id) REFERENCES tiles_processed(tile_id),
    FOREIGN KEY (run_id) REFERENCES scan_runs(run_id)
);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"#;

pub fn metadata_bootstrap_rows(now: &str) -> Vec<(&'static str, String)> {
    vec![
        ("database_version", "1.0".into()),
        ("created_at", now.to_string()),
    ]
}

pub fn expected_tables() -> &'static [&'static str] {
    &[
        "scan_runs",
        "tiles_processed",
        "raw_detections",
        "metadata",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_schema_has_core_tables() {
        assert!(MINIMAL_SCHEMA.contains("scan_runs"));
        assert!(MINIMAL_SCHEMA.contains("raw_detections"));
    }
}
