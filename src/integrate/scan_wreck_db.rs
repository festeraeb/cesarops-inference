//! Scan Bagrecovery for wreck SQLite DBs — port of `scan_wreck_db.py`.

use serde::{Deserialize, Serialize};

pub const MIN_DB_BYTES: u64 = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DbCandidate {
    pub path: String,
    pub size_kb: u64,
    pub parent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableCoordProbe {
    pub table: String,
    pub row_count: u64,
    pub lat_col: Option<String>,
    pub lon_col: Option<String>,
    pub real_coord_rows: u64,
}

pub fn qualifies_db(size_bytes: u64) -> bool {
    size_bytes >= MIN_DB_BYTES
}

pub fn guess_lat_lon_columns(columns: &[String]) -> (Option<String>, Option<String>) {
    let lat = columns.iter().find(|c| c.to_lowercase().contains("lat")).cloned();
    let lon = columns.iter().find(|c| c.to_lowercase().contains("lon")).cloned();
    (lat, lon)
}

pub fn real_coord_count_sql(table: &str, lat: &str, lon: &str) -> String {
    format!(
        "SELECT COUNT(*) FROM [{table}] WHERE [{lat}] IS NOT NULL AND [{lat}] != 0 AND [{lat}] != 45.0"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_size_filter() {
        assert!(!qualifies_db(5000));
        assert!(qualifies_db(20000));
    }
}
