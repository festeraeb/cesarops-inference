//! Census / runs DB connector — port of `utils/database_connector.py` (types + SQL).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DbPaths {
    pub census_db: PathBuf,
    pub runs_db: PathBuf,
}

impl Default for DbPaths {
    fn default() -> Self {
        Self {
            census_db: PathBuf::from("wreckhunter2000/LAKE_MICHIGAN_CENSUS_2026.db"),
            runs_db: PathBuf::from("outputs/run_zero/cesarops_runs.db"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StationaryAnchor {
    pub id: i64,
    pub lat: f64,
    pub lon: f64,
    pub triple_lock_status: String,
    pub combined_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NewArrival {
    pub id: i64,
    pub lat: f64,
    pub lon: f64,
    pub triple_lock_status: String,
    pub score: Option<f64>,
    pub priority: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CensusStatus {
    pub census_present: bool,
    pub stationary_anchors: u64,
    pub new_arrivals: u64,
    pub anomaly_hits: u64,
    pub swot_passes: u64,
}

pub fn census_exists(paths: &DbPaths) -> bool {
    paths.census_db.exists()
}

pub fn runs_exists(paths: &DbPaths) -> bool {
    paths.runs_db.exists()
}

/// INSERT for cuda batch logging (census `anomaly_hits`).
pub fn sql_log_scan_run(
    epoch_date: &str,
    run_name: &str,
    tile_count: u32,
    detection_count: u32,
) -> (&'static str, Vec<serde_json::Value>) {
    (
        r#"INSERT INTO anomaly_hits (
            epoch_date, lat, lon, concept, score, classification,
            scene_id, thermal_zscore, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        vec![
            serde_json::json!(epoch_date),
            serde_json::json!(42.5),
            serde_json::json!(-87.0),
            serde_json::json!(format!("cuda_batch_{tile_count}_tiles")),
            serde_json::json!(0.8),
            serde_json::json!("cesarops_cuda_test"),
            serde_json::json!(run_name),
            serde_json::json!(detection_count as f64),
            serde_json::json!(chrono_like_now()),
        ],
    )
}

pub fn sql_stationary_anchors() -> &'static str {
    r#"SELECT id, lat, lon, triple_lock_status, swot_persistent_anomaly,
       combined_score, thermal_sink_l8, sar_stability_s1
       FROM stationary_anchors ORDER BY id"#
}

pub fn sql_new_arrivals(limit: Option<u32>) -> String {
    let mut q = String::from(
        r#"SELECT id, lat, lon, triple_lock_status, flagged_at,
           score, priority, thermal_sink_l8, sar_stability_s1
           FROM new_arrivals ORDER BY id DESC"#,
    );
    if let Some(n) = limit {
        q.push_str(&format!(" LIMIT {n}"));
    }
    q
}

pub fn sql_update_triple_lock(anchor_id: i64, status: &str, updated_at: &str) -> (&'static str, Vec<serde_json::Value>) {
    (
        r#"UPDATE stationary_anchors SET triple_lock_status = ?, updated_at = ? WHERE id = ?"#,
        vec![
            serde_json::json!(status),
            serde_json::json!(updated_at),
            serde_json::json!(anchor_id),
        ],
    )
}

fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_paths_relative() {
        let p = DbPaths::default();
        assert!(p.census_db.to_string_lossy().contains("CENSUS"));
    }

    #[test]
    fn log_scan_sql_has_placeholders() {
        let (sql, args) = sql_log_scan_run("2026-05-26", "run_a", 10, 100);
        assert!(sql.contains("anomaly_hits"));
        assert_eq!(args.len(), 9);
    }
}
