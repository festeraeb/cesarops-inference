//! Repo file inventory — port of `scripts/inventory_all_files.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const CORE_SCRIPTS: &[&str] = &[
    "database_connector.py",
    "cesarops_engine.py",
    "cuda_test_kmz.py",
    "tpu_server.py",
    "live_feed_server.py",
    "three_tile_offset_analysis.py",
    "validate_detection.py",
    "deep_wreck_validation.py",
    "smart_daily_scan.py",
    "find_swot_dates.py",
    "prioritized_pull_v2.py",
];

pub const SCRIPTS_FOLDER: &[&str] = &[
    "scripts/wipe_database.py",
    "scripts/inventory_geotiffs.py",
    "scripts/process_tiles.py",
    "scripts/check_xenon_cuda.py",
    "scripts/compare_runs.py",
    "scripts/populate_database.py",
    "scripts/validate_database.py",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileInventoryReport {
    pub timestamp: String,
    pub core_scripts: Vec<PathBuf>,
    pub scripts_folder: Vec<PathBuf>,
    pub geotiffs: Vec<PathBuf>,
    pub bag_files: Vec<PathBuf>,
    pub archive_candidates: Vec<PathBuf>,
}

pub fn classify_path(path: &str) -> &'static str {
    let lower = path.to_lowercase();
    if lower.ends_with(".tif") || lower.ends_with(".tiff") {
        "geotiff"
    } else if lower.ends_with(".bag") {
        "bag"
    } else if lower.ends_with(".py") {
        "script"
    } else {
        "other"
    }
}

pub fn is_core_script(name: &str) -> bool {
    CORE_SCRIPTS
        .iter()
        .any(|c| *c == name || c.ends_with(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_tiff() {
        assert_eq!(classify_path("tile.tif"), "geotiff");
    }
}
