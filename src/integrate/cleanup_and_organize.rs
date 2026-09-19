//! Cleanup plan for cesarops-clean branch — port of `cleanup_and_organize.py`.

use serde::{Deserialize, Serialize};

pub const KEEP_CORE: &[&str] = &[
    "database_connector.py",
    "cesarops_engine.py",
    "cuda_test_kmz.py",
    "tpu_server.py",
    "live_feed_server.py",
    "three_tile_offset_analysis.py",
    "validate_detection.py",
    "deep_wreck_validation.py",
];

pub const KEEP_DOCS: &[&str] = &[
    "FRESH_START_PLAN.md",
    "TODO_RECOVERY.md",
    "DATABASE_STATUS.md",
    "FILE_INVENTORY.md",
    "README.md",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CleanupPlan {
    pub clean_dir: String,
    pub archive_prefix: String,
    pub core_scripts: Vec<String>,
    pub docs: Vec<String>,
    pub skip_names: Vec<String>,
}

pub fn default_cleanup_plan(timestamp: &str) -> CleanupPlan {
    CleanupPlan {
        clean_dir: "cesarops-clean".into(),
        archive_prefix: format!("cesarops-archive-{timestamp}"),
        core_scripts: KEEP_CORE.iter().map(|s| s.to_string()).collect(),
        docs: KEEP_DOCS.iter().map(|s| s.to_string()).collect(),
        skip_names: vec![
            "cesarops-clean".into(),
            "cesarops-archive".into(),
            ".git".into(),
        ],
    }
}

pub fn should_archive(path_name: &str, plan: &CleanupPlan) -> bool {
    !plan.skip_names.iter().any(|s| s == path_name)
        && !path_name.starts_with("cesarops-archive-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_git() {
        let p = default_cleanup_plan("20260101");
        assert!(!should_archive(".git", &p));
        assert!(should_archive("old_script.py", &p));
    }
}
