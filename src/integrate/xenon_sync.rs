//! Xenon DB sync starter — port of `wreckhunter/sync_xenon_db.py`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const XENON_USER: &str = "cesarops";
pub const XENON_HOST: &str = "10.0.0.55";
pub const XENON_PATH: &str = "~/cesarops-wreckhunter-build";

pub const DB_SYNC_FILES: &[&str] = &[
    "init_database.py",
    "cesarops_comprehensive_schema.sql",
    "database_connector.py",
    "triple_lock_fusion.py",
    "lake_michigan_scan.py",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct XenonSyncPlan {
    pub ssh_test: String,
    pub files: Vec<PathBuf>,
    pub remote_init_cmd: String,
}

pub fn build_sync_plan(base: &Path) -> XenonSyncPlan {
    let files: Vec<PathBuf> = DB_SYNC_FILES
        .iter()
        .map(|f| base.join(f))
        .filter(|p| p.exists())
        .collect();
    XenonSyncPlan {
        ssh_test: format!("ssh {XENON_USER}@{XENON_HOST} \"echo Connected\""),
        files,
        remote_init_cmd: format!(
            "ssh {XENON_USER}@{XENON_HOST} \"cd {XENON_PATH}; python3 init_database.py\""
        ),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct XenonSyncSummary {
    pub synced: u32,
    pub failed: u32,
    pub skipped: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_init_uses_xenon_path() {
        let plan = build_sync_plan(Path::new("."));
        assert!(plan.remote_init_cmd.contains("init_database.py"));
    }
}
