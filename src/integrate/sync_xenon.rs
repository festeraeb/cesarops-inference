//! Deploy DB tooling to Xenon — port of `deploy_xenon.py`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const XENON_USER: &str = "cesarops";
pub const XENON_HOST: &str = "10.0.0.55";
pub const XENON_REMOTE_PATH: &str = "~/cesarops-wreckhunter-build";

pub const DEPLOY_FILES: &[&str] = &[
    "database_connector.py",
    "cesarops_comprehensive_schema.sql",
    "init_database.py",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScpJob {
    pub local: PathBuf,
    pub remote: String,
}

pub fn xenon_scp_command(local: &Path) -> String {
    format!(
        "scp \"{}\" {}@{}:{}/",
        local.display(),
        XENON_USER,
        XENON_HOST,
        XENON_REMOTE_PATH
    )
}

pub fn plan_deploy_jobs(base_dir: &Path) -> Vec<ScpJob> {
    DEPLOY_FILES
        .iter()
        .filter_map(|name| {
            let local = base_dir.join(name);
            if local.exists() {
                Some(ScpJob {
                    local,
                    remote: format!("{XENON_REMOTE_PATH}/{name}"),
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scp_includes_host() {
        let cmd = xenon_scp_command(Path::new("/tmp/database_connector.py"));
        assert!(cmd.contains("10.0.0.55"));
    }
}
