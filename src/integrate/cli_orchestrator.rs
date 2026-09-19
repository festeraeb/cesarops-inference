//! CESAROPS CLI orchestration — port of `cesarops_cli.py` (TIFF discovery + batch plan).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const THERMAL_GLOB_PATTERNS: &[&str] = &["**/*B10.tif", "**/*B11.tif"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiffJob {
    pub path: PathBuf,
    pub stem: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchScanPlan {
    pub data_root: PathBuf,
    pub output_dir: PathBuf,
    pub jobs: Vec<TiffJob>,
    pub max_jobs: usize,
}

/// Collect thermal TIFF paths under `data_dir` (non-recursive glob via walk).
pub fn find_thermal_tiffs(data_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !data_dir.is_dir() {
        return out;
    }
    walk_tiffs(data_dir, &mut out);
    out.sort();
    out.dedup();
    out
}

fn walk_tiffs(dir: &Path, acc: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_tiffs(&path, acc);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let lower = name.to_ascii_lowercase();
            if lower.ends_with("b10.tif") || lower.ends_with("b11.tif") {
                acc.push(path);
            }
        }
    }
}

pub fn build_batch_plan(data_root: &Path, output_dir: &Path, max_jobs: usize) -> BatchScanPlan {
    let tiffs = find_thermal_tiffs(data_root);
    let jobs: Vec<TiffJob> = tiffs
        .into_iter()
        .take(max_jobs)
        .filter_map(|path| {
            let stem = path.file_stem()?.to_string_lossy().into_owned();
            Some(TiffJob { path, stem })
        })
        .collect();
    BatchScanPlan {
        data_root: data_root.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        jobs,
        max_jobs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn finds_b10_under_tree() {
        let tmp = std::env::temp_dir().join("cesarops_cli_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("nested")).unwrap();
        fs::write(tmp.join("nested/scene_B10.tif"), b"").unwrap();
        let found = find_thermal_tiffs(&tmp);
        assert_eq!(found.len(), 1);
        let _ = fs::remove_dir_all(&tmp);
    }
}
