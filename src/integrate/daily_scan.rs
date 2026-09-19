//! Daily scan helpers — port of `daily_scan.py` (discovery + summary).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DailyScanSummary {
    pub tiff_count: usize,
    pub detection_count: u64,
    pub classified_internal: u64,
}

pub fn find_tiffs(root: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_tiffs(root, suffix, &mut out);
    out.sort();
    out.dedup();
    out
}

fn walk_tiffs(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    if let Ok(read) = std::fs::read_dir(dir) {
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_tiffs(&path, suffix, out);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(suffix))
                .unwrap_or(false)
            {
                out.push(path);
            }
        }
    }
}

pub fn summarize_detection_counts(per_tile: &[u64]) -> DailyScanSummary {
    let detection_count: u64 = per_tile.iter().sum();
    DailyScanSummary {
        tiff_count: per_tile.len(),
        detection_count,
        classified_internal: detection_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn summarizes_counts() {
        let s = summarize_detection_counts(&[3, 5, 2]);
        assert_eq!(s.tiff_count, 3);
        assert_eq!(s.detection_count, 10);
    }

    #[test]
    fn finds_tiff_suffix() {
        let tmp = std::env::temp_dir().join(format!("daily_scan_{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("tile_B11.tif"), b"x").unwrap();
        let hits = find_tiffs(&tmp, ".tif");
        assert_eq!(hits.len(), 1);
        let _ = fs::remove_dir_all(tmp);
    }
}
