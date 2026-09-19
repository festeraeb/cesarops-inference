//! Batch download swarm — port of `wreckhunter/batch_download_manager.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakeDownloadSpec {
    pub key: String,
    pub label: String,
    pub bbox: [f64; 4],
}

pub fn lake_catalog() -> Vec<LakeDownloadSpec> {
    vec![
        LakeDownloadSpec {
            key: "superior".into(),
            label: "Lake Superior".into(),
            bbox: [46.5, -92.0, 48.0, -84.5],
        },
        LakeDownloadSpec {
            key: "michigan".into(),
            label: "Lake Michigan".into(),
            bbox: [41.5, -88.0, 46.0, -85.5],
        },
        LakeDownloadSpec {
            key: "straits".into(),
            label: "Straits of Mackinac".into(),
            bbox: [45.65, -85.0, 46.10, -84.10],
        },
        LakeDownloadSpec {
            key: "huron".into(),
            label: "Lake Huron".into(),
            bbox: [42.5, -84.0, 46.0, -81.0],
        },
        LakeDownloadSpec {
            key: "erie".into(),
            label: "Lake Erie".into(),
            bbox: [41.3, -83.5, 42.5, -78.8],
        },
        LakeDownloadSpec {
            key: "ontario".into(),
            label: "Lake Ontario".into(),
            bbox: [43.2, -79.5, 44.2, -76.0],
        },
    ]
}

pub fn summer_fall_dates(year: i32) -> (String, String) {
    (format!("{year}-06-01"), format!("{year}-10-31"))
}

pub fn build_download_tasks(
    lakes: &[String],
    start_year: i32,
    end_year: i32,
) -> Vec<(String, i32, String)> {
    let mut out = Vec::new();
    for lake in lakes {
        for year in start_year..=end_year {
            out.push((lake.clone(), year, format!("{lake}-{year}")));
        }
    }
    out
}

pub fn filter_chunks_for_node<T: Clone>(chunks: &[T], node_idx: u32) -> Vec<T> {
    chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as u32) % 2 == node_idx)
        .map(|(_, c)| c.clone())
        .collect()
}
