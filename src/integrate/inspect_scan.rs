//! Scan JSON inspector — port of `inspect_scan.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanDetection {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    #[serde(rename = "type")]
    pub det_type: String,
    pub line5_candidate: Option<bool>,
    pub known_wreck_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanJson {
    pub detections: Vec<ScanDetection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScanInspectSummary {
    pub total: usize,
    pub by_type: HashMap<String, usize>,
    pub bright_optical_blue: usize,
    pub thermal_cold_sink: usize,
    pub thermal_warm: usize,
}

pub fn summarize_scan(data: &ScanJson) -> ScanInspectSummary {
    let mut by_type = HashMap::new();
    for d in &data.detections {
        *by_type.entry(d.det_type.clone()).or_insert(0) += 1;
    }
    let bright_optical_blue = data
        .detections
        .iter()
        .filter(|d| d.det_type == "optical_blue" && d.zscore > 0.0)
        .count();
    let thermal_cold_sink = data
        .detections
        .iter()
        .filter(|d| d.det_type == "thermal" && d.zscore > -10.0 && d.zscore < -1.5)
        .count();
    let thermal_warm = data
        .detections
        .iter()
        .filter(|d| d.det_type == "thermal" && d.zscore > 0.5)
        .count();
    ScanInspectSummary {
        total: data.detections.len(),
        by_type,
        bright_optical_blue,
        thermal_cold_sink,
        thermal_warm,
    }
}

/// Port of `show_line5.py` — top Line 5 corridor candidates by |zscore|.
pub fn top_line5_candidates(data: &ScanJson, limit: usize) -> Vec<&ScanDetection> {
    let mut rows: Vec<_> = data
        .detections
        .iter()
        .filter(|d| d.line5_candidate == Some(true))
        .collect();
    rows.sort_by(|a, b| {
        b.zscore
            .abs()
            .partial_cmp(&a.zscore.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);
    rows
}

pub fn line5_summary(data: &ScanJson) -> (usize, usize) {
    let total = data.detections.len();
    let line5 = data
        .detections
        .iter()
        .filter(|d| d.line5_candidate == Some(true))
        .count();
    (total, line5)
}

pub fn top_optical_blue(data: &ScanJson, limit: usize) -> Vec<&ScanDetection> {
    let mut rows: Vec<_> = data
        .detections
        .iter()
        .filter(|d| d.det_type == "optical_blue" && d.zscore > 0.0)
        .collect();
    rows.sort_by(|a, b| b.zscore.partial_cmp(&a.zscore).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(limit);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_types() {
        let data = ScanJson {
            detections: vec![
                ScanDetection {
                    lat: 45.0,
                    lon: -85.0,
                    zscore: 2.0,
                    det_type: "optical_blue".into(),
                    line5_candidate: None,
                    known_wreck_name: None,
                },
                ScanDetection {
                    lat: 45.1,
                    lon: -85.1,
                    zscore: -3.0,
                    det_type: "thermal".into(),
                    line5_candidate: None,
                    known_wreck_name: None,
                },
            ],
        };
        let s = summarize_scan(&data);
        assert_eq!(s.total, 2);
        assert_eq!(s.bright_optical_blue, 1);
    }

    #[test]
    fn line5_sorted_by_abs_z() {
        let data = ScanJson {
            detections: vec![
                ScanDetection {
                    lat: 45.8,
                    lon: -84.73,
                    zscore: 1.0,
                    det_type: "hydrocarbon".into(),
                    line5_candidate: Some(true),
                    known_wreck_name: None,
                },
                ScanDetection {
                    lat: 45.81,
                    lon: -84.72,
                    zscore: -3.5,
                    det_type: "hydrocarbon".into(),
                    line5_candidate: Some(true),
                    known_wreck_name: None,
                },
            ],
        };
        let (total, line5) = line5_summary(&data);
        assert_eq!(total, 2);
        assert_eq!(line5, 2);
        let top = top_line5_candidates(&data, 1);
        assert_eq!(top[0].zscore, -3.5);
    }
}
