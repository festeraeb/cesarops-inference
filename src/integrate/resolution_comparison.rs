//! Resolution comparison metrics — port of `analyze_resolution_comparison.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolutionRun {
    pub run_id: u64,
    pub run_name: String,
    pub detection_count: u64,
    pub duration_seconds: f64,
    pub chunking_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolutionMetrics {
    pub full_avg_detections: f64,
    pub reduced_avg_detections: f64,
    pub full_avg_time: f64,
    pub reduced_avg_time: f64,
    pub speedup: f64,
    pub position_match_rate: f64,
    pub avg_score_diff: f64,
}

pub fn analyze_resolution_comparison(runs: &[ResolutionRun]) -> ResolutionMetrics {
    let full: Vec<_> = runs.iter().filter(|r| r.chunking_enabled).collect();
    let reduced: Vec<_> = runs.iter().filter(|r| !r.chunking_enabled).collect();
    let avg = |rows: &[&ResolutionRun], f: fn(&ResolutionRun) -> f64| -> f64 {
        if rows.is_empty() {
            0.0
        } else {
            rows.iter().map(|r| f(r)).sum::<f64>() / rows.len() as f64
        }
    };
    let full_avg_detections = avg(&full, |r| r.detection_count as f64);
    let reduced_avg_detections = avg(&reduced, |r| r.detection_count as f64);
    let full_avg_time = avg(&full, |r| r.duration_seconds);
    let reduced_avg_time = avg(&reduced, |r| r.duration_seconds);
    let speedup = if reduced_avg_time > 0.0 {
        full_avg_time / reduced_avg_time
    } else {
        0.0
    };
    ResolutionMetrics {
        full_avg_detections,
        reduced_avg_detections,
        full_avg_time,
        reduced_avg_time,
        speedup,
        position_match_rate: 0.0,
        avg_score_diff: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_speedup() {
        let runs = vec![
            ResolutionRun {
                run_id: 1,
                run_name: "full".into(),
                detection_count: 10,
                duration_seconds: 20.0,
                chunking_enabled: true,
            },
            ResolutionRun {
                run_id: 2,
                run_name: "reduced".into(),
                detection_count: 9,
                duration_seconds: 10.0,
                chunking_enabled: false,
            },
        ];
        let m = analyze_resolution_comparison(&runs);
        assert!((m.speedup - 2.0).abs() < 0.001);
    }
}
