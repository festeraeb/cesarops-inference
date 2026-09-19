//! Detection sorter/filter logic from `detection_sorter.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSite {
    pub site_id: String,
    pub confidence_score: f64,
    pub total_detections: u32,
    pub release_status: String,
    pub identified: bool,
    pub material: String,
    pub tools_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SorterFilters {
    pub min_confidence: f64,
    pub min_detections: u32,
    pub release_status: Vec<String>,
    pub identified: Option<bool>,
    pub materials: Vec<String>,
    pub tools: Vec<String>,
}

impl Default for SorterFilters {
    fn default() -> Self {
        Self {
            min_confidence: 0.0,
            min_detections: 1,
            release_status: vec!["CONFIRMED".into(), "PUBLIC".into()],
            identified: None,
            materials: vec![],
            tools: vec![],
        }
    }
}

pub fn apply_filters(sites: &[DetectionSite], filters: &SorterFilters, admin_mode: bool) -> Vec<DetectionSite> {
    let mut out: Vec<DetectionSite> = sites
        .iter()
        .filter(|s| s.confidence_score >= filters.min_confidence)
        .filter(|s| s.total_detections >= filters.min_detections)
        .filter(|s| {
            if admin_mode || filters.release_status.is_empty() {
                true
            } else {
                filters.release_status.iter().any(|x| x == &s.release_status)
            }
        })
        .filter(|s| filters.identified.is_none_or(|v| s.identified == v))
        .filter(|s| filters.materials.is_empty() || filters.materials.iter().any(|m| m == &s.material))
        .filter(|s| {
            filters.tools.is_empty() || filters.tools.iter().all(|needle| s.tools_used.iter().any(|t| t == needle))
        })
        .cloned()
        .collect();
    out.sort_by(|a, b| {
        b.confidence_score
            .partial_cmp(&a.confidence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.total_detections.cmp(&a.total_detections))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<DetectionSite> {
        vec![
            DetectionSite {
                site_id: "A".into(),
                confidence_score: 0.95,
                total_detections: 8,
                release_status: "PUBLIC".into(),
                identified: true,
                material: "steel".into(),
                tools_used: vec!["M2200".into(), "SAR".into()],
            },
            DetectionSite {
                site_id: "B".into(),
                confidence_score: 0.45,
                total_detections: 1,
                release_status: "INTERNAL".into(),
                identified: false,
                material: "unknown".into(),
                tools_used: vec!["M2200".into()],
            },
        ]
    }

    #[test]
    fn public_filter_hides_internal() {
        let out = apply_filters(&sample(), &SorterFilters::default(), false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].site_id, "A");
    }

    #[test]
    fn admin_can_view_internal() {
        let out = apply_filters(&sample(), &SorterFilters::default(), true);
        assert_eq!(out.len(), 2);
    }
}
