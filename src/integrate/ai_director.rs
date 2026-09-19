//! AI Director planning primitives from `ai_director.py`.
//! Pure Rust request parsing + preset/tool selection (no remote LLM call here).

use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct BBoxPreset {
    pub name: &'static str,
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
    pub targets: &'static [&'static str],
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolMeta {
    pub script: &'static str,
    pub description: &'static str,
    pub default_threshold: Option<f32>,
    pub best_for: &'static [&'static str],
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectorPlan {
    pub bbox_name: Option<String>,
    pub bbox: Option<[f64; 4]>,
    pub tools: Vec<String>,
    pub sensitivity: f32,
}

pub fn bounding_boxes() -> HashMap<&'static str, BBoxPreset> {
    HashMap::from([
        (
            "fox_islands",
            BBoxPreset {
                name: "Fox Islands",
                lat_min: 45.80,
                lat_max: 46.00,
                lon_min: -84.60,
                lon_max: -84.40,
                targets: &["Gilcher"],
            },
        ),
        (
            "beaver_islands",
            BBoxPreset {
                name: "Beaver Islands",
                lat_min: 45.60,
                lat_max: 45.80,
                lon_min: -85.60,
                lon_max: -85.40,
                targets: &["Parnell"],
            },
        ),
        (
            "lake_michigan_south",
            BBoxPreset {
                name: "Lake Michigan South (Andaste)",
                lat_min: 42.30,
                lat_max: 43.20,
                lon_min: -88.50,
                lon_max: -87.40,
                targets: &["Andaste", "Chicorah"],
            },
        ),
    ])
}

pub fn available_tools() -> HashMap<&'static str, ToolMeta> {
    HashMap::from([
        (
            "thermal",
            ToolMeta {
                script: "lake_michigan_scan.py",
                description: "Thermal cold-sink detection",
                default_threshold: Some(2.5),
                best_for: &["steel masses", "large vessels"],
            },
        ),
        (
            "optical",
            ToolMeta {
                script: "lake_michigan_scan.py",
                description: "Optical glint detection",
                default_threshold: Some(2.0),
                best_for: &["aluminum", "aircraft"],
            },
        ),
        (
            "sar",
            ToolMeta {
                script: "lake_michigan_scan.py",
                description: "SAR VV/VH ratio",
                default_threshold: Some(2.0),
                best_for: &["heavy steel", "dense masses"],
            },
        ),
        (
            "triple_lock",
            ToolMeta {
                script: "triple_lock_fusion.py",
                description: "Multi-sensor fusion",
                default_threshold: Some(2.5),
                best_for: &["high confidence verification"],
            },
        ),
    ])
}

pub fn parse_request_fallback(request: &str) -> DirectorPlan {
    let req = request.to_lowercase();
    let bboxes = bounding_boxes();

    let bbox_name = bboxes.iter().find_map(|(k, v)| {
        if req.contains(&k.replace('_', " ").to_lowercase())
            || v.targets.iter().any(|t| req.contains(&t.to_lowercase()))
        {
            Some((*k).to_string())
        } else {
            None
        }
    });

    let mut tools = Vec::new();
    for (keys, tool) in [
        (&["thermal", "cold", "heat", "sink"][..], "thermal"),
        (&["optical", "glint", "aluminum", "aircraft"][..], "optical"),
        (&["sar", "vv", "vh", "radar"][..], "sar"),
        (&["fusion", "triple", "lock", "verify"][..], "triple_lock"),
    ] {
        if keys.iter().any(|k| req.contains(k)) {
            tools.push(tool.to_string());
        }
    }
    if tools.is_empty() {
        tools = vec!["thermal".into(), "optical".into()];
    }

    let sensitivity = if req.contains("conservative") || req.contains("strict") {
        3.0
    } else if req.contains("aggressive") || req.contains("sensitive") {
        1.0
    } else {
        2.0
    };

    DirectorPlan {
        bbox_name,
        bbox: None,
        tools,
        sensitivity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_bbox_and_tools() {
        let p = parse_request_fallback("Find triple lock near Andaste, aggressive");
        assert_eq!(p.bbox_name.as_deref(), Some("lake_michigan_south"));
        assert!(p.tools.contains(&"triple_lock".to_string()));
        assert!((p.sensitivity - 1.0).abs() < f32::EPSILON);
    }
}
