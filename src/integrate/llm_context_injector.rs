//! LLM context injection from JSON knowledge files — port of `llm_context_injector.py`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SatelliteSource {
    pub name: String,
    pub description: String,
    pub requires_api_key: bool,
    pub env_key: String,
    pub collections: Vec<String>,
    pub best_for: Vec<String>,
    pub notes: String,
}

pub fn build_satellite_context(sources: &BTreeMap<String, SatelliteSource>, guidelines: &BTreeMap<String, Value>) -> String {
    let mut lines = vec![
        "SATELLITE DATA SOURCES AVAILABLE:".into(),
        "These are the data sources CESAROPS can pull from. When recommending data \
         acquisition, reference these source IDs and their auth requirements."
            .into(),
        String::new(),
    ];
    for (src_id, src) in sources {
        lines.push(format!("  [{src_id}] {}", src.name));
        if !src.description.is_empty() {
            lines.push(format!("    Description: {}", src.description));
        }
        if src.requires_api_key && !src.env_key.is_empty() {
            lines.push(format!("    Auth: requires {}", src.env_key));
        }
        if !src.collections.is_empty() {
            lines.push(format!("    Collections: {}", src.collections.join(", ")));
        }
        if !src.best_for.is_empty() {
            lines.push(format!("    Best for: {}", src.best_for.join(", ")));
        }
        if !src.notes.is_empty() {
            lines.push(format!("    Notes: {}", src.notes));
        }
        lines.push(String::new());
    }
    if !guidelines.is_empty() {
        lines.push("DATA ACQUISITION GUIDELINES:".into());
        for (area, info) in guidelines {
            lines.push(format!("  {area}:"));
            if let Some(bbox) = info.get("bbox") {
                lines.push(format!("    BBOX: {bbox}"));
            }
            if let Some(months) = info.get("best_months").and_then(|v| v.as_array()) {
                let m: Vec<_> = months.iter().filter_map(|x| x.as_str()).collect();
                lines.push(format!("    Best months: {}", m.join(", ")));
            }
            if let Some(sensors) = info.get("recommended_sensors").and_then(|v| v.as_array()) {
                let s: Vec<_> = sensors.iter().filter_map(|x| x.as_str()).collect();
                lines.push(format!("    Recommended: {}", s.join(", ")));
            }
            if let Some(reason) = info.get("reason").and_then(|v| v.as_str()) {
                lines.push(format!("    Reason: {reason}"));
            }
            lines.push(String::new());
        }
    }
    lines.join("\n")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnownTarget {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub confidence: String,
    pub lock_type: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanLesson {
    pub date: String,
    pub scan_type: String,
    pub what_worked: Vec<String>,
    pub what_failed: Vec<String>,
}

pub fn build_knowledge_context(
    targets: &[KnownTarget],
    lessons: &[ScanLesson],
    rules: &[String],
    techniques: &[(String, String, String)],
) -> String {
    if targets.is_empty() && lessons.is_empty() && rules.is_empty() {
        return "KNOWLEDGE BASE: Empty. Run a scan to start building knowledge.\n".into();
    }
    let mut lines = vec!["CESAROPS KNOWLEDGE BASE — What We Know So Far:\n".into()];
    if !targets.is_empty() {
        lines.push("KNOWN TARGETS:".into());
        for t in targets {
            let status = if t.lock_type == "not_scanned" {
                "○ not yet scanned"
            } else {
                "✓ scanned"
            };
            lines.push(format!(
                "  [{}] {} ({:.4}, {:.4}) confidence={} {status}",
                t.id, t.name, t.lat, t.lon, t.confidence
            ));
            if !t.notes.is_empty() {
                lines.push(format!("    → {}", t.notes));
            }
        }
        lines.push(String::new());
    }
    if !lessons.is_empty() {
        lines.push("SCAN LESSONS LEARNED:".into());
        for lesson in lessons.iter().rev().take(3).rev() {
            lines.push(format!("  [{}] {}", lesson.date, lesson.scan_type));
            for w in lesson.what_worked.iter().take(5) {
                lines.push(format!("    ✓ {w}"));
            }
            for f in lesson.what_failed.iter().take(5) {
                lines.push(format!("    ✗ {f}"));
            }
        }
        lines.push(String::new());
    }
    if !rules.is_empty() {
        lines.push("AGENT BEHAVIOR RULES:".into());
        for r in rules {
            lines.push(format!("  • {r}"));
        }
        lines.push(String::new());
    }
    let not_tested: Vec<_> = techniques
        .iter()
        .filter(|(_, _, status)| status == "not_tested")
        .collect();
    if !not_tested.is_empty() {
        lines.push("TECHNIQUES TO TRY (idle research):".into());
        for (priority, name, notes) in not_tested {
            let note = if notes.len() > 100 { &notes[..100] } else { notes.as_str() };
            lines.push(format!("  [{priority}] {name} — {note}"));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WarpConfig {
    pub working_memory_mb: u32,
    pub block_size: u32,
    pub resampling: String,
}

pub fn build_warp_context(cfg: Option<&WarpConfig>) -> String {
    match cfg {
        Some(c) => format!(
            "GDAL WARP CONFIG: working_memory_mb={}, block_size={}, resampling={}",
            c.working_memory_mb, c.block_size, c.resampling
        ),
        None => "GDAL WARP CONFIG: (not set, defaults: wm=4000, block=512, resample=lanczos)".into(),
    }
}

pub fn build_full_context(
    satellite: Option<&str>,
    learning: Option<&str>,
    warp: Option<&str>,
    include_weather: bool,
) -> String {
    let mut parts = Vec::new();
    if let Some(s) = satellite {
        parts.push(s.to_string());
    }
    if let Some(l) = learning {
        parts.push(l.to_string());
    }
    if let Some(w) = warp {
        parts.push(w.to_string());
    }
    if include_weather {
        parts.push(
            "WEATHER CHECK: Before recommending a scan, the agent should verify \
             current weather conditions (cloud cover, wind, precipitation) for the \
             target area. Use web search to check NOAA/NWS forecasts. \
             Clear skies = good for optical/thermal. Overcast = SAR only. \
             Heavy rain/wind = postpone scan."
                .into(),
        );
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satellite_context_includes_id() {
        let mut src = BTreeMap::new();
        src.insert(
            "hls".into(),
            SatelliteSource {
                name: "HLS".into(),
                description: "Harmonized Landsat Sentinel".into(),
                requires_api_key: true,
                env_key: "EARTHDATA_TOKEN".into(),
                collections: vec!["HLSL30.v2.0".into()],
                best_for: vec!["optical".into()],
                notes: String::new(),
            },
        );
        let ctx = build_satellite_context(&src, &BTreeMap::new());
        assert!(ctx.contains("[hls]"));
        assert!(ctx.contains("EARTHDATA_TOKEN"));
    }
}
