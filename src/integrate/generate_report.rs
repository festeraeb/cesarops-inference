//! Mission report builder — port of `generate_report.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrossrefRow {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    pub confidence: String,
    pub types: Vec<String>,
    pub known_wreck_name: Option<String>,
    pub line5_candidate: Option<bool>,
}

pub fn geo_context(lat: f64, lon: f64) -> &'static str {
    if lat < 45.77 && lon < -84.65 {
        "Open Straits south, west approach — pre-bridge zone"
    } else if lat < 45.77 {
        "Open Straits south — Michigan shoreline approach"
    } else if (45.77..=45.87).contains(&lat) && (-84.77..=-84.68).contains(&lon) {
        "Mackinac Bridge corridor — primary wreck zone"
    } else if lat > 45.95 && lon > -84.45 {
        "Eastern Straits outlet — Bois Blanc / Lake Huron approach"
    } else if lat > 45.95 {
        "Northern Straits / Lake Michigan-Huron transition"
    } else if (45.87..=45.97).contains(&lat) && lon < -84.55 {
        "Mid-Straits north channel — St. Ignace side"
    } else {
        "Mid-Straits"
    }
}

pub fn split_confidence(rows: &[CrossrefRow]) -> (Vec<&CrossrefRow>, Vec<&CrossrefRow>) {
    let hi: Vec<_> = rows.iter().filter(|r| r.confidence.eq_ignore_ascii_case("HIGH")).collect();
    let med: Vec<_> = rows.iter().filter(|r| r.confidence.eq_ignore_ascii_case("MEDIUM")).collect();
    (hi, med)
}

pub fn format_high_confidence_lines(rows: &[CrossrefRow]) -> Vec<String> {
    let (hi, _) = split_confidence(rows);
    let mut out = vec![
        "HIGH CONFIDENCE ANOMALIES (appear in BOTH 2015 and 2024 independently)".into(),
        "-".repeat(70),
    ];
    for (i, r) in hi.iter().enumerate() {
        let sensors = r.types.join("+");
        let geo = geo_context(r.lat, r.lon);
        out.push(format!(
            "  [{:02}] lat={:.5}  lon={:.5}  z={:.2}",
            i + 1,
            r.lat,
            r.lon,
            r.zscore
        ));
        out.push(format!("        sensors: {sensors}"));
        out.push(format!("        context: {geo}"));
        out.push(String::new());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_corridor() {
        assert!(geo_context(45.80, -84.72).contains("Bridge"));
    }
}
