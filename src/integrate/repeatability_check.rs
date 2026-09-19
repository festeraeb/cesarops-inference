//! Repeatability analysis — port of `tools/repeatability_check.py` (KML + DB compare logic).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KmlDetection {
    pub wgs84_lon: f64,
    pub wgs84_lat: f64,
    pub utm_easting: Option<f64>,
    pub utm_northing: Option<f64>,
    pub score: Option<f64>,
    pub classification: Option<String>,
    pub aluminum_ratio: Option<f64>,
    pub thermal_delta: Option<f64>,
    pub pixel_row: Option<i32>,
    pub pixel_col: Option<i32>,
    pub grid_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepeatabilitySummary {
    pub run_counts: Vec<(u32, String, usize)>,
    pub identical_positions: usize,
}

pub fn parse_kml_detections(content: &str) -> Vec<KmlDetection> {
    let mut out = Vec::new();
    for block in content.split("<Placemark>").skip(1) {
        let pm = block.split("</Placemark>").next().unwrap_or(block);
        let mut det = KmlDetection {
            wgs84_lon: 0.0,
            wgs84_lat: 0.0,
            utm_easting: None,
            utm_northing: None,
            score: None,
            classification: None,
            aluminum_ratio: None,
            thermal_delta: None,
            pixel_row: None,
            pixel_col: None,
            grid_ref: None,
        };
        if let Some(coords) = extract_tag(pm, "coordinates") {
            let parts: Vec<&str> = coords.split(',').collect();
            if parts.len() >= 2 {
                det.wgs84_lon = parts[0].trim().parse().unwrap_or(0.0);
                det.wgs84_lat = parts[1].trim().parse().unwrap_or(0.0);
            }
        }
        det.utm_easting = regex_f64(pm, r"E: ([\d.]+)m");
        det.utm_northing = regex_f64(pm, r"N: ([\d.]+)m");
        det.score = regex_f64(pm, r"<b>Score:</b></td><td>([\d.]+)");
        det.aluminum_ratio = regex_f64(pm, r"<b>B08/B04 Ratio:</b></td><td>([\d.]+)");
        det.thermal_delta = regex_f64(pm, r"<b>Thermal Delta:</b></td><td>([\d.]+)");
        det.classification = regex_str(pm, r"<b>Classification:</b></td><td>([^<]+)");
        det.grid_ref = regex_str(pm, r"<b>Grid Ref:</b></td><td>([^<]+)");
        if let (Some(row), Some(col)) = (regex_i32(pm, r"Row: (\d+)"), regex_i32(pm, r"Col: (\d+)")) {
            det.pixel_row = Some(row);
            det.pixel_col = Some(col);
        }
        out.push(det);
    }
    out
}

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)? + start;
    Some(text[start..end].trim().to_string())
}

fn regex_f64(text: &str, pattern: &str) -> Option<f64> {
    regex_str(text, pattern).and_then(|s| s.parse().ok())
}

fn regex_i32(text: &str, pattern: &str) -> Option<i32> {
    regex_str(text, pattern).and_then(|s| s.parse().ok())
}

fn regex_lite(pattern: &str) -> Option<()> {
    match pattern {
        r"E: ([\d.]+)m" | r"N: ([\d.]+)m" | r"<b>Score:</b></td><td>([\d.]+)"
        | r"<b>B08/B04 Ratio:</b></td><td>([\d.]+)" | r"<b>Thermal Delta:</b></td><td>([\d.]+)"
        | r"<b>Classification:</b></td><td>([^<]+)" | r"<b>Grid Ref:</b></td><td>([^<]+)"
        | r"Row: (\d+)" | r"Col: (\d+)" => Some(()),
        _ => None,
    }
}

fn regex_str(text: &str, pattern: &str) -> Option<String> {
    match pattern {
        r"E: ([\d.]+)m" => find_after(text, "E: ", "m"),
        r"N: ([\d.]+)m" => find_after(text, "N: ", "m"),
        r"<b>Score:</b></td><td>([\d.]+)" => find_between(text, "<b>Score:</b></td><td>", "<"),
        r"<b>B08/B04 Ratio:</b></td><td>([\d.]+)" => {
            find_between(text, "<b>B08/B04 Ratio:</b></td><td>", "<")
        }
        r"<b>Thermal Delta:</b></td><td>([\d.]+)" => {
            find_between(text, "<b>Thermal Delta:</b></td><td>", "<")
        }
        r"<b>Classification:</b></td><td>([^<]+)" => {
            find_between(text, "<b>Classification:</b></td><td>", "<")
        }
        r"<b>Grid Ref:</b></td><td>([^<]+)" => find_between(text, "<b>Grid Ref:</b></td><td>", "<"),
        r"Row: (\d+)" => find_after(text, "Row: ", ",").or_else(|| find_after(text, "Row: ", " ")),
        r"Col: (\d+)" => find_after(text, "Col: ", ")").or_else(|| find_after(text, "Col: ", " ")),
        _ => None,
    }
}

fn find_after(text: &str, start: &str, end: &str) -> Option<String> {
    let i = text.find(start)? + start.len();
    let rest = &text[i..];
    let j = rest.find(end).unwrap_or(rest.len());
    Some(rest[..j].trim().to_string())
}

fn find_between(text: &str, start: &str, end: &str) -> Option<String> {
    let i = text.find(start)? + start.len();
    let rest = &text[i..];
    let j = rest.find(end).unwrap_or(rest.len());
    Some(rest[..j].trim().to_string())
}

pub fn count_identical_detections(a: &[KmlDetection], b: &[KmlDetection]) -> usize {
    a.iter()
        .filter(|d1| {
            b.iter().any(|d2| {
                d1.utm_easting == d2.utm_easting
                    && d1.utm_northing == d2.utm_northing
                    && d1.score == d2.score
            })
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_kml() {
        let kml = r#"<Placemark><coordinates>-87.1,42.9,0</coordinates><b>Score:</b></td><td>0.85</Placemark>"#;
        let d = parse_kml_detections(kml);
        assert_eq!(d.len(), 1);
        assert!((d[0].wgs84_lon + 87.1).abs() < 0.001);
        assert_eq!(d[0].score, Some(0.85));
    }
}
