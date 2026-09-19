//! SWOT pass date extraction — port of `satellite/swot/find_swot_dates.py` (pure logic).

use serde::{Deserialize, Serialize};

pub const SWOT_PRODUCT: &str = "SWOT_L2_LR_SSH_2.0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakeBbox {
    pub lon_min: f64,
    pub lat_min: f64,
    pub lon_max: f64,
    pub lat_max: f64,
}

pub fn lake_michigan_bbox() -> LakeBbox {
    LakeBbox {
        lon_min: -87.9,
        lat_min: 41.5,
        lon_max: -85.5,
        lat_max: 46.0,
    }
}

pub fn bbox_query_string(b: &LakeBbox) -> String {
    format!("{},{},{},{}", b.lon_min, b.lat_min, b.lon_max, b.lat_max)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DateRange {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwotDateCatalog {
    pub product: String,
    pub bbox: LakeBbox,
    pub total_dates: usize,
    pub dates: Vec<String>,
    pub date_ranges: Vec<DateRange>,
}

pub fn extract_dates_from_cmr_entries(entries: &[serde_json::Value]) -> Vec<String> {
    let mut dates = std::collections::BTreeSet::new();
    for entry in entries {
        if let Some(ts) = entry.get("time_start").and_then(|v| v.as_str()) {
            if ts.len() >= 10 {
                dates.insert(ts[..10].to_string());
            }
        }
    }
    dates.into_iter().collect()
}

pub fn group_consecutive_dates(dates: &[String], max_gap_days: i64) -> Vec<DateRange> {
    if dates.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = dates[0].clone();
    let mut end = dates[0].clone();
    for pair in dates.windows(2) {
        let prev = chrono_like_days(&pair[0]);
        let curr = chrono_like_days(&pair[1]);
        if curr - prev <= max_gap_days {
            end = pair[1].clone();
        } else {
            ranges.push(DateRange {
                start: start.clone(),
                end: end.clone(),
            });
            start = pair[1].clone();
            end = pair[1].clone();
        }
    }
    ranges.push(DateRange { start, end });
    ranges
}

fn chrono_like_days(ymd: &str) -> i64 {
    let parts: Vec<i64> = ymd.split('-').filter_map(|p| p.parse().ok()).collect();
    if parts.len() != 3 {
        return 0;
    }
    parts[0] * 372 + parts[1] * 31 + parts[2]
}

pub fn build_swot_catalog(entries: &[serde_json::Value]) -> SwotDateCatalog {
    let dates = extract_dates_from_cmr_entries(entries);
    let date_ranges = group_consecutive_dates(&dates, 3);
    SwotDateCatalog {
        product: SWOT_PRODUCT.into(),
        bbox: lake_michigan_bbox(),
        total_dates: dates.len(),
        dates,
        date_ranges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_unique_sorted_dates() {
        let entries = vec![
            json!({"time_start": "2024-01-01T12:00:00Z"}),
            json!({"time_start": "2024-01-02T12:00:00Z"}),
            json!({"time_start": "2024-01-01T08:00:00Z"}),
        ];
        let cat = build_swot_catalog(&entries);
        assert_eq!(cat.total_dates, 2);
        assert_eq!(cat.dates, vec!["2024-01-01", "2024-01-02"]);
    }

    #[test]
    fn groups_date_ranges() {
        let dates = vec![
            "2024-01-01".into(),
            "2024-01-02".into(),
            "2024-01-10".into(),
        ];
        let ranges = group_consecutive_dates(&dates, 3);
        assert_eq!(ranges.len(), 2);
    }
}
