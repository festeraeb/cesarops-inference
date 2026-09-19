//! Lake Michigan dual-scan planning (low-water windows + tile quality).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LakeBounds {
    pub north: f64,
    pub south: f64,
    pub west: f64,
    pub east: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DateWindow {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct YearWindows {
    pub year: u16,
    pub windows: Vec<DateWindow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityScore {
    pub score: f64,
    pub note: &'static str,
}

pub fn lake_michigan_bounds() -> LakeBounds {
    LakeBounds {
        north: 46.10,
        south: 41.60,
        west: -88.10,
        east: -84.70,
    }
}

pub fn low_water_years() -> Vec<YearWindows> {
    vec![
        YearWindows {
            year: 2012,
            windows: vec![
                DateWindow { start: "2012-05-20".into(), end: "2012-06-15".into() },
                DateWindow { start: "2012-09-01".into(), end: "2012-09-30".into() },
            ],
        },
        YearWindows {
            year: 2013,
            windows: vec![
                DateWindow { start: "2013-05-20".into(), end: "2013-06-15".into() },
                DateWindow { start: "2013-09-01".into(), end: "2013-09-30".into() },
            ],
        },
        YearWindows {
            year: 2019,
            windows: vec![
                DateWindow { start: "2019-05-20".into(), end: "2019-06-15".into() },
                DateWindow { start: "2019-09-01".into(), end: "2019-09-30".into() },
            ],
        },
        YearWindows {
            year: 2020,
            windows: vec![
                DateWindow { start: "2020-05-20".into(), end: "2020-06-15".into() },
                DateWindow { start: "2020-09-01".into(), end: "2020-09-30".into() },
            ],
        },
        YearWindows {
            year: 2021,
            windows: vec![
                DateWindow { start: "2021-05-20".into(), end: "2021-06-15".into() },
                DateWindow { start: "2021-09-01".into(), end: "2021-09-30".into() },
            ],
        },
        YearWindows {
            year: 2024,
            windows: vec![
                DateWindow { start: "2024-05-20".into(), end: "2024-06-15".into() },
                DateWindow { start: "2024-09-01".into(), end: "2024-09-30".into() },
            ],
        },
        YearWindows {
            year: 2025,
            windows: vec![
                DateWindow { start: "2025-05-20".into(), end: "2025-06-15".into() },
                DateWindow { start: "2025-09-01".into(), end: "2025-09-30".into() },
            ],
        },
    ]
}

pub fn generate_landsat_path_rows() -> Vec<String> {
    let mut out = Vec::new();
    for path in 22..=26 {
        for row in 29..=33 {
            out.push(format!("{path:03}{row:03}"));
        }
    }
    out
}

pub fn sentinel_tiles() -> Vec<&'static str> {
    vec!["16TDM", "16TDN", "16TEM", "16TEN", "17TMT", "17TMU"]
}

pub fn calculate_tile_quality(cloud_cover: f64, sensor: &str, date_iso: &str) -> QualityScore {
    if cloud_cover > 20.0 {
        return QualityScore { score: 0.0, note: "EXCESSIVE CLOUD COVER (>20%)" };
    }
    let mut score: f64 = 1.0;
    if cloud_cover > 10.0 {
        score -= 0.3;
    } else if cloud_cover > 5.0 {
        score -= 0.15;
    }
    if sensor.contains("Landsat") {
        score += 0.1;
    } else if sensor.contains("Sentinel") {
        score += 0.05;
    }
    let d = date_iso;
    if d.ends_with("-03-15") || d.contains("-03-") || d.contains("-04-") {
        score += 0.05;
    }
    if d.contains("-05-") || d.contains("-06-") {
        score += 0.05;
    }
    QualityScore { score: score.min(1.0), note: "OK" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_sizes_match_expectations() {
        assert_eq!(generate_landsat_path_rows().len(), 25);
        assert_eq!(sentinel_tiles().len(), 6);
    }

    #[test]
    fn rejects_heavy_cloud() {
        let q = calculate_tile_quality(45.0, "Landsat-9", "2024-06-10");
        assert_eq!(q.score, 0.0);
    }
}
