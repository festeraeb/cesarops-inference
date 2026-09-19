//! Fix `known_wrecks.json`, merge Straits dive-community wrecks, crossref anomalies — port of `fix_and_add_wrecks.py`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const MATCH_KM: f64 = 0.457;
pub const NEARBY_KM: f64 = 1.5;
pub const STRAITS_BBOX: (f64, f64, f64, f64) = (45.70, -84.80, 46.05, -84.10);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WreckEntry {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
    pub depth_ft: i64,
    pub depth_range: String,
    pub year_lost: Option<i64>,
    pub wreck_type: String,
    pub length_ft: Option<i64>,
    pub confidence: String,
    pub gps_status: String,
    pub location_desc: String,
    pub lake: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnomalyHit {
    pub lat: f64,
    pub lon: f64,
    pub zscore: f64,
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NearbyMatch {
    pub label: String,
    pub anomaly: AnomalyHit,
    pub wreck_key: String,
    pub wreck_name: String,
    pub distance_km: f64,
    pub flag: String,
}

pub fn dms_to_decimal(deg: i32, dec_min: f64) -> f64 {
    deg as f64 + dec_min / 60.0
}

pub fn bbox_from_center(lat: f64, lon: f64, radius: f64) -> (f64, f64, f64, f64) {
    (lat - radius, lat + radius, lon - radius, lon + radius)
}

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    r * 2.0 * a.sqrt().asin()
}

pub fn straits_wreck_catalog() -> BTreeMap<String, WreckEntry> {
    let rows: &[(&str, &str, f64, f64, i64, i64, Option<i64>, &str, Option<i64>, &str, &str, &str)] = &[
        ("cayuga", "Cayuga", dms_to_decimal(45, 43.239), -dms_to_decimal(85, 11.401), 75, 102, None, "freighter", None, "high", "published", "Straits of Mackinac, western"),
        ("cedarville", "Cedarville", dms_to_decimal(45, 47.235), -dms_to_decimal(84, 40.248), 40, 110, Some(1965), "steel_freighter", None, "high", "published", "SE of Mackinac Bridge"),
        ("eber_ward", "Eber Ward", dms_to_decimal(45, 48.763), -dms_to_decimal(84, 49.133), 111, 145, Some(1909), "wooden_bulk_freighter", None, "high", "published_tried", "West of bridge mid-channel"),
        ("fred_mcbrier", "Fred McBrier", dms_to_decimal(45, 48.342), -dms_to_decimal(85, 55.301), 96, 104, None, "freighter", None, "high", "published", "Western Straits approach"),
        ("maitland", "Maitland", dms_to_decimal(45, 48.249), -dms_to_decimal(85, 52.555), 85, 85, None, "wooden_bark", Some(137), "medium", "published_untried", "West of bridge mid-channel"),
        ("minneapolis", "Minneapolis", dms_to_decimal(45, 48.511), -dms_to_decimal(84, 43.904), 124, 124, None, "freighter", None, "high", "published", "East of Mackinac Bridge"),
        ("newell_eddy", "Newell Eddy", dms_to_decimal(45, 46.890), -dms_to_decimal(84, 13.810), 165, 165, None, "freighter", None, "high", "published", "Eastern Straits near Bois Blanc Island"),
        ("northwest", "Northwest", dms_to_decimal(45, 47.450), -dms_to_decimal(84, 51.465), 75, 75, None, "vessel", None, "high", "published", "West of bridge mid-channel"),
        ("rock_maze", "Rock Maze", dms_to_decimal(45, 51.803), -dms_to_decimal(84, 36.410), 0, 35, None, "reef_site", None, "high", "published", "North Straits, shallow dive site"),
        ("sandusky", "Sandusky", dms_to_decimal(45, 47.959), -dms_to_decimal(84, 50.249), 70, 85, None, "schooner", None, "medium", "published_tried_not_close", "West of bridge mid-channel"),
        ("m_stalker", "M. Stalker", dms_to_decimal(45, 47.620), -dms_to_decimal(84, 41.062), 85, 85, None, "vessel", None, "high", "published", "SE of Mackinac Bridge"),
        ("st_andrew", "St. Andrew", dms_to_decimal(45, 42.051), -dms_to_decimal(84, 31.795), 62, 62, None, "vessel", None, "high", "published", "SE Straits"),
        ("uganda", "Uganda", dms_to_decimal(45, 50.553), -dms_to_decimal(85, 2.998), 185, 207, None, "vessel", None, "high", "published", "West of bridge, deep channel"),
        ("william_barnum", "William H. Barnum", dms_to_decimal(45, 44.708), -dms_to_decimal(84, 37.866), 58, 75, None, "wooden_freighter", Some(218), "medium", "published_untried", "SE of bridge off Mackinaw City"),
        ("william_young", "William Young", dms_to_decimal(45, 48.777), -dms_to_decimal(84, 41.923), 120, 120, None, "schooner", None, "high", "published", "East of bridge off upper shore"),
    ];
    let mut out = BTreeMap::new();
    for (key, name, lat, lon, d_min, d_max, year, wtype, length, conf, gps_status, desc) in rows {
        let (lat_min, lat_max, lon_min, lon_max) = bbox_from_center(*lat, *lon, 0.015);
        let depth_range = if d_min != d_max {
            format!("{d_min}-{d_max}")
        } else {
            d_min.to_string()
        };
        out.insert(
            (*key).into(),
            WreckEntry {
                name: (*name).into(),
                lat: (*lat * 1_000_000.0).round() / 1_000_000.0,
                lon: (*lon * 1_000_000.0).round() / 1_000_000.0,
                lat_min: (lat_min * 1_000_000.0).round() / 1_000_000.0,
                lat_max: (lat_max * 1_000_000.0).round() / 1_000_000.0,
                lon_min: (lon_min * 1_000_000.0).round() / 1_000_000.0,
                lon_max: (lon_max * 1_000_000.0).round() / 1_000_000.0,
                depth_ft: (d_min + d_max) / 2,
                depth_range,
                year_lost: *year,
                wreck_type: (*wtype).into(),
                length_ft: *length,
                confidence: (*conf).into(),
                gps_status: (*gps_status).into(),
                location_desc: (*desc).into(),
                lake: "michigan_huron_straits".into(),
                source: "dive_community".into(),
            },
        );
    }
    out
}

pub fn merge_trailing_fragment(content: &str) -> Result<(Map<String, Value>, usize), serde_json::Error> {
    let trimmed = content.trim();
    let end = find_json_end(trimmed).ok_or_else(|| {
        serde_json::from_str::<Value>("not json").err().unwrap()
    })?;
    let mut root: Map<String, Value> = serde_json::from_str(&trimmed[..end])?;
    let remainder = trimmed[end..].trim();
    let mut merged = 0usize;
    if !remainder.is_empty() {
        let wrapped = format!("{{{remainder}}}");
        if let Ok(Value::Object(extra)) = serde_json::from_str(&wrapped) {
            if let Some(Value::Object(wrecks)) = root.get_mut("wrecks") {
                for (k, v) in extra {
                    if !wrecks.contains_key(&k) {
                        wrecks.insert(k, v);
                        merged += 1;
                    }
                }
            }
        }
    }
    Ok((root, merged))
}

fn find_json_end(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

pub fn apply_straits_catalog(wrecks: &mut BTreeMap<String, WreckEntry>) -> (usize, usize) {
    let catalog = straits_wreck_catalog();
    let mut added = 0usize;
    let mut updated = 0usize;
    for (key, entry) in catalog {
        if wrecks.contains_key(&key) {
            wrecks.insert(key, entry);
            updated += 1;
        } else {
            wrecks.insert(key, entry);
            added += 1;
        }
    }
    (added, updated)
}

pub fn wrecks_in_bbox(wrecks: &BTreeMap<String, WreckEntry>, bbox: (f64, f64, f64, f64)) -> Vec<(String, WreckEntry)> {
    let (lat_min, lon_min, lat_max, lon_max) = bbox;
    wrecks
        .iter()
        .filter(|(_, w)| w.lat >= lat_min && w.lat <= lat_max && w.lon >= lon_min && w.lon <= lon_max)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

pub fn crossref_nearby_matches(hits: &[AnomalyHit], wrecks: &BTreeMap<String, WreckEntry>) -> Vec<NearbyMatch> {
    let mut out = Vec::new();
    for (i, hit) in hits.iter().enumerate() {
        let label = if hit.confidence.eq_ignore_ascii_case("HIGH") {
            format!("H{}", i + 1)
        } else {
            format!("M{}", i.saturating_sub(12) + 1)
        };
        let mut best: Option<(&String, &WreckEntry, f64)> = None;
        for (key, wreck) in wrecks {
            let dist_km = haversine_m(hit.lat, hit.lon, wreck.lat, wreck.lon) / 1000.0;
            if best.as_ref().map(|(_, _, d)| dist_km < *d).unwrap_or(true) {
                best = Some((key, wreck, dist_km));
            }
        }
        if let Some((key, wreck, dist_km)) = best {
            if dist_km <= NEARBY_KM {
                let flag = if dist_km <= MATCH_KM {
                    "<<< MATCH".into()
                } else {
                    "< nearby".into()
                };
                out.push(NearbyMatch {
                    label,
                    anomaly: hit.clone(),
                    wreck_key: key.clone(),
                    wreck_name: wreck.name.clone(),
                    distance_km: dist_km,
                    flag,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dms_roundtrip() {
        assert!((dms_to_decimal(45, 43.239) - 45.72065).abs() < 0.001);
    }

    #[test]
    fn catalog_has_cedarville() {
        let cat = straits_wreck_catalog();
        assert!(cat.contains_key("cedarville"));
        assert_eq!(cat["cedarville"].name, "Cedarville");
    }

    #[test]
    fn nearby_match_within_mile() {
        let mut wrecks = straits_wreck_catalog();
        let cedar = wrecks.remove("cedarville").unwrap();
        wrecks.insert("cedarville".into(), cedar);
        let hits = vec![AnomalyHit {
            lat: 45.787,
            lon: -84.671,
            zscore: 4.2,
            confidence: "HIGH".into(),
        }];
        let matches = crossref_nearby_matches(&hits, &wrecks);
        assert!(!matches.is_empty());
    }
}
