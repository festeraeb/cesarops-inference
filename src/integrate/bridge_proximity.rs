//! Mackinac bridge proximity table — port of `bridge_proximity.py`.

use serde::{Deserialize, Serialize};

pub const MACKINAC_BRIDGE: (f64, f64) = (45.80140, -84.72740);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnownWreckEntry {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub depth_ft: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeProximityRow {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub depth_ft: String,
    pub km: f64,
    pub miles: f64,
    pub within_2mi: bool,
    pub within_5mi: bool,
}

pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

pub fn wreck_centroid(raw: &serde_json::Value, key: &str) -> Option<(f64, f64)> {
    let w = raw.get(key)?;
    if let (Some(lat), Some(lon)) = (w.get("lat"), w.get("lon")) {
        return Some((lat.as_f64()?, lon.as_f64()?));
    }
    let lat = (w.get("lat_min")?.as_f64()? + w.get("lat_max")?.as_f64()?) / 2.0;
    let lon = (w.get("lon_min")?.as_f64()? + w.get("lon_max")?.as_f64()?) / 2.0;
    Some((lat, lon))
}

pub fn bridge_proximity_table(wrecks: &[KnownWreckEntry]) -> Vec<BridgeProximityRow> {
    let mut rows: Vec<_> = wrecks
        .iter()
        .map(|w| {
            let km = haversine_km(MACKINAC_BRIDGE.0, MACKINAC_BRIDGE.1, w.lat, w.lon);
            let miles = km * 0.621371;
            BridgeProximityRow {
                name: w.name.clone(),
                lat: w.lat,
                lon: w.lon,
                depth_ft: w
                    .depth_ft
                    .map(|d| format!("{d}"))
                    .unwrap_or_else(|| "?".into()),
                km,
                miles,
                within_2mi: miles < 2.0,
                within_5mi: miles < 5.0,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.km.partial_cmp(&b.km).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_by_distance() {
        let rows = bridge_proximity_table(&[
            KnownWreckEntry {
                id: "a".into(),
                name: "far".into(),
                lat: 46.0,
                lon: -84.0,
                depth_ft: Some(100.0),
            },
            KnownWreckEntry {
                id: "b".into(),
                name: "near".into(),
                lat: 45.81,
                lon: -84.73,
                depth_ft: None,
            },
        ]);
        assert_eq!(rows[0].name, "near");
    }
}
