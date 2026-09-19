//! Cross-reference analyzer port of `analyze_crossref.py`.
//! Computes nearest known wrecks and simple confidence summaries for anomalies.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wreck {
    pub name: String,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub lat_min: Option<f64>,
    pub lat_max: Option<f64>,
    pub lon_min: Option<f64>,
    pub lon_max: Option<f64>,
    pub depth_ft: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySite {
    pub lat: f64,
    pub lon: f64,
    pub sensors: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestWreck {
    pub name: String,
    pub distance_km: f64,
    pub depth_ft: Option<f64>,
}

#[inline]
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0_f64;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

pub fn wreck_coords(w: &Wreck) -> Option<(f64, f64)> {
    if let (Some(lat), Some(lon)) = (w.lat, w.lon) {
        return Some((lat, lon));
    }
    if let (Some(lat_min), Some(lat_max), Some(lon_min), Some(lon_max)) =
        (w.lat_min, w.lat_max, w.lon_min, w.lon_max)
    {
        return Some(((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0));
    }
    None
}

pub fn nearest_wrecks(site: &AnomalySite, wrecks: &[Wreck], n: usize) -> Vec<NearestWreck> {
    let mut ranked: Vec<NearestWreck> = wrecks
        .iter()
        .filter_map(|w| {
            let (lat, lon) = wreck_coords(w)?;
            Some(NearestWreck {
                name: w.name.clone(),
                distance_km: haversine_km(site.lat, site.lon, lat, lon),
                depth_ft: w.depth_ft,
            })
        })
        .collect();
    ranked.sort_by(|a, b| a.distance_km.total_cmp(&b.distance_km));
    ranked.truncate(n);
    ranked
}

pub fn confidence_flag(distance_km: f64) -> &'static str {
    if distance_km < 1.5 {
        "WITHIN_1_5KM"
    } else if distance_km < 3.0 {
        "WITHIN_3KM"
    } else {
        "NONE"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_sorting_and_flag() {
        let wrecks = vec![
            Wreck {
                name: "Far".into(),
                lat: Some(45.0),
                lon: Some(-85.0),
                lat_min: None,
                lat_max: None,
                lon_min: None,
                lon_max: None,
                depth_ft: Some(300.0),
            },
            Wreck {
                name: "Near".into(),
                lat: Some(45.94),
                lon: Some(-84.37),
                lat_min: None,
                lat_max: None,
                lon_min: None,
                lon_max: None,
                depth_ft: Some(120.0),
            },
        ];
        let site = AnomalySite {
            lat: 45.9394,
            lon: -84.3705,
            sensors: "optical".into(),
        };
        let n = nearest_wrecks(&site, &wrecks, 1);
        assert_eq!(n[0].name, "Near");
        assert_eq!(confidence_flag(n[0].distance_km), "WITHIN_1_5KM");
    }
}
