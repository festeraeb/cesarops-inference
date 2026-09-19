//! Tile geometry sidecar builder — port of `tile_geometry.py`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const BLUE_1E_DEPTH_NADIR_M: f64 = 25.0;
pub const THRESH_OPTICAL_CLOUD: i32 = 20;
pub const THRESH_BATHY_CLOUD: i32 = 10;
pub const THRESH_OPTICAL_SUN_EL: f64 = 20.0;
pub const THRESH_BATHY_SUN_EL: f64 = 35.0;
pub const STRAITS_CENTER: (f64, f64) = (45.88, -84.55);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    Optical,
    Sar,
    Thermal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolarAngles {
    pub sun_elevation_deg: f64,
    pub sun_azimuth_deg: f64,
    pub solar_zenith_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileGeometrySidecar {
    pub granule_id: String,
    pub tif_file: String,
    pub platform: String,
    pub sensor_type: String,
    pub datetime_utc: Option<String>,
    pub cloud_cover_pct: i32,
    pub sun_elevation_deg: f64,
    pub sun_azimuth_deg: f64,
    pub solar_zenith_deg: f64,
    pub cos_solar_zenith: f64,
    pub depth_correction: f64,
    pub blue_1e_depth_m: f64,
    pub view_zenith_deg: f64,
    pub usable_optical: bool,
    pub usable_bathy: bool,
    pub usable_thermal: bool,
    pub usable_sar: bool,
    pub source: String,
}

pub fn platform_view_zenith(platform: &str) -> f64 {
    match platform {
        "sentinel-2a" | "sentinel-2b" => 10.3,
        "landsat-8" | "landsat-9" => 7.5,
        _ => 7.5,
    }
}

pub fn compute_solar_angles(year: i32, month: u32, day: u32, hour: u32, minute: u32, lat: f64, lon: f64) -> SolarAngles {
    let doy = day_of_year(year, month, day);
    let b = (360.0 / 365.0) * (doy as f64 - 1.0);
    let b_rad = b.to_radians();
    let eot_min = (0.000075 + 0.001868 * b_rad.cos() - 0.032077 * b_rad.sin()
        - 0.014615 * (2.0 * b_rad).cos() - 0.04089 * (2.0 * b_rad).sin())
        * 229.18;
    let decl_deg = (0.006918 - 0.399912 * b_rad.cos() + 0.070257 * b_rad.sin()
        - 0.006758 * (2.0 * b_rad).cos() + 0.000907 * (2.0 * b_rad).sin()
        - 0.002697 * (3.0 * b_rad).cos() + 0.00148 * (3.0 * b_rad).sin())
        .to_degrees();
    let decl = decl_deg.to_radians();
    let lat_r = lat.to_radians();
    let utc_minutes = hour as f64 * 60.0 + minute as f64;
    let tst = utc_minutes + eot_min + (lon * 4.0);
    let hour_angle = ((tst / 4.0) - 180.0).to_radians();
    let sin_el = (lat_r.sin() * decl.sin() + lat_r.cos() * decl.cos() * hour_angle.cos())
        .clamp(-1.0, 1.0);
    let elevation = sin_el.asin().to_degrees();
    let cos_az = ((decl.sin() - lat_r.sin() * sin_el) / (lat_r.cos() * sin_el.acos().cos() + 1e-10))
        .clamp(-1.0, 1.0);
    let mut azimuth = cos_az.acos().to_degrees();
    if hour_angle > 0.0 {
        azimuth = 360.0 - azimuth;
    }
    let zenith = 90.0 - elevation;
    SolarAngles {
        sun_elevation_deg: (elevation * 100.0).round() / 100.0,
        sun_azimuth_deg: (azimuth * 100.0).round() / 100.0,
        solar_zenith_deg: (zenith * 100.0).round() / 100.0,
    }
}

fn day_of_year(year: i32, month: u32, day: u32) -> i32 {
    let days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut doy = day as i32;
    for m in 1..month {
        doy += days[m as usize];
        if m == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
            doy += 1;
        }
    }
    doy
}

pub fn platform_from_filename(fname: &str) -> (String, SensorType) {
    let f = fname.to_uppercase();
    let mut sensor = SensorType::Optical;
    let platform = if f.starts_with("S2A") {
        "sentinel-2a".into()
    } else if f.starts_with("S2B") {
        "sentinel-2b".into()
    } else if f.starts_with("LC08") {
        "landsat-8".into()
    } else if f.starts_with("LC09") {
        "landsat-9".into()
    } else if f.starts_with("S1A") || f.starts_with("S1B") || f.starts_with("S1C") {
        sensor = SensorType::Sar;
        "sentinel-1".into()
    } else if f.contains("LWIR") || f.contains("THERMAL") || f.contains("B10") || f.contains("ST_B10") {
        sensor = SensorType::Thermal;
        "landsat-9".into()
    } else {
        "unknown".into()
    };
    if f.contains("LWIR") || f.contains("THERMAL") || f.contains(".B10.") || f.contains(".B11.") {
        sensor = SensorType::Thermal;
    }
    (platform, sensor)
}

pub fn parse_date_from_filename(fname: &str) -> Option<(i32, u32, u32)> {
    for part in fname.split(|c: char| !c.is_ascii_digit()) {
        if part.len() == 8 {
            if let (Ok(y), Ok(m), Ok(d)) = (
                part[0..4].parse::<i32>(),
                part[4..6].parse::<u32>(),
                part[6..8].parse::<u32>(),
            ) {
                if (1..=12).contains(&m) && (1..=31).contains(&d) {
                    return Some((y, m, d));
                }
            }
        }
    }
    None
}

pub fn cloud_from_stac_props(props: &BTreeMap<String, String>) -> (i32, &'static str) {
    for (k, v) in props {
        if k.to_lowercase().contains("cloud") {
            if let Ok(n) = v.parse::<f64>() {
                return (n.round() as i32, "stac+computed");
            }
        }
    }
    (-1, "computed")
}

pub fn build_geometry_sidecar(
    tif_file: &str,
    stac_props: Option<&BTreeMap<String, String>>,
    acq: Option<(i32, u32, u32, u32, u32)>,
) -> TileGeometrySidecar {
    let (platform, sensor) = platform_from_filename(tif_file);
    let (lat, lon) = STRAITS_CENTER;
    let acq = acq.or_else(|| parse_date_from_filename(tif_file).map(|(y, m, d)| (y, m, d, 16, 30)));
    let (cloud_pct, mut source) = stac_props
        .map(cloud_from_stac_props)
        .unwrap_or((-1, "computed"));
    let cloud_pct = if sensor == SensorType::Sar { 0 } else { cloud_pct };
    let angles = acq
        .map(|(y, m, d, h, min)| compute_solar_angles(y, m, d, h, min, lat, lon))
        .unwrap_or(SolarAngles {
            sun_elevation_deg: 30.0,
            sun_azimuth_deg: 150.0,
            solar_zenith_deg: 60.0,
        });
    let cos_zen = angles.solar_zenith_deg.to_radians().cos();
    let depth_correction = if cos_zen > 0.01 { 1.0 / cos_zen } else { 99.0 };
    let sensor_str = match sensor {
        SensorType::Optical => "optical",
        SensorType::Sar => "sar",
        SensorType::Thermal => "thermal",
    };
    let (usable_optical, usable_bathy, usable_thermal, usable_sar) = match sensor {
        SensorType::Sar => (false, false, false, true),
        SensorType::Thermal => (
            false,
            false,
            cloud_pct >= 0
                && cloud_pct < THRESH_OPTICAL_CLOUD
                && angles.sun_elevation_deg > THRESH_OPTICAL_SUN_EL,
            false,
        ),
        SensorType::Optical => (
            cloud_pct >= 0 && cloud_pct < THRESH_OPTICAL_CLOUD && angles.sun_elevation_deg > THRESH_OPTICAL_SUN_EL,
            cloud_pct >= 0 && cloud_pct < THRESH_BATHY_CLOUD && angles.sun_elevation_deg > THRESH_BATHY_SUN_EL,
            false,
            false,
        ),
    };
    if stac_props.is_some() && source == "computed" {
        source = "stac+computed";
    }
    TileGeometrySidecar {
        granule_id: tif_file.rsplit_once('.').map(|(s, _)| s.into()).unwrap_or_else(|| tif_file.into()),
        tif_file: tif_file.into(),
        platform: platform.clone(),
        sensor_type: sensor_str.into(),
        datetime_utc: acq.map(|(y, m, d, h, min)| format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:00Z")),
        cloud_cover_pct: cloud_pct,
        sun_elevation_deg: angles.sun_elevation_deg,
        sun_azimuth_deg: angles.sun_azimuth_deg,
        solar_zenith_deg: angles.solar_zenith_deg,
        cos_solar_zenith: (cos_zen * 10000.0).round() / 10000.0,
        depth_correction: (depth_correction * 1000.0).round() / 1000.0,
        blue_1e_depth_m: (BLUE_1E_DEPTH_NADIR_M * cos_zen * 10.0).round() / 10.0,
        view_zenith_deg: platform_view_zenith(&platform),
        usable_optical,
        usable_bathy,
        usable_thermal,
        usable_sar,
        source: source.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2b_is_optical() {
        let (p, s) = platform_from_filename("S2B_16TFR_20240903_0_L2A.blue.tif");
        assert_eq!(p, "sentinel-2b");
        assert_eq!(s, SensorType::Optical);
    }

    #[test]
    fn low_cloud_usable_optical() {
        let mut props = BTreeMap::new();
        props.insert("eo:cloud_cover".into(), "6".into());
        let geo = build_geometry_sidecar("S2B_16TFR_20240903.blue.tif", Some(&props), Some((2024, 9, 3, 16, 40)));
        assert!(geo.usable_optical);
    }
}
