//! Daily satellite pull planner — port of `daily_satellite_pull.py` (orchestration config).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SatelliteSource {
    NdbcBuoy,
    SwotSsh,
    Icesat2,
    LandsatThermal,
    Sentinel1Sar,
    Sentinel2Optical,
}

impl SatelliteSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::NdbcBuoy => "NDBC Buoy",
            Self::SwotSsh => "SWOT SSH",
            Self::Icesat2 => "ICESat-2 ATL13",
            Self::LandsatThermal => "Landsat-8/9 Thermal",
            Self::Sentinel1Sar => "Sentinel-1 SAR",
            Self::Sentinel2Optical => "Sentinel-2 Optical",
        }
    }

    pub fn repeat_days(self) -> Option<u32> {
        match self {
            Self::NdbcBuoy => None,
            Self::SwotSsh => Some(21),
            Self::Icesat2 => Some(91),
            Self::LandsatThermal => Some(16),
            Self::Sentinel1Sar => Some(6),
            Self::Sentinel2Optical => Some(5),
        }
    }
}

pub fn all_sources() -> [SatelliteSource; 6] {
    [
        SatelliteSource::NdbcBuoy,
        SatelliteSource::SwotSsh,
        SatelliteSource::Icesat2,
        SatelliteSource::LandsatThermal,
        SatelliteSource::Sentinel1Sar,
        SatelliteSource::Sentinel2Optical,
    ]
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullPlan {
    pub start_iso: String,
    pub end_iso: String,
    pub sources: Vec<SatelliteSource>,
    pub output_subdir: String,
}

/// Build a pull plan from ISO date strings (`YYYY-MM-DD`).
pub fn plan_date_range(start_iso: &str, end_iso: &str) -> PullPlan {
    PullPlan {
        start_iso: start_iso.to_string(),
        end_iso: end_iso.to_string(),
        sources: all_sources().into(),
        output_subdir: format!("{start_iso}_{end_iso}"),
    }
}
