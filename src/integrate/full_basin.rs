//! Full-basin Lake Michigan scan config — port of `full_basin_scan.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LakeBounds {
    pub north: f64,
    pub south: f64,
    pub east: f64,
    pub west: f64,
}

pub fn lake_michigan_bounds() -> LakeBounds {
    LakeBounds {
        north: 45.5,
        south: 41.5,
        east: -86.0,
        west: -87.9,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownWreck {
    pub name: &'static str,
    pub lat: f64,
    pub lon: f64,
    pub length_ft: f64,
    pub wreck_type: &'static str,
    pub year: u16,
}

pub fn known_wrecks() -> &'static [KnownWreck] {
    &[
        KnownWreck {
            name: "SS Andaste",
            lat: 42.4125,
            lon: -87.25,
            length_ft: 266.9,
            wreck_type: "Whaleback",
            year: 1929,
        },
        KnownWreck {
            name: "SS Gilcher",
            lat: 43.2,
            lon: -86.5,
            length_ft: 352.0,
            wreck_type: "Steel Freighter",
            year: 1907,
        },
        KnownWreck {
            name: "Pere Marquette 18",
            lat: 43.1,
            lon: -86.3,
            length_ft: 338.0,
            wreck_type: "Car Ferry",
            year: 1910,
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureFilter {
    BessemerSteelLock,
    CarFerryGrid,
    WoodenGhostSieve,
    AviationCluster,
    ConstructionMonster,
}

pub fn signature_filter_name(f: SignatureFilter) -> &'static str {
    match f {
        SignatureFilter::BessemerSteelLock => "Bessemer Steel Lock",
        SignatureFilter::CarFerryGrid => "Car-Ferry Grid",
        SignatureFilter::WoodenGhostSieve => "Wooden Ghost Sieve",
        SignatureFilter::AviationCluster => "Aviation Cluster",
        SignatureFilter::ConstructionMonster => "Construction Monster",
    }
}
