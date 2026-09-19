//! Tile selection by scan mode — port of `tile_selector.py`.

use crate::integrate::tile_geometry::TileGeometrySidecar;
use serde::{Deserialize, Serialize};

pub const MODES: &[&str] = &["historic_wreck", "bathy_3d", "sar_search", "sar_after_event"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModeConfig {
    pub cloud_max: i32,
    pub sun_el_min: f64,
    pub sensor: String,
    pub rank_by: String,
    pub description: String,
    pub event_window_days: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedTile {
    pub tif: String,
    pub rank: f64,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloud_cover_pct: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_elevation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_correction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datetime_utc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExcludedTile {
    pub tif: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileSelectionResult {
    pub mode: String,
    pub mode_description: String,
    pub selected_tiles: Vec<SelectedTile>,
    pub excluded_tiles: Vec<ExcludedTile>,
    pub no_sar_tiles: bool,
    pub no_optical_tiles: bool,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn mode_config(mode: &str) -> Option<ModeConfig> {
    match mode {
        "historic_wreck" => Some(ModeConfig {
            cloud_max: 20,
            sun_el_min: 20.0,
            sensor: "optical".into(),
            rank_by: "depth_correction".into(),
            description: "Wreck detection: any optical tile with low cloud".into(),
            event_window_days: None,
        }),
        "bathy_3d" => Some(ModeConfig {
            cloud_max: 10,
            sun_el_min: 35.0,
            sensor: "optical".into(),
            rank_by: "cloud_cover_pct".into(),
            description: "Stumpf depth mapping: strict optical quality required".into(),
            event_window_days: None,
        }),
        "sar_search" => Some(ModeConfig {
            cloud_max: 15,
            sun_el_min: 20.0,
            sensor: "both".into(),
            rank_by: "cloud_cover_pct".into(),
            description: "SAR always; optical added if available and cloud<15%".into(),
            event_window_days: None,
        }),
        "sar_after_event" => Some(ModeConfig {
            cloud_max: 100,
            sun_el_min: 0.0,
            sensor: "sar".into(),
            rank_by: "datetime_utc".into(),
            description: "Delta-detect: SAR tiles within event_date ± 3 days only".into(),
            event_window_days: Some(3),
        }),
        _ => None,
    }
}

fn days_from_event(tile_dt: &str, event_date: &str) -> Result<f64, String> {
    let tile = tile_dt.replace('Z', "+00:00");
    let event = format!("{event_date}T00:00:00+00:00");
    let tile_parsed = chrono_like_days(&tile)?;
    let event_parsed = chrono_like_days(&event)?;
    Ok((tile_parsed - event_parsed).abs())
}

fn chrono_like_days(iso: &str) -> Result<f64, String> {
    let date_part = iso.split('T').next().unwrap_or(iso);
    let parts: Vec<_> = date_part.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("bad date: {iso}"));
    }
    let y: i32 = parts[0].parse().map_err(|_| iso.to_string())?;
    let m: u32 = parts[1].parse().map_err(|_| iso.to_string())?;
    let d: u32 = parts[2].parse().map_err(|_| iso.to_string())?;
    Ok(y as f64 * 365.25 + m as f64 * 30.4 + d as f64)
}

pub fn select_tiles(geo_list: &[TileGeometrySidecar], mode: &str, event_date: Option<&str>) -> TileSelectionResult {
    let Some(cfg) = mode_config(mode) else {
        return TileSelectionResult {
            mode: mode.into(),
            mode_description: String::new(),
            selected_tiles: vec![],
            excluded_tiles: vec![],
            no_sar_tiles: false,
            no_optical_tiles: false,
            summary: String::new(),
            error: Some(format!("Unknown mode: {mode}. Valid: {MODES:?}")),
        };
    };

    let mut selected = Vec::new();
    let mut excluded = Vec::new();
    let window = cfg.event_window_days.unwrap_or(3) as f64;

    for geo in geo_list {
        let tif = geo.tif_file.clone();
        let cloud = geo.cloud_cover_pct;
        let sun = geo.sun_elevation_deg;
        let stype = geo.sensor_type.as_str();
        let depth = geo.depth_correction;

        if mode == "sar_after_event" {
            if stype != "sar" {
                excluded.push(ExcludedTile {
                    tif,
                    reason: "sar_after_event mode requires SAR tiles only".into(),
                });
                continue;
            }
            if event_date.is_none() {
                selected.push(SelectedTile {
                    tif,
                    rank: 0.0,
                    reason: "SAR tile (no event date filter)".into(),
                    datetime_utc: geo.datetime_utc.clone(),
                    cloud_cover_pct: None,
                    sun_elevation: None,
                    depth_correction: None,
                });
                continue;
            }
            let Some(dt) = geo.datetime_utc.as_deref() else {
                excluded.push(ExcludedTile {
                    tif,
                    reason: "no acquisition datetime — cannot verify event window".into(),
                });
                continue;
            };
            match days_from_event(dt, event_date.unwrap()) {
                Ok(delta) if delta <= window => {
                    selected.push(SelectedTile {
                        tif,
                        rank: delta,
                        reason: format!("SAR tile {delta:.1}d from event"),
                        datetime_utc: geo.datetime_utc.clone(),
                        cloud_cover_pct: None,
                        sun_elevation: None,
                        depth_correction: None,
                    });
                }
                Ok(delta) => excluded.push(ExcludedTile {
                    tif,
                    reason: format!("{delta:.1}d from event > {window}d window"),
                }),
                Err(e) => excluded.push(ExcludedTile {
                    tif,
                    reason: format!("datetime parse error: {e}"),
                }),
            }
            continue;
        }

        if mode == "sar_search" {
            if stype == "sar" {
                selected.push(SelectedTile {
                    tif,
                    rank: 0.0,
                    reason: "SAR tile always included for sar_search".into(),
                    cloud_cover_pct: None,
                    sun_elevation: None,
                    depth_correction: None,
                    datetime_utc: None,
                });
            } else if cloud < 0 {
                excluded.push(ExcludedTile {
                    tif,
                    reason: "cloud cover unknown — cannot assess optical usability".into(),
                });
            } else if cloud <= cfg.cloud_max && sun >= cfg.sun_el_min {
                selected.push(SelectedTile {
                    tif,
                    rank: cloud as f64,
                    reason: format!("optical overlay: cloud={cloud}% sun={sun:.1}°"),
                    cloud_cover_pct: Some(cloud),
                    sun_elevation: Some(sun),
                    depth_correction: Some(depth),
                    datetime_utc: None,
                });
            } else {
                let mut parts = Vec::new();
                if cloud > cfg.cloud_max {
                    parts.push(format!("cloud={cloud}% > {}% threshold", cfg.cloud_max));
                }
                if sun < cfg.sun_el_min {
                    parts.push(format!("sun={sun:.1}° < {}° threshold", cfg.sun_el_min));
                }
                excluded.push(ExcludedTile {
                    tif,
                    reason: parts.join("; "),
                });
            }
            continue;
        }

        if stype == "sar" {
            excluded.push(ExcludedTile {
                tif,
                reason: format!("mode={mode} does not use SAR tiles"),
            });
            continue;
        }
        if stype == "thermal" {
            excluded.push(ExcludedTile {
                tif,
                reason: format!("mode={mode} does not use thermal tiles"),
            });
            continue;
        }
        if cloud < 0 {
            excluded.push(ExcludedTile {
                tif,
                reason: "cloud cover unknown — skipping".into(),
            });
            continue;
        }
        if cloud > cfg.cloud_max {
            excluded.push(ExcludedTile {
                tif,
                reason: format!("cloud={cloud}% > {}% threshold", cfg.cloud_max),
            });
            continue;
        }
        if sun < cfg.sun_el_min {
            excluded.push(ExcludedTile {
                tif,
                reason: format!("sun={sun:.1}° < {}° minimum", cfg.sun_el_min),
            });
            continue;
        }
        let rank = if cfg.rank_by == "depth_correction" {
            depth
        } else {
            cloud as f64
        };
        selected.push(SelectedTile {
            tif,
            rank,
            cloud_cover_pct: Some(cloud),
            sun_elevation: Some(sun),
            depth_correction: Some(depth),
            reason: format!("cloud={cloud}% sun={sun:.1}° depth_corr={depth:.3}"),
            datetime_utc: None,
        });
    }

    selected.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap_or(std::cmp::Ordering::Equal));
    for (i, s) in selected.iter_mut().enumerate() {
        s.rank = (i + 1) as f64;
    }

    let sar_count = selected
        .iter()
        .filter(|s| s.reason.to_uppercase().contains("SAR") || s.tif.to_lowercase().contains("sar"))
        .count();
    let no_sar = matches!(mode, "sar_search" | "sar_after_event") && sar_count == 0;
    let no_optical = matches!(mode, "historic_wreck" | "bathy_3d") && selected.is_empty();

    let mut summary_parts = Vec::new();
    if !selected.is_empty() {
        summary_parts.push(format!("{} tile(s) selected", selected.len()));
    }
    if !excluded.is_empty() {
        summary_parts.push(format!("{} excluded", excluded.len()));
    }
    if no_sar {
        summary_parts.push("WARNING: no SAR tiles available".into());
    }
    if no_optical {
        summary_parts.push("WARNING: no usable optical tiles".into());
    }
    let summary = if summary_parts.is_empty() {
        "no tiles processed".into()
    } else {
        summary_parts.join("; ")
    };

    TileSelectionResult {
        mode: mode.into(),
        mode_description: cfg.description,
        selected_tiles: selected,
        excluded_tiles: excluded,
        no_sar_tiles: no_sar,
        no_optical_tiles: no_optical,
        summary,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::tile_geometry::build_geometry_sidecar;

    #[test]
    fn historic_wreck_selects_low_cloud() {
        let mut props = std::collections::BTreeMap::new();
        props.insert("eo:cloud_cover".into(), "6".into());
        let geo = build_geometry_sidecar("S2B_16TFR_20240903.blue.tif", Some(&props), Some((2024, 9, 3, 16, 40)));
        let r = select_tiles(&[geo], "historic_wreck", None);
        assert!(!r.selected_tiles.is_empty());
        assert!(!r.no_optical_tiles);
    }
}
