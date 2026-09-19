//! Global scanner settings — port of `global_controls.py` presets/thresholds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LakePreset {
    pub sensitivity: f32,
    pub curvelet_scales: u32,
    pub curvelet_angles: u32,
    pub min_temp_k: f32,
    pub max_temp_k: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetPreset {
    pub sensitivity: f32,
    pub min_length_ft: Option<f32>,
    pub max_length_ft: Option<f32>,
    pub min_mass_tons: Option<f32>,
    pub aluminum_ratio: Option<f32>,
    pub thermal_sink: Option<f32>,
    pub depth_ft: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VramSettings {
    pub chunk_size: u32,
    pub overlap_percent: u32,
    pub use_streams: bool,
    pub max_vram_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobalScannerSettings {
    pub sensitivity: f32,
    pub curvelet_scales: u32,
    pub curvelet_angles: u32,
    pub lake_presets: HashMap<String, LakePreset>,
    pub target_presets: HashMap<String, TargetPreset>,
    pub vram_settings: VramSettings,
}

impl Default for GlobalScannerSettings {
    fn default() -> Self {
        let mut lake_presets = HashMap::new();
        lake_presets.insert(
            "michigan".into(),
            LakePreset {
                sensitivity: 2.5,
                curvelet_scales: 4,
                curvelet_angles: 16,
                min_temp_k: 275.0,
                max_temp_k: 295.0,
            },
        );
        lake_presets.insert(
            "erie".into(),
            LakePreset {
                sensitivity: 4.5,
                curvelet_scales: 3,
                curvelet_angles: 8,
                min_temp_k: 278.0,
                max_temp_k: 305.0,
            },
        );
        let mut target_presets = HashMap::new();
        target_presets.insert(
            "andaste".into(),
            TargetPreset {
                sensitivity: 1.8,
                min_length_ft: Some(250.0),
                max_length_ft: Some(350.0),
                min_mass_tons: None,
                aluminum_ratio: None,
                thermal_sink: None,
                depth_ft: Some(180.0),
            },
        );
        Self {
            sensitivity: 3.0,
            curvelet_scales: 4,
            curvelet_angles: 16,
            lake_presets,
            target_presets,
            vram_settings: VramSettings {
                chunk_size: 512,
                overlap_percent: 10,
                use_streams: true,
                max_vram_gb: 3.5,
            },
        }
    }
}

impl GlobalScannerSettings {
    pub fn overlap_pixels(&self) -> u32 {
        self.vram_settings.chunk_size * self.vram_settings.overlap_percent / 100
    }

    pub fn apply_lake(&mut self, lake_name: &str) -> bool {
        let key = lake_name.to_lowercase();
        if let Some(p) = self.lake_presets.get(&key).cloned() {
            self.sensitivity = p.sensitivity;
            self.curvelet_scales = p.curvelet_scales;
            self.curvelet_angles = p.curvelet_angles;
            true
        } else {
            false
        }
    }

    pub fn apply_target(&mut self, target: &str) -> Option<TargetPreset> {
        self.target_presets.get(&target.to_lowercase()).cloned().map(|p| {
            self.sensitivity = p.sensitivity;
            p
        })
    }

    pub fn threshold(&self, mean: f32, std: f32) -> f32 {
        mean + self.sensitivity * std
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lake_preset_updates_sensitivity() {
        let mut s = GlobalScannerSettings::default();
        assert!(s.apply_lake("michigan"));
        assert_eq!(s.sensitivity, 2.5);
    }

    #[test]
    fn overlap_pixels_default() {
        let s = GlobalScannerSettings::default();
        assert_eq!(s.overlap_pixels(), 51);
    }
}
