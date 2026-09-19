//! M2200 hardware synthetic TIFF test — port of `gpu_engine/tests/test_m2200_hardware.py`.

use serde::{Deserialize, Serialize};

pub const SYNTH_WIDTH: u32 = 1000;
pub const SYNTH_HEIGHT: u32 = 1000;
pub const AMBIENT_K: f32 = 285.0;
pub const ANOMALY_DELTA_K: f32 = 10.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareTestSpec {
    pub width: u32,
    pub height: u32,
    pub anomaly_count: u32,
    pub output_tiff: String,
}

impl Default for HardwareTestSpec {
    fn default() -> Self {
        Self {
            width: SYNTH_WIDTH,
            height: SYNTH_HEIGHT,
            anomaly_count: 10,
            output_tiff: "test_thermal.tif".into(),
        }
    }
}

pub fn scale_to_u16(kelvin: f32) -> u16 {
    ((kelvin - 250.0) * 200.0).clamp(0.0, 65535.0) as u16
}

pub fn expected_anomaly_pixels(spec: &HardwareTestSpec) -> u32 {
    spec.anomaly_count * 11 * 11
}
