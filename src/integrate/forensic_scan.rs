//! Integrated forensic scan logic — port of `integrated_forensic_scan.py` / `full_basin_scan.py`.
//! Zion depth scaling, Straits offset, detection records (no Python runtime).

use serde::{Deserialize, Serialize};

pub const ZION_CONSTANT: f64 = 1.47;
pub const DEPTH_THRESHOLD_FT: f64 = 400.0;
pub const STRAITS_LAT_THRESHOLD: f64 = 45.8;
pub const STRAITS_CORRECTION_FACTOR: f64 = 0.92;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicDetection {
    pub lat: f64,
    pub lon: f64,
    pub length_ft_original: f64,
    pub length_ft_corrected: f64,
    pub mass_tons: f64,
    pub thermal_zscore: f64,
    pub signature_type: String,
    pub confidence: f64,
    pub filter_matched: String,
    pub source_tile: String,
    pub candidate_type: Option<String>,
    pub island_count: u32,
    pub jitter_detected: bool,
    pub specular_ratio: f64,
    pub two_date_verified: bool,
    pub curvelet_applied: bool,
    pub straits_offset_applied: bool,
    pub notes: String,
}

/// Apply 1.47× inverse length scaling when depth exceeds threshold (Zion Constant).
#[inline]
pub fn apply_depth_scaling(length_ft: f64, depth_ft: f64) -> f64 {
    if depth_ft > DEPTH_THRESHOLD_FT {
        length_ft / ZION_CONSTANT
    } else {
        length_ft
    }
}

/// Circular Slosh correction north of Mackinac Straits.
#[inline]
pub fn apply_straits_correction(lat: f64, length_ft: f64) -> (f64, bool) {
    if lat > STRAITS_LAT_THRESHOLD {
        (length_ft * STRAITS_CORRECTION_FACTOR, true)
    } else {
        (length_ft, false)
    }
}

/// Full post-process pipeline for one detection row.
pub fn finalize_detection(
    mut det: ForensicDetection,
    depth_ft: f64,
) -> ForensicDetection {
    det.length_ft_corrected = apply_depth_scaling(det.length_ft_original, depth_ft);
    let (corrected, straits) = apply_straits_correction(det.lat, det.length_ft_corrected);
    det.length_ft_corrected = corrected;
    det.straits_offset_applied = straits;
    if straits {
        det.notes.push_str(" | Straits Offset applied (Circular Slosh)");
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_scaling_above_threshold() {
        assert!((apply_depth_scaling(352.0, 500.0) - 239.46).abs() < 0.1);
    }

    #[test]
    fn straits_correction_north() {
        let (len, applied) = apply_straits_correction(46.0, 100.0);
        assert!(applied);
        assert!((len - 92.0).abs() < f64::EPSILON);
    }
}
