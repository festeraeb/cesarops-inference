//! Fuel-leak / bubble discrimination from laptop `analyze_fuel_leaks.py` (Line 5 monitoring).
//! Pure Rust — no Python runtime.

/// (NIR - SWIR2) / (NIR + SWIR2). Positive → fuel sheen; negative → bubbles/foam.
#[inline]
pub fn leak_index(nir: f64, swir2: f64) -> f64 {
    let denom = nir + swir2;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    (nir - swir2) / denom
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelClass {
    FuelSheen,
    BubblesFoam,
    Ambiguous,
}

/// Classify from band reflectance (Sentinel-2 B08 / B12 proxies).
pub fn classify_pixel(nir: f64, swir2: f64, threshold: f64) -> (PixelClass, f64) {
    let idx = leak_index(nir, swir2);
    let class = if idx > threshold {
        PixelClass::FuelSheen
    } else if idx < -threshold {
        PixelClass::BubblesFoam
    } else {
        PixelClass::Ambiguous
    };
    (class, idx)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectionFuelAnalysis {
    pub utm_easting: f64,
    pub utm_northing: f64,
    pub wgs84_lat: f64,
    pub wgs84_lon: f64,
    pub original_classification: String,
    pub b08_nir: Option<f64>,
    pub b12_swir2: Option<f64>,
    pub leak_index: Option<f64>,
    pub bubble_mask_classification: String,
    pub is_fuel_sheen_candidate: bool,
    pub is_bubble_foam: bool,
}

/// Analyze one detection JSON row when NIR/SWIR2 present (or proxies).
pub fn analyze_detection(
    detection: &serde_json::Map<String, serde_json::Value>,
    threshold: f64,
) -> DetectionFuelAnalysis {
    let get_f = |k: &str| -> f64 {
        detection
            .get(k)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    };
    let nir = detection
        .get("b08_nir")
        .or_else(|| detection.get("aluminum_ratio"))
        .and_then(|v| v.as_f64());
    let swir2 = detection.get("b12_swir2").and_then(|v| v.as_f64());

    let mut out = DetectionFuelAnalysis {
        utm_easting: get_f("utm_easting"),
        utm_northing: get_f("utm_northing"),
        wgs84_lat: get_f("wgs84_lat"),
        wgs84_lon: get_f("wgs84_lon"),
        original_classification: detection
            .get("classification")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string(),
        b08_nir: nir,
        b12_swir2: swir2,
        leak_index: None,
        bubble_mask_classification: "PENDING".into(),
        is_fuel_sheen_candidate: false,
        is_bubble_foam: false,
    };

    if let (Some(n), Some(s)) = (nir, swir2) {
        let (class, idx) = classify_pixel(n, s, threshold);
        out.leak_index = Some(idx);
        out.is_fuel_sheen_candidate = class == PixelClass::FuelSheen;
        out.is_bubble_foam = class == PixelClass::BubblesFoam;
        out.bubble_mask_classification = format!("{class:?}");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuel_sheen_positive_index() {
        let (c, idx) = classify_pixel(0.8, 0.1, 0.1);
        assert_eq!(c, PixelClass::FuelSheen);
        assert!(idx > 0.0);
    }

    #[test]
    fn foam_negative_index() {
        let (c, _) = classify_pixel(0.2, 0.9, 0.1);
        assert_eq!(c, PixelClass::BubblesFoam);
    }
}
