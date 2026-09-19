//! Monster material-density audit heuristics.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoProfile {
    pub key: &'static str,
    pub thermal_decay_rate: f64,
    pub typical_cargo_tons: u32,
}

pub fn cargo_profiles() -> Vec<CargoProfile> {
    vec![
        CargoProfile { key: "steel_rails", thermal_decay_rate: 0.45, typical_cargo_tons: 8000 },
        CargoProfile { key: "iron_ore", thermal_decay_rate: 0.55, typical_cargo_tons: 10000 },
        CargoProfile { key: "granite_stone", thermal_decay_rate: 0.85, typical_cargo_tons: 4000 },
        CargoProfile { key: "coal_bulk", thermal_decay_rate: 0.90, typical_cargo_tons: 5000 },
    ]
}

pub fn calculate_thermal_decay(thermal_signature: &str, mass_tons: f64) -> f64 {
    let base = 0.70;
    let mass_factor = mass_tons.log10() / 4.0;
    let mass_adj = -0.15 * mass_factor;
    let sig_adj = if thermal_signature.to_ascii_lowercase().contains("strong_steel") {
        -0.10
    } else if thermal_signature.to_ascii_lowercase().contains("moderate") {
        0.0
    } else {
        0.15
    };
    (base + mass_adj + sig_adj).clamp(0.1, 1.2)
}

pub fn compare_thermal_decay(target_decay: f64, reference_decay: f64) -> (&'static str, &'static str) {
    let diff = target_decay - reference_decay;
    if diff < -0.15 {
        ("MUCH COLDER - Metal cargo (Rails/Ore)", "HIGH")
    } else if diff < -0.05 {
        ("COLDER - Dense metal cargo", "MODERATE")
    } else if diff.abs() < 0.05 {
        ("SIMILAR - Mixed cargo", "LOW")
    } else if diff < 0.15 {
        ("WARMER - Stone/Organic cargo", "MODERATE")
    } else {
        ("MUCH WARMER - Organic cargo (Coal/Grain)", "HIGH")
    }
}

pub fn best_profile_for_decay(target_decay: f64) -> Option<CargoProfile> {
    cargo_profiles().into_iter().min_by(|a, b| {
        (a.thermal_decay_rate - target_decay)
            .abs()
            .partial_cmp(&(b.thermal_decay_rate - target_decay).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steel_signature_prefers_lower_decay() {
        let d = calculate_thermal_decay("strong_steel", 14_474.0);
        assert!(d < 0.7);
        assert_eq!(best_profile_for_decay(d).unwrap().key, "steel_rails");
    }
}
