//! Monster candidate scoring — port of `analysis/monster_candidate.py`.

use serde::{Deserialize, Serialize};

pub const MONSTER_LENGTH_FT: f32 = 343.0;
pub const MONSTER_LAT: f64 = 42.4180;
pub const MONSTER_LON: f64 = -87.2350;
pub const PIXELS_PER_FT: f32 = 1.0 / 98.4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MonsterCandidate {
    pub rank: u32,
    pub row: u32,
    pub col: u32,
    pub pixels: u32,
    pub zscore: f32,
    pub notes: String,
}

pub fn target_pixel_area() -> u32 {
    let side = MONSTER_LENGTH_FT * PIXELS_PER_FT;
    (side * side) as u32
}

pub fn is_monster_sized(pixels: u32) -> bool {
    (100..=500).contains(&pixels)
}

pub fn best_match(candidates: &[MonsterCandidate]) -> Option<&MonsterCandidate> {
    candidates
        .iter()
        .filter(|c| is_monster_sized(c.pixels))
        .max_by(|a, b| a.zscore.partial_cmp(&b.zscore).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_in_range() {
        assert!(is_monster_sized(422));
        assert!(!is_monster_sized(50));
    }
}
