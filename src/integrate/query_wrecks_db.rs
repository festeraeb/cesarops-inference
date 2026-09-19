//! Wrecks.db query helpers — port of `query_wrecks_db.py`.

use serde::{Deserialize, Serialize};

pub const FEATURE_TABLE_CANDIDATES: &[&str] = &["features", "wrecks", "vessels", "shipwrecks"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableSummary {
    pub name: String,
    pub row_count: u64,
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoordColumnGuess {
    pub lat_col: Option<String>,
    pub lon_col: Option<String>,
    pub name_col: Option<String>,
}

pub fn pick_feature_table(tables: &[String]) -> Option<String> {
    tables
        .iter()
        .find(|t| FEATURE_TABLE_CANDIDATES.contains(&t.as_str()))
        .cloned()
        .or_else(|| tables.first().cloned())
}

pub fn guess_coord_columns(columns: &[String]) -> CoordColumnGuess {
    let lat_col = columns.iter().find(|c| c.to_lowercase().contains("lat")).cloned();
    let lon_col = columns.iter().find(|c| c.to_lowercase().contains("lon")).cloned();
    let name_col = columns.iter().find(|c| {
        let l = c.to_lowercase();
        ["vessel_name", "ship_name", "name", "title"]
            .iter()
            .any(|k| l.contains(k))
    }).cloned();
    CoordColumnGuess {
        lat_col,
        lon_col,
        name_col,
    }
}

pub fn is_real_coord(lat: f64, lon: f64) -> bool {
    lat != 0.0 && lat != 45.0 && lon != -83.0
}

pub fn verified_coords_sql(table: &str, lat: &str, lon: &str) -> String {
    format!(
        "SELECT COUNT(*) FROM [{table}] WHERE [{lat}] IS NOT NULL AND [{lat}] != 0 \
         AND [{lat}] != 45.0 AND [{lon}] != -83.0"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_features_table() {
        assert_eq!(
            pick_feature_table(&["meta".into(), "features".into()]).as_deref(),
            Some("features")
        );
    }

    #[test]
    fn rejects_placeholder_coords() {
        assert!(!is_real_coord(45.0, -83.0));
        assert!(is_real_coord(42.5, -87.0));
    }
}
