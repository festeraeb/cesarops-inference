//! Census DB wipe SQL — port of `wipe_database.py`.

pub const WIPE_TABLES: &[&str] = &[
    "anomaly_hits",
    "stationary_anchors",
    "new_arrivals",
    "swot_passes",
];

pub const RESET_SEQUENCE_TABLES: &[&str] = &[
    "anomaly_hits",
    "stationary_anchors",
    "new_arrivals",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WipeCounts {
    pub anomaly_hits: u64,
    pub stationary_anchors: u64,
    pub new_arrivals: u64,
}

pub fn wipe_sql_statements() -> Vec<String> {
    let mut stmts: Vec<String> = WIPE_TABLES.iter().map(|t| format!("DELETE FROM {t};")).collect();
    for t in RESET_SEQUENCE_TABLES {
        stmts.push(format!("DELETE FROM sqlite_sequence WHERE name='{t}';"));
    }
    stmts
}

pub fn wipe_successful(after: &WipeCounts) -> bool {
    after.anomaly_hits == 0 && after.stationary_anchors == 0 && after.new_arrivals == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_delete_statements() {
        assert!(wipe_sql_statements().iter().any(|s| s.contains("anomaly_hits")));
    }
}
