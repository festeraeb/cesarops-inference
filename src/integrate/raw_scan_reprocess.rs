//! Raw scan reprocess orchestrator — port of `wreckhunter/tools/raw_scan_reprocess.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReprocessMode {
    FetchOnly,
    AuditOnly,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum YearWindow {
    Rossa,
    Baseline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReprocessPlan {
    pub tiles: Vec<String>,
    pub mode: ReprocessMode,
    pub year_window: YearWindow,
}

pub fn parse_tiles_arg(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_uppercase())
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn parse_year_window(s: &str) -> YearWindow {
    match s.to_lowercase().as_str() {
        "baseline" => YearWindow::Baseline,
        _ => YearWindow::Rossa,
    }
}

pub fn parse_mode(s: &str) -> ReprocessMode {
    match s.to_lowercase().as_str() {
        "fetch-only" => ReprocessMode::FetchOnly,
        "audit-only" => ReprocessMode::AuditOnly,
        _ => ReprocessMode::All,
    }
}
