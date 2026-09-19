//! Erie multi-year HLS download plan — port of `wreckhunter/download_erie_multiyear.py`.

use serde::{Deserialize, Serialize};

pub const ERIE_BBOX: [f64; 4] = [41.3, -83.5, 42.5, -78.8];
pub const SKIP_MONTHS: &[u32] = &[1, 2, 3];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErieMonthTask {
    pub year: i32,
    pub month: u32,
    pub max_results: u32,
    pub label: String,
}

pub fn active_months() -> Vec<u32> {
    (1..=12).filter(|m| !SKIP_MONTHS.contains(m)).collect()
}

pub fn max_results_for(year: i32, month: u32) -> u32 {
    if year == 2015 && month == 10 {
        50
    } else {
        4
    }
}

pub fn build_erie_tasks(years: std::ops::RangeInclusive<i32>) -> Vec<ErieMonthTask> {
    let mut tasks = Vec::new();
    for year in years {
        for month in active_months() {
            let max_results = max_results_for(year, month);
            let tag = if max_results >= 50 { "ALL" } else { "top4" };
            tasks.push(ErieMonthTask {
                year,
                month,
                max_results,
                label: format!("Erie | {year}-{month:02} | {tag}"),
            });
        }
    }
    tasks
}

pub fn month_date_range(year: i32, month: u32, last_day: u32) -> (String, String) {
    (
        format!("{year}-{month:02}-01"),
        format!("{year}-{month:02}-{last_day:02}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oct_2015_gets_all_granules() {
        assert_eq!(max_results_for(2015, 10), 50);
        assert_eq!(max_results_for(2016, 10), 4);
    }
}
