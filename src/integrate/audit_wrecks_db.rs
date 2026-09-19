//! Wrecks.db coordinate audit — port of `wreckhunter/tools/audit_wrecks_db.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoordCluster {
    pub lat_rounded: f64,
    pub lon_rounded: f64,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WreckSingleton {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WrecksAuditReport {
    pub tables: Vec<String>,
    pub gps_row_count: u32,
    pub top_clusters: Vec<CoordCluster>,
    pub high_density_cluster_wrecks: u32,
    pub singletons: Vec<WreckSingleton>,
}

pub fn round_coord(v: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (v * factor).round() / factor
}

/// Cluster rows by rounded lat/lon; return top-N by count.
pub fn top_coordinate_clusters(
    rows: &[(f64, f64)],
    decimals: u32,
    limit: usize,
) -> Vec<CoordCluster> {
    let mut counts = std::collections::HashMap::<String, (f64, f64, u32)>::new();
    for &(lat, lon) in rows {
        let lat_r = round_coord(lat, decimals);
        let lon_r = round_coord(lon, decimals);
        let key = format!("{lat_r},{lon_r}");
        counts
            .entry(key)
            .and_modify(|e| e.2 += 1)
            .or_insert((lat_r, lon_r, 1));
    }
    let mut clusters: Vec<CoordCluster> = counts
        .values()
        .map(|(lat_rounded, lon_rounded, count)| CoordCluster {
            lat_rounded: *lat_rounded,
            lon_rounded: *lon_rounded,
            count: *count,
        })
        .collect();
    clusters.sort_by(|a, b| b.count.cmp(&a.count));
    clusters.truncate(limit);
    clusters
}

pub fn wrecks_in_high_density_clusters(clusters: &[CoordCluster], threshold: u32) -> u32 {
    clusters
        .iter()
        .filter(|c| c.count > threshold)
        .map(|c| c.count)
        .sum()
}

/// Unique GPS: no other wreck within ~0.001° on same rounded bucket.
pub fn find_singletons(
    wrecks: &[(String, f64, f64)],
    decimals: u32,
) -> Vec<WreckSingleton> {
    let mut bucket_counts = std::collections::HashMap::<String, u32>::new();
    for (_, lat, lon) in wrecks {
        let key = format!(
            "{},{}",
            round_coord(*lat, decimals),
            round_coord(*lon, decimals)
        );
        *bucket_counts.entry(key).or_insert(0) += 1;
    }
    wrecks
        .iter()
        .filter(|(_, lat, lon)| {
            let key = format!(
                "{},{}",
                round_coord(*lat, decimals),
                round_coord(*lon, decimals)
            );
            bucket_counts.get(&key) == Some(&1)
        })
        .map(|(name, lat, lon)| WreckSingleton {
            name: name.clone(),
            lat: *lat,
            lon: *lon,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_duplicate_coords() {
        let rows = vec![(42.1, -87.0), (42.1004, -87.0002), (45.0, -86.0)];
        let top = top_coordinate_clusters(&rows, 1, 10);
        assert!(!top.is_empty());
        assert!(top[0].count >= 2);
    }
}
