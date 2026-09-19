//! Multi-node deploy + comprehensive scan — port of `deploy_and_scan.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeConfig {
    pub name: String,
    pub host: String,
    pub user: String,
    pub work_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanBbox {
    pub lat_min: f64,
    pub lon_min: f64,
    pub lat_max: f64,
    pub lon_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanArea {
    pub name: String,
    pub description: String,
    pub bbox: ScanBbox,
    pub start_date: String,
    pub end_date: String,
    pub sensors: Vec<String>,
}

pub fn default_nodes() -> Vec<NodeConfig> {
    vec![
        NodeConfig {
            name: "pi".into(),
            host: "10.0.0.226".into(),
            user: "pi".into(),
            work_dir: "/home/pi/cesarops".into(),
        },
        NodeConfig {
            name: "xenon".into(),
            host: "10.0.0.40".into(),
            user: "cesarops".into(),
            work_dir: "/home/cesarops/cesarops/cesarops-core".into(),
        },
    ]
}

pub fn deploy_python_files() -> &'static [&'static str] {
    &[
        "ai_director.py",
        "lake_michigan_scan.py",
        "cesarops_orchestrator.py",
        "hard_pixel_audit.py",
        "swot_ssh_extractor.py",
        "remote_dispatch.py",
        "background_probe.py",
        "cuda_env.py",
        "cesarops_engine.py",
        "database_connector.py",
        "init_database.py",
        "universal_downloader.py",
    ]
}

pub fn deploy_config_files() -> &'static [&'static str] {
    &[".env", "known_wrecks.json", "satellite_data_sources.json"]
}

pub fn default_scan_area() -> ScanArea {
    ScanArea {
        name: "Northern Great Lakes Comprehensive Scan".into(),
        description: "Northern Lake Huron + Michigan + Straits + UP + North Channel + Georgian Bay + Green Bay".into(),
        bbox: ScanBbox {
            lat_min: 44.5,
            lon_min: -92.0,
            lat_max: 47.0,
            lon_max: -80.0,
        },
        start_date: "2024-06-01".into(),
        end_date: "2025-09-30".into(),
        sensors: vec!["thermal".into(), "optical".into(), "sar".into(), "swot".into()],
    }
}

pub fn mkdir_outputs_cmd(work_dir: &str) -> String {
    format!("mkdir -p {work_dir}/outputs/probes")
}

pub fn xenon_cupy_check_cmd(work_dir: &str) -> String {
    format!(
        "cd {work_dir} && source ~/cesarops/venv/bin/activate 2>/dev/null || true && \
         python3 -c \"import cupy; print('CuPy version:', cupy.__version__)\" 2>&1"
    )
}

pub fn comprehensive_scan_cmd(work_dir: &str, area: &ScanArea) -> String {
    let b = &area.bbox;
    let sensors = area.sensors.join(",");
    let ts = area.start_date.replace('-', "");
    format!(
        "source ~/cesarops/venv/bin/activate && cd {work_dir} && \
         export CESAROPS_DATA_DIR=/home/cesarops/cesarops/Sync && \
         python ai_director.py \
         --bbox {lat_min},{lon_min},{lat_max},{lon_max} \
         --tools {sensors} \
         --sensitivity 1.0 \
         --execute \
         --no-llm \
         --output outputs/probes/comprehensive_scan_{ts}.json 2>&1",
        lat_min = b.lat_min,
        lon_min = b.lon_min,
        lat_max = b.lat_max,
        lon_max = b.lon_max,
        sensors = sensors,
        ts = ts,
    )
}

pub fn deploy_plan(node: &NodeConfig) -> Vec<String> {
    let mut steps = vec![mkdir_outputs_cmd(&node.work_dir)];
    for f in deploy_python_files().iter().chain(deploy_config_files().iter()) {
        steps.push(format!("deploy {}/{}", node.work_dir, f));
    }
    if node.name == "xenon" {
        steps.push(xenon_cupy_check_cmd(&node.work_dir));
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_cmd_includes_bbox() {
        let area = default_scan_area();
        let cmd = comprehensive_scan_cmd("/home/cesarops/cesarops/cesarops-core", &area);
        assert!(cmd.contains("44.5"));
        assert!(cmd.contains("ai_director.py"));
    }
}
