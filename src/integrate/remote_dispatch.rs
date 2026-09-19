//! Remote task dispatcher — SSH command builders — port of `remote_dispatch.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteNode {
    pub host: String,
    pub user: String,
    pub work_dir: String,
    pub has_password: bool,
    pub has_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_s: f64,
}

pub fn default_pi_node() -> RemoteNode {
    RemoteNode {
        host: std::env::var("PI_HOST").unwrap_or_else(|_| "10.0.0.100".into()),
        user: std::env::var("PI_USER").unwrap_or_else(|_| "pi".into()),
        work_dir: std::env::var("PI_WORK").unwrap_or_else(|_| "/home/pi/cesarops/sync".into()),
        has_password: std::env::var("PI_PASS").is_ok(),
        has_key: std::env::var("PI_KEY").is_ok(),
    }
}

pub fn default_xenon_node() -> RemoteNode {
    RemoteNode {
        host: std::env::var("XENON_HOST").unwrap_or_else(|_| "10.0.0.40".into()),
        user: std::env::var("XENON_USER").unwrap_or_else(|_| "cesarops".into()),
        work_dir: std::env::var("XENON_WORK").unwrap_or_else(|_| "/home/cesarops/cesarops/sync".into()),
        has_password: std::env::var("XENON_PASS").is_ok(),
        has_key: std::env::var("XENON_KEY").is_ok(),
    }
}

pub fn build_pi_slice_task(
    work_dir: &str,
    area_name: &str,
    sources: &[&str],
    tile_size: u32,
    target_resolution: f64,
    mission_json: Option<&str>,
) -> String {
    let sources_str = sources.join(" ");
    let mut cmd = format!(
        "cd {work_dir} && echo '[PI] Starting slice pipeline for {area_name}' && \
         mkdir -p tiles/cpu tiles/tpu tiles/gpu tiles/hybrid && \
         ./slicer vrt {sources_str} --output tiles --tile-size {tile_size} \
         --target-resolution {target_resolution}"
    );
    if let Some(m) = mission_json {
        cmd.push_str(&format!(" --mission {m}"));
    }
    cmd.push_str(
        " && echo '[PI] Slicing complete — tiles staged in delegate folders' \
         && ls -la tiles/*/ | tail -20",
    );
    cmd
}

pub fn build_xenon_process_task(work_dir: &str, delegate: Option<&str>) -> String {
    match delegate {
        Some(d) => format!(
            "cd {work_dir} && echo '[XENON] Starting tile processing' && \
             echo '[XENON] Processing {d} tiles...' && \
             python cesarops_engine.py --tiles-dir tiles/{d} --delegate {d} && \
             echo '[XENON] Processing complete'"
        ),
        None => format!(
            "cd {work_dir} && echo '[XENON] Starting tile processing' && \
             for delegate in tpu gpu cpu hybrid; do \
               count=$(ls tiles/$delegate/*.bin 2>/dev/null | wc -l); \
               if [ $count -gt 0 ]; then \
                 echo '[XENON] Processing $delegate: $count tiles'; \
                 python cesarops_engine.py --tiles-dir tiles/$delegate --delegate $delegate; \
               fi; \
             done && echo '[XENON] Processing complete'"
        ),
    }
}

pub fn build_xenon_tpu_health(work_dir: &str) -> String {
    format!(
        "cd {work_dir} && curl -s http://localhost:5001/health 2>/dev/null || echo '{{\"status\": \"unreachable\"}}'"
    )
}

pub fn build_xenon_tpu_start(work_dir: &str) -> String {
    let search_paths = [
        work_dir,
        &format!("{work_dir}/../cesarops-core"),
        "/home/cesarops/cesarops-core",
        "/home/cesarops/cesarops/cesarops-core",
        "/opt/cesarops",
    ];
    let find_cmd: Vec<_> = search_paths
        .iter()
        .map(|p| format!("[ -f {p}/tpu_server.py ] && echo {p}"))
        .collect();
    format!(
        "TPU_DIR=$( {} | head -1 ) && \
         if [ -z \"$TPU_DIR\" ]; then echo '{{\"status\": \"tpu_server_not_found\"}}'; exit 0; fi && \
         if ! curl -s http://localhost:5001/health > /dev/null 2>&1; then \
           cd $TPU_DIR && nohup python tpu_server.py --port 5001 > /tmp/tpu_server.log 2>&1 & sleep 5; \
         fi && \
         curl -s http://localhost:5001/health 2>/dev/null || echo '{{\"status\": \"start_failed\"}}'",
        find_cmd.join(" || ")
    )
}

pub fn build_data_inventory(work_dir: &str, lakes: &[&str]) -> String {
    let lake_list = lakes.join(" ");
    format!(
        "python3 -c \"import json, os; root='{work_dir}/downloads'; inv={{}}; \
         [inv.update({{lake: {{'exists': os.path.isdir(os.path.join(root,lake)), \
         'files': len(os.listdir(os.path.join(root,lake))) if os.path.isdir(os.path.join(root,lake)) else 0}}}} \
         for lake in '{lake_list}'.split()])]; print(json.dumps(inv))\""
    )
}

pub fn node_configured(node: &RemoteNode) -> bool {
    node.has_password || node.has_key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_slice_includes_area() {
        let cmd = build_pi_slice_task("/home/pi/sync", "Straits", &["a.tif"], 1024, 10.0, None);
        assert!(cmd.contains("Straits"));
        assert!(cmd.contains("./slicer"));
    }

    #[test]
    fn xenon_process_all_delegates() {
        let cmd = build_xenon_process_task("/home/cesarops/sync", None);
        assert!(cmd.contains("tpu gpu cpu hybrid"));
    }
}
