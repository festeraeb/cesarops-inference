```
=== FILE: Cargo.toml (root, full replacement) ===
[package]
name = "cesarops-adaptive"
version = "0.1.0"
edition = "2021"
description = "Unified Adaptive Detection Pipeline for CESAROPS"

[dependencies]
ndarray = "0.15"
geo = "0.24"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[workspace]
members = [
    "cesarops-inference",
    "cesarops-forge-v2",
    "cesarops-detection",
    "cesarops-aeromagnetic-worker",
]

=== FILE: cesarops-detection/scripts/start.sh ===
#!/usr/bin/env bash
# Start the cesarops-detection HTTP service on port 5580.
# Sets the right LAN IPs for the vision workers.
set -e

# Vision workers — use LAN IPs (cesarops2 = 10.0.0.129, cesarops3 = 10.0.0.41)
export SCOUT_URL="http://10.0.0.41:5570"          # cesarops3 LAN — Florence-2 on 1060
export VALIDATOR_URL="http://10.0.0.129:5571"     # cesarops2 LAN — Moondream2 on P1000
export JITTER_URL="http://192.168.122.10:8080"    # TPU VM (kept for reference; degrade gracefully if down)
export DETECTION_PORT="5580"

cd "$(dirname "$0")/.."
exec ../target/release/cesarops-detection

=== DIFF: cesarops-forge-v2/src/tools.rs ===
--- a/cesarops-forge-v2/src/tools.rs
+++ b/cesarops-forge-v2/src/tools.rs
@@ -1,5 +1,6 @@
+use serde_json::{json, Value};
+use std::time::Duration;
+
+async fn detection_health(args: &Value, state: &AppState) -> String {
+    let client = reqwest::Client::builder()
+        .timeout(Duration::from_secs(5))
+        .build()
+        .unwrap_or_default();
+
+    match client.get("http://127.0.0.1:5580/health").send().await {
+        Ok(resp) => {
+            if let Ok(json) = resp.json::<Value>().await {
+                let workers = json.get("workers");
+                let scout = if workers.and_then(|w| w.get("scout_1060")).and_then(|v| v.as_bool()).unwrap_or(false) { "OK" } else { "OFFLINE" };
+                let validator = if workers.and_then(|w| w.get("validator_p1000")).and_then(|v| v.as_bool()).unwrap_or(false) { "OK" } else { "OFFLINE" };
+                let jitter = if workers.and_then(|w| w.get("jitter_tpu")).and_then(|v| v.as_bool()).unwrap_or(false) { "OK" } else { "OFFLINE" };
+                
+                format!(
+                    "Detection service: ONLINE\n\
+                     Scout (1060):     {}\n\
+                     Validator (P1000): {}\n\
+                     Jitter (TPU):     {} {}",
+                    scout, validator, jitter,
+                    if jitter == "OFFLINE" { "— pipeline runs in 2-lock degraded mode" } else { "" }
+                )
+            } else {
+                "Detection service: ONLINE (but health JSON malformed)".to_string()
+            }
+        }
+        Err(_) => "Detection service OFFLINE. Start with: cesarops-detection/scripts/start.sh".to_string(),
+    }
+}
+
+async fn detection_scan(args: &Value, state: &AppState) -> String {
+    let client = reqwest::Client::builder()
+        .timeout(Duration::from_secs(10))
+        .build()
+        .unwrap_or_default();
+
+    let region = args.get("region").and_then(|v| v.as_str());
+    let tiles = args.get("tiles");
+
+    if region.is_none() || tiles.is_none() {
+        return "Error: Missing required arguments 'region' (string) or 'tiles' (array)".to_string();
+    }
+
+    let body = json!({
+        "region": region,
+        "tiles": tiles
+    });
+
+    match client.post("http://127.0.0.1:5580/scan").json(&body).send().await {
+        Ok(resp) => {
+            if let Ok(json) = resp.json::<Value>().await {
+                let job_id = json.get("job_id").and_then(|v| v.as_str()).unwrap_or("unknown");
+                format!("Job submitted. ID: {}. Use 'detection_poll' with this ID to check progress.", job_id)
+            } else {
+                "Error: Failed to parse submission response".to_string()
+            }
+        }
+        Err(e) => format!("Error submitting scan: {}", e),
+    }
+}
+
+async fn detection_poll(args: &Value, state: &AppState) -> String {
+    let client = reqwest::Client::builder()
+        .timeout(Duration::from_secs(5))
+        .build()
+        .unwrap_or_default();
+
+    let job_id = args.get("job_id").and_then(|v| v.as_str());
+    if let Some(id) = job_id {
+        let url = format!("http://127.0.0.1:5580/scan/{}", id);
+        match client.get(&url).send().await {
+            Ok(resp) => {
+                if let Ok(json) = resp.json::<Value>().await {
+                    let mut out = json.to_string();
+                    if out.len() > 2000 {
+                        out.truncate(2000);
+                        out.push_str("... [truncated]");
+                    }
+                    out
+                } else {
+                    "Error: Failed to parse poll response".to_string()
+                }
+            }
+            Err(e) => format!("Error polling job: {}", e),
+        }
+    } else {
+        "Error: Missing required argument 'job_id'".to_string()
+    }
+}
+
+/* ... inside execute() match arm ... */
+    "detection_health" => { reset_think_counter(); detection_health(arguments, state).await },
+    "detection_scan" => { reset_think_counter(); detection_scan(arguments, state).await },
+    "detection_poll" => { reset_think_counter(); detection_poll(arguments, state).await },
+/* ... inside fallback error message ... */
+    _ => format!(
+        "Unknown tool: '{}'. Available: [..., detection_health, detection_scan, detection_poll, scan_region, magnetic_dipole_detect, download_satellite_window, weather_window]",
+        name
+    ),
```
