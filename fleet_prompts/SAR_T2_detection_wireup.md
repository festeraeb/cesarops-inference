You are a Rust + cargo workspace + systemd specialist. Wire the cesarops-detection crate into the active fleet so we can run the triple-lock detection pipeline tonight.

## Current state

`cesarops-detection/` exists at `/home/cesarops/wreckhunter2000-1/cesarops-detection/` with its own `Cargo.toml` (NOT in workspace members). It implements:
- `pipeline.rs` — Triple-Lock orchestrator (Scout → Validator → Jitter)
- `dispatcher.rs` — axum HTTP server on port 5580 with /scan, /scan/{id}, /workers, /health
- `workers.rs` — HTTP clients for vision workers
- `types.rs` — JSON types

Existing endpoints:
- `POST /scan` → submit scan job (returns job_id)
- `GET /scan/{id}` → poll status + results
- `GET /workers` → worker health
- `GET /health` → service health

The workers it expects:
- Scout (Florence-2): SCOUT_URL env, default `http://100.105.77.74:5570` (cesarops3)
- Validator (Moondream2): VALIDATOR_URL env, default `http://100.102.158.111:5571` (cesarops2 — old Tailscale IP, real LAN is `10.0.0.129:5571`)
- Jitter (TPU VM): JITTER_URL env, default `http://192.168.122.10:8080`

## Deliverables

### 1. Workspace registration

Update the root `Cargo.toml` at `/home/cesarops/wreckhunter2000-1/Cargo.toml` to include `cesarops-detection` and `cesarops-aeromagnetic-worker` as workspace members so they build with `cargo build --release` from the root.

NOTE: The root Cargo.toml currently looks like (paste content):
```
[package]
name = "cesarops-adaptive"
version = "0.1.0"
edition = "2021"
description = "Unified Adaptive Detection Pipeline for CESAROPS"

[dependencies]
ndarray = "0.15"
...
```

This is a hybrid (top-level package + missing workspace). Convert it to a true workspace by adding:

```toml
[workspace]
members = [
    "cesarops-inference",
    "cesarops-forge-v2",
    "cesarops-detection",
    "cesarops-aeromagnetic-worker",
]
```

Keep the top-level `cesarops-adaptive` package intact (it's a valid binary that depends on ndarray/geo). The workspace + package together is a "virtual workspace with one root package."

### 2. Environment override script

Create `cesarops-detection/scripts/start.sh`:

```bash
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
```

Make executable (chmod +x).

### 3. Health-check tool for forge

The forge already has a `cluster/discover` endpoint that probes Tailscale peers. Add a tool to `cesarops-forge-v2/src/tools.rs` that lets the AI ask: "Is the triple-lock detection service ready right now?"

```rust
async fn detection_health(args: &Value, state: &AppState) -> String {
    // No args required.
    //
    // GET http://127.0.0.1:5580/health
    // Parse JSON {"workers": {"scout_1060": bool, "validator_p1000": bool, "jitter_tpu": bool}}
    //
    // Return human-readable summary:
    //   "Detection service: ONLINE
    //    Scout (1060):     OK
    //    Validator (P1000): OK
    //    Jitter (TPU):     OFFLINE — pipeline runs in 2-lock degraded mode"
    //
    // Timeout: 5 seconds.
    // If detection service unreachable: return "Detection service OFFLINE.
    //   Start with: cesarops-detection/scripts/start.sh"
}
```

Wire into the match: `"detection_health" => { reset_think_counter(); detection_health(arguments, state).await },`

### 4. Submit-scan tool for forge

```rust
async fn detection_scan(args: &Value, state: &AppState) -> String {
    // Args:
    //   region: string (required)         — region label, e.g. "lake_erie_central"
    //   tiles:  array of {lat, lon, image_b64}  — list of tiles to process
    //
    // POST http://127.0.0.1:5580/scan
    //   { "region": region, "tiles": tiles }
    //
    // Response: {"job_id": "uuid", "status": "running", "tiles": N, "message": "..."}
    //
    // Return: the job_id string + a hint to poll with detection_poll.
    // Timeout: 10 seconds (just submission).
}

async fn detection_poll(args: &Value, state: &AppState) -> String {
    // Args:
    //   job_id: string (required)
    //
    // GET http://127.0.0.1:5580/scan/{job_id}
    //
    // Return: pretty-printed JSON of confirmed detections + status.
    // Truncate at 2000 chars.
    // Timeout: 5 seconds.
}
```

Wire both into the match.

### 5. Update existing tool list error message

The fallback `_ => format!("Unknown tool: '{}'. Available: ...", name)` in `tools.rs::execute()` lists current tools. Update it to include all the new SAR tools that will land:
- detection_health
- detection_scan
- detection_poll
- (the SAR_T1 tools: scan_region, magnetic_dipole_detect, download_satellite_window, weather_window will be added by another batch)

## Constraints

- DO NOT change `cesarops-detection/` source code itself — it works as-is.
- The workspace conversion in root `Cargo.toml` MUST keep the existing `cesarops-adaptive` package buildable.
- Use `reqwest::Client` from the existing forge dependencies (already imported in tools.rs).
- All output return via `format!()` — no panics, no unwrap.
- Truncate detection_poll output if response > 2000 chars.

## Output format

Provide three artifacts:

```
=== FILE: Cargo.toml (root, full replacement) ===
// new workspace + package combined

=== FILE: cesarops-detection/scripts/start.sh ===
// the bash script

=== DIFF: cesarops-forge-v2/src/tools.rs ===
// match arm additions + 3 new async fn definitions (detection_health,
// detection_scan, detection_poll)
```

Code only. No preamble.
