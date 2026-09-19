You are a Rust + axum + reqwest specialist. Add 4 wreck-detection tools to our forge-v2 tool registry. The forge-v2 is the AI orchestration layer for cesarops.com — a dual-use remote-sensing platform (wreck hunting + search and rescue + downed-aircraft detection). The toolchain doesn't care if the missing thing is a 19th-century schooner or a Cessna — same physics, same sensors, different validation databases.

## Existing pattern in src/tools.rs

Existing tools follow this signature:

```rust
pub async fn execute(name: &str, arguments: &Value, state: &AppState) -> String {
    match name {
        "write_file" => write_file(arguments, state).await,
        "read_file" => read_file(arguments, state).await,
        "cargo_check" => cargo_check(arguments, state).await,
        "think_harder" => think_harder(arguments, state).await,
        "remember" => remember(arguments, state).await,
        "run_command" => run_command(arguments, state).await,
        "speed_check" => speed_check(arguments, state).await,
        _ => format!("Unknown tool: '{}'. Available: ...", name),
    }
}
```

Each tool is `async fn name(args: &Value, state: &AppState) -> String`. Use `tokio::process::Command::new("python")` or `Command::new("cargo")` to shell out. Truncate output with `truncate_output(&s, 2000)` before returning. Return errors as `format!("Error: ...")` strings — DO NOT panic.

Existing helper functions you can use:
- `truncate_output(&str, max: usize) -> String` — caps long output
- `resolve_path(relative, state) -> PathBuf` — resolves project-relative paths

## Deliverables

Add 4 tools to `cesarops-forge-v2/src/tools.rs`:

### 1. `scan_region`

Calls the proven Python wreck-detection scanner.

```rust
async fn scan_region(args: &Value, state: &AppState) -> String {
    // Required args:
    //   bbox: "lat_min,lon_min,lat_max,lon_max"  (string of 4 floats)
    //   days: u32 — look-back window in days, default 14
    // Optional:
    //   mode: string — "wreck" (default) | "sar" | "downed_aircraft"
    //   region_name: string — friendly label for logs/output
    //
    // Shell out to:
    //   python /mnt/data-external/cesarops/repo/scan_engine.py \
    //     --bbox <bbox> --days <days> --mode <mode> \
    //     --output /tmp/scan_<unix_timestamp>.json
    //
    // Return: JSON content of /tmp/scan_*.json (anomaly list with
    //         lat/lon/confidence/triple_lock_score), or error string.
    //
    // Timeout: 30 minutes (scan_engine.py is slow on big regions)
}
```

### 2. `magnetic_dipole_detect`

Calls the Rust aeromagnetic worker (already built and shipping in
`cesarops-aeromagnetic-worker/`).

```rust
async fn magnetic_dipole_detect(args: &Value, state: &AppState) -> String {
    // Required args:
    //   grid_path: string — path to magnetic grid file (CSV or NPY)
    //   pixel_size_m: f32 — meters per pixel, default 25.0
    //   inner_radius: u32 — dipole detection inner radius (default 5)
    //   outer_radius: u32 — outer radius (default 15)
    //
    // Shell out to:
    //   /codebase/repos/wreckhunter2000-1/target/release/cesarops-aeromagnetic-worker \
    //     --grid <grid_path> --pixel-size <pixel_size_m> \
    //     --inner <inner_radius> --outer <outer_radius>
    //
    // Return: stdout (dipole detection list) or error string.
    // Timeout: 5 minutes
}
```

### 3. `download_satellite_window`

Calls the multi-source satellite downloader (proven Python).

```rust
async fn download_satellite_window(args: &Value, state: &AppState) -> String {
    // Required args:
    //   bbox: "lat_min,lon_min,lat_max,lon_max"
    //   provider: "auto" (default) | "sentinel" | "landsat" | "ecostress" | "swot"
    //   days: u32 — window size in days, default 14
    //   output_dir: optional override (defaults to /tmp/cesarops_downloads/)
    //
    // Shell out to:
    //   python /mnt/data-external/cesarops/repo/universal_downloader.py \
    //     --bbox <bbox> --provider <provider> --days <days> --output-dir <dir>
    //
    // Return: comma-separated list of downloaded file paths.
    // Timeout: 30 minutes
}
```

### 4. `weather_window`

Calls weather classifier — wreck/SAR scans need post-storm windows
when sediment is settled but features are stirred up.

```rust
async fn weather_window(args: &Value, state: &AppState) -> String {
    // Required args:
    //   bbox: "lat_min,lon_min,lat_max,lon_max"
    //   check: string — "post_storm" (default) | "calm" | "any"
    //
    // Shell out to:
    //   python /mnt/data-external/cesarops/repo/weather_service.py \
    //     --bbox <bbox> --classify <check>
    //
    // Return: JSON {"window": "post_storm"|"calm"|"none",
    //               "last_storm_hours_ago": float,
    //               "wind_speed_kts": float,
    //               "recommendation": "scan_now"|"wait_<hours>"}
    // Timeout: 60 seconds
}
```

## Wiring into execute()

Add these branches to the match in `pub async fn execute()`:

```rust
"scan_region" => { reset_think_counter(); scan_region(arguments, state).await },
"magnetic_dipole_detect" => { reset_think_counter(); magnetic_dipole_detect(arguments, state).await },
"download_satellite_window" => { reset_think_counter(); download_satellite_window(arguments, state).await },
"weather_window" => { reset_think_counter(); weather_window(arguments, state).await },
```

Update the `_ => format!("Unknown tool ...")` line to list these new tools too.

## Constraints

- DO NOT change existing tool functions
- DO NOT add new dependencies — use what's already in Cargo.toml (tokio, reqwest, serde_json, tracing)
- Use `tokio::process::Command::new("python")` not `std::process::Command` — async, doesn't block runtime
- DO NOT use `unwrap()` anywhere — all errors return as String via format!()
- All four functions return `String` (consumed as tool output by the AI)
- Max output ~2000 chars (truncate)
- For long-running commands (scan_region, download_satellite_window): use `.timeout(...)` on a tokio::time::timeout wrapper, not `Command` timeout (Command doesn't have one)

## Output format

Provide the diff to `cesarops-forge-v2/src/tools.rs` as:

```
=== DIFF: src/tools.rs ===
// existing match arms...
+        "scan_region" => { reset_think_counter(); scan_region(arguments, state).await },
+        "magnetic_dipole_detect" => { reset_think_counter(); magnetic_dipole_detect(arguments, state).await },
+        "download_satellite_window" => { reset_think_counter(); download_satellite_window(arguments, state).await },
+        "weather_window" => { reset_think_counter(); weather_window(arguments, state).await },
// existing fallback...

=== APPEND: src/tools.rs ===
// 4 full async fn implementations matching the pattern of existing tools
```

The DIFF block goes near the top match, the APPEND block goes at the end of the file before the helpers section.

Begin now. Code only, no preamble.
