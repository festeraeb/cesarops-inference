```rust
=== DIFF: src/tools.rs ===
// ... existing match arms ...
+        "scan_region" => { reset_think_counter(); scan_region(arguments, state).await },
+        "magnetic_dipole_detect" => { reset_think_counter(); magnetic_dipole_detect(arguments, state).await },
+        "download_satellite_window" => { reset_think_counter(); download_satellite_window(arguments, state).await },
+        "weather_window" => { reset_think_counter(); weather_window(arguments, state).await },
// ... existing fallback ...
+        "Unknown tool: {}. Available tools: search_vessels, get_vessel_details, scan_region, magnetic_dipole_detect, download_satellite_window, weather_window",
```

```rust
=== APPEND: src/tools.rs ===
async fn scan_region(args: &Value, _state: &AppState) -> String {
    let bbox = match args.get("bbox").and_then(|v| v.as_str()) {
        Some(b) => b,
        None => return "Error: Missing required argument 'bbox'".to_string(),
    };
    let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(14) as u32;
    let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("wreck");
    
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let output_path = format!("/tmp/scan_{}.json", timestamp);
    let script_path = "/mnt/data-external/cesarops/repo/scan_engine.py";

    let mut cmd = tokio::process::Command::new("python");
    cmd.arg(script_path)
        .arg("--bbox").arg(bbox)
        .arg("--days").arg(days.to_string())
        .arg("--mode").arg(mode)
        .arg("--output").arg(&output_path);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1800),
        cmd.output()
    ).await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            match std::fs::read_to_string(&output_path) {
                Ok(content) => {
                    let mut s = content;
                    if s.len() > 2000 { s.truncate(2000); }
                    s
                }
                Err(e) => format!("Error reading output file: {}", e),
            }
        }
        Ok(Ok(output)) => format!("Scan failed: {}", String::from_utf8_lossy(&output.stderr)),
        Ok(Err(e)) => format!("Execution error: {}", e),
        Err(_) => "Error: Scan timed out after 30 minutes".to_string(),
    }
}

async fn magnetic_dipole_detect(args: &Value, _state: &AppState) -> String {
    let grid_path = match args.get("grid_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Error: Missing required argument 'grid_path'".to_string(),
    };
    let pixel_size = args.get("pixel_size_m").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32;
    let inner = args.get("inner_radius").and_then(|v| v.as_u64()).unwrap_or(5) as u32;
    let outer = args.get("outer_radius").and_then(|v| v.as_u64()).unwrap_or(15) as u32;

    let binary_path = "/codebase/repos/wreckhunter2000-1/target/release/cesarops-aeromagnetic-worker";

    let mut cmd = tokio::process::Command::new(binary_path);
    cmd.arg("--grid").arg(grid_path)
        .arg("--pixel-size").arg(pixel_size.to_string())
        .arg("--inner").arg(inner.to_string())
        .arg("--outer").arg(outer.to_string());

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        cmd.output()
    ).await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let mut s = String::from_utf8_lossy(&output.stdout).to_string();
            if s.len() > 2000 { s.truncate(2000); }
            s
        }
        Ok(Ok(output)) => format!("Detection error: {}", String::from_utf8_lossy(&output.stderr)),
        Ok(Err(e)) => format!("Execution error: {}", e),
        Err(_) => "Error: Magnetic detection timed out".to_string(),
    }
}

async fn download_satellite_window(args: &Value, _state: &AppState) -> String {
    let bbox = match args.get("bbox").and_then(|v| v.as_str()) {
        Some(b) => b,
        None => return "Error: Missing required argument 'bbox'".to_string(),
    };
    let provider = args.get("provider").and_then(|v| v.as_str()).unwrap_or("auto");
    let days = args.get("days").and_then(|v| v.as_u64()).unwrap_or(14) as u32;
    let output_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("/tmp/cesarops_downloads/");
    
    let script_path = "/mnt/data-external/cesarops/repo/universal_downloader.py";

    let mut cmd = tokio::process::Command::new("python");
    cmd.arg(script_path)
        .arg("--bbox").arg(bbox)
        .arg("--provider").arg(provider)
        .arg("--days").arg(days.to_string())
        .arg("--output-dir").arg(output_dir);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1800),
        cmd.output()
    ).await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let mut s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.len() > 2000 { s.truncate(2000); }
            s
        }
        Ok(Ok(output)) => format!("Download error: {}", String::from_utf8_lossy(&output.stderr)),
        Ok(Err(e)) => format!("Execution error: {}", e),
        Err(_) => "Error: Download timed out after 30 minutes".to_string(),
    }
}

async fn weather_window(args: &Value, _state: &AppState) -> String {
    let bbox = match args.get("bbox").and_then(|v| v.as_str()) {
        Some(b) => b,
        None => return "Error: Missing required argument 'bbox'".to_string(),
    };
    let check = args.get("check").and_then(|v| v.as_str()).unwrap_or("post_storm");
    
    let script_path = "/mnt/data-external/cesarops/repo/weather_service.py";

    let mut cmd = tokio::process::Command::new("python");
    cmd.arg(script_path)
        .arg("--bbox").arg(bbox)
        .arg("--classify").arg(check);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        cmd.output()
    ).await;

    match result {
        Ok(Ok(output)) if output.status.success() => {
            let mut s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if s.len() > 2000 { s.truncate(2000); }
            s
        }
        Ok(Ok(output)) => format!("Weather error: {}", String::from_utf8_lossy(&output.stderr)),
        Ok(Err(e)) => format!("Execution error: {}", e),
        Err(_) => "Error: Weather check timed out".to_string(),
    }
}
```
