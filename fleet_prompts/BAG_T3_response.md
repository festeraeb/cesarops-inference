```toml
[package]
name = "cesarops-bag-scan"
version = "0.1.0"
edition = "2021"

[dependencies]
gdal = "0.17"
rayon = "1.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

```rust
use clap::Parser;
use gdal::Dataset;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .bag file
    path: String,

    /// Z-score threshold for anomaly detection
    #[arg(short, long, default_value_t = 2.5)]
    threshold: f32,

    /// Sensitivity for redaction detection (0.0 to 1.0)
    #[arg(short, long, default_value_t = 0.6)]
    redaction_sensitivity: f32,
}

#[derive(Serialize)]
struct WreckCandidate {
    x: f64,
    y: f64,
    size_px: usize,
    avg_depth: f32,
    gradient_mag: f32,
    variance: f32,
}

#[derive(Serialize)]
struct RedactionSignature {
    x: f64,
    y: f64,
    kind: String, // "smoothing", "removal", "alteration"
    area_px: usize,
}

#[derive(Serialize)]
struct ScanResult {
    file: String,
    grid_size: (u32, u32),
    resolution_m: f64,
    nodata_pct: f32,
    wreck_candidates: Vec<WreckCandidate>,
    redaction_signatures: Vec<RedactionSignature>,
    process_time_ms: u128,
}

const NODATA_LIMIT: f32 = 1_000_000.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let start = Instant::now();

    let dataset = Dataset::open(&args.path)?;
    let band = dataset.rasterband(1)?;
    let (width, height) = band.size();
    let transform = dataset.geo_transform()?;
    let resolution = transform[1].abs();

    let data: Vec<f32> = band.read_as::<f32>((0, 0), width, height, None)?;

    let mut nodata_count = 0;
    for &val in &data {
        if val.is_nan() || val > NODATA_LIMIT {
            nodata_count += 1;
        }
    }
    let nodata_pct = (nodata_count as f32 / (width * height) as f32) * 100.0;

    let mut wreck_candidates = Vec::new();
    let mut redaction_signatures = Vec::new();
    let mut visited = vec![false; (width * height) as usize];

    // Multi-resolution scan: 1x, 2x, 4x
    for skip in [1, 2, 4] {
        let mut local_visited = vec![false; (width * height) as usize];
        
        for y in (0..height).step_by(skip as usize) {
            for x in (0..width).step_by(skip as usize) {
                let idx = (y * width + x) as usize;
                if visited[idx] || local_visited[idx] || data[idx] > NODATA_LIMIT || data[idx].is_nan() {
                    continue;
                }

                // Compute local stats (11x11 neighborhood)
                let (mean, std, grad) = compute_local_stats(&data, x, y, width, height, skip);
                let z_score = if std > 0.001 { (data[idx] - mean).abs() / std } else { 0.0 };

                if z_score > args.threshold {
                    // Flood fill to find connected anomaly
                    let (region, area) = flood_fill(&data, x, y, width, height, skip, &mut local_visited, args.threshold);
                    if area > 5 {
                        let (tx, ty) = transform_to_geo(x, y, &transform);
                        wreck_candidates.push(WreckCandidate {
                            x: tx,
                            y: ty,
                            size_px: area,
                            avg_depth: mean,
                            gradient_mag: grad,
                            variance: std * std,
                        });
                        // Mark as visited for higher res passes
                        for &i in &region { visited[i] = true; }
                    }
                } else {
                    // Check for Redaction: Smoothing/Removal
                    if std < 0.001 { // Constant depth (Removal)
                        let (tx, ty) = transform_to_geo(x, y, &transform);
                        redaction_signatures.push(RedactionSignature { x: tx, y: ty, kind: "removal".to_string(), area_px: 1 });
                    } else if std < 0.05 * mean.abs().max(1.0) { // Smoothing
                        let (tx, ty) = transform_to_geo(x, y, &transform);
                        redaction_signatures.push(RedactionSignature { x: tx, y: ty, kind: "smoothing".to_string(), area_px: 1 });
                    }
                }
            }
        }
    }

    let result = ScanResult {
        file: args.path.clone(),
        grid_size: (width, height),
        resolution_m: resolution,
        nodata_pct,
        wreck_candidates,
        redaction_signatures,
        process_time_ms: start.elapsed().as_millis(),
    };

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn compute_local_stats(data: &[f32], cx: u32, cy: u32, w: u32, h: u32, skip: u32) -> (f32, f32, f32) {
    let mut sum = 0.0;
    let mut count = 0;
    let r = 5 * skip;

    for dy in -(r as i32)..=(r as i32) {
        for dx in -(r as i32)..=(r as i32) {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let val = data[(ny as u32 * w + nx as u32) as usize];
                if val < NODATA_LIMIT {
                    sum += val;
                    count += 1;
                }
            }
        }
    }

    if count == 0 { return (0.0, 0.0, 0.0); }
    let mean = sum / count as f32;
    let mut var_sum = 0.0;
    for dy in -(r as i32)..=(r as i32) {
        for dx in -(r as i32)..=(r as i32) {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let val = data[(ny as u32 * w + nx as u32) as usize];
                if val < NODATA_LIMIT {
                    var_sum += (val - mean).powi(2);
                }
            }
        }
    }
    let std = (var_sum / count as f32).sqrt();
    // Simple gradient approximation
    let grad = if cx > 0 && cx < w - 1 { (data[(cy as u32 * w + cx + 1) as usize] - data[(cy as u32 * w + cx - 1) as usize]).abs() } else { 0.0 };

    (mean, std, grad)
}

fn flood_fill(data: &[f32], sx: u32, sy: u32, w: u32, h: u32, skip: u32, visited: &mut [bool], thresh: f32) -> (Vec<usize>, usize) {
    let mut stack = vec![(sx, sy)];
    let mut region = Vec::new();
    let mut count = 0;

    while let Some((x, y)) = stack.pop() {
        let idx = (y * w + x) as usize;
        if visited[idx] || data[idx] > NODATA_LIMIT || data[idx].is_nan() { continue; }
        
        visited[idx] = true;
        region.push(idx);
        count += 1;

        for dy in [-(skip as i32), 0, skip as i32] {
            for dx in [-(skip as i32), 0, skip as i32] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    let n_idx = (ny as u32 * w + nx as u32) as usize;
                    if !visited[n_idx] && data[n_idx] < NODATA_LIMIT {
                        let (m, s, _) = compute_local_stats(data, nx as u32, ny as u32, w, h, skip);
                        if (data[n_idx] - m).abs() / (s.max(0.1)) < thresh {
                            stack.push((nx as u32, ny as u32));
                        }
                    }
                }
            }
        }
    }
    (region, count)
}

fn transform_to_geo(x: u32, y: u32, transform: &[f64; 6]) -> (f64, f64) {
    let gx = transform[0] + x as f64 * transform[1] + y as f64 * transform[2];
    let gy = transform[3] + x as f64 * transform[4] + y as f64 * transform[5];
    (gx, gy)
}
```
