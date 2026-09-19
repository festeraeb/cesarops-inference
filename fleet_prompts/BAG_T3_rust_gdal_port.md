# Task: Port BAG Scanner to Rust using GDAL crate (NOT hdf5)

You are a senior Rust developer. Write a complete BAG file scanner that reads NOAA BAG files using the `gdal` crate (which reads BAG natively, same as Python's rasterio).

## Key insight: BAG files are read by GDAL directly
- `gdal` crate opens .bag files like any raster
- Band 1 = elevation (f32, negative = below water)
- Band 2 = uncertainty (f32)
- NoData value is typically 1000000.0 or NaN

## What to implement:

### 1. Read BAG file via GDAL
```rust
use gdal::Dataset;
let dataset = Dataset::open(path)?;
let band1 = dataset.rasterband(1)?; // elevation
let elevation: Vec<f32> = band1.read_as::<f32>((0,0), band1.size(), band1.size(), None)?;
let (width, height) = band1.size();
let transform = dataset.geo_transform()?; // [origin_x, pixel_w, 0, origin_y, 0, pixel_h]
```

### 2. Multi-resolution anomaly detection
- Scan at 3 skip patterns: 1, 2, 4 (native, 2x, 4x downsampled)
- At each pixel: compute local z-score vs 11x11 neighborhood
- If z-score > threshold (default 2.5): grow connected region via flood-fill
- Score region by: size, gradient magnitude, depth variance

### 3. Redaction signature detection
- **Smoothing**: regions where local variance is < 5% of surrounding variance
- **Removal**: constant-depth patches (std < 0.001 within region)
- **Alteration**: sharp boundary between two statistically different populations

### 4. Output JSON to stdout

### CLI:
```
cesarops-bag-scan /path/to/file.bag --threshold 2.5 --redaction-sensitivity 0.6
```

## Cargo.toml:
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

## Output format (JSON to stdout):
```json
{
  "file": "H10957_MB_1m_MLLW_1of10.bag",
  "grid_size": [4000, 3000],
  "resolution_m": 1.0,
  "nodata_pct": 12.5,
  "wreck_candidates": [...],
  "redaction_signatures": [...],
  "process_time_ms": 450
}
```

## Constraints:
- Use `rayon` for parallel row processing (the main speedup over Python)
- Handle NoData (1000000.0 or > 1e6) by masking
- Pixel-to-geo coordinate conversion using the GeoTransform
- Under 350 lines for main.rs
- Must compile with `gdal = "0.17"` and system libgdal-dev installed

## Output: Complete single-file `src/main.rs` + `Cargo.toml`
