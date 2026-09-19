# Task: Complete Rust BAG Scanner with HDF5 Reading + Redaction Detection

You are a senior Rust developer. Rewrite `cesarops-slicer/src/bathymetry_specialist.rs` to be a complete BAG file scanner that reads real NOAA BAG files (HDF5 format) and detects both wreck anomalies AND redaction signatures.

## Current state (skeleton):
- Uses simulated data (vec![-50.0; width*height])
- Has rayon gradient analysis
- Pushes to sled anomaly queue
- No real HDF5 reading

## What to implement:

### 1. Real BAG/HDF5 reading
BAG files are HDF5 with this structure:
- `/BAG_root/elevation` — 2D f32 array of depths (negative = below water)
- `/BAG_root/uncertainty` — 2D f32 array of measurement uncertainty
- `/BAG_root/tracking_list` — metadata about data sources
- `/BAG_root/metadata` — XML metadata with projection info

Use the `hdf5` crate to read these datasets.

### 2. Multi-resolution anomaly detection (from Python version)
- Scan at 3 resolutions: native, 2x downsampled, 4x downsampled
- At each resolution: compute local gradient (Sobel), find pixels > threshold
- Grow anomaly regions (flood-fill connected pixels above threshold)
- Score each region by: size, gradient magnitude, shape complexity, depth variance

### 3. Redaction signature detection (the key feature)
Detect where NOAA has artificially smoothed/removed bathymetry data:

**Smoothing signatures:**
- Unnaturally low local variance in a region surrounded by high variance
- Gaussian-like smoothing kernel artifacts (detectable via frequency analysis)
- Sharp boundary between smoothed and natural terrain

**Removal signatures:**
- Constant-depth patches (all pixels exactly the same value)
- NoData holes surrounded by valid data
- Uncertainty values that don't match the surrounding pattern

**Alteration signatures:**
- Depth values that break the natural gradient continuity
- Statistical outliers in the uncertainty layer
- Tracking list entries that show manual edits

### 4. Output
```json
{
  "file": "H13607_MB_50cm.bag",
  "grid_size": [4000, 4000],
  "resolution_m": 0.5,
  "wreck_candidates": [
    {
      "id": "wc-001",
      "center_lat": 45.816,
      "center_lon": -84.717,
      "size_m2": 450.0,
      "confidence": 0.87,
      "gradient_score": 3.2,
      "depth_m": -32.5,
      "type": "structural_anomaly"
    }
  ],
  "redaction_signatures": [
    {
      "id": "rs-001",
      "center_lat": 45.820,
      "center_lon": -84.720,
      "area_m2": 2500.0,
      "type": "smoothing",
      "confidence": 0.92,
      "evidence": "variance_drop_ratio: 0.05, boundary_sharpness: 0.94"
    }
  ],
  "process_time_ms": 1234
}
```

### 5. CLI interface
```
cesarops-bag-scan --bag /path/to/file.bag \
  --anomaly-threshold 2.5 \
  --redaction-sensitivity 0.6 \
  --output results.json
```

## Dependencies to add to Cargo.toml:
```toml
hdf5 = "0.8"  # HDF5 reading
ndarray = "0.15"  # 2D array ops
rayon = "1.8"  # parallel processing
```

## Constraints:
- Keep under 400 lines
- Use rayon for parallel gradient computation
- Handle missing/NoData values gracefully (BAG uses 1e6 as NoData)
- If hdf5 crate fails to open file, fall back to error message (don't panic)
- Output JSON to stdout (pipe-friendly for forge tool integration)

## Output format:
Complete Rust file for `bathymetry_specialist.rs`. No Cargo.toml changes (just the .rs file).
