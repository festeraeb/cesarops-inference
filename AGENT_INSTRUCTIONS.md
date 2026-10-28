# AGENT INSTRUCTIONS: cesarops-inference Multi-Sensor Vision & Bayesian Arbitration Migration

This document provides exact instructions, file references, line numbers, and architectural patterns for the local agent tasked with integrating raster vision inference and multi-sensor Bayesian fusion into the modern Rust engine of `cesarops-inference`.

---

## 1. Context & Architectural Goal

`cesarops-inference` contains an advanced Rust compute engine for tensor dispatch, MoE routing, and attention kernels (e.g. [`src/moe_expert_loader.rs`](file:///codebase/repos/cesarops-inference/src/moe_expert_loader.rs) and [`src/attention_dispatch.rs`](file:///codebase/repos/cesarops-inference/src/attention_dispatch.rs)).

However, the repository currently lacks the marine computer vision and multi-sensor arbitration pipeline present in the legacy Python scripts:
1. Multibeam bathymetry raster patch inference using ONNX Runtime from [`noaa_mbes_shipwreckfinder_model/inference_onnx.py`](file:///codebase/repos/noaa_mbes_shipwreckfinder_model/inference_onnx.py).
2. Multi-sensor Bayesian probability arbitration from [`Detection-Analysis-and-Refinement/scripts/eval_physics_and_clusters.py`](file:///codebase/repos/Detection-Analysis-and-Refinement/scripts/eval_physics_and_clusters.py).

---

## 2. Legacy Code Sections to Extract & Convert to Rust

### A. Multibeam Bathymetry ONNX Model Inference
- **Legacy Source**: [`noaa_mbes_shipwreckfinder_model/inference_onnx.py`](file:///codebase/repos/noaa_mbes_shipwreckfinder_model/inference_onnx.py)
- **Line Numbers**: Lines 25 – 85 (`def run_inference_on_tile`)
- **What it Does**:
  - Ingests bathymetric depth tiles (e.g. NOAA National Centers for Environmental Information 1m/2m bathymetry).
  - Normalizes depth values, extracts 256x256 sliding window patches with 50% overlap.
  - Feeds patches into the trained UNet / HRNet ONNX model (`model.onnx`).
  - Merges patch predictions with Gaussian spatial blending to output a georeferenced shipwreck probability mask.
- **Agent Task for Rust Conversion**:
  - Add `ort` (ONNX Runtime) or `tract-onnx` dependency to `Cargo.toml`.
  - Create `src/vision_bathymetry.rs`:
    ```rust
    pub struct BathymetryInferenceEngine {
        session: ort::Session,
    }

    pub struct DetectionBoundingBox {
        pub min_lat: f64,
        pub max_lat: f64,
        pub min_lon: f64,
        pub max_lon: f64,
        pub max_probability: f32,
        pub area_sq_meters: f32,
    }

    impl BathymetryInferenceEngine {
        pub fn infer_raster_patch(&self, patch: &[f32], width: usize, height: usize) -> Result<Vec<f32>, InferenceError>;
    }
    ```

---

### B. Multi-Sensor Bayesian Probability Arbitration
- **Legacy Source**: [`Detection-Analysis-and-Refinement/scripts/eval_physics_and_clusters.py`](file:///codebase/repos/Detection-Analysis-and-Refinement/scripts/eval_physics_and_clusters.py)
- **Line Numbers**: Lines 30 – 95 (`def fuse_sensor_probabilities`)
- **What it Does**:
  - In marine search, individual sensors (optical satellite, aeromagnetics, sidescan sonar, thermal infrared) have distinct false-alarm probabilities:
    - Aeromagnetics detects ferrous steel mass, blind to wooden hulls.
    - Side-scan sonar detects physical seabed relief, sensitive to rock outcroppings.
    - Thermal infrared detects cold-water upwelling sinks, sensitive to seasonal stratification.
  - Combines sensor likelihoods using independent Bayesian evidence updating:
    $$P(\text{Wreck} | S_1, \dots, S_n) = \frac{P_0 \prod_{i=1}^n \frac{P(S_i | \text{Wreck})}{P(S_i | \text{SeaFloor})}}{P_0 \prod_{i=1}^n \frac{P(S_i | \text{Wreck})}{P(S_i | \text{SeaFloor})} + (1 - P_0)}$$
- **Agent Task for Rust Conversion**:
  - Create `src/bayesian_fusion.rs`:
    ```rust
    #[derive(Debug, Clone)]
    pub struct SensorObservation {
        pub sensor_family: String,
        pub raw_score: f32,
        pub spatial_offset_m: f32,
    }

    pub fn compute_fused_wreck_probability(
        prior_p0: f64,
        observations: &[SensorObservation],
    ) -> f64 {
        let mut log_odds = (prior_p0 / (1.0 - prior_p0)).ln();
        for obs in observations {
            let likelihood_ratio = match obs.sensor_family.to_lowercase().as_str() {
                "aeromag" => (obs.raw_score as f64 * 4.5).max(1.0),
                "sonar" => (obs.raw_score as f64 * 6.0).max(1.0),
                "thermal" => (obs.raw_score as f64 * 2.2).max(1.0),
                "optical" => (obs.raw_score as f64 * 1.8).max(1.0),
                _ => 1.0,
            };
            // Distance decay penalty
            let distance_penalty = (-obs.spatial_offset_m as f64 / 250.0).exp();
            log_odds += (1.0 + (likelihood_ratio - 1.0) * distance_penalty).ln();
        }
        1.0 / (1.0 + (-log_odds).exp())
    }
    ```
