

=== FILE: cesarops-forge-v2/src/orchestrator.rs ===
```rust
use axum::Json;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;
use tokio::time::{timeout, Duration};

// ---------------------------------------------------------------------------
// Domain Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScenarioClass {
    WeatherCheck,
    SatelliteDownload,
    MagneticDipoleDetect,
    ThermoclineJitter,
    GalvanicPlume,
    TripleLockDetection,
    WreckHunt,
    DownedAircraft,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DelegateType {
    VulkanGpu,
    Cpu,
    CoralTpuInt8,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorScenario {
    pub raw_text: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub workers: Vec<WorkerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerState {
    pub name: String,
    pub model_type: String,
    pub current_task: Option<String>,
    pub load: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionPlan {
    pub scenario_class: ScenarioClass,
    pub modules: Vec<ModuleSpec>,
    pub stitching: Option<StitchingStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSpec {
    pub id: String,
    pub name: String,
    pub delegate: DelegateType,
    pub bbox: Option<[f64; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteAssignment {
    pub module_id: String,
    pub endpoint: String,
    pub fallback_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetoolReceipt {
    pub worker_name: String,
    pub previous_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleResult {
    pub module_id: String,
    pub status: String,
    pub data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReport {
    pub scenario_class: Scenario
