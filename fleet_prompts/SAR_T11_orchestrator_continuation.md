CONTINUATION PROMPT — finish src/orchestrator.rs after a prior dispatch was truncated.

You previously started `cesarops-forge-v2/src/orchestrator.rs` but the response was cut off mid-struct (~2.2 KB of types only, no functions, no handlers). The full design is locked in (see fleet_prompts/SAR_T9_response.md for architecture).

## What was already produced (re-emit verbatim, then continue)

```rust
use axum::Json;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;
use tokio::time::{timeout, Duration};

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
```

## What's missing — write all of this NOW

The `MissionReport` struct was being emitted when truncation happened. Continue from there. Required:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReport {
    pub scenario_class: ScenarioClass,
    pub modules: Vec<ModuleResult>,
    pub stitching_summary: Option<String>,
    pub runtime_seconds: f32,
    pub status: String,  // "ok" | "partial" | "failed"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StitchingStrategy {
    pub n_tiles: usize,
    pub stacks_per_p100: usize,    // 2 for dual-stack-of-10
    pub tiles_per_stack: usize,    // 10
    pub merge_method: String,      // "mean_of_means" | "median_merge" | "weighted_confidence"
}
```

Then the functions, all `pub async fn ...`:

1. **`probe_cluster() -> ClusterSnapshot`** — POST to `http://127.0.0.1:9100/cluster/discover` (the forge's existing discovery endpoint), parse the JSON response into `Vec<WorkerState>`. Each entry has name, model_type from probe response info, current_task = None for now, load = 50.0 placeholder.

2. **`plan_from_scenario(scenario: &OperatorScenario) -> Result<MissionPlan, String>`** — POST to `http://127.0.0.1:5001/api/v1/generate` (Gemma-4-MoE). Build a planning prompt instructing the LLM to emit JSON with this shape:
   ```json
   {
     "scenario_class": "WreckHunt",
     "modules": [
       {"id":"weather_window","name":"weather","delegate":"Cpu","bbox":[lat1,lon1,lat2,lon2]},
       {"id":"satellite_download","name":"satellite_download","delegate":"Cpu","bbox":[...]},
       {"id":"thermocline_jitter","name":"thermocline","delegate":"VulkanGpu","bbox":[...]},
       {"id":"magnetic_dipole","name":"mag_dipole","delegate":"VulkanGpu","bbox":[...]},
       {"id":"galvanic_plume","name":"galvanic","delegate":"VulkanGpu","bbox":[...]},
       {"id":"triple_lock","name":"detection","delegate":"Hybrid","bbox":[...]}
     ],
     "stitching": {"n_tiles":20,"stacks_per_p100":2,"tiles_per_stack":10,"merge_method":"mean_of_means"}
   }
   ```
   Parse strictly. If JSON fails, retry once with the same prompt.

3. **`assign_specialists(modules: &[ModuleSpec], cluster: &ClusterSnapshot) -> Vec<RouteAssignment>`** — for each module: VulkanGpu → endpoint `"rust://cesarops_inference/{module_name}"`. Cpu + LLM-needs → pick least-loaded online worker preferring 1060/1070/P1000 over P100s. CoralTpuInt8 → fallback to VulkanGpu with comment in fallback_endpoints. Hybrid → split into two assignments.

4. **`retool_for_mission(plan: &MissionPlan) -> Result<RetoolReceipt, String>`** — when scenario_class is WreckHunt or DownedAircraft AND a tile-compute module exists in plan: POST to `http://127.0.0.1:9100/cluster/worker/GemmaBig/stop` to free P100#0. Return RetoolReceipt with name="GemmaBig" and previous_config=current model path. For non-SAR scenarios: return Ok with empty receipt (no-op).

5. **`restore_cluster(receipt: &RetoolReceipt) -> Result<(), String>`** — POST to `http://127.0.0.1:9100/cluster/worker/{receipt.worker_name}/start` to resurrect the worker.

6. **`dispatch_modules(plan: &MissionPlan, routing: &[RouteAssignment]) -> Vec<ModuleResult>`** — use `tokio::task::JoinSet` to fan out. For each route assignment:
   - If endpoint starts with `"rust://cesarops_inference/"` → for v1, return placeholder `ModuleResult{ status: "ok", data: Some("placeholder, wire as cesarops-inference dep in v2") }` and log a TODO with the path.
   - If HTTP endpoint → POST to `/api/v1/generate` with a domain-specific prompt template per module.id.
   - Wrap each task with `tokio::time::timeout(Duration::from_secs(60), ...)`. Timeouts → status="failed".

7. **`execute_mission(scenario: OperatorScenario) -> MissionReport`** — glue: probe_cluster → plan_from_scenario → assign_specialists → retool_for_mission → dispatch_modules → restore_cluster → aggregate ModuleResults into MissionReport.

8. **Three handler functions** with correct axum 0.8 signatures:
   ```rust
   pub async fn orchestrator_probe() -> Json<ClusterSnapshot>;
   pub async fn orchestrator_plan(Json(scenario): Json<OperatorScenario>) -> Json<serde_json::Value>;
   pub async fn orchestrator_execute(Json(scenario): Json<OperatorScenario>) -> Json<MissionReport>;
   ```
   `orchestrator_plan` returns `{"plan": MissionPlan}` or `{"error": "..."}`.

## Constraints unchanged

- No new Cargo.toml deps (existing forge deps only)
- All public functions return Result<_, String> for error reporting
- `parking_lot::RwLock<Arc<...>>` if shared state needed (none expected at v1)
- Match agent_dispatch.rs and validator.rs style
- ~400-500 LOC for the missing portion

## Output format

Single block:

```
=== FILE: cesarops-forge-v2/src/orchestrator.rs ===
// FULL FILE — re-emit the types from above + everything new
// Don't elide, don't placeholder. The file goes straight to disk.

=== DIFF: cesarops-forge-v2/src/main.rs ===
+ mod orchestrator;
+ ... 3 new routes
```

Begin now. Code only.
