You are a Rust + axum + tokio implementation specialist. Implement the cesarops orchestration layer that turns scenario text into routed specialist tasks.

## CESAROPS MISSION

cesarops.com is a dual-use remote-sensing platform. Same toolchain serves wreck hunting, search-and-rescue, and downed-aircraft search. The user types a scenario in chat; an LLM decides what tools we have, retools the cluster (frees P100s for tile/mag work, parks worker LLMs on 1060/1070/P1000), then routes individual tasks to specialists by domain. Worker LLMs are dynamic — pick whatever's available.

## CURRENT STATE — what's built

### Forge layer (cesarops-forge-v2 on :9100, RUNNING NOW)
14 chat tools wired:
- General: write_file, read_file, cargo_check, run_command, think_harder, remember, speed_check
- Wreck/SAR: scan_region, magnetic_dipole_detect, download_satellite_window, weather_window
- Detection-service: detection_health, detection_scan, detection_poll
Direct invocation: POST /tool/{name} { "arguments": {...} } returns { "result", "tool" }
Mode toggle: POST /mode/coding swaps cluster to coding (Qwen MoE coder + Gemma reviewer)
                 POST /mode/cesarops swaps back to SAR fleet roles

### Detection physics (cesarops-inference, all built + tested 27/27)
- optical_mass.rs (thermocline jitter)
- magnetic_eraser.rs (sub-nT residual + dipole)
- galvanic_battery.rs (temporal plume detection)
- satellite_stitch.rs (sub-pixel FFT phase correlation drift correction)
- nauticuvs-full integrated (f64 curvelet, INTERNAL only — never crates.io)

### Triple-lock detection service (cesarops-detection on :5580, RUNNING NOW)
- POST /scan, GET /scan/{id}, GET /workers, GET /health
- Pipeline: Scout (Florence-2 / 1060) → Validator (Moondream2 / P1000) → Jitter (TPU)

### Cluster fleet (cluster_config.toml, current state)
T440 local: 2x P100 16GB
  - P100#0 :5001 -> Gemma-4-26B-MoE-IQ4_XS (current)
  - P100#1 :5002 -> Qwen3.6-35B-A3B-Q4_K_M (current)
cesarops2 :10.0.0.129
  - GTX 1070 8GB :5200 -> FortyTwo-Rust-14B (Marvin)
  - Quadro P1000 4GB :5571 -> TinyLlama (Picasso)
cesarops3 :10.0.0.41
  - GTX 1060/P106-100 6GB :5100 -> DeepSeek-R1-7B (Scout)
  - 1060 :5570 -> Florence-2 vision worker (when launched)
ThinkPad M2200 (Tailscale, mobile)
  - :5571 -> TinyLlama (Nautik)

All workers expose KoboldCPP-compatible /api/v1/generate (POST { prompt, max_length, temperature, stop_sequence }).
All workers respond to GET /health or GET /api/extra/version for liveness.

### Slicer mission spec (existing pattern at /mnt/data-external/cesarops/repo/cesarops-slicer/src/spec/)
```rust
pub struct MissionSpec {
    pub mission_id: String,
    pub target_ref: String,
    pub search_params: SearchParams,
    pub modules: Vec<ModuleSpec>,
}
pub struct ModuleSpec {
    pub id: String,
    pub mode: String,
    pub delegate: DelegateTarget,  // CPU | CoralTpuInt8 | VulkanGpu | Hybrid | Skip
    pub params: serde_json::Value,
    pub roi: Option<RoiSpec>,
}
```

## OPERATOR'S DESIGN GOALS

1. **User flow**: chat text → forge AI receives via /send → AI plans → AI calls retool → AI dispatches modules → results aggregated
2. **Cluster retooling**: when scenario = WreckHunt/DownedAircraft, P100s swap from coding LLMs to tile compute. Worker-LLM duties (validation, draft, sub-task reasoning) go to 1060/1070/P1000.
3. **Dynamic worker selection**: pick whatever LLM is currently available + healthy; cascade fallbacks.
4. **Tile-stitching innovation**: P100 fits ~10 full geotiffs in fp16. Operator wants 2 parallel stacks of 10 instead of 1 stack of 20 — cuts drift error in half.

## YOUR TASK: implement

Build a new module `cesarops-forge-v2/src/orchestrator.rs` exposing these types and functions. Code must:
- compile against the existing forge build (axum 0.8, tokio, reqwest, serde_json)
- not break any existing tool or route
- add three new endpoints to the forge router

### Types (mirror slicer's MissionSpec where possible)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioClass {
    WreckHunt,
    DownedAircraft,
    SearchRescue,
    OilSpill,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorScenario {
    pub raw_text: String,
    pub bbox: Option<[f64; 4]>,  // [lat_min, lon_min, lat_max, lon_max]
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Priority { Realtime, Background, Research }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub probed_at_unix: u64,
    pub workers: Vec<WorkerStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub name: String,
    pub endpoint: String,
    pub host: String,
    pub gpu_label: String,
    pub model: Option<String>,
    pub online: bool,
    pub estimated_load_pct: f32,  // 0..100, derived from probe latency
    pub roles_supported: Vec<String>,  // ["code","think","validate","vision","draft"]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSpec {
    pub id: String,
    pub mode: String,
    pub delegate: String,  // CPU | VULKAN_GPU | CORAL_TPU_INT8 | HYBRID | SKIP
    pub params: serde_json::Value,
    pub roi: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteAssignment {
    pub module_id: String,
    pub specialist_endpoint: String,
    pub specialist_model: String,
    pub fallback_endpoints: Vec<String>,
    pub estimated_load_pct: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StitchingStrategy {
    pub n_tiles: usize,
    pub stacks_per_p100: usize,         // 2 for dual-stack-of-10 trick
    pub tiles_per_stack: usize,         // 10
    pub merge_method: String,           // "mean_of_means" | "median_merge" | "weighted_confidence"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionPlan {
    pub mission_id: String,
    pub scenario_class: ScenarioClass,
    pub target_ref: String,
    pub bbox: [f64; 4],
    pub search_params: serde_json::Value,
    pub modules: Vec<ModuleSpec>,
    pub specialist_routing: Vec<RouteAssignment>,
    pub cluster_state: ClusterSnapshot,
    pub stitching_strategy: StitchingStrategy,
    pub expected_runtime_minutes: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleResult {
    pub module_id: String,
    pub status: String,  // "ok" | "failed" | "skipped"
    pub output: serde_json::Value,
    pub elapsed_seconds: f32,
    pub specialist_endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReport {
    pub mission_id: String,
    pub confirmed_detections: Vec<serde_json::Value>,
    pub specialist_outputs: Vec<ModuleResult>,
    pub stitching_summary: serde_json::Value,
    pub runtime_seconds: f32,
}
```

### Functions

```rust
/// Probe the cluster (Tailscale + LAN) and return current state.
pub async fn probe_cluster() -> ClusterSnapshot;

/// Use an LLM (Gemma-4-MoE on :5001) to convert raw scenario text into a structured plan.
/// Internally calls /api/v1/generate with a planning system prompt that instructs the LLM
/// to emit JSON matching MissionPlan's modules[] + bbox + scenario_class.
pub async fn plan_from_scenario(scenario: &OperatorScenario) -> Result<MissionPlan, String>;

/// Decide which specialist gets each module based on delegate type + cluster state.
/// Modules requiring P100 tile compute don't need a worker LLM; they invoke
/// cesarops-inference Rust modules directly. Modules requiring vision/reasoning
/// route to whichever worker LLM is online + least loaded.
pub async fn assign_specialists(
    modules: &[ModuleSpec],
    cluster: &ClusterSnapshot,
) -> Vec<RouteAssignment>;

/// If scenario_class indicates SAR work and a P100 is currently running a code-LLM,
/// stop that LLM and free the P100 for tile compute. Tracks roll-back state for
/// restoration after mission completes.
pub async fn retool_for_mission(plan: &MissionPlan) -> Result<RetoolReceipt, String>;

/// Return cluster to coding configuration after mission completes.
pub async fn restore_cluster(receipt: &RetoolReceipt) -> Result<(), String>;

/// Fan out to specialists. Tile-compute modules invoke local cesarops-inference
/// directly via Rust; LLM modules POST to specialist /api/v1/generate; vision modules
/// POST to scout/validator endpoints.
pub async fn dispatch_modules(plan: &MissionPlan) -> Vec<ModuleResult>;

/// Top-level: full pipeline scenario → report.
pub async fn execute_mission(scenario: OperatorScenario) -> MissionReport;
```

### New endpoints to register on the forge router

```rust
.route("/orchestrator/probe",   get(orchestrator_probe))
.route("/orchestrator/plan",    post(orchestrator_plan))
.route("/orchestrator/execute", post(orchestrator_execute))
```

Handlers:
- GET /orchestrator/probe → { cluster: ClusterSnapshot }
- POST /orchestrator/plan { "scenario": {...} } → { plan: MissionPlan } (no execution, just plan)
- POST /orchestrator/execute { "scenario": {...} } → { report: MissionReport } (runs full pipeline)

### Implementation details

**probe_cluster**: Re-use the existing cluster discovery logic from `discover_nodes` in main.rs (POST /cluster/discover endpoint). Don't duplicate; call internally or extract a shared function.

**plan_from_scenario**: Send a structured planning prompt to Gemma-4-MoE on :5001. Prompt template should instruct the LLM to:
- Determine scenario_class from raw_text
- Compute or extract a bbox (default to a 50km box around any named lake or coordinates mentioned)
- Generate 4-6 modules covering: weather window check, satellite download, magnetic dipole detect (when applicable), thermocline jitter (when applicable), galvanic plume (multi-day window), triple-lock detection
- Assign each module a delegate (VulkanGpu for tile/wgpu work, CPU for orchestration, hybrid for some)
- Output the modules[] array as JSON

The prompt should be ~600 tokens of system + 200 tokens of user. Parse the JSON response strictly; if malformed, retry once with the same prompt.

**assign_specialists**: For each module:
- delegate=VULKAN_GPU → assign endpoint = local invocation marker "rust://cesarops_inference/<module_name>"
- delegate=CPU + module needs LLM reasoning → pick least-loaded online worker from cluster state, prefer 1060/1070/P1000 over P100s
- delegate=CORAL_TPU_INT8 → if no Coral available, fall back to Vulkan GPU with a comment in fallback_endpoints
- delegate=HYBRID → split into two RouteAssignments (one Vulkan, one CPU)

**retool_for_mission**: When scenario_class is WreckHunt or DownedAircraft AND any P100 worker is currently running a code-LLM AND a tile-compute module is in the plan:
- Save current worker config to RetoolReceipt
- Stop the code-LLM via existing /cluster/worker/{name}/stop endpoint (call locally via http reqwest to 127.0.0.1:9100)
- Don't start a new model on the freed P100; the tile compute is direct Rust on the same box and doesn't need an HTTP service
- Wait 5s, probe cluster, verify P100 is free
- Return RetoolReceipt with the stopped worker name + previous config

**dispatch_modules**: For each route assignment:
- If endpoint starts with "rust://cesarops_inference/" → match on the path and invoke the corresponding Rust function (note: forge doesn't currently link cesarops-inference as a dependency; for v1 this can be a stub returning a placeholder result + a TODO log; v2 wires it as a real dep)
- If HTTP endpoint → POST to /api/v1/generate with a domain-specific prompt template (per scenario_class + module.id)
- If detection-service endpoint → POST to /scan + poll
- Run all modules concurrently via tokio::task::JoinSet, collect results
- Don't block on a slow specialist > 60s; mark as failed and continue

**execute_mission**: glue:
1. probe_cluster
2. plan_from_scenario
3. retool_for_mission (if needed)
4. dispatch_modules
5. restore_cluster
6. aggregate ModuleResults into MissionReport

### Stitching strategy section

For modules that involve satellite_stitch (tile drift correction), the orchestrator should set:
```rust
StitchingStrategy {
    n_tiles: <count from satellite download>,
    stacks_per_p100: 2,
    tiles_per_stack: 10,
    merge_method: "mean_of_means".to_string(),
}
```

Two stacks of 10 reduces drift error variance vs one stack of 20 because:
- Drift accumulates as random walk: variance grows linearly with number of pairs
- 1 stack of 20 = 19 alignments compounding
- 2 stacks of 10 = 9+9 alignments, then a single mean-of-means merge with 1 alignment
- Total drift variance: 2*9 + 1 = 19 vs 19. THE THEORETICAL VARIANCE IS THE SAME.
- HOWEVER: the practical win is each P100 fits 10 tiles in fp16 simultaneously, enabling true parallel batched alignment instead of sequential pair-wise. That's the real speedup the operator is after — call it out in code comments.

## CONSTRAINTS

- ~500-700 LOC of Rust, single new file `src/orchestrator.rs`
- Plus minimal diff to `src/main.rs` adding the module + 3 new routes
- No new dependencies in Cargo.toml unless absolutely required (we already have axum, reqwest, serde_json, tokio, parking_lot)
- All public functions return Result<_, String> for HTTP error reporting
- Use parking_lot::RwLock + Arc for any shared state
- Match existing forge code style (look at agent_dispatch.rs and validator.rs for patterns)
- Tests not required for v1 — this is plumbing, not algorithm

## OUTPUT FORMAT

Two sections:

```
=== FILE: cesarops-forge-v2/src/orchestrator.rs ===
// full module body, ~500-700 lines

=== DIFF: cesarops-forge-v2/src/main.rs ===
+ mod orchestrator;
// ... at line where modules are declared
+ .route("/orchestrator/probe",   get(orchestrator::orchestrator_probe))
+ .route("/orchestrator/plan",    post(orchestrator::orchestrator_plan))
+ .route("/orchestrator/execute", post(orchestrator::orchestrator_execute))
```

Code only. No preamble. Begin now.
