You are an architect designing the dynamic orchestration layer for the cesarops.com SAR/wreck-detection platform.

## CESAROPS MISSION (the why)

cesarops.com is a dual-use remote-sensing platform. Same toolchain serves:
1. Wreck hunting — detect sunken vessels in 500ft of water from satellite
2. Search & rescue — locate missing vessels / aircraft in same waters
3. Downed aircraft — magnetic + glint paint + thermal signatures

The physics doesn't change between schooner and Cessna — only the validation database (known_wrecks.json vs known_aircraft.json) and spectral priors. This is the unique insight: the toolchain is mission-agnostic.

The user experience target:
> "User sits and types the scenario of what they're looking for. That hits an API. An LLM decides what tools we have, retools the cluster if needed, clears the P100s for tile and mag work, then routes individual tasks to specialists. Worker LLMs run on whatever is available (1060/1070/P1000 typically; P100s freed up for compute)."

## CURRENT STATE — what's built

### Forge layer (cesarops-forge-v2 on port 9100)
- Web UI + chat at /
- Cluster panel at /cluster (worker control + Wreck Detection / SAR section)
- 14 chat tools: write_file, read_file, cargo_check, run_command, think_harder, remember, speed_check, scan_region, magnetic_dipole_detect, download_satellite_window, weather_window, detection_health, detection_scan, detection_poll
- Direct invocation: POST /tool/{name} { "arguments": {...} }
- Coding-mode toggle: POST /mode/coding swaps cluster to layered code review (Qwen-MoE coder + Gemma reviewer)
- Tool-routing translator + corrector cascade (TinyLlama on P1000 fallback)
- Live cluster discovery via tailscale + LAN HTTP probes

### Detection physics (cesarops-inference, all building, all tested)
- optical_mass.rs — thermocline jitter detection (7/7 tests)
- magnetic_eraser.rs — sub-nT residual + dipole extraction (7/7 tests)
- galvanic_battery.rs — temporal plume detection (7/7 tests)
- satellite_stitch.rs — sub-pixel FFT phase correlation drift correction (6/6 tests)
- nauticuvs-full integrated (f64 curvelet, INTERNAL only — never crates.io)

### Triple-lock detection service (cesarops-detection on port 5580, RUNNING NOW)
- POST /scan, GET /scan/{id}, GET /workers, GET /health
- Pipeline: Scout (Florence-2 / 1060) → Validator (Moondream2 / P1000) → Jitter (TPU)
- Degrades gracefully to 2-lock when jitter VM offline
- Exposed to forge AI via detection_scan + detection_poll tools

### Slicer + tile mission spec (cesarops-slicer at /mnt/data-external/cesarops/repo/cesarops-slicer/)
- src/spec/mission.rs — JSON mission spec format (Qwen knob-turner)
- src/spec/delegate.rs — DelegateTarget enum: CPU / CoralTpuInt8 / VulkanGpu / Hybrid / Skip
- Specialists: thermal_specialist.rs, optical_structural_worker.rs, sar_slick_worker.rs, research_ingestion_specialist.rs
- vrt_slicer.rs + slicer.rs for tile slicing
- Spec format example:
  {
    "mission_id": "LAKE_MI_MONSTER_001",
    "target_ref": "Andoste_Vicinity",
    "search_params": { "bounds": [...], "depth_target_feet": 450 },
    "modules": [{ "id": "SiltMasker_B4_B3", "mode": "TRANSPARENCY_HOLE", "delegate": "CORAL_TPU_INT8", "params": {...}, "roi": {...} }],
    "output_strategy": { "sync_node": "PI_JANITOR_SYNCTHING", "storage": "...", "frontend": "..." }
  }

### Cluster fleet (cluster_config.toml)
- T440 (local): 2x P100 16GB — currently Gemma-4-26B-MoE on :5001 + Qwen3.6-35B-A3B on :5002
- cesarops2 (10.0.0.129): GTX 1070 8GB (FortyTwo Rust 14B on :5200) + Quadro P1000 4GB (TinyLlama on :5571)
- cesarops3 (10.0.0.41): GTX 1060 6GB / P106-100 (DeepSeek-R1-7B on :5100)
- ThinkPad M2200 (Tailscale, mobile): TinyLlama validator on :5571
- All workers expose KoboldCPP-compatible /api/v1/generate

### n8n workflows on disk (legacy patterns to preserve)
- n8n_moe_tool_router.json: webhook → parse tool call → IF route → execute → return
- n8n_r1_worker.json: autonomous loop (call LLM → parse → execute tool → loop until done)
- These show the orchestration shape that worked before — we want to fold the same logic into native Rust now

## OPERATOR'S DESIGN GOALS

1. **User flow**: chat input → forge AI receives → AI decides scenario class → AI plans which tools/specialists to invoke → AI retools the cluster (swap models, free P100s for tile/mag work) → AI routes subtasks to specialist LLMs by domain → results aggregated back

2. **Cluster retooling rule**: when a SAR scan task is detected, P100s should swap from coding LLMs (Gemma/Qwen) to tile compute. The smaller cards (1060/1070/P1000) handle worker-LLM duties (validation, draft generation, sub-task reasoning).

3. **Dynamic worker selection**: must pick whatever LLM is currently available. If 1060 is offline, route 1060 work to 1070. If everything is busy, queue.

4. **Tile-stitching innovation**: P100 has ~16GB VRAM. With aggressive packing, 10 full geotiffs fit per P100 in fp16. Operator's idea: instead of stitching 20 sequential tiles into one drift-corrected stack (which compounds drift error), do 2 parallel stacks of 10, then merge the two means. Per stack: 10 tiles aligned to a master, sub-pixel correction via satellite_stitch.rs (already built). Cuts drift error in half.

5. **Specialist domains** (slicer convention): each "module" in the mission spec gets a delegate target. Examples:
   - SiltMasker -> CORAL_TPU_INT8 (when available) or CPU
   - SAR_temporal_diff -> P100 (tile compute)
   - Magnetic dipole detection -> aeromagnetic-worker (Pascal P100 wgpu)
   - Vision validation -> Florence-2 / Moondream2 (1060 / P1000)
   - LLM reasoning -> route to whichever 1060/1070/P1000 is least loaded

## YOUR TASK

Design a `cesarops-orchestrator` module that:

### 1. Mission Intake (the entry point)

```rust
pub struct OperatorScenario {
    pub raw_text: String,         // "I'm looking for a B-29 lost in Lake Huron 1948"
    pub bbox: Option<(f64, f64, f64, f64)>,  // optional manual override
    pub priority: Priority,       // Realtime | Background | Research
}

pub enum Priority { Realtime, Background, Research }

pub fn intake(scenario: OperatorScenario) -> MissionPlan;
```

### 2. Mission Plan (what the AI produces)

Same shape as the slicer's MissionSpec but extended with:
- A "cluster_state" field showing which LLMs/cards are available NOW
- A "specialist_routing" field mapping each module → which worker endpoint will run it
- A "stitching_strategy" field for the dual-stack-of-10 plan when prefill batch fits

```rust
pub struct MissionPlan {
    pub mission_id: String,
    pub scenario_class: ScenarioClass,  // WreckHunt | DownedAircraft | SearchRescue | OilSpill | Custom
    pub target_ref: String,
    pub search_params: SearchParams,
    pub modules: Vec<ModuleSpec>,        // from slicer
    pub specialist_routing: Vec<RouteAssignment>,
    pub cluster_state: ClusterSnapshot,
    pub stitching_strategy: StitchingStrategy,
    pub expected_runtime_minutes: f32,
}

pub struct StitchingStrategy {
    pub n_tiles: usize,                  // total tiles in window
    pub stacks_per_p100: usize,          // 2 for the dual-stack-of-10 trick
    pub tiles_per_stack: usize,          // 10
    pub merge_method: MergeMethod,       // MeanOfMeans | MedianMerge | WeightedConfidence
}

pub struct RouteAssignment {
    pub module_id: String,
    pub specialist_endpoint: String,     // http://10.0.0.129:5571 etc.
    pub specialist_model: String,        // TinyLlama / Florence-2 / Moondream2
    pub fallback_endpoints: Vec<String>, // ordered cascade
    pub estimated_load_pct: f32,
}
```

### 3. Cluster Retooling Logic

When the scenario class = WreckHunt or DownedAircraft, the cluster needs P100s freed for tile compute. The orchestrator:

```rust
pub async fn retool_cluster_for_mission(plan: &MissionPlan) -> Result<RetoolResult, RetoolError>;
```

Steps:
- Probe current cluster state (which LLMs running where)
- Identify which P100 workers need to swap to tile-compute mode
- Issue forge /cluster/worker/{name}/stop for those LLMs
- Route the displaced LLM workloads to smaller cards (which the assignment already calculated)
- Wait for swap completion + verify VRAM headroom on P100s

### 4. Task Dispatch — fan out to specialists

```rust
pub async fn dispatch_modules(plan: &MissionPlan) -> Vec<ModuleResult>;
```

Each module routes to its specialist endpoint with the appropriate prompt template. Tile-compute modules invoke the local cesarops-inference modules (optical_mass, magnetic_eraser, galvanic_battery, satellite_stitch) directly via Rust calls — not HTTP. Vision modules go to remote endpoints.

### 5. Result Aggregation

Confirmed detections from triple-lock + module results aggregated into a final report:

```rust
pub struct MissionReport {
    pub mission_id: String,
    pub confirmed_detections: Vec<Detection>,
    pub specialist_outputs: Vec<ModuleResult>,
    pub stitching_summary: StitchingResult,
    pub runtime_seconds: f32,
}
```

## DELIVERABLE FROM YOU

Don't write code. Write a **design document** that answers:

1. **Architecture diagram** (ASCII art) showing the data flow from scenario text to mission report
2. **The cluster retooling state machine** — exact states, transitions, and rollback on failure
3. **The dual-stack-of-10 stitching strategy** in detail — how it interacts with the existing satellite_stitch.rs API, including math on why 2x10 reduces drift error vs 1x20
4. **Specialist routing table** mapping each module type → preferred specialist + fallback cascade
5. **Failure modes and mitigations** — what happens when a P100 fails to free, when a worker LLM doesn't respond, when GeoTIFF download fails partway, when a specialist returns malformed output
6. **Where in the existing forge codebase the new orchestrator integrates** — does it sit beside loop_engine.rs? Replace it? Wrap it? Look at the operator's existing tool-routing pattern
7. **Three concrete test scenarios** with expected execution traces:
   - "Find me a 19th-century schooner around Sleeping Bear Dunes, Lake Michigan"
   - "Search and rescue: missing fishing boat last seen near Whitefish Point, north of Sault Ste. Marie"
   - "Locate a downed Cessna 172 in eastern Lake Erie south of Buffalo"

## CONSTRAINTS

- This is a DESIGN doc; no Rust code beyond the type signatures shown above
- Length: 600-1200 lines markdown
- Specific to OUR fleet topology (no generic "any GPU" hand-waving)
- Must reference the slicer's MissionSpec / DelegateTarget pattern as the foundation
- Must explain how this replaces the n8n MoE Tool Router + R1 Worker patterns natively
- Keep all module wiring backward-compatible with the existing forge tool registry

Output format: Markdown, structured under the 7 numbered sections above. Begin now.
