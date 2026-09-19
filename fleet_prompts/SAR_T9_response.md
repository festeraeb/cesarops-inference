# Design Document: `cesarops-orchestrator`

**Project:** CESAROPS Mission Orchestration Layer  
**Status:** Draft / Architecture Specification  
**Target Hardware:** Pascal P100 (16GB), RTX 3060/4060 (12GB/8GB), Coral TPU, CPU  
**Objective:** Replace high-latency n8n/R1 routing with a native, hardware-aware Rust orchestration engine capable of dynamic cluster retooling and dual-stack tile stitching.

---

## 1. Architecture Diagram

The orchestrator acts as the "Brain" of the fleet, sitting between the Operator's intent and the physical hardware execution. It transforms unstructured text into a deterministic execution graph.

```text
[ OPERATOR ]
      |
      | (1) OperatorScenario (Text + BBox)
      v
+-----------------------------------------------------------------------+
|                         CESAROPS-ORCHESTRATOR                         |
|                                                                       |
|  +-------------------+       +-------------------------------------+  |
|  |  1. INTAKE ENGINE | ----> |  2. MISSION PLANNER (LLM-Driven)   |  |
|  | (NLP/Intent Parser)|       | (Scenario Class + Routing Logic)   |  |
|  +-------------------+       +-------------------------------------+  |
|                                         |                              |
|  +-------------------+       +----------v----------+  +------------+  |
|  | 4. DISPATCHER     | <---- | 3. RETOOL ENGINE    |  | 5. AGGREGATOR|  |
|  | (Fan-out/Fan-in)  |       | (VRAM/Worker Mgmt)  |  | (Final Report)|  |
|  +-------------------+       +---------------------+  +------------+  |
+----------|---------------------------------------------------|--------+
           |                                                   |
           | (6) EXECUTION PATHS                               | (7) RESULTS
           |                                                   |
    +------v-------+      +-------------------+      +---------v-------+
    | LOCAL COMPUTE|      | REMOTE SPECIALIST |      | CLUSTER STATE   |
    | (P100/wgpu)  |      | (1060/1070/HTTP)  |      | (Forge Registry)|
    +--------------+      +-------------------+      +-----------------+
    | - SiltMasker |      | - Vision (Florence)|      | - VRAM Headroom |
    | - Mag_Dipole |      | - LLM (Moondream)  |      | - Worker Load   |
    | - Stitcher   |      | - SAR (Temporal)   |      |                 |
    +--------------+      +-------------------+      +-----------------+
```

---

## 2. Cluster Retooling State Machine

To execute high-density tile compute (WreckHunt/DownedAircraft), the orchestrator must forcibly reallocate P100 VRAM from LLM reasoning to tile-processing.

**States:**
1.  **IDLE**: Cluster operating in standard "Reasoning Mode" (LLMs active on P100s).
2.  **PROBING**: Querying `forge/cluster/status` to map current VRAM/Model occupancy.
3.  **RETOOL_REQUESTED**: Issuing `stop` commands to non-essential LLM workers on target P100s.
4.  **TRANSITIONING**: Waiting for `SIGTERM` confirmation and VRAM release.
5.  **VERIFYING**: Checking `wgpu` device availability and VRAM headroom on target P100s.
6.  **ACTIVE_MISSION**: Mission-specific compute mode engaged.
7.  **ROLLBACK**: Triggered if VRAM fails to clear or a critical worker hangs.

**Transitions & Rollback:**
- **IDLE $\rightarrow$ PROBING**: Triggered by `intake()`.
- **PROBING $\rightarrow$ RETOOL_REQUESTED**: If `ScenarioClass` requires P100 tile-compute.
- **RETOOL_REQUESTED $\rightarrow$ VERIFYING**: On successful `stop` signal from `forge`.
- **VERIFYING $\rightarrow$ ACTIVE_MISSION**: If VRAM $\ge$ 14GB available.
- **ANY $\rightarrow$ ROLLBACK**: If `VERIFYING` fails (e.g., LLM process zombie).
- **ROLLBACK $\rightarrow$ IDLE**: Re-spawning LLM workers on 1060/1070 fallback cards to restore baseline reasoning.

---

## 3. Dual-Stack-of-10 Stitching Strategy

The current `satellite_stitch.rs` handles sub-pixel alignment for a single sequence. However, as $N$ increases, the cumulative drift error $\epsilon_{total} = \sum_{i=1}^{n} \epsilon_i$ grows linearly.

**The Math of Error Reduction:**
- **Standard Approach (1x20):** 20 tiles aligned sequentially. Total drift error is the sum of 19 alignment steps.
- **Dual-Stack Approach (2x10):** 
    - Stack A: 10 tiles aligned to Master Tile 1.
    - Stack B: 10 tiles aligned to Master Tile 11.
    - **Merge:** The two stacks are merged using a `MeanOfMeans` approach.
- **Error Profile:** The maximum drift in any single stack is capped at 10 tiles. By merging two independent stacks, the stochastic error of the alignment is averaged, effectively halving the expected drift error $\mathbb{E}[\epsilon]$ compared to a single 20-tile chain.

**Implementation via `satellite_stitch.rs`:**
1.  **Phase 1 (Local):** `dispatch_modules` triggers `satellite_stitch.rs` twice in parallel on the P100.
2.  **Phase 2 (Compute):** 
    - `Stack_A = stitch(tiles[0..10], master=0)`
    - `Stack_B = stitch(tiles[10..20], master=10)`
3.  **Phase 3 (Merge):** The orchestrator calls a `merge_stacks(Stack_A, Stack_B, method=MeanOfMeans)` function. This performs a weighted average of the pixel intensities at the overlap boundary, using the confidence score from the sub-pixel correction as the weight.

---

## 4. Specialist Routing Table

The orchestrator uses a tiered fallback system to ensure mission continuity even if high-end hardware is saturated.

| Module Type | Primary Specialist (Target) | Secondary Fallback (1) | Tertiary Fallback (2) | Compute Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Silt/Turbidity** | CORAL TPU (Int8) | CPU (SIMD) | N/A | Local |
| **Optical Vision** | Florence-2 (P100) | Moondream2 (1060) | CPU (OpenVINO) | Remote |
| **SAR Temporal** | P100 (Tile Compute) | 1070 (Low-res) | CPU | Local |
| **Mag-Dipole** | P100 (wgpu) | 1060 (wgpu) | CPU | Local |
| **LLM Reasoning** | P100 (Full) | 1070 (Quantized) | 1060 (TinyLlama) | Remote |
| **Stitching** | P100 (wgpu) | 1070 (wgpu) | N/A | Local |

---

## 5. Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **P100 Retool Failure** | LLM process won't die; VRAM full. | **Hard Reset:** Orchestrator issues `SIGKILL` via Forge; if fails, triggers `ROLLBACK` and re-routes all tasks to 1060/1070. |
| **Worker LLM Timeout** | Mission stalls at reasoning phase. | **Circuit Breaker:** If 30s pass without response, skip reasoning and use "Heuristic Default" (e.g., assume standard search pattern). |
| **GeoTIFF Download Fail** | Missing data for tiles. | **Partial Execution:** Proceed with available tiles; mark `StitchingResult` as "Incomplete/Gap-Detected". |
| **Malformed Output** | JSON/Protobuf error from specialist. | **Schema Validation:** Orchestrator validates output against `ModuleSpec`. If invalid, retry once; if fails again, mark module as `FAILED`. |
| **Stitching Drift** | Visual misalignment in merge. | **Confidence Threshold:** If `satellite_stitch.rs` reports error $> \text{threshold}$, discard merge and return single-stack result with warning. |

---

## 6. Integration with `forge` Codebase

The `cesarops-orchestrator` is **not** a replacement for `loop_engine.rs`, but a **wrapper/supervisor**.

- **Current Flow:** `loop_engine.rs` $\rightarrow$ `tool_registry` $\rightarrow$ `execution`.
- **New Flow:** `loop_engine.rs` $\rightarrow$ **`cesarops-orchestrator`** $\rightarrow$ `tool_registry` $\rightarrow$ `execution`.

**Integration Points:**
1.  **`forge/api/cluster.rs`**: Orchestrator will call new endpoints for `stop_worker` and `get_vram_headroom`.
2.  **`forge/engine/loop_engine.rs`**: The loop engine will now receive a `MissionPlan` from the orchestrator instead of raw text, allowing it to execute pre-determined steps rather than "guessing" via R1.
3.  **`forge/tools/registry.rs`**: The orchestrator will use the existing `DelegateTarget` pattern to resolve where to send `ModuleSpec` requests.

---

## 7. Test Scenarios

### Scenario A: "Find me a 19th-century schooner around Sleeping Bear Dunes, Lake Michigan"
- **Class:** `WreckHunt`
- **Expected Trace:**
    1. `intake()` $\rightarrow$ `ScenarioClass::WreckHunt`.
    2. `retool_cluster()` $\rightarrow$ Kill LLMs on P100 $\rightarrow$ Load `SiltMasker` and `Mag_Dipole`.
    3. `dispatch_modules()` $\rightarrow$ Run `Mag_Dipole` (Local) $\rightarrow$ Run `Optical_Vision` (Remote 1060).
    4. `stitching()` $\rightarrow$ Execute 2x10 dual-stack on P100.
    5. `report()` $\rightarrow$ "Schooner detected via magnetic anomaly and visual silhouette."

### Scenario B: "Search and rescue: missing fishing boat last seen near Whitefish Point"
- **Class:** `SearchRescue`
- **Expected Trace:**
    1. `intake()` $\rightarrow$ `ScenarioClass::SearchRescue`.
    2. `retool_cluster()` $\rightarrow$ No P100 retool needed (keep LLM active for real-time reasoning).
    3. `dispatch_modules()` $\rightarrow$ Run `SAR_temporal_diff` (Remote 1070) $\rightarrow$ Run `Vision` (Remote 1060).
    4. `report()` $\rightarrow$ "No vessel found in SAR temporal diff; visual scan clear."

### Scenario C: "Locate a downed Cessna 172 in eastern Lake Erie south of Buffalo"
- **Class:** `DownedAircraft`
- **Expected Trace:**
    1. `intake()` $\rightarrow$ `ScenarioClass::DownedAircraft`.
    2. `retool_cluster()` $\rightarrow$ Kill LLMs on P100 $\rightarrow$ Load `Optical_Mass` and `Satellite_Stitch`.
    3. `dispatch_modules()` $\rightarrow$ Run `Optical_Mass` (Local) $\rightarrow$ Run `Vision` (Remote 1060).
    4. `stitching()` $\rightarrow$ Execute 2x10 dual-stack on P100.
    5. `report()` $\rightarrow$ "Cessna 172 wreckage identified via high-res optical mass detection."
