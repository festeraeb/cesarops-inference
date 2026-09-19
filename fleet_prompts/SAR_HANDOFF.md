# SAR Orchestrator — Operator Handoff

**Status as of session end (May 17, ~05:15 local):**

- Forge running on `:9100` with SAR tools + coding-mode toggle + invoke_tool route + Wreck Detection panel. Build pid 713297.
- Detection service running on `:5580`, end-to-end verified earlier (POST /scan returns job_id, GET /scan/{id} reports Completed).
- 27 commits today. All on local master. Push to GitHub is operator's EOD job.
- 4 detection-physics stubs replaced with real implementations (optical_mass + magnetic_eraser + galvanic_battery + satellite_stitch), 27/27 unit tests passing.
- nauticuvs-full revived as INTERNAL (publish=false), aeromagnetic worker fixed.

## What's done on the orchestrator

**T9 — design doc (Gemma-4-MoE on P100#0)** — COMPLETE at `fleet_prompts/SAR_T9_response.md`. All 7 sections present:
1. Architecture diagram (ASCII)
2. Cluster retooling state machine (7 states + rollback)
3. Dual-stack-of-10 stitching math + `satellite_stitch.rs` integration plan
4. Specialist routing table (5 module types × primary + 2 fallbacks)
5. Failure modes + mitigations (5 categories)
6. Forge integration plan (wraps loop_engine.rs, doesn't replace)
7. Three test scenarios with execution traces (Sleeping Bear schooner, Whitefish Point SAR, Cessna 172)

This is the spec. Code from it.

**T10 — implementation (Qwen3.6-A3B on P100#1)** — TRUNCATED at `fleet_prompts/SAR_T10_response.md`. Only ~2.2 KB returned, types-only, cut off mid-`MissionReport` struct. No functions, no handlers.

## How to finish the orchestrator next session

Three options, in order of recommendation:

### Option 1: Re-fire the continuation prompt to either P100

The completion prompt is at `fleet_prompts/SAR_T11_orchestrator_continuation.md`. It includes the verbatim types T10 already produced + a precise spec for the missing functions. Fire it at whichever P100 is least loaded:

```bash
cd cesarops-inference

# Pick whichever returns first / produces cleaner code
bash fleet_prompts/dispatch.sh http://127.0.0.1:5001 \
    fleet_prompts/SAR_T11_orchestrator_continuation.md \
    fleet_prompts/SAR_T11_response.md

# OR parallel against both:
bash fleet_prompts/dispatch_parallel.sh \
    fleet_prompts/SAR_T11_orchestrator_continuation.md SAR_T11
```

Expected output: `=== FILE: cesarops-forge-v2/src/orchestrator.rs ===` block + `=== DIFF: cesarops-forge-v2/src/main.rs ===` block.

### Option 2: Hand-write from T9

T9 design is detailed enough to drop a working orchestrator straight into the forge. The structure is:

1. Create `cesarops-forge-v2/src/orchestrator.rs` with the types from T10 (verbatim) + the missing types listed in T11 (`MissionReport`, `StitchingStrategy`)
2. Implement the 7 async functions per T11 spec (probe_cluster, plan_from_scenario, assign_specialists, retool_for_mission, restore_cluster, dispatch_modules, execute_mission)
3. Add 3 axum handlers (`orchestrator_probe`, `orchestrator_plan`, `orchestrator_execute`) per the signatures in T11 section 8
4. Patch `cesarops-forge-v2/src/main.rs`:
   - Add `mod orchestrator;` at the top
   - Add 3 routes in the `Router::new()` chain near `.route("/cluster/freeform/apply", ...)`:
     ```rust
     .route("/orchestrator/probe", get(orchestrator::orchestrator_probe))
     .route("/orchestrator/plan", post(orchestrator::orchestrator_plan))
     .route("/orchestrator/execute", post(orchestrator::orchestrator_execute))
     ```
5. Build: `cargo build --release -p cesarops-forge-v2`
6. Restart forge: `pkill cesarops-forge-v2 && cd cesarops-forge-v2 && nohup ./target/release/cesarops-forge-v2 > /tmp/forge_v2.log 2>&1 &`
7. Smoke: `curl http://127.0.0.1:9100/orchestrator/probe`

Estimate: 2-3 hours focused, no AI assistance.

### Option 3: Send to operator's friend's coding agent cluster

Same prompt as Option 1 (`SAR_T11_orchestrator_continuation.md`) but to whichever specialist endpoint the operator's friend uses. Their cluster has been more reliable on long-form code generation than our local Gemma/Qwen for files of this size.

## Verification once orchestrator lands

```bash
# 1. Build
cd /home/cesarops/wreckhunter2000-1
cargo build --release -p cesarops-forge-v2

# 2. Restart forge
pkill cesarops-forge-v2
cd cesarops-forge-v2 && nohup ./target/release/cesarops-forge-v2 > /tmp/forge_v2.log 2>&1 &
sleep 3

# 3. Probe
curl -sS http://127.0.0.1:9100/orchestrator/probe | python3 -m json.tool | head -30

# 4. Plan only (no execution)
curl -sS -X POST http://127.0.0.1:9100/orchestrator/plan \
  -H 'Content-Type: application/json' \
  -d '{"raw_text":"Find me a 19th-century schooner around Sleeping Bear Dunes, Lake Michigan","priority":1}'

# 5. Full execution
curl -sS -X POST http://127.0.0.1:9100/orchestrator/execute \
  -H 'Content-Type: application/json' \
  -d '{"raw_text":"Search for missing fishing boat near Whitefish Point","priority":2}' | python3 -m json.tool
```

Expected behavior on a fresh request:
1. Forge calls probe_cluster → gets cluster snapshot from `/cluster/discover`
2. Forge calls plan_from_scenario → POSTs to Gemma on :5001 → gets back MissionPlan JSON
3. Forge calls retool_for_mission → if WreckHunt/DownedAircraft, stops GemmaBig on P100#0 (frees ~16 GB)
4. Forge calls dispatch_modules → fans out via JoinSet, each module ≤60s timeout
5. Forge calls restore_cluster → restarts GemmaBig
6. Returns MissionReport JSON

## Known v1 limitations (track for v2)

- Tile-compute modules (delegate=VulkanGpu) currently return placeholder ModuleResults. Real wiring requires linking `cesarops-inference` as a workspace dep of `cesarops-forge-v2` so the orchestrator can invoke `optical_mass`, `magnetic_eraser`, `galvanic_battery`, `satellite_stitch` directly via Rust calls. That's a Cargo.toml + `cesarops-inference` lib re-export change. ~30 min of work.
- Vision worker endpoints (Florence-2, Moondream2) aren't running yet on cesarops3/cesarops2. The orchestrator routes there but those POSTs will timeout. The detection service handles vision-worker-offline gracefully (Standby action). For now: treat vision modules as "soft skips" — if endpoint not reachable, mark module status=skipped, mission continues.
- Coral TPU isn't installed. CORAL_TPU_INT8 delegates auto-fall-back to VulkanGpu per T9 routing table.
- The `plan_from_scenario` LLM prompt may produce malformed JSON occasionally. Single retry is built in; on second failure, return a default-template MissionPlan with empty modules — better to have a usable shell than 500-error.

## Where to look next

- T9 design doc: `cesarops-inference/fleet_prompts/SAR_T9_response.md`
- T10 truncated impl: `cesarops-inference/fleet_prompts/SAR_T10_response.md`
- T11 continuation prompt: `cesarops-inference/fleet_prompts/SAR_T11_orchestrator_continuation.md`
- Forge state file: `cesarops-forge-v2/mode_state.json`
- Detection service: `cesarops-detection/scripts/start.sh` (still running on :5580)
- Cluster config: `cesarops-forge-v2/cluster_config.toml`
- Slicer pattern reference: `/mnt/data-external/cesarops/repo/cesarops-slicer/src/spec/`

End of handoff.
