# Task: Add LLM-driven mission planning to orchestrator

You are a senior Rust developer. Add an optional LLM planning step to the orchestrator in `cesarops-forge-v2/src/orchestrator.rs`.

## Context

Currently `plan_from_scenario` uses a heuristic keyword classifier. This works for known scenario types but can't handle novel requests like "check if there's a thermal plume near the power plant outfall at coordinates 42.1, -80.3".

The intake brain (Picasso, TinyLlama on P1000) is always available at `http://10.0.0.129:5571`. If it's offline, fall back to scout at `http://10.0.0.41:5100`.

## What to write

A new function `llm_refine_plan` that:
1. Takes the heuristic plan + the operator's raw_text
2. Sends a structured prompt to the intake brain asking it to refine/override the plan
3. Parses the LLM response as JSON
4. If parsing succeeds, merges LLM suggestions into the plan (add/remove modules, adjust bbox)
5. If parsing fails or LLM is unreachable, returns the heuristic plan unchanged (graceful degradation)

## Function signature

```rust
async fn llm_refine_plan(
    heuristic_plan: MissionPlan,
    scenario: &OperatorScenario,
) -> MissionPlan
```

## LLM prompt format

Send to koboldcpp `/api/v1/generate`:
```json
{
  "prompt": "You are a SAR mission planner. Given this scenario and initial plan, output ONLY a JSON object with optional overrides.\n\nScenario: {raw_text}\nBBox: {bbox}\n\nCurrent plan:\n- Class: {class}\n- Modules: {module_names}\n- Stitching: {stitching_summary}\n\nIf the plan looks correct, output: {\"action\": \"accept\"}\nIf you want to add a module, output: {\"action\": \"add_module\", \"module\": {\"id\": \"...\", \"name\": \"...\", \"tool_name\": \"...\", \"tool_args\": {...}}}\nIf you want to change the scenario class, output: {\"action\": \"reclassify\", \"class\": \"WreckHunt|DownedAircraft|SearchRescue|...\"}\n\nOutput ONLY valid JSON, nothing else.",
  "max_length": 256,
  "temperature": 0.1,
  "stop_sequence": ["\n\n", "}}\n"]
}
```

## Integration point

Call `llm_refine_plan` inside `execute_mission` AFTER `retry_plan` succeeds but BEFORE `assign_specialists`. Add a note to the mission report indicating whether LLM refinement was applied or skipped.

## Constants already available:
```rust
const INTAKE_ENDPOINT: &str = "http://10.0.0.129:5571";
const INTAKE_FALLBACK: &str = "http://10.0.0.41:5570";
```

## Output format

Output ONLY the Rust function `llm_refine_plan` + the updated section of `execute_mission` that calls it. No imports, no other code.

```rust
async fn llm_refine_plan(heuristic_plan: MissionPlan, scenario: &OperatorScenario) -> MissionPlan {
    // ...
}

// Updated execute_mission snippet (just the section between plan and assign_specialists):
// let plan = llm_refine_plan(plan, &scenario).await;
// notes.push(...);
```

## Constraints
- Timeout: 15s for the LLM call (TinyLlama is fast)
- If LLM returns "accept", return plan unchanged
- If LLM returns "add_module", append the module to plan.modules
- If LLM returns "reclassify", rebuild the plan with the new class
- If anything fails (timeout, parse error, unreachable), return plan unchanged + add note
- Under 70 lines
