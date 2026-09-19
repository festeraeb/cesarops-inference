# Task: Add cesarops.com webhook intake route to forge

You are a senior Rust developer. Add a webhook intake endpoint to `cesarops-forge-v2/src/main.rs` that receives mission requests from cesarops.com and routes them through the orchestrator.

## Context

cesarops.com will POST mission requests to the forge. The forge needs a public-facing endpoint that:
1. Accepts the webhook payload
2. Validates it minimally
3. Routes it to the orchestrator's `execute_mission`
4. Returns the MissionReport as JSON

## Endpoint

`POST /webhook/mission`

## Request body (from cesarops.com):
```json
{
  "source": "cesarops.com",
  "user_id": "operator-1",
  "scenario": "Find a sunken freighter near Thunder Bay, Lake Huron",
  "bbox": [45.0, -83.5, 45.3, -83.2],
  "priority": 1,
  "callback_url": "https://cesarops.com/api/mission/callback"
}
```

## Behavior:
1. Extract `scenario`, `bbox`, `priority` from the payload
2. Build an `OperatorScenario` (the orchestrator's input type)
3. Spawn the mission execution as a background task (don't block the webhook response)
4. Return immediately with `{"status": "accepted", "mission_id": "<uuid>"}`
5. When the mission completes, POST the MissionReport to `callback_url` (best-effort, don't fail if callback is unreachable)

## Also add:

`GET /webhook/missions` — returns a list of recent missions (last 50) with their status.

Store missions in a simple `Arc<Mutex<Vec<MissionRecord>>>` on AppState.

```rust
struct MissionRecord {
    id: String,           // uuid
    source: String,
    scenario_text: String,
    status: String,       // "running" | "completed" | "failed"
    submitted_at: u64,    // unix timestamp
    completed_at: Option<u64>,
    report: Option<MissionReport>,
}
```

## Function signatures:

```rust
async fn webhook_mission(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value>

async fn list_missions(
    State(state): State<AppState>,
) -> Json<serde_json::Value>
```

## Output format

Output the Rust code for:
1. The `MissionRecord` struct
2. The `webhook_mission` handler
3. The `list_missions` handler
4. The background task that runs the mission and posts callback

No imports, no Router setup. Just the structs and functions.

```rust
#[derive(Clone, Serialize, Deserialize)]
struct MissionRecord { ... }

async fn webhook_mission(...) -> ... { ... }
async fn list_missions(...) -> ... { ... }
async fn run_mission_background(...) { ... }
```

## Constraints:
- Generate mission_id with: `format!("{:016x}", std::time::SystemTime::now()...as_nanos() & 0xFFFFFFFFFFFFFFFF)`
- Use `tokio::spawn` for background execution
- Callback POST: 10s timeout, ignore errors
- Keep mission history capped at 50 entries (drop oldest)
- Under 100 lines total
