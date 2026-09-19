# Task: Implement Node Registry Handlers for cesarops-forge-v2

You are a senior Rust developer. Write 3 axum handler functions that will be added to `cesarops-forge-v2/src/main.rs`.

## Context

We have a node daemon (`cesarops-node`) that registers with the forge and sends heartbeats. The forge needs handlers to receive these. The forge already uses:
- axum 0.8
- serde/serde_json
- tokio with Mutex
- tracing

The forge's AppState already has this field:
```rust
pub node_registry: Arc<Mutex<std::collections::HashMap<String, NodeRegistration>>>,
```

With these types already defined:
```rust
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NodeRegistration {
    pub node_id: String,
    pub hardware: serde_json::Value,
    pub available_models: Vec<String>,
    pub listen_port: u16,
    pub last_heartbeat: Option<NodeHeartbeat>,
    pub last_seen: u64,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NodeHeartbeat {
    pub state: String,
    pub model: Option<String>,
    pub port: Option<u16>,
    pub gpu: serde_json::Value,
    pub queue_depth: u32,
}
```

## Write these 3 functions:

### 1. `node_register` — POST /cluster/node/register

Receives JSON body:
```json
{
  "node_id": "cesarops2",
  "hardware": { "gpu": "Quadro P1000", "vram_mb": 4096, "backend": "cuda" },
  "available_models": ["TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"],
  "listen_port": 5500
}
```

Behavior:
- Insert/update the node in `state.node_registry`
- Set `last_seen` to current unix timestamp
- Log the registration with tracing::info
- Return `{"status": "registered", "node_id": "..."}`

### 2. `node_heartbeat` — POST /cluster/node/heartbeat

Receives JSON body:
```json
{
  "node_id": "cesarops2",
  "state": "serving",
  "model": "TinyLlama-1.1B-Chat-v1.0-Q4_K_M",
  "port": 5571,
  "gpu": { "vram_used_mb": 1200, "vram_total_mb": 4096, "temp_c": 45, "util_pct": 12 },
  "queue_depth": 0
}
```

Behavior:
- Look up node_id in registry. If not found, return `{"error": "unknown node, register first"}`
- Update `last_heartbeat` and `last_seen`
- Return `{"status": "ok"}`

### 3. `list_registered_nodes` — GET /cluster/nodes

No body. Returns the full registry as a JSON array:
```json
[
  {
    "node_id": "cesarops2",
    "hardware": {...},
    "available_models": [...],
    "listen_port": 5500,
    "last_heartbeat": { "state": "serving", "model": "...", ... },
    "last_seen": 1716000000,
    "online": true
  }
]
```

The `online` field is computed: `last_seen > (now - 30)` (30 second staleness threshold).

## Function signatures

All handlers take `State(state): State<AppState>` and optionally `Json(body): Json<serde_json::Value>`.

## Output format

Output ONLY the Rust code for the 3 functions. No imports, no main, no Router setup — just the 3 `async fn` blocks. Use this format:

```rust
// === node_register ===
async fn node_register(...) -> ... {
    ...
}

// === node_heartbeat ===
async fn node_heartbeat(...) -> ... {
    ...
}

// === list_registered_nodes ===
async fn list_registered_nodes(...) -> ... {
    ...
}
```

## Constraints
- Get current unix timestamp via `std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()`
- Use `state.node_registry.lock().await` to access the HashMap
- Keep it simple — no fancy error types, just Json<serde_json::Value> returns
- Total: under 80 lines
