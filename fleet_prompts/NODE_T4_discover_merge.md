# Task: Update discover_nodes to merge node registry data

You are a senior Rust developer. Rewrite the `discover_nodes` function for cesarops-forge-v2.

## Current behavior

The existing `discover_nodes()` reads `cluster_config.toml` `[[known_nodes]]` entries and probes each node's ports via HTTP. It returns a JSON array of node objects.

## New behavior

The updated function should:
1. First, collect all nodes from the **DII node registry** (populated by cesarops-node heartbeats)
2. Then, merge in any `[[known_nodes]]` from cluster_config.toml that AREN'T already in the registry (legacy nodes without the daemon)
3. For registry nodes: use the heartbeat data directly (no HTTP probe needed — the heartbeat IS the probe)
4. For legacy nodes: keep the existing HTTP probe behavior

## Function signature

```rust
async fn discover_nodes(State(state): State<AppState>) -> Json<serde_json::Value>
```

Note: it now takes `State(state)` to access `state.node_registry`.

## Output JSON format (per node)

```json
{
  "name": "cesarops2",
  "ip": "10.0.0.129",
  "online": true,
  "source": "registry",  // or "legacy_probe"
  "gpu": "Quadro P1000 4GB",
  "state": "serving",
  "model": "TinyLlama-1.1B-Chat-v1.0-Q4_K_M",
  "port": 5571,
  "vram_used_mb": 1200,
  "vram_total_mb": 4096,
  "last_seen_secs_ago": 5,
  "services_online": true
}
```

For registry nodes:
- `online` = last_seen within 30 seconds
- `state`, `model`, `port`, `gpu` come from last_heartbeat
- `source` = "registry"

For legacy nodes (from cluster_config.toml):
- Keep existing HTTP probe logic
- `source` = "legacy_probe"
- `state` = "unknown" (we don't have heartbeat data)

## Available state

```rust
// Access registry:
let registry = state.node_registry.lock().await;
// registry is HashMap<String, NodeRegistration>

// NodeRegistration has:
// - node_id, hardware (Value), available_models, listen_port
// - last_heartbeat: Option<NodeHeartbeat>
// - last_seen: u64 (unix timestamp)

// NodeHeartbeat has:
// - state, model, port, gpu (Value), queue_depth
```

## Constraints
- Keep the existing `get_tailscale_peers()` call for legacy nodes
- Keep the existing reqwest probe logic for legacy nodes
- Staleness threshold: 30 seconds
- Current timestamp: `std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()`
- Config path: `/codebase/repos/wreckhunter2000-1/cesarops-forge-v2/cluster_config.toml`
- Output the complete function body (just the one `async fn discover_nodes(...)` function)
- Under 120 lines
