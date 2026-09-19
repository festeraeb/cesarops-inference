# Task: Replace SSH bring-up with POST /spawn in orchestrator

You are a senior Rust developer. Rewrite the `bring_up_remote` function in `cesarops-forge-v2/src/orchestrator.rs`.

## Current behavior (to be replaced)

The current `bring_up_remote` function SSHes into a remote host and runs `~/start_<name>.sh`. This is fragile and requires SSH keys + scripts on every host.

## New behavior

Each remote node now runs `cesarops-node` daemon on port 5500. The orchestrator should:
1. POST to `http://{host}:5500/spawn` with the model config
2. Wait for the response (the daemon handles port-polling internally)
3. Return success/failure

## Function to write

```rust
async fn bring_up_remote(w: &SecondaryWorker) -> BringUpOutcome {
    // ...
}
```

## SecondaryWorker struct (already defined):
```rust
struct SecondaryWorker {
    name: String,
    host: String,
    port: u16,      // the inference port (e.g. 5571)
    enabled: bool,
}
```

## cesarops-node /spawn API:
- URL: `POST http://{host}:5500/spawn`
- Body: `{"model_path": "/path/to/model.gguf", "port": 5571, "gpu_layers": 999, "context_size": 2048}`
- Success response: `{"status": "ok", "model": "...", "port": 5571}`
- Error response: `{"error": "already serving ..., call /stop first"}` or `{"error": "spawn: ..."}`

## BringUpOutcome enum (already defined):
```rust
enum BringUpOutcome {
    Started,
    Failed(String),
    HostUnreachable,
}
```

## Logic:
1. First check if the node daemon is reachable: GET `http://{host}:5500/status` with 3s timeout
2. If unreachable → return `HostUnreachable`
3. If reachable, check status response: if `state == "serving"` → already up, return `Started`
4. If idle, POST `/spawn` with the worker's model config
5. The model_path and other params need to come from cluster_config.toml. Read the [[worker]] entry matching `w.name` to get model, gpu_layers (default 999), contextsize (default 8192).
6. If spawn returns `{"status": "ok"}` → return `Started`
7. If spawn returns error → return `Failed(error_message)`

## Available helpers:
- `fn load_worker_ports() -> HashMap<String, u16>` — already exists
- Config path: `/codebase/repos/wreckhunter2000-1/cesarops-forge-v2/cluster_config.toml`
- Use `reqwest::Client` with 120s timeout for the spawn call (model loading takes time)
- Use `tokio::time::timeout` for the overall operation

## Also write a helper:
```rust
fn load_worker_spawn_config(name: &str) -> Option<SpawnConfig>
```
That reads cluster_config.toml and returns the model_path, gpu_layers, context_size for a given worker name.

## Output format:
Just the two functions. No imports, no other code.

```rust
struct SpawnConfig {
    model_path: String,
    gpu_layers: u32,
    context_size: u32,
}

fn load_worker_spawn_config(name: &str) -> Option<SpawnConfig> {
    // ...
}

async fn bring_up_remote(w: &SecondaryWorker) -> BringUpOutcome {
    // ...
}
```

## Constraints:
- Under 60 lines total
- No SSH, no shell scripts
- Use reqwest for HTTP calls
- Timeout: 3s for status check, 120s for spawn
