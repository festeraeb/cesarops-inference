# Task: Implement `cesarops-node` Daemon (Rust)

You are a senior Rust developer. Write the complete implementation of `cesarops-node` — a lightweight daemon that runs on each GPU node in a heterogeneous cluster.

## What it does

Each node in our cluster (T440 with P100s, cesarops2 with 1070+P1000, cesarops3 with P106-100) runs this daemon. It:
1. Registers with the central forge (http://forge:9100/cluster/node/register)
2. Heartbeats every 10s with GPU load, VRAM, queue depth
3. Exposes a local HTTP API for the forge to control model loading

## Required files

### 1. `cesarops-node/Cargo.toml`

```toml
[package]
name = "cesarops-node"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }
toml = "0.8"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
sysinfo = "0.30"
```

### 2. `cesarops-node/src/main.rs`

Implement:

```rust
// State machine
enum NodeState { Init, Registering, Idle, Loading, Serving, Error(String) }

// Config loaded from cesarops-node.toml
struct NodeConfig {
    forge_url: String,        // e.g. "http://10.0.0.1:9100"
    node_name: String,        // e.g. "cesarops2"
    models_dir: String,       // e.g. "/mnt/storage/models"
    listen_port: u16,         // default 5500
    gpu_id: u32,              // which GPU index to use
    backend: String,          // "cuda" | "vulkan" | "cpu"
    koboldcpp_path: String,   // path to koboldcpp binary
    heartbeat_interval_secs: u64, // default 10
    max_restarts: u32,        // default 3
}

// Shared app state
struct AppState {
    config: NodeConfig,
    state: RwLock<NodeState>,
    child: RwLock<Option<Child>>,  // managed koboldcpp process
    active_model: RwLock<Option<String>>,
    active_port: RwLock<Option<u16>>,
    gpu_info: RwLock<GpuInfo>,
}

struct GpuInfo {
    name: String,
    vram_total_mb: u64,
    vram_used_mb: u64,
    temperature_c: u32,
    utilization_pct: u32,
}
```

**Endpoints to implement:**

- `GET /status` → `{ state, model, port, gpu: { vram_used, vram_total, temp, util }, uptime_secs }`
- `POST /spawn` body: `{ model_path, port, gpu_layers, context_size }` → starts koboldcpp child, transitions to Loading then Serving once port binds
- `POST /stop` → kills child process, transitions to Idle, returns `{ stopped: model_name }`
- `GET /models` → scans models_dir for *.gguf files, returns `[{ name, size_gb }]`
- `GET /health` → deep check: GPU accessible, disk space, child alive if serving

**Background tasks:**

- `heartbeat_loop`: every N seconds, POST to `{forge_url}/cluster/node/heartbeat` with:
  ```json
  { "node_id": "cesarops2", "state": "serving", "model": "TinyLlama...", "port": 5571,
    "gpu": { "vram_used_mb": 1200, "vram_total_mb": 4096, "temp_c": 45, "util_pct": 12 },
    "queue_depth": 0 }
  ```
- `child_monitor`: watches the spawned process, detects crashes, auto-restarts up to max_restarts
- `gpu_poll`: refreshes GpuInfo every 5s via nvidia-smi or sysinfo

**Process spawn logic for /spawn:**

```
1. If already serving → return error "already serving {model}, call /stop first"
2. Set state = Loading
3. Build command: koboldcpp --model {path} --port {port} --use{backend} --gpulayers {layers} --contextsize {ctx} --quiet --maingpu {gpu_id}
4. Spawn as child process, capture stdout/stderr
5. Poll port every 1s for up to 120s
6. If port binds → state = Serving, return success
7. If timeout → kill child, state = Error, return error
```

### 3. `cesarops-node/cesarops-node.toml` (example config)

```toml
forge_url = "http://10.0.0.1:9100"
node_name = "cesarops2"
models_dir = "/mnt/storage/models"
listen_port = 5500
gpu_id = 1
backend = "cuda"
koboldcpp_path = "/home/cesarops/benchmark/koboldcpp"
heartbeat_interval_secs = 10
max_restarts = 3
```

## Output format

Output the complete implementation as:
```
=== FILE: cesarops-node/Cargo.toml ===
(contents)

=== FILE: cesarops-node/src/main.rs ===
(contents)

=== FILE: cesarops-node/cesarops-node.toml ===
(contents)
```

## Constraints
- Rust 2021, stable toolchain only
- Use `std::process::Command` for child management (not nix crate)
- Use `tokio::process::Command` for async spawn with stdout/stderr capture
- GPU info: shell out to `nvidia-smi --query-gpu=...` (works on all our CUDA nodes)
- For Vulkan nodes (T440 P100s): fall back to parsing `/sys/class/drm/` or just report "vulkan" with no temp/util
- Error handling: use `Result<T, String>` for simplicity (no anyhow/thiserror)
- Keep it under 500 lines total for main.rs
