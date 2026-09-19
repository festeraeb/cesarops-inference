# Design Document: `cesarops-node` Daemon

**Version:** 1.0.0  
**Status:** Draft  
**Author:** Senior Systems Architect  

## 1. Architecture Overview

The `cesarops-node` acts as a local agent that bridges the gap between the central Forge Orchestrator and the heterogeneous hardware. It abstracts the complexity of process management, hardware telemetry, and security signing.

```text
[ Forge Orchestrator ]
       ^  |
       |  | (1) Signed Registration / Heartbeats (HTTPS/POST)
       |  v
+-------------------------------------------------------+
|                  cesarops-node (Daemon)               |
|                                                       |
|  [ Axum API ] <---- (2) Local Control (HTTP/JSON) ----+ [ User/CLI ]
|       |                                               |
|       | (3) Spawn/Kill/Monitor                        |
|       v                                               |
|  [ Process Manager ]                                  |
|       |                                               |
|       +--> [ koboldcpp / cesarops-inference ] --------+ [ GPU/VRAM ]
|       |                                               |
|  [ Telemetry Engine ] <--- (4) sysinfo / NVML/Vulkan --+ [ CPU/RAM ]
|                                                       |
+-------------------------------------------------------+
```

## 2. Data Types

All messages are serialized using `serde` with `snake_case` naming conventions.

### 2.1. Hardware & Registration
```rust
pub enum Backend {
    Cuda,
    Vulkan,
    Cpu,
}

pub struct GpuProfile {
    pub name: String,
    pub vram_mb: u64,
    pub backend: Backend,
    pub compute_capability: String, // e.g., "7.5"
}

pub struct HardwareProfile {
    pub gpu: GpuProfile,
    pub cpu_cores: u32,
    pub total_ram_mb: u64,
    pub hostname: String,
}

pub struct RegistrationPayload {
    pub node_id: String, // Ed25519 Public Key Fingerprint
    pub hardware: HardwareProfile,
    pub available_models: Vec<String>,
    pub listen_address: String,
    pub listen_port: u16,
}
```

### 2.2. Heartbeat & Telemetry
```rust
pub struct HeartbeatPayload {
    pub node_id: String,
    pub timestamp: u64,
    pub load_avg: f32,
    pub vram_used_mb: u64,
    pub active_model: Option<String>,
    pub queue_depth: u32,
    pub status: NodeStatus,
}

pub enum NodeStatus {
    Idle,
    Loading,
    Serving,
    Error(String),
}
```

### 2.3. Control Commands (Local API)
```rust
pub struct SpawnRequest {
    pub model_path: PathBuf,
    pub port: u16,
    pub backend: Backend,
    pub gpu_layers: u32,
    pub context_size: u32,
}

pub struct HealthResponse {
    pub gpu_temp_c: f32,
    pub vram_pressure_pct: f32,
    pub disk_free_gb: f32,
    pub uptime_secs: u64,
}
```

## 3. State Machine

The daemon operates as a finite state machine (FSM) to ensure predictable behavior during model transitions.

1.  **INIT**: Loading config, generating/loading Ed25519 keys, initializing `sysinfo`.
2.  **REGISTERING**: Sending initial registration to Forge. Waiting for acknowledgment.
3.  **IDLE**: No inference process running. Listening for `/spawn` commands.
4.  **LOADING**: Spawning child process. Monitoring stdout for "Model Loaded" signal.
5.  **SERVING**: Child process is healthy and accepting requests.
6.  **ERROR**: Process crashed or hardware failure. Attempting restart based on `max_restarts` policy.

## 4. API Contract

### 4.1. Forge Communication (Outbound)
*All requests must include an `X-CesarOps-Signature` header containing the Ed25519 signature of the JSON body.*

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/cluster/node/register` | Initial handshake with hardware profile. |
| `POST` | `/cluster/node/heartbeat` | Periodic status update (every 10s). |

### 4.2. Local API (Inbound - Port 5500)
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Returns current `NodeStatus` and VRAM usage. |
| `GET` | `/models` | Returns list of `.gguf` files in `models_dir`. |
| `POST` | `/spawn` | Starts a new inference process. |
| `POST` | `/stop` | Kills the current inference process. |
| `POST` | `/health` | Returns deep hardware telemetry. |

## 5. Process Management Strategy

The daemon uses `tokio::process::Command` to manage child processes.

-   **Spawning**: When `/spawn` is called, the daemon validates the model path exists and matches the whitelist. It executes the binary with arguments derived from the request.
-   **Monitoring**: The daemon captures `stdout` and `stderr`. It parses logs for specific patterns (e.g., `llama.cpp: loaded model`) to transition from `LOADING` to `SERVING`.
-   **Termination**: `/stop` sends `SIGTERM` to the child process. If the child does not exit within 5 seconds, `SIGKILL` is issued.
-   **Resilience**: If the child process exits with a non-zero code, the daemon increments a `restart_counter`. If `restart_counter < max_restarts`, it attempts a re-spawn. Otherwise, it enters `ERROR` state and notifies the Forge via the next heartbeat.

## 6. Config File Schema

**File:** `/etc/cesarops/cesarops-node.toml` (or `~/.cesarops-node/config.toml`)

```toml
# cesarops-node configuration

[node]
node_name = "cesarops2"
forge_url = "http://10.0.0.1:9100"
models_dir = "/mnt/models"
default_port = 5500

[hardware]
gpu_id = 0
backend = "cuda" # Options: cuda, vulkan, cpu
max_vram_pct = 90

[runtime]
heartbeat_interval_secs = 10
max_restarts = 3
allowed_models = [
    "mistral-7b-v0.1.gguf",
    "llama-3-8b-instruct.gguf"
]

[security]
# Keypair is auto-generated in ~/.cesarops-node/ if not present
key_path = "~/.cesarops-node/id_ed25519"
```

## 7. Build & Deploy Plan

### 7.1. Build Pipeline
-   **Target**: `x86_64-unknown-linux-musl` for a fully static binary.
-   **Toolchain**: `cargo build --release`.
-   **Optimization**: `strip` the binary and use `LTO = true` in `Cargo.toml` to meet the < 10MB requirement.
-   **Cross-Compilation**: Use `cross` (Docker-based) to target MUSL from any host.

### 7.2. Deployment
1.  **Binary Distribution**: Single static binary `cesarops-node`.
2.  **Systemd Unit**:
    ```ini
    [Unit]
    Description=CesarOps Node Daemon
    After=network.target

    [Service]
    Type=simple
    ExecStart=/usr/bin/cesarops-node --config /etc/cesarops/cesarops-node.toml
    Restart=always
    User=cesarops
    Group=cesarops

    [Install]
    WantedBy=multi-user.target
    ```
3.  **Permissions**: Ensure the `cesarops` user has permissions to access `/dev/dri/` (Vulkan) and NVIDIA drivers.
