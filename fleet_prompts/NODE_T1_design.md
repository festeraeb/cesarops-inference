# Task: Design Document for `cesarops-node` Daemon

You are a senior Rust systems architect. Write a complete design document for a lightweight node daemon called `cesarops-node`.

## Context

We have a heterogeneous GPU cluster:
- T440 server: 2x Tesla P100-PCIE-16GB (local, Vulkan)
- cesarops2: GTX 1070 8GB + Quadro P1000 4GB (LAN 10.0.0.129, CUDA)
- cesarops3: P106-100 6GB mining card (LAN 10.0.0.41, CUDA)
- Nautik9 laptop: Quadro M2200 4GB (Tailscale, mobile)

Currently the forge orchestrator uses SSH + shell scripts to start/stop workers on remote nodes. This is fragile and doesn't support:
- Capability registration
- Heartbeat / health monitoring
- Dynamic model loading/unloading
- VRAM accounting
- Queue depth reporting

## Requirements

Design a `cesarops-node` daemon that:

1. **Runs on every node** as a single static binary (cross-compiled for x86_64-linux)
2. **Registers with the forge** on startup: POST to `http://{forge_ip}:9100/cluster/node/register` with:
   - node_id (hostname or Ed25519 pubkey fingerprint)
   - hardware profile (GPU name, VRAM, CUDA/Vulkan capability, CPU cores, RAM)
   - available models (scanned from a configured models_dir)
   - listen address + port
3. **Heartbeats** every 10s: POST to `/cluster/node/heartbeat` with load, VRAM usage, active model, queue depth
4. **Exposes local HTTP API** (Axum, port 5500 by default):
   - `GET /status` — current state (idle/loading/serving/error), VRAM, model, queue
   - `POST /spawn` — load a model: `{model_path, port, backend, gpu_layers, context_size}`
   - `POST /stop` — unload current model (kill koboldcpp/cesarops-inference child)
   - `POST /health` — deep health check (GPU temp, memory pressure, disk)
   - `GET /models` — list available .gguf files in models_dir
5. **Manages child processes**: spawns koboldcpp or cesarops-inference as a child, monitors stdout/stderr, restarts on crash (configurable max_restarts)
6. **Config file**: `cesarops-node.toml` with:
   - forge_url, node_name, models_dir, default_port, gpu_id, backend (vulkan/cuda/cpu)
   - Optional: allowed_models whitelist, max_vram_pct, heartbeat_interval_secs
7. **Security**: Ed25519 keypair generated on first run, stored in `~/.cesarops-node/`. All registration/heartbeat requests signed. Forge validates signature.

## Deliverables

Write the design doc with these sections:
1. Architecture overview (ASCII diagram showing node ↔ forge communication)
2. Data types (Rust structs/enums for all messages)
3. State machine (node lifecycle: Init → Registering → Idle → Loading → Serving → Error)
4. API contract (request/response JSON for each endpoint)
5. Process management strategy (how spawn/stop/restart works)
6. Config file schema (full cesarops-node.toml example)
7. Build/deploy plan (single binary, cross-compile targets, systemd unit)

## Constraints
- Rust 2021 edition
- Dependencies: axum 0.8, tokio, serde, reqwest, ed25519-dalek, sysinfo
- No async runtime other than tokio
- Must compile on stable Rust (no nightly features)
- Binary size target: < 10 MB stripped
- The daemon does NOT run inference itself — it manages koboldcpp/cesarops-inference as child processes

Output ONLY the design document in markdown. No code yet.
