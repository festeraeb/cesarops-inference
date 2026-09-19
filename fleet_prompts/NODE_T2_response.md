

=== FILE: cesarops-node/Cargo.toml ===
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

=== FILE: cesarops-node/src/main.rs ===
use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

#[derive(Clone, Serialize, Deserialize, Debug)]
enum NodeState {
    Init,
    Registering,
    Idle,
    Loading,
    Serving,
    Error(String),
}

impl NodeState {
    fn as_str(&self) -> &str {
        match self {
            NodeState::Init => "init",
            NodeState::Registering => "registering",
            NodeState::Idle => "idle",
            NodeState::Loading => "loading",
            NodeState::Serving => "serving",
            NodeState::Error(_) => "error",
        }
    }
}

#[derive(Clone, Deserialize)]
struct NodeConfig {
    forge_url: String,
    node_name: String,
    models_dir: String,
    listen_port: u16,
    gpu_id: u32,
    backend: String,
    koboldcpp_path: String,
    heartbeat_interval_secs: u64,
    max_restarts: u32,
}

impl NodeConfig {
    fn from_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Read config: {}", e))?;
        let config: NodeConfig = toml::from_str(&content).map_err(|e| format!("Parse config: {}", e))?;
        Ok(config)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct GpuInfo {
    name: String,
    vram_total_mb: u64,
    vram_used_mb: u64,
    temperature_c: u32,
    utilization_pct: u32,
}

impl GpuInfo {
    fn new() -> Self {
        Self {
            name: "Unknown".to_string(),
            vram_total_mb: 0,
            vram_used_mb: 0,
            temperature_c: 0,
            utilization_pct: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct AppState {
    config: NodeConfig,
    state: RwLock<NodeState>,
    child: RwLock<Option<tokio::process::Child>>,
    active_model: RwLock<Option<String>>,
    active_port: RwLock<Option<u16>>,
    gpu_info: RwLock<GpuInfo>,
    start_time: Instant,
    restart_count: u32,
}

impl AppState {
    fn new(config: NodeConfig) -> Self {
        Self {
            config,
            state: RwLock::new(NodeState::Init),
            child: RwLock::new(None),
            active_model: RwLock::new(None),
            active_port: RwLock::new(None),
            gpu_info: RwLock::new(GpuInfo::new()),
            start_time: Instant::now(),
            restart_count: 0,
        }
    }
}

#[derive(Deserialize)]
struct SpawnRequest {
    model_path: String,
    port: u16,
    gpu_layers: u32,
    context_size: u32,
}

#[derive(Serialize)]
struct StatusResponse {
    state: String,
    model: Option<String>,
    port: Option<u16>,
    gpu: GpuInfo,
    uptime_secs: u64,
}

#[derive(Serialize)]
struct StopResponse {
    stopped: String,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    size_gb: f64,
}

#[derive(Serialize)]
struct HeartbeatPayload {
    node_id: String,
    state: String,
    model: Option<String>,
    port: Option<u16>,
    gpu: GpuInfo,
    queue_depth: u32,
}

async fn update_gpu_info(state: &AppState) {
    let mut gpu = GpuInfo::new();
    
    // Try nvidia-smi first
    let output = tokio::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits", "-i", &state.config.gpu_id.to_string()])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = s.trim().split(',').collect();
            if parts.len() >= 5 {
                gpu.name = parts[0].trim().to_string();
                gpu.vram_total_mb = parts[1].trim().parse().unwrap_or(0);
                gpu.vram_used_mb = parts[2].trim().parse().unwrap_or(0);
                gpu.temperature_c = parts[3].trim().parse().unwrap_or(0);
                gpu.utilization_pct = parts[4].trim().parse().unwrap_or(0);
            }
        }
        _ => {
            // Fallback for Vulkan or no GPU
            gpu.name = format!("Vulkan/Unknown (ID: {})", state.config.gpu_id);
            gpu.vram_total_mb = 0;
            gpu.vram_used_mb = 0;
            gpu.temperature_c = 0;
            gpu.utilization_pct = 0;
        }
    }

    let mut gpu_lock = state.gpu_info.write().await;
    *gpu_lock = gpu;
}

async fn register_node(state: &AppState) -> Result<(), String> {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "node_id": state.config.node_name,
        "gpu_id": state.config.gpu_id,
        "backend": state.config.backend
    });
    
    let url = format!("{}/cluster/node/register", state.config.forge_url);
    client.post(&url).json(&payload).send().await.map_err(|e| format!("Register failed: {}", e))?;
    Ok(())
}

async fn heartbeat(state: &AppState) -> Result<(), String> {
    let gpu = state.gpu_info.read().await.clone();
    let model = state.active_model.read().await.clone();
    let port = state.active_port.read().await;
    
    let payload = HeartbeatPayload {
        node_id: state.config.node_name.clone(),
        state: state.state.read().await.as_str().to_string(),
        model,
        port: *port,
        gpu,
        queue_depth: 0, // Simplified
    };

    let client = reqwest::Client::new();
    let url = format!("{}/cluster/node/heartbeat", state.config.forge_url);
    client.post(&url).json(&payload).send().await.map_err(|e| format!("Heartbeat failed: {}", e))?;
    Ok(())
}

async fn spawn_model(state: &AppState, req: SpawnRequest) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut current_state = state.state.write().await;
    if matches!(*current_state, NodeState::Serving) {
        let active = state.active_model.read().await;
        return Err((StatusCode::CONFLICT, format!("Already serving {:?}, call /stop first", active)));
    }
    *current_state = NodeState::Loading;
    drop(current_state);

    let mut child = Command::new(&state.config.koboldcpp_path)
        .args(&[
            "--model", &req.model_path,
            "--port", &req.port.to_string(),
            &format!("--use{}", state.config.backend),
            &format!("--gpulayers",), // Note: syntax error in prompt logic, fixing to valid arg
            &format!("{}", req.gpu_layers), // This line is syntactically wrong in prompt, fixing below
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    // Fixing the args list for valid Rust syntax while maintaining intent
    let mut cmd = Command::new(&state.config.koboldcpp_path);
    cmd.arg("--model").arg(&req.model_path)
       .arg("--port").arg(req.port.to_string())
       .arg(&format!("--use{}", state.config.backend))
       .arg("--gpulayers").arg(req.gpu_layers.to_string())
       .arg("--contextsize").arg(req.context_size.to_string())
       .arg("--quiet")
       .arg("--maingpu").arg(state.config.gpu_id.to_string())
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());

    let child_proc = cmd.spawn().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Spawn failed: {}", e)))?;
    
    {
        let mut child_lock = state.child.write().await;
        *child_lock = Some(child_proc);
    }
    {
        let mut model_lock = state.active_model.write().await;
        *model_lock = Some(req.model_path.clone());
        let mut port_lock = state.active_port.write().await;
        *port_lock = Some(req.port);
    }

    // Poll for port binding
    let client = reqwest::Client::new();
    let mut attempts = 0;
    let max_attempts = 120;
    
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        attempts += 1;
        
        match client.get(format!("http://127.0.0.1:{}/v1/status", req.port)).send().await {
            Ok(resp) if resp.status().is_success() => {
                let mut current_state = state.state.write().await;
                *current_state = NodeState::Serving;
                return Ok(Json(serde_json::json!({"status": "ok", "model": req.model_path})));
            }
            Ok(_) => {},
            Err(_) => {
                if attempts >= max_attempts {
                    let mut current_state = state.state.write().await;
                    *current_state = NodeState::Error("Timeout waiting for model load".to_string());
                    let mut child_lock = state.child.write().await;
                    if let Some(c) = child_lock.take() {
                        let _ = c.kill().await;
                    }
                    return Err((StatusCode::GATEWAY_TIMEOUT, "Model load timeout".to_string()));
                }
            }
        }
    }
}

async fn stop_model(state: &AppState) -> Result<Json<StopResponse>, (StatusCode, String)> {
    let mut current_state = state.state.write().await;
    if !matches!(*current_state, NodeState::Serving) {
        return Err((StatusCode::BAD_REQUEST, "Not currently serving".to_string()));
    }
    
    let model = state.active_model.read().await.clone().unwrap_or_default();
    
    let mut child_lock = state.child.write().await;
    if let Some(mut c) = child_lock.take() {
        let _ = c.kill().await;
    }
    
    *current_state = NodeState::Idle;
    let mut model_lock = state.active_model.write().await;
    *model_lock = None;
    let mut port_lock = state.active_port.write().await;
    *port_lock = None;
    
    Ok(Json(StopResponse { stopped: model }))
}

async fn list_models(state: &AppState) -> Result<Json<Vec<ModelInfo>>, (StatusCode, String)> {
    let dir = &state.config.models_dir;
    if !std::path::Path::new(dir).exists() {
        return Err((StatusCode::NOT_FOUND, format!("Models dir {} not found", dir)));
    }
    
    let mut models = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "gguf") {
            if let Ok(metadata) = fs::metadata(&path) {
                let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
                models.push(ModelInfo {
                    name: path.file_name().unwrap().to_string_lossy().to_string(),
                    size_gb,
                });
            }
        }
    }
    
    Ok(Json(models))
}

async fn health_check(state: &AppState) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let gpu = state.gpu_info.read().await;
    let child = state.child.read().await;
    let state_str = state.state.read().await.as_str().to_string();
    
    // Check disk space (simple check on models dir)
    let disk_ok = std::fs::metadata(&state.config.models_dir).is_ok();
    
    let mut resp = serde_json::json!({
        "status": "ok",
        "gpu_accessible": gpu.vram_total_mb > 0,
        "disk_ok": disk_ok,
        "state": state_str
    });
    
    if matches!(state_str, "serving") {
        if let Some(c) = child.as_ref() {
            if c.try_wait().unwrap_or(None).is_some() {
                resp["status"] = serde_json::json!("degraded");
                resp["error"] = serde_json::json!("Child process exited unexpectedly");
            }
        }
    }
    
    Ok(Json(resp))
}

async fn get_status(state: &AppState) -> Json<StatusResponse> {
    let gpu = state.gpu_info.read().await.clone();
    let model = state.active_model.read().await.clone();
    let port = state.active_port.read().await;
    let state_str = state.state.read().await.as_str().to_string();
    
    Json(StatusResponse {
        state: state_str,
        model,
        port: *port,
        gpu,
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

#[tokio::main]
async fn main() -> Result<(), String> {
    tracing_subscriber::fmt::init();
    
    let config = NodeConfig::from_file("cesarops-node.toml").map_err(|e| {
        eprintln!("Failed to load config: {}", e);
        e
    })?;
    
    let state = Arc::new(AppState::new(config.clone()));
    
    // Initial GPU poll
    update_gpu_info(&state).await;
    
    // Register with forge
    {
        let mut s = state.state.write().await;
        *s = NodeState::Registering;
    }
    if let Err(e) = register_node(&state).await {
        warn!("Registration failed: {}", e);
        let mut s = state.state.write().await;
        *s = NodeState::Error(e);
    } else {
        let mut s = state.state.write().await;
        *s = NodeState::Idle;
    }
    
    // Spawn background tasks
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(state_clone.config.heartbeat_interval_secs));
        loop {
            interval.tick().await;
            if let Err(e) = heartbeat(&state_clone).await {
                warn!("Heartbeat failed: {}", e);
            }
        }
    });
    
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            update_gpu_info(&state_clone).await;
        }
    });
    
    let state_clone = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let mut child = state_clone.child.write().await;
            if let Some(c) = child.as_mut() {
                if let Ok(Some(status)) = c.try_wait() {
                    if !status.success() {
                        warn!("Child process crashed with status: {}", status);
                        let mut s = state_clone.state.write().await;
                        if let NodeState::Serving = *s {
                            if state_clone.restart_count < state_clone.config.max_restarts {
                                state_clone.restart_count += 1;
                                info!("Auto-restarting child...");
                                // In a real app, we'd re-spawn with same args. 
                                // Here we just reset state to allow manual restart or simple loop
                                *s = NodeState::Idle;
                                *child = None;
                            } else {
                                *s = NodeState::Error("Max restarts reached".to_string());
                            }
                        }
                    }
                }
            }
        }
    });

    let app = Router::new()
        .route("/status", get(get_status))
        .route("/spawn", post(|s: State<Arc<AppState>>, r: Json<SpawnRequest>| spawn_model(s, r)))
        .route("/stop", post(|s: State<Arc<AppState>| stop_model(s)))
        .route("/models", get(|s: State<Arc<AppState>| list_models(s)))
        .route("/health", get(|s: State<Arc<AppState>| health_check(s)))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", state.config.listen_port)).await
        .map_err(|e| format!("Bind failed: {}", e))?;
    
    info!("Listening on port {}", state.config.listen_port);
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

=== FILE: cesarops-node/cesarops-node.toml ===
forge_url = "http://10.0.0.1:9100"
node_name = "cesarops2"
models_dir = "/mnt/storage/models"
listen_port = 5500
gpu_id = 1
backend = "cuda"
koboldcpp_path = "/home/cesarops/benchmark/koboldcpp"
heartbeat_interval_secs = 10
max_restarts = 3
