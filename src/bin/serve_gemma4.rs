//! llama.cpp-compatible HTTP server for Gemma-4-26B-MoE (native wgpu path).
//!
//!   serve_gemma4 --model PATH --port 5002 --gpu 1
//!
//! Endpoints: GET /health, POST /v1/completions

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use cesarops_inference::gemma4_runner::{Gemma4Runner, SamplingParams};
use cesarops_inference::gemma_tokenizer::GemmaTokenizer;
use cesarops_inference::hardware;
use cesarops_inference::loader;

#[derive(Clone)]
struct AppState {
    runner: Arc<Gemma4Runner>,
    tokenizer: Arc<GemmaTokenizer>,
    model_name: String,
    infer_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    temperature: f32,
}

fn default_max_tokens() -> u32 {
    64
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
    model: String,
    object: &'static str,
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "engine": "cesarops-inference-gemma4"}))
}

async fn completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let _guard = st.infer_lock.lock();
    let prompt_ids = st.tokenizer.encode(&req.prompt, true);
    if prompt_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut params = SamplingParams::default();
    params.max_new_tokens = req.max_tokens as usize;
    params.temperature = req.temperature;
    if let Some(eos) = st.tokenizer.eos_id {
        params.stop_tokens = vec![eos];
    }

    let new_ids = st
        .runner
        .generate(&prompt_ids, &params)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let text = st.tokenizer.decode(&new_ids);

    Ok(Json(CompletionResponse {
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "length".into(),
        }],
        model: st.model_name.clone(),
        object: "text_completion",
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    cesarops_inference::pamp::init_from_env();

    let mut model: Option<PathBuf> = None;
    let mut port: u16 = 5002;
    let mut gpu: usize = 1;
    let mut host = "0.0.0.0".to_string();

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--port" => {
                port = args[i + 1].parse()?;
                i += 2;
            }
            "--gpu" => {
                gpu = args[i + 1].parse()?;
                i += 2;
            }
            "--host" => {
                host = args[i + 1].clone();
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "serve_gemma4 --model PATH [--port 5002] [--gpu 1] [--host 0.0.0.0]"
                );
                return Ok(());
            }
            other => {
                anyhow::bail!("unknown arg: {other}");
            }
        }
    }

    let model = model.ok_or_else(|| anyhow::anyhow!("--model required"))?;
    let model_name = model
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    tracing::info!("loading GGUF once from {:?}", model);
    let profile = hardware::audit_system();
    let weights = loader::load(&model, &profile)?;
    let tokenizer = Arc::new(
        GemmaTokenizer::from_gguf(&weights).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?,
    );

    tracing::info!("building Gemma4Runner on GPU {gpu} (shared mmap)...");
    let runner = Arc::new(
        Gemma4Runner::from_weights(weights, gpu).map_err(|e| anyhow::anyhow!("load: {e}"))?,
    );

    let state = AppState {
        runner,
        tokenizer,
        model_name,
        infer_lock: Arc::new(Mutex::new(())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("serve_gemma4 listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
