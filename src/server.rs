//! HTTP API Server — KoboldCPP-compatible drop-in replacement.
//!
//! Endpoints:
//!   POST /api/v1/generate — same JSON format as KoboldCPP
//!   GET  /api/v1/model    — model info
//!   GET  /health          — liveness check

use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::info;

use crate::arena::InferenceArena;
use crate::bridge;
use crate::kv_cache::KvCache;
use crate::loader::ModelWeights;
use crate::sampling::{self, SamplingParams};
use crate::tokenizer::ZeroAllocBpeTokenizer;
use crate::transformer::{TransformerConfig, TransformerDecoder};
use crate::weight_cache::WeightCache;

/// Server state shared across handlers — read-only after construction.
/// The model lives behind Arc references so concurrent reads (health, model_info)
/// don't block inference, and inference itself is gated by `infer_lock`
/// (single-flight) plus `idem_cache` (retry-storm immunity).
pub struct InferenceState {
    pub model_name: String,
    pub weights: Arc<ModelWeights>,
    pub decoder: Arc<TransformerDecoder>,
    pub tokenizer: Arc<ZeroAllocBpeTokenizer>,
    pub prefix_cache: Arc<parking_lot::RwLock<crate::kv_prefix_cache::KvPrefixCache>>,
    /// Single-flight permit. Only one forward pass runs at a time.
    pub infer_lock: Arc<Semaphore>,
    /// Idempotency cache: request_id → completed text. Bounded by TTL+size.
    pub idem_cache: Arc<Mutex<IdemCache>>,
}

#[derive(Default)]
pub struct IdemCache {
    map: HashMap<String, IdemEntry>,
}

struct IdemEntry {
    text: String,
    inserted_at: Instant,
}

impl IdemCache {
    pub fn get_fresh(&mut self, key: &str, ttl: Duration) -> Option<String> {
        let now = Instant::now();
        // GC stale entries (cheap pass since cache is bounded).
        self.map.retain(|_, v| now.duration_since(v.inserted_at) < ttl);
        self.map.get(key).map(|e| e.text.clone())
    }
    pub fn insert(&mut self, key: String, text: String) {
        // Soft cap to prevent unbounded growth on long-running servers.
        if self.map.len() > 1024 {
            self.map.clear();
        }
        self.map.insert(key, IdemEntry { text, inserted_at: Instant::now() });
    }
}

/// KoboldCPP-compatible generate request.
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_length")]
    pub max_length: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_rep_pen")]
    pub rep_pen: f32,
    #[serde(default)]
    pub stop_sequence: Vec<String>,
    /// If true (default), wrap prompt in Qwen2.5 ChatML template.
    /// Set to false for raw completion mode.
    #[serde(default = "default_chat_template")]
    pub use_chat_template: bool,
    /// Optional client-supplied request id for idempotent retries.
    #[serde(default)]
    pub request_id: Option<String>,
}

fn default_chat_template() -> bool { true }

fn default_max_length() -> u32 { 512 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }
fn default_rep_pen() -> f32 { 1.1 }

/// KoboldCPP-compatible generate response.
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub results: Vec<GenerateResult>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResult {
    pub text: String,
}

/// Model info response.
#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub result: String,
    /// KV prefix cache stats (hit/miss/eviction telemetry).
    pub prefix_cache: PrefixCacheStats,
}

#[derive(Debug, Serialize, Default)]
pub struct PrefixCacheStats {
    pub capacity_tokens: usize,
    pub total_tokens: usize,
    pub nodes_alive: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub model: String,
}

async fn health(State(state): State<Arc<InferenceState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "cesarops-inference".to_string(),
        model: state.model_name.clone(),
    })
}

async fn model_info(State(state): State<Arc<InferenceState>>) -> Json<ModelResponse> {
    let cache = state.prefix_cache.read();
    let stats = cache.stats();
    let capacity = cache.capacity();
    drop(cache);
    Json(ModelResponse {
        result: format!("cesarops-inference/{}", state.model_name),
        prefix_cache: PrefixCacheStats {
            capacity_tokens: capacity,
            total_tokens: stats.total_tokens,
            nodes_alive: stats.nodes_alive,
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
        },
    })
}

async fn generate(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    info!(
        "Generate: prompt_len={}, max_length={}, temp={}, request_id={:?}",
        req.prompt.len(), req.max_length, req.temperature, req.request_id
    );

    // ── Idempotent retry guard ──────────────────────────────────────────
    // If the client passes a request_id and we've already completed it
    // recently, return the cached text instead of starting a duplicate
    // forward pass. This neutralizes the HTTP-retry-on-timeout storm that
    // caused the "stuck on layer 0" symptom.
    if let Some(ref rid) = req.request_id {
        let mut cache = state.idem_cache.lock().await;
        if let Some(text) = cache.get_fresh(rid, Duration::from_secs(600)) {
            info!("[idem] cache hit request_id={}", rid);
            return Json(GenerateResponse {
                results: vec![GenerateResult { text }],
            });
        }
    }

    // ── Single-flight gate ──────────────────────────────────────────────
    // GPU forward passes share device buffers (`buf_a`, `buf_c`, etc.) so
    // concurrent inference would clobber each other. Permit count = 1 today;
    // bump when per-stream scratch lands.
    let _permit = state
        .infer_lock
        .acquire()
        .await
        .expect("infer semaphore closed");

    // Cheap clones — they're all Arc.
    let weights = Arc::clone(&state.weights);
    let decoder = Arc::clone(&state.decoder);
    let tokenizer = Arc::clone(&state.tokenizer);
    let prefix_cache = Arc::clone(&state.prefix_cache);

    // KV prefix-cache telemetry. v1: bookkeeping only (no actual KV state
    // restore yet). The cache records hits/misses on prompt prefix and
    // commits the prefix at end-of-generation so subsequent identical
    // prompts at least register as hits in the stats endpoint.
    let prompt_template = if req.use_chat_template {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            req.prompt
        )
    } else {
        req.prompt.clone()
    };
    let prompt_tokens_for_cache = tokenizer.encode(&prompt_template);
    let prompt_hashes = crate::kv_prefix_cache::hash_sequence(&prompt_tokens_for_cache, 0);

    {
        let mut cache = prefix_cache.write();
        match cache.prefix_match(&prompt_hashes) {
            Some((len, _slice)) => info!("[cache] hit prefix_len={}", len),
            None => info!("[cache] miss"),
        }
    }

    // Run actual inference on a blocking thread so the tokio runtime stays
    // responsive for /health and /api/v1/model while the GPU is busy.
    let prompt = req.prompt.clone();
    let max_length = req.max_length;
    let temperature = req.temperature;
    let top_p = req.top_p;
    let rep_pen = req.rep_pen;
    let use_chat = req.use_chat_template;
    let response_text = tokio::task::spawn_blocking(move || {
        run_inference(
            &weights, &decoder, &tokenizer,
            &prompt, max_length, temperature, top_p, rep_pen, use_chat,
        )
    })
    .await
    .unwrap_or_else(|e| format!("inference panic: {}", e));

    // Commit prefix to cache so the next identical prompt registers as hit.
    // v2 will store actual KV state here for true prefill skip.
    {
        let mut cache = prefix_cache.write();
        cache.commit(
            &prompt_hashes,
            crate::kv_prefix_cache::KvSlice {
                start_pos: 0,
                len: prompt_tokens_for_cache.len() as u32,
                layer_data_handle: 0,
            },
        );
    }

    // Stash the result under the supplied request_id so retries return cached text.
    if let Some(ref rid) = req.request_id {
        let mut cache = state.idem_cache.lock().await;
        cache.insert(rid.clone(), response_text.clone());
    }

    Json(GenerateResponse {
        results: vec![GenerateResult { text: response_text }],
    })
}

/// Run the actual inference loop: embed → forward → sample → decode.
fn run_inference(
    weights: &ModelWeights,
    decoder: &TransformerDecoder,
    tokenizer: &ZeroAllocBpeTokenizer,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    rep_pen: f32,
    use_chat_template: bool,
) -> String {
    let h = decoder.config.hidden_size;
    let vocab_size = decoder.config.vocab_size;

    // --- Apply Qwen2.5 ChatML template if requested ---
    let effective_prompt = if use_chat_template {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    } else {
        prompt.to_string()
    };

    // --- Tokenize prompt ---
    let prompt_tokens = tokenizer.encode(&effective_prompt);
    info!("Prompt tokens: {} (real_bpe={}, chat_template={}), ids={:?}", prompt_tokens.len(), tokenizer.has_real_vocab, use_chat_template, &prompt_tokens[..prompt_tokens.len().min(20)]);

    // --- Get embedding weight ---
    let embed_name = find_embed_name(weights);
    let embed_data = if let Some(bytes) = weights.tensor_bytes(&embed_name) {
        let region = weights.tensors.get(&embed_name).unwrap();
        let n_elements: usize = region.shape.iter().product();
        bridge::dequantize_tensor(bytes, region.quant_type, n_elements)
    } else {
        info!("WARNING: No embedding tensor found, using random init");
        vec![0.01f32; vocab_size * h]
    };

    // --- Get initial hidden state from last prompt token ---
    let last_prompt_token = *prompt_tokens.last().unwrap_or(&0) as usize;
    let mut hidden_state = get_embedding(&embed_data, last_prompt_token, h, vocab_size);

    // --- Process prompt through KV cache (prefill) ---
    let mut kv_cache = KvCache::new(decoder.config.num_layers, 4096);

    // Process all prompt tokens through the transformer to build KV cache
    for (i, &token_id) in prompt_tokens.iter().enumerate() {
        let mut h_state = get_embedding(&embed_data, token_id as usize, h, vocab_size);
        let _logits = decoder.forward(&mut h_state, i, weights, &mut kv_cache);
        kv_cache.advance();
        // Keep the last hidden state for generation
        if i == prompt_tokens.len() - 1 {
            hidden_state = h_state;
        }
    }
    info!("Prefill complete: {} tokens cached", kv_cache.len());

    // --- Autoregressive generation loop ---
    let mut generated_tokens: Vec<u32> = Vec::new();

    let params = SamplingParams {
        temperature,
        top_p,
        rep_pen,
        rep_pen_range: 256,
        stop_sequences: Vec::new(),
        banned_tokens: std::collections::HashSet::new(),
    };

    let diag = crate::diagnostics::Diagnostics::from_env();

    for step in 0..max_tokens {
        let position = prompt_tokens.len() + step as usize;

        // Forward pass with KV cache
        let mut logits = decoder.forward(&mut hidden_state, position, weights, &mut kv_cache);
        kv_cache.advance();

        // Gated NaN check + logits stats. No-op in Off mode (default).
        // Replaces two unconditional CPU scans per token + per-token info!() log.
        diag.check_nan("hidden_state", &hidden_state);
        diag.log_logits(&logits);

        // Sample next token
        let recent: Vec<u32> = prompt_tokens.iter()
            .chain(generated_tokens.iter())
            .rev()
            .take(params.rep_pen_range)
            .copied()
            .collect();
        let next_token = sampling::sample(&mut logits, &params, &recent);

        // Check for EOS (Qwen EOS = 151645)
        if next_token == 151645 || next_token == 151643 {
            info!("EOS at step {}", step);
            break;
        }

        generated_tokens.push(next_token);

        // Embed the new token for next iteration
        hidden_state = get_embedding(&embed_data, next_token as usize, h, vocab_size);

        // Log progress every token (debug mode only — gated to avoid
        // per-token tracing overhead in production).
        if diag.debug_enabled() {
            tracing::debug!("Step {}: token_id={}", step, next_token);
        }
    }

    // Decode all generated tokens at once
    let output = tokenizer.decode(&generated_tokens);
    info!("Generated {} tokens, decoded to {} chars", generated_tokens.len(), output.len());
    output
}

/// Look up a token's embedding vector from the embedding matrix.
/// GGUF column-major: shape [1536, 151936] means 1536 is contiguous (the stride).
/// Each token's embedding is a contiguous block of 1536 elements.
/// Token t's embedding = data[t * hidden_size .. (t+1) * hidden_size]
fn get_embedding(embed_data: &[f32], token_id: usize, hidden_size: usize, vocab_size: usize) -> Vec<f32> {
    let token_id = token_id.min(vocab_size - 1);
    let start = token_id * hidden_size;
    let end = start + hidden_size;
    if end <= embed_data.len() {
        embed_data[start..end].to_vec()
    } else {
        (0..hidden_size).map(|i| 0.001 * ((i % 7) as f32 - 3.0) / 3.0).collect()
    }
}

/// Find the embedding tensor name (different models use different names).
fn find_embed_name(weights: &ModelWeights) -> String {
    let candidates = [
        "token_embd.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embeddings.word_embeddings.weight",
    ];
    for name in &candidates {
        if weights.tensors.contains_key(*name) {
            return name.to_string();
        }
    }
    // Return first candidate as default
    candidates[0].to_string()
}

/// Generate check endpoint (KoboldCPP compatibility).
async fn generate_check(State(_state): State<Arc<InferenceState>>) -> Json<GenerateResponse> {
    Json(GenerateResponse {
        results: vec![GenerateResult { text: String::new() }],
    })
}

/// Build and run the inference server.
pub async fn run_server(
    model_name: String,
    port: u16,
    weights: Arc<ModelWeights>,
    arena: Arc<InferenceArena>,
    tokenizer: Arc<ZeroAllocBpeTokenizer>,
    gpu_context: Option<Arc<crate::gpu_context::GpuContext>>,
) -> anyhow::Result<()> {
    // Derive intermediate_size from FFN gate tensor shape
    // GGUF shape for ffn_gate is [hidden_dim, intermediate_size] or [intermediate_size, hidden_dim]
    let intermediate_size = weights.tensors.get("blk.0.ffn_gate.weight")
        .map(|r| {
            // The dimension that ISN'T hidden_dim is the intermediate_size
            if r.shape.len() >= 2 {
                let dim0 = r.shape[0];
                let dim1 = r.shape[1];
                if dim0 == weights.hidden_dim { dim1 } else { dim0 }
            } else {
                weights.hidden_dim * 4
            }
        })
        .unwrap_or(weights.hidden_dim * 4);
    
    info!("Derived intermediate_size={} from FFN gate tensor", intermediate_size);

    let config = TransformerConfig {
        vocab_size: weights.vocab_size,
        hidden_size: weights.hidden_dim,
        intermediate_size,
        num_layers: weights.n_layers,
        num_heads: weights.n_heads,
        num_kv_heads: weights.n_kv_heads,
        head_dim: weights.hidden_dim / weights.n_heads,
        max_seq_len: 4096,
        rope_theta: 1000000.0,
        rms_norm_eps: 1e-6,
    };

    // Pre-dequantize all weights into a CPU cache (eliminates per-token dequant overhead)
    let weight_cache = Arc::new(WeightCache::from_model(&weights));

    // If GPU is available, upload weights to VRAM ONCE so per-token matmuls
    // don't re-upload tens of MB to the device. This is the primary fix for
    // the 12-second-per-layer pathology.
    let gpu_weights: Option<Arc<crate::weight_cache::GpuWeightCache>> = if let Some(ref gpu) = gpu_context {
        Some(Arc::new(crate::weight_cache::GpuWeightCache::upload(&weight_cache, Arc::clone(gpu))))
    } else {
        None
    };

    let mut decoder_builder =
        TransformerDecoder::new(config, arena).with_weight_cache(Arc::clone(&weight_cache));
    if let Some(ref gpu) = gpu_context {
        decoder_builder = decoder_builder.with_gpu(Arc::clone(gpu));
    }
    if let Some(ref gw) = gpu_weights {
        decoder_builder = decoder_builder.with_gpu_weights(Arc::clone(gw));
    }
    let decoder = Arc::new(decoder_builder);

    // KV prefix cache. v1: telemetry + bookkeeping only (real prefill skip
    // lands when KvCache state retention across requests is wired). Sized
    // for 16 GB P100 / Qwen 1.5B GQA: 57344 bytes/token at fp32, ~75k token cap.
    // Auto-shrinks for smaller VRAM via from_vram_budget.
    let prefix_cache = Arc::new(parking_lot::RwLock::new(
        crate::kv_prefix_cache::KvPrefixCache::from_vram_budget(
            16 * 1024 * 1024 * 1024,
            57344,
        ),
    ));

    let state = Arc::new(InferenceState {
        model_name: model_name.clone(),
        weights,
        decoder,
        tokenizer,
        prefix_cache,
        infer_lock: Arc::new(Semaphore::new(1)),
        idem_cache: Arc::new(Mutex::new(IdemCache::default())),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/model", get(model_info))
        .route("/api/v1/generate", post(generate))
        .route("/api/extra/generate/check", get(generate_check))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("cesarops-inference server on {}", addr);
    info!("  POST /api/v1/generate  — KoboldCPP-compatible");
    info!("  GET  /api/v1/model     — model info");
    info!("  GET  /health           — liveness");
    info!("  Model: {}", model_name);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
