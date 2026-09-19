# Task: NautiInferer v4 — API + Streaming + Worker Runtime (Rust)

Write the API layer, streaming infrastructure, and worker runtime for a production distributed inference control plane.

## Files to produce:

### 1. `src/main.rs` — Entry point
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // init tracing, load config, match mode (Coordinator vs Worker), run
}
```

### 2. `src/api/routes.rs` — Axum router setup
Routes:
- POST /v1/inference — submit inference job (returns streaming SSE)
- POST /v1/inference/cancel — cancel a running job
- GET /v1/models — list available models across fleet
- GET /v1/nodes — list connected worker nodes
- POST /internal/worker/connect — worker registration (websocket upgrade)
- POST /internal/worker/heartbeat — worker heartbeat
- GET /health — health check
- GET /metrics — prometheus metrics

### 3. `src/api/inference.rs` — Inference request handler
- Validate request (typed InferenceRequest struct)
- Check quota
- Schedule to best node via scheduler
- Return SSE stream of TokenChunks
- On client disconnect → cancel job

### 4. `src/streaming/sse.rs` — SSE response builder
- Typed TokenChunk: { job_id, delta, token_index, finished, model }
- Proper SSE framing: "data: {json}\n\n"
- Heartbeat keepalive every 15s (": keepalive\n\n")

### 5. `src/streaming/cancellation.rs` — Cancellation tokens
- CancellationToken wrapper (tokio_util)
- Propagates from client disconnect → scheduler → worker

### 6. `src/worker/runtime.rs` — Worker-side runtime
- Connects to coordinator via websocket
- Receives job assignments
- Spawns local inference (calls koboldcpp/llamacpp endpoint)
- Streams tokens back to coordinator
- Handles cancellation

### 7. `src/worker/local_engine.rs` — Local inference engine
```rust
pub struct LocalInferenceEngine {
    pub http: reqwest::Client,
    pub endpoint: String, // e.g. http://127.0.0.1:5001
}
```
- generate_stream: POST to koboldcpp /api/extra/generate/stream
- Parse SSE response, yield TokenChunks
- Timeout + retry logic

### 8. `src/worker/auth.rs` — Worker authentication
- Load ed25519 keypair from file (generate if missing)
- Sign challenge nonce from coordinator
- Send signed response

### 9. `src/protocol/control.rs` — Control protocol messages
```rust
pub enum ControlFrame {
    AssignJob { job_id: Uuid, request: InferenceRequest },
    CancelJob { job_id: Uuid },
    Heartbeat,
    Shutdown,
}

pub enum WorkerFrame {
    TokenChunk { job_id: Uuid, delta: String, index: u32, finished: bool },
    JobError { job_id: Uuid, error: String },
    HeartbeatAck { active_jobs: u32, free_vram_mb: u64 },
}
```

### 10. `src/protocol/auth.rs` — Auth protocol
```rust
pub struct ChallengeFrame { pub nonce: [u8; 32] }
pub struct ResponseFrame { pub signature: [u8; 64], pub public_key: [u8; 32] }
```

## Constraints:
- Axum 0.8 with State extractor
- All streaming via async-stream or tokio-stream
- Bounded channels (256 capacity)
- Proper error handling (thiserror + anyhow)
- tracing with structured fields
- Under 600 lines total across all files
- Production-quality: no unwrap() in handlers, proper error responses

## Output format:
```
=== FILE: src/main.rs ===
(content)

=== FILE: src/api/routes.rs ===
(content)
...
```
