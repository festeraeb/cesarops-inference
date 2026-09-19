# Task: NautiInferer v4 — Scheduler + Node Management (Rust)

Write the core scheduler and node management modules for a production distributed inference control plane.

## Files to produce:

### 1. `src/types/mod.rs` — Core types
```rust
// NodeId, JobId, typed errors, config
```

### 2. `src/types/errors.rs` — thiserror errors

### 3. `src/node/registry.rs` — Node registry using DashMap
- `DashMap<NodeId, Arc<NodeRuntime>>` — no global mutex
- NodeRuntime: metadata (RwLock), mailbox (bounded mpsc), stats, last_heartbeat (AtomicU64), active_jobs (AtomicU32)
- register_node, deregister_node, get_node, list_nodes

### 4. `src/node/heartbeat.rs` — Heartbeat sweeper
- Background task: every 15s, remove nodes with last_heartbeat > 30s stale
- Update node stats on heartbeat receive

### 5. `src/node/manager.rs` — Node lifecycle
- Handle worker connect/disconnect
- Challenge-response auth (ed25519): server sends nonce, worker signs, server verifies
- Capability advertisement on connect

### 6. `src/scheduler/mod.rs` — Job scheduler
- `DashMap<JobId, ActiveJob>` for tracking
- ActiveJob: stream_tx, cancellation_token, node_id, created_at
- schedule_job: picks best node using scoring function
- cancel_job: propagates cancellation

### 7. `src/scheduler/scoring.rs` — Node scoring
```rust
pub fn compute_score(stats: &NodeStatistics) -> f64 {
    let queue_penalty = stats.active_jobs as f64 * 1.5;
    let latency_penalty = stats.avg_latency_ms / 1000.0;
    let vram_bonus = stats.free_vram_mb as f64 / 1024.0;
    let throughput_bonus = stats.tokens_per_sec;
    throughput_bonus + vram_bonus - queue_penalty - latency_penalty
}
```

### 8. `src/scheduler/affinity.rs` — KV-prefix affinity routing
- Hash prompt prefix → check if a node already has it cached
- Route continuations to same node to minimize prefill

### 9. `src/scheduler/quotas.rs` — Token quota management
- SQLite-backed: api_key → remaining_tokens
- reserve_tokens, release_tokens, check_quota

### 10. `src/config.rs` — Configuration
```rust
pub struct Config {
    pub mode: RuntimeMode, // Coordinator or Worker
    pub listen_port: u16,
    pub db_url: String,
    pub auth_keypair_path: String,
}
pub enum RuntimeMode { Coordinator, Worker }
```

## Constraints:
- No global Mutex anywhere — use DashMap, RwLock, atomics
- Bounded channels only (capacity 256)
- All errors via thiserror
- tracing for logging (info!, warn!, error! with structured fields)
- Under 600 lines total across all files
- Each file should be self-contained with proper use statements

## Output format:
```
=== FILE: src/types/mod.rs ===
(content)

=== FILE: src/types/errors.rs ===
(content)

=== FILE: src/node/registry.rs ===
(content)
...
```
