```rust
=== FILE: src/types/mod.rs ===
use std::sync::Arc;
use serde::{Serialize, Deserialize};

pub mod errors;

pub type NodeId = String;
pub type JobId = String;
pub type ApiKey = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub model_names: Vec<String>,
    pub max_vram_mb: u32,
    pub supported_precision: String,
}

#[derive(Debug, Clone, Default)]
pub struct NodeStatistics {
    pub free_vram_mb: u32,
    pub avg_latency_ms: f64,
    pub tokens_per_sec: f64,
    pub active_jobs: u32,
}

pub mod node {
    pub mod registry;
    pub mod heartbeat;
    pub mod manager;
}

pub mod scheduler {
    pub mod scoring;
    pub mod affinity;
    pub mod quotas;
}

=== FILE: src/types/errors.rs ===
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NautiError {
    #[error("Node registry error: {0}")]
    RegistryError(String),
    #[error("Authentication failed: {0}")]
    AuthError(String),
    #[error("Job error: {0}")]
    JobError(String),
    #[error("Quota exceeded for key: {0}")]
    QuotaExceeded(String),
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Internal error: {0}")]
    Internal(String),
}

=== FILE: src/node/registry.rs ===
use crate::types::{NodeId, NodeCapabilities, NodeStatistics};
use crate::types::errors::NautiError;
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use tokio::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct NodeRuntime {
    pub id: NodeId,
    pub capabilities: NodeCapabilities,
    pub stats: RwLock<NodeStatistics>,
    pub mailbox: mpsc::Sender<String>, // Simplified command stream
    pub last_heartbeat: AtomicU64,
    pub active_jobs: AtomicU32,
}

pub struct NodeRegistry {
    nodes: DashMap<NodeId, Arc<NodeRuntime>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self { nodes: DashMap::new() }
    }

    pub fn register_node(&self, id: NodeId, caps: NodeCapabilities, tx: mpsc::Sender<String>) -> Result<Arc<NodeRuntime>, NautiError> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let runtime = Arc::new(NodeRuntime {
            id: id.clone(),
            capabilities: caps,
            stats: RwLock::new(NodeStatistics::default()),
            mailbox: tx,
            last_heartbeat: AtomicU64::new(now),
            active_jobs: AtomicU32::new(0),
        });
        self.nodes.insert(id, runtime.clone());
        Ok(runtime)
    }

    pub fn deregister_node(&self, id: &NodeId) {
        self.nodes.remove(id);
    }

    pub fn get_node(&self, id: &NodeId) -> Option<Arc<NodeRuntime>> {
        self.nodes.get(id).map(|r| r.value().clone())
    }

    pub fn list_nodes(&self) -> Vec<Arc<NodeRuntime>> {
        self.nodes.iter().map(|r| r.value().clone()).collect()
    }
}

=== FILE: src/node/heartbeat.rs ===
use crate::node::registry::NodeRegistry;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time;
use tracing::{info, warn};

pub async fn start_heartbeat_sweeper(registry: Arc<NodeRegistry>) {
    let mut interval = time::interval(Duration::from_secs(15));
    loop {
        interval.tick().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let nodes = registry.list_nodes();

        for node in nodes {
            let last = node.last_heartbeat.load(std::sync::atomic::Ordering::Relaxed);
            if now > last + 30 {
                warn!(node_id = %node.id, "Node heartbeat stale, removing");
                registry.deregister_node(&node.id);
            }
        }
    }
}

=== FILE: src/node/manager.rs ===
use crate::types::{NodeId, NodeCapabilities};
use crate::types::errors::NautiError;
use crate::node::registry::NodeRegistry;
use ed25519_dalek::{Verifier, VerifyingKey, Signature};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error};

pub struct NodeManager {
    registry: Arc<NodeRegistry>,
    public_key: VerifyingKey,
}

impl NodeManager {
    pub fn new(registry: Arc<NodeRegistry>, public_key: VerifyingKey) -> Self {
        Self { registry, public_key }
    }

    pub async fn handle_connection(
        &self, 
        id: NodeId, 
        caps: NodeCapabilities, 
        nonce: &[u8], 
        signature: &[u8],
        tx: mpsc::Sender<String>
    ) -> Result<(), NautiError> {
        // Challenge-Response Verification
        let sig = Signature::from_bytes(signature.try_into().map_err(|_| NautiError::AuthError("Invalid sig format".into()))?);
        self.public_key.verify(nonce, &sig).map_err(|_| NautiError::AuthError("Signature mismatch".into()))?;

        info!(node_id = %id, "Node authenticated successfully");
        self.registry.register_node(id, caps, tx)?;
        Ok(())
    }
}

=== FILE: src/scheduler/mod.rs ===
use crate::types::{JobId, NodeId};
use crate::types::errors::NautiError;
use crate::node::registry::NodeRegistry;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

pub mod scoring;
pub mod affinity;
pub mod quotas;

pub struct ActiveJob {
    pub node_id: NodeId,
    pub cancel_token: CancellationToken,
}

pub struct Scheduler {
    registry: Arc<NodeRegistry>,
    active_jobs: DashMap<JobId, ActiveJob>,
}

impl Scheduler {
    pub fn new(registry: Arc<NodeRegistry>) -> Self {
        Self {
            registry,
            active_jobs: DashMap::new(),
        }
    }

    pub async fn schedule_job(&self, job_id: JobId, prompt: &str) -> Result<NodeId, NautiError> {
        let nodes = self.registry.list_nodes();
        let best_node = nodes.iter()
            .map(|n| (n.clone(), scoring::compute_score(&n.stats.read().unwrap())))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or_else(|| NautiError::JobError("No available nodes".into()))?;

        let node = best_node.0;
        self.active_jobs.insert(job_id.clone(), ActiveJob {
            node_id: node.id.clone(),
            cancel_token: CancellationToken::new(),
        });

        Ok(node.id)
    }

    pub fn cancel_job(&self, job_id: &JobId) {
        if let Some((_, job)) = self.active_jobs.remove(job_id) {
            job.cancel_token.cancel();
        }
    }
}

=== FILE: src/scheduler/scoring.rs ===
use crate::types::NodeStatistics;

pub fn compute_score(stats: &NodeStatistics) -> f64 {
    let queue_penalty = stats.active_jobs as f64 * 1.5;
    let latency_penalty = stats.avg_latency_ms / 1000.0;
    let vram_bonus = stats.free_vram_mb as f64 / 1024.0;
    let throughput_bonus = stats.tokens_per_sec;
    throughput_bonus + vram_bonus - queue_penalty - latency_penalty
}

=== FILE: src/scheduler/affinity.rs ===
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn get_affinity_prefix(prompt: &str) -> String {
    let mut s = DefaultHasher::new();
    prompt.chars().take(20).collect::<String>().hash(&mut s);
    format!("{:x}", s.finish())
}

=== FILE: src/scheduler/quotas.rs ===
use crate::types::{ApiKey, NautiError};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};

pub struct QuotaManager {
    conn: Arc<Mutex<Connection>>,
}

impl QuotaManager {
    pub fn new(db_url: &str) -> Result<Self, NautiError> {
        let conn = Connection::open(db_url)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS quotas (api_key TEXT PRIMARY KEY, tokens INTEGER)",
            [],
        )?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    pub fn reserve_tokens(&self, key: &ApiKey, amount: u64) -> Result<(), NautiError> {
        let conn = self.conn.lock().unwrap();
        let remaining: u64 = conn.query_row(
            "SELECT tokens FROM quotas WHERE api_key = ?",
            params![key],
            |row| row.get(0),
        ).unwrap_or(0);

        if remaining < amount {
            return Err(NautiError::QuotaExceeded(key.clone()));
        }

        conn.execute(
            "UPDATE quotas SET tokens = tokens - ? WHERE api_key = ?",
            params![amount, key],
        )?;
        Ok(())
    }

    pub fn release_tokens(&self, key: &ApiKey, amount: u64) -> Result<(), NautiError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE quotas SET tokens = tokens + ? WHERE api_key = ?",
            params![amount, key],
        )?;
        Ok(())
    }
}

=== FILE: src/config.rs ===
pub enum RuntimeMode { Coordinator, Worker }

pub struct Config {
    pub mode: RuntimeMode,
    pub listen_port: u16,
    pub db_url: String,
    pub auth_keypair_path: String,
}
```
