//! Predictive Async MoE Placement (PAMP) — observation and hint interfaces.
//!
//! PAMP is an **external** orchestrator. This module provides zero-cost hooks
//! for the inference hot path to emit expert-selection telemetry and (later)
//! consume best-effort placement hints. Correctness never depends on PAMP.
//!
//! See `docs/gemma4_moe_optimization_plan.md` §18.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// Metadata passed alongside a MoE forward dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeDispatchMeta {
    pub layer: u32,
    pub token_seq: u64,
}

/// One router top-k decision for a single layer and token.
#[derive(Debug, Clone)]
pub struct ExpertSelectionEvent {
    pub layer: u32,
    pub token_seq: u64,
    pub expert_indices: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

/// Best-effort prefetch / residency suggestion from the PAMP service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementHint {
    pub layer: u32,
    pub expert: u32,
    /// Target device index in the `ExpertDevice` list.
    pub target_device: u32,
    /// Higher = more urgent prefetch.
    pub priority: u8,
}

/// Observes expert selections without blocking inference.
pub trait PampObserver: Send + Sync {
    fn on_expert_selection(&self, event: ExpertSelectionEvent);
}

/// Default no-op observer — zero overhead when PAMP is disabled.
pub struct NoopObserver;

impl PampObserver for NoopObserver {
    fn on_expert_selection(&self, _event: ExpertSelectionEvent) {}
}

/// Lock-protected ring buffer for async drain by a background thread or HTTP exporter.
pub struct RingBufferObserver {
    cap: usize,
    buf: Mutex<VecDeque<ExpertSelectionEvent>>,
}

impl RingBufferObserver {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(64),
            buf: Mutex::new(VecDeque::with_capacity(cap.min(4096))),
        }
    }

    /// Drain up to `max` events (non-blocking for inference; may block the caller).
    pub fn drain(&self, max: usize) -> Vec<ExpertSelectionEvent> {
        let mut guard = self.buf.lock().expect("pamp ring poisoned");
        let n = max.min(guard.len());
        guard.drain(..n).collect()
    }

    pub fn len(&self) -> usize {
        self.buf.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl PampObserver for RingBufferObserver {
    fn on_expert_selection(&self, event: ExpertSelectionEvent) {
        if let Ok(mut guard) = self.buf.lock() {
            if guard.len() >= self.cap {
                guard.pop_front();
            }
            guard.push_back(event);
        }
    }
}

/// Consumes placement hints produced by the external PAMP service.
pub trait PampHintConsumer: Send + Sync {
    fn poll_hints(&self, max: usize) -> Vec<PlacementHint>;
}

static GLOBAL_OBSERVER: OnceLock<Arc<dyn PampObserver>> = OnceLock::new();

/// Install a process-wide observer (call once at startup).
pub fn set_global_observer(observer: Arc<dyn PampObserver>) {
    let _ = GLOBAL_OBSERVER.set(observer);
}

fn global_observer() -> &'static dyn PampObserver {
    GLOBAL_OBSERVER
        .get()
        .map(|a| a.as_ref() as &dyn PampObserver)
        .unwrap_or(&NoopObserver)
}

/// Build the observer from environment (`PAMP_OBSERVE=1` enables ring buffer).
pub fn observer_from_env() -> Arc<dyn PampObserver> {
    if std::env::var("PAMP_OBSERVE").ok().as_deref() == Some("1") {
        let cap = std::env::var("PAMP_RING_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        Arc::new(RingBufferObserver::new(cap))
    } else {
        Arc::new(NoopObserver)
    }
}

/// Call once at process startup when `PAMP_OBSERVE=1`.
pub fn init_from_env() {
    if std::env::var("PAMP_OBSERVE").ok().as_deref() == Some("1") {
        set_global_observer(observer_from_env());
    }
}

/// Non-blocking telemetry emit — called from the MoE router after top-k.
pub fn emit_expert_selection(
    meta: Option<&MoeDispatchMeta>,
    expert_indices: &[usize],
    expert_weights: &[f32],
) {
    let Some(meta) = meta else {
        return;
    };
    global_observer().on_expert_selection(ExpertSelectionEvent {
        layer: meta.layer,
        token_seq: meta.token_seq,
        expert_indices: expert_indices.to_vec(),
        expert_weights: expert_weights.to_vec(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_drops_oldest_at_capacity() {
        let obs = RingBufferObserver::new(2);
        for i in 0..3 {
            obs.on_expert_selection(ExpertSelectionEvent {
                layer: i,
                token_seq: i as u64,
                expert_indices: vec![i as usize],
                expert_weights: vec![1.0],
            });
        }
        let drained = obs.drain(10);
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].layer, 1);
        assert_eq!(drained[1].layer, 2);
    }
}
