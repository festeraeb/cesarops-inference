// src/kv_cache.rs
//! Simple KV Cache — stores K and V vectors per layer per position.
//! Uses Vec<f32> storage (heap). NUMA-pinned arena version comes later.
//!
//! This gives the transformer context: each new token can attend to all
//! previous tokens' K/V vectors, not just itself.

/// Per-layer KV storage.
pub struct LayerKvCache {
    /// K vectors: [seq_len][kv_dim]
    pub keys: Vec<Vec<f32>>,
    /// V vectors: [seq_len][kv_dim]
    pub values: Vec<Vec<f32>>,
}

/// Full model KV cache across all layers.
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache {
                keys: Vec::with_capacity(max_seq_len),
                values: Vec::with_capacity(max_seq_len),
            })
            .collect();

        Self {
            layers,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Push a new K/V pair for a given layer at the current position.
    pub fn push(&mut self, layer_idx: usize, k: Vec<f32>, v: Vec<f32>) {
        if layer_idx < self.layers.len() {
            self.layers[layer_idx].keys.push(k);
            self.layers[layer_idx].values.push(v);
        }
    }

    /// Advance the sequence position (call once after all layers process a token).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get the current sequence length (number of cached positions).
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Get all cached K vectors for a layer: &[Vec<f32>] of length seq_len.
    pub fn get_keys(&self, layer_idx: usize) -> &[Vec<f32>] {
        &self.layers[layer_idx].keys
    }

    /// Get all cached V vectors for a layer.
    pub fn get_values(&self, layer_idx: usize) -> &[Vec<f32>] {
        &self.layers[layer_idx].values
    }

    // ───────────────────────────────────────────────────────────────────────
    // Speculative decoding support — snapshot / rollback / commit
    // ───────────────────────────────────────────────────────────────────────

    /// Record the current sequence position. Use before a speculative batch
    /// so we can roll back on rejection.
    #[inline]
    pub fn snapshot_pos(&self) -> u32 {
        self.seq_len as u32
    }

    /// Reset the sequence pointer to a prior snapshot position. Truncates
    /// each layer's K and V vectors so subsequent forward() calls overwrite
    /// the rolled-back range cleanly.
    pub fn rollback_to(&mut self, pos: u32) {
        let target = pos as usize;
        debug_assert!(
            target <= self.seq_len,
            "rollback_to({}) past current seq_len={}",
            target,
            self.seq_len
        );
        if target >= self.seq_len {
            return;
        }
        for layer in &mut self.layers {
            layer.keys.truncate(target);
            layer.values.truncate(target);
        }
        self.seq_len = target;
    }

    /// Mark all positions up to `pos` as committed. Currently a no-op (no
    /// separate committed/speculative state in this storage), but the API
    /// is retained so SpeculativeDecoder can record acceptance points
    /// without a state-machine rewrite later.
    #[inline]
    pub fn commit_through(&mut self, pos: u32) {
        debug_assert!(pos as usize <= self.seq_len);
    }
}
