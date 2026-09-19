//! Token-hash trie with generational arena backing for KV cache prefix reuse.
//!
//! Use case: chat workloads with long system prompts (800-2000 tokens) repeat
//! 90%+ of the prefix on every turn. This cache lets us skip prefill for
//! the cached portion.
//!
//! Architecture: Vec<KvNode> as arena, NodeId = u32 indices, FxHashMap for
//! per-node child lookups, VecDeque for LRU ordering.
//!
//! Concurrency: wrap in `parking_lot::RwLock` at the call site. Read-locked
//! for prefix_match, write-locked for commit/evict.
//!
//! Eviction: bounded by `max_tokens`, evict LRU leaf nodes first.

use std::collections::VecDeque;
use rustc_hash::FxHashMap;

pub type TokenHash = u64;
pub type NodeId = u32;

const ROOT: NodeId = 0;
const NULL_NODE: NodeId = u32::MAX;

/// Opaque handle into the engine's KvCache buffer for a cached prefix span.
/// Currently a position range; extended later when GPU-side KV bytes get
/// referenced via wgpu::Buffer handles.
pub type KvDataHandle = u64;

#[derive(Clone, Debug)]
pub struct KvSlice {
    pub start_pos: u32,
    pub len: u32,
    pub layer_data_handle: KvDataHandle,
}

#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub nodes_alive: usize,
    pub total_tokens: usize,
    pub evictions: u64,
    pub hits: u64,
    pub misses: u64,
}

struct KvNode {
    parent: NodeId,
    children: FxHashMap<TokenHash, NodeId>,
    kv_slice: Option<KvSlice>,
    depth: u32,
    last_access_epoch: u64,
    /// Number of children + 1-if-has-kv_slice. When this hits 0 the node
    /// is eligible for eviction and removal.
    refcount: u32,
}

impl KvNode {
    fn new_root() -> Self {
        Self {
            parent: NULL_NODE,
            children: FxHashMap::default(),
            kv_slice: None,
            depth: 0,
            last_access_epoch: 0,
            refcount: 1,  // root never evicts
        }
    }

    fn new_child(parent: NodeId, depth: u32, epoch: u64) -> Self {
        Self {
            parent,
            children: FxHashMap::default(),
            kv_slice: None,
            depth,
            last_access_epoch: epoch,
            refcount: 0,
        }
    }
}

pub struct KvPrefixCache {
    arena: Vec<KvNode>,
    free_list: Vec<NodeId>,
    /// Nodes with kv_slice, sorted approximately by last_access_epoch (oldest front).
    /// Real LRU ordering is enforced lazily during eviction by epoch comparison.
    lru: VecDeque<NodeId>,
    total_tokens: usize,
    max_tokens: usize,
    epoch: u64,
    stats: CacheStats,
}

impl KvPrefixCache {
    /// Create a cache with explicit token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            arena: vec![KvNode::new_root()],
            free_list: Vec::new(),
            lru: VecDeque::new(),
            total_tokens: 0,
            max_tokens,
            epoch: 0,
            stats: CacheStats::default(),
        }
    }

    /// Auto-derive token budget from VRAM bytes available, given per-token
    /// KV bytes for this model (computed once at model load).
    ///
    /// For Qwen 1.5B GQA (28 layers, 2 KV heads, head_dim 128, fp32 K+V):
    ///   per_token_bytes = 28 * 2 * 2 * 128 * 4 = 57344 bytes
    /// On a 16 GB P100 with ~25% reserved for cache:
    ///   max_tokens = (16 * 1024^3 / 4) / 57344 ≈ 75000 tokens
    pub fn from_vram_budget(vram_bytes: u64, per_token_kv_bytes: u64) -> Self {
        let budget = vram_bytes / 4; // reserve 25% for prefix cache
        let max_tokens = (budget / per_token_kv_bytes.max(1)) as usize;
        Self::new(max_tokens.max(256))
    }

    /// Find the longest cached prefix of `tokens`. Returns the prefix length
    /// (number of tokens matched) and the KvSlice covering exactly that prefix.
    ///
    /// Updates LRU access epoch for hit nodes (LRU promotion).
    pub fn prefix_match(&mut self, tokens: &[TokenHash]) -> Option<(usize, KvSlice)> {
        self.epoch = self.epoch.wrapping_add(1);

        let mut node = ROOT;
        let mut last_kv: Option<(usize, KvSlice)> = None;
        let mut last_kv_node: Option<NodeId> = None;
        let mut depth = 0;

        for &t in tokens {
            let next = self.arena[node as usize].children.get(&t).copied();
            match next {
                Some(n) => {
                    node = n;
                    depth += 1;
                    self.arena[n as usize].last_access_epoch = self.epoch;
                    if let Some(ref kv) = self.arena[n as usize].kv_slice {
                        last_kv = Some((depth, kv.clone()));
                        last_kv_node = Some(n);
                    }
                }
                None => break,
            }
        }

        if let Some(node_id) = last_kv_node {
            // LRU promotion: remove from current position, push to back.
            // Linear scan but bounded by lru.len() which is bounded by max_tokens.
            if let Some(pos) = self.lru.iter().position(|&n| n == node_id) {
                self.lru.remove(pos);
                self.lru.push_back(node_id);
            }
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }

        last_kv
    }

    /// Commit a token sequence + KV slice into the cache.
    /// Trims via evict_lru if total_tokens exceeds budget.
    pub fn commit(&mut self, tokens: &[TokenHash], kv: KvSlice) {
        self.epoch = self.epoch.wrapping_add(1);

        let mut node = ROOT;
        for &t in tokens {
            let next = self.arena[node as usize].children.get(&t).copied();
            node = match next {
                Some(n) => {
                    self.arena[n as usize].last_access_epoch = self.epoch;
                    n
                }
                None => self.alloc_child(node, t),
            };
        }

        let kv_len = kv.len as usize;
        let arena_node = &mut self.arena[node as usize];
        if arena_node.kv_slice.is_none() {
            arena_node.refcount += 1;
            self.lru.push_back(node);
            self.total_tokens += kv_len;
        } else if let Some(ref old_kv) = arena_node.kv_slice {
            // Replace existing slice — adjust token total
            self.total_tokens =
                self.total_tokens.saturating_sub(old_kv.len as usize) + kv_len;
        }
        self.arena[node as usize].kv_slice = Some(kv);

        if self.total_tokens > self.max_tokens {
            self.evict_lru();
        }
    }

    /// Evict LRU leaf nodes with kv_slice until we're under budget.
    /// Internal nodes (children > 0, no kv_slice) are kept as path-only.
    pub fn evict_lru(&mut self) {
        let mut evicted = 0u64;

        while self.total_tokens > self.max_tokens && !self.lru.is_empty() {
            let node = match self.lru.pop_front() {
                Some(n) => n,
                None => break,
            };

            let n_idx = node as usize;
            if self.arena[n_idx].kv_slice.is_none() {
                // Already evicted via another path; skip.
                continue;
            }

            let len = self.arena[n_idx].kv_slice.as_ref().unwrap().len as usize;
            self.arena[n_idx].kv_slice = None;
            self.arena[n_idx].refcount = self.arena[n_idx].refcount.saturating_sub(1);
            self.total_tokens = self.total_tokens.saturating_sub(len);
            evicted += 1;

            // Garbage collect the chain up to root if leaf has no children either
            self.try_collect_chain(node);
        }

        self.stats.evictions += evicted;
    }

    fn try_collect_chain(&mut self, mut node: NodeId) {
        while node != ROOT {
            let n_idx = node as usize;
            if self.arena[n_idx].refcount > 0 || !self.arena[n_idx].children.is_empty() {
                break;
            }
            // Detach from parent
            let parent = self.arena[n_idx].parent;
            if parent == NULL_NODE {
                break;
            }
            // Find and remove the child link from parent
            let child_key = self.arena[parent as usize]
                .children
                .iter()
                .find_map(|(k, &v)| if v == node { Some(*k) } else { None });
            if let Some(k) = child_key {
                self.arena[parent as usize].children.remove(&k);
                self.arena[parent as usize].refcount =
                    self.arena[parent as usize].refcount.saturating_sub(1);
            }
            // Return node to free list
            self.arena[n_idx].kv_slice = None;
            self.arena[n_idx].children.clear();
            self.arena[n_idx].parent = NULL_NODE;
            self.free_list.push(node);
            node = parent;
        }
    }

    fn alloc_child(&mut self, parent: NodeId, edge: TokenHash) -> NodeId {
        let p_depth = self.arena[parent as usize].depth;
        let id = if let Some(reused) = self.free_list.pop() {
            self.arena[reused as usize] = KvNode::new_child(parent, p_depth + 1, self.epoch);
            reused
        } else {
            let id = self.arena.len() as NodeId;
            self.arena.push(KvNode::new_child(parent, p_depth + 1, self.epoch));
            id
        };

        self.arena[parent as usize].children.insert(edge, id);
        self.arena[parent as usize].refcount += 1;
        id
    }

    pub fn stats(&self) -> CacheStats {
        let mut s = self.stats.clone();
        s.nodes_alive = self.arena.len() - self.free_list.len();
        s.total_tokens = self.total_tokens;
        s
    }

    pub fn capacity(&self) -> usize {
        self.max_tokens
    }
}

/// Hash a single token + role marker into a TokenHash.
/// Uses the golden-ratio mixer (simple, fast, no crypto, low collision).
/// Role marker distinguishes system / user / assistant turns so identical
/// content under different roles doesn't collide.
#[inline]
pub fn hash_token(token_id: u32, role_marker: u32) -> TokenHash {
    let mut h = (token_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= (role_marker as u64).wrapping_add(0x85eb_ca6b_2bb1_4f25);
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h
}

/// Hash an entire token sequence into a list of TokenHashes for prefix
/// match. Each position's hash includes the same role marker for the
/// whole sequence (caller chooses role per-segment for chat templates).
pub fn hash_sequence(tokens: &[u32], role_marker: u32) -> Vec<TokenHash> {
    tokens.iter().map(|&t| hash_token(t, role_marker)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(start: u32, len: u32) -> KvSlice {
        KvSlice {
            start_pos: start,
            len,
            layer_data_handle: 0,
        }
    }

    #[test]
    fn match_returns_longest_prefix() {
        let mut cache = KvPrefixCache::new(1024);
        let toks = vec![hash_token(1, 0), hash_token(2, 0), hash_token(3, 0)];
        cache.commit(&toks, slice(0, 3));

        let result = cache.prefix_match(&toks);
        assert!(result.is_some());
        let (depth, kv) = result.unwrap();
        assert_eq!(depth, 3);
        assert_eq!(kv.len, 3);
    }

    #[test]
    fn match_returns_partial_prefix() {
        let mut cache = KvPrefixCache::new(1024);
        let stored = vec![hash_token(1, 0), hash_token(2, 0)];
        cache.commit(&stored, slice(0, 2));

        let query = vec![hash_token(1, 0), hash_token(2, 0), hash_token(3, 0)];
        let (depth, kv) = cache.prefix_match(&query).unwrap();
        assert_eq!(depth, 2);
        assert_eq!(kv.len, 2);
    }

    #[test]
    fn match_misses_completely() {
        let mut cache = KvPrefixCache::new(1024);
        cache.commit(&[hash_token(1, 0)], slice(0, 1));

        let query = vec![hash_token(99, 0)];
        assert!(cache.prefix_match(&query).is_none());
    }

    #[test]
    fn lru_eviction_under_budget() {
        let mut cache = KvPrefixCache::new(5);  // tiny budget
        cache.commit(&[hash_token(1, 0)], slice(0, 3));
        cache.commit(&[hash_token(2, 0)], slice(0, 3));  // triggers evict
        let stats = cache.stats();
        assert!(stats.evictions > 0);
        assert!(stats.total_tokens <= 5);
    }

    #[test]
    fn lru_promotion_via_match() {
        let mut cache = KvPrefixCache::new(5);
        cache.commit(&[hash_token(1, 0)], slice(0, 2));
        cache.commit(&[hash_token(2, 0)], slice(0, 2));

        // Touch first entry -> promotes its access epoch
        let _ = cache.prefix_match(&[hash_token(1, 0)]);

        // Adding a third entry should evict #2 (LRU), not #1
        cache.commit(&[hash_token(3, 0)], slice(0, 2));

        // Confirm #1 still hits
        assert!(cache.prefix_match(&[hash_token(1, 0)]).is_some());
    }

    #[test]
    fn role_marker_prevents_collision() {
        let user_hash = hash_token(42, 0);
        let assistant_hash = hash_token(42, 1);
        assert_ne!(user_hash, assistant_hash);
    }

    #[test]
    fn vram_budget_construction() {
        // 16 GB, 57344 bytes/token -> ~73k tokens
        let cache = KvPrefixCache::from_vram_budget(16 * 1024 * 1024 * 1024, 57344);
        assert!(cache.capacity() > 50_000);
    }
}
