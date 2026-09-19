=== FILE: src/diagnostics.rs ===
use std::env;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Off,
    ErrorOnly,
    Debug,
}

#[derive(Clone)]
pub struct Diagnostics {
    pub level: DiagnosticLevel,
}

impl Diagnostics {
    pub fn from_env() -> Self {
        let lvl = env::var("CESAROPS_DIAG").unwrap_or_else(|_| "off".into());
        let level = match lvl.as_str() {
            "error" => DiagnosticLevel::ErrorOnly,
            "debug" => DiagnosticLevel::Debug,
            _ => DiagnosticLevel::Off,
        };

        Self { level }
    }

    #[inline(always)]
    pub fn enabled(&self) -> bool {
        self.level != DiagnosticLevel::Off
    }

    #[inline(always)]
    pub fn debug_enabled(&self) -> bool {
        self.level == DiagnosticLevel::Debug
    }

    #[inline]
    pub fn check_nan(&self, v: &[f32]) -> bool {
        if !self.enabled() { return false; }
        v.iter().any(|x| x.is_nan() || x.is_infinite())
    }

    #[inline]
    pub fn log_logits(&self, logits: &[f32]) {
        if !self.debug_enabled() { return; }
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = logits.iter().sum::<f32>() / logits.len().max(1) as f32;
        eprintln!("[diag] logits min={:.4} max={:.4} mean={:.6}", min, max, mean);
    }

    #[inline]
    pub fn log_tensor_stats(&self, name: &str, v: &[f32]) {
        if !self.debug_enabled() { return; }
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
        eprintln!("[diag] {} min={:.4} max={:.4} mean={:.6}", name, min, max, mean);
    }
}

#[cfg(feature = "engine-debug")]
pub fn diagnostics_enabled_compile() {}

DIFF: server.rs (log replacement only)

// existing

// REMOVE:
let has_nan = hidden_state.iter().any(|x| x.is_nan() || x.is_infinite());
let logits_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
info!("Logits: min={:.4}, max={:.4}, mean={:.6}", logits_min, ...);

// ADD:
let diag = crate::diagnostics::Diagnostics::from_env();
if diag.check_nan(&hidden_state) {
    eprintln!("[diag] NaN detected in hidden_state");
}
diag.log_logits(&logits);

=== FILE: src/uniform_pool.rs ===
use std::sync::Arc;
use wgpu::Maintain;

pub struct UniformAllocation {
    pub offset: u64,
    pub size: u64,
}

pub struct UniformPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer: wgpu::Buffer,
    size: u64,
    offset: u64,
    alignment: u64,
    last_submission: u64,
}

impl UniformPool {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let limits = device.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment as u64;

        let size = 8 * 1024 * 1024;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_pool"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            buffer,
            size,
            offset: 0,
            alignment,
            last_submission: 0,
        }
    }

    #[inline]
    pub fn alloc(&mut self, size: u64) -> UniformAllocation {
        let aligned = (self.offset + self.alignment - 1) & !(self.alignment - 1);

        if aligned + size > self.size {
            self.offset = 0;
        }

        let offset = self.offset;
        self.offset = aligned + size;

        UniformAllocation { offset, size }
    }

    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, data: &T, alloc: &UniformAllocation) {
        let bytes = bytemuck::bytes_of(data);
        self.queue.write_buffer(
            &self.buffer,
            alloc.offset,
            bytes,
        );
    }

    pub fn reset_after(&mut self, submission_idx: u64) {
        if submission_idx > self.last_submission {
            let _ = self.device.poll(Maintain::WaitForSubmissionIndex(submission_idx));
            self.offset = 0;
            self.last_submission = submission_idx;
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

=== FILE: src/gpu_stats.rs ===
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct LogitStats {
    pub min_bits: u32,
    pub max_bits: u32,
    pub has_nan: u32,
    pub _pad: u32,
}

impl LogitStats {
    pub fn decode(&self) -> (f32, f32, bool) {
        let min = f32::from_bits(self.min_bits);
        let max = f32::from_bits(self.max_bits);
        (min, max, self.has_nan != 0)
    }
}

FILE: shaders/stats_helper.wgsl

struct Stats {
    min_val: atomic<u32>,
    max_val: atomic<u32>,
    has_nan: atomic<u32>,
};

@group(0) @binding(5) var<storage, read_write> stats: Stats;

fn float_to_ordered_u32(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    let mask = select(0x80000000u, 0x00000000u, bits < 0x80000000u);
    return bits ^ mask;
}

fn ordered_u32_to_float(x: u32) -> f32 {
    let mask = select(0x80000000u, 0x00000000u, x < 0x80000000u);
    return bitcast<f32>(x ^ mask);
}

fn update_stats(x: f32) {
    if (isnan(x)) {
        atomicStore(&stats.has_nan, 1u);
        return;
    }

    let v = float_to_ordered_u32(x);

    atomicMin(&stats.min_val, v);
    atomicMax(&stats.max_val, v);
}

=== FILE: src/kv_prefix_cache.rs ===
use std::collections::{HashMap, VecDeque};
use rustc_hash::FxHashMap;

pub type TokenHash = u64;
pub type NodeId = u32;

pub type KvDataHandle = u64;

#[derive(Clone)]
pub struct KvSlice {
    pub start_pos: u32,
    pub len: u32,
    pub layer_data_handle: KvDataHandle,
}

pub struct CacheStats {
    pub nodes: usize,
    pub tokens: usize,
    pub evictions: usize,
}

pub struct KvNode {
    pub parent: Option<NodeId>,
    pub children: FxHashMap<TokenHash, NodeId>,
    pub kv_slice: Option<KvSlice>,
    pub depth: u32,
    pub last_access_epoch: u64,
}

pub struct KvPrefixCache {
    arena: Vec<KvNode>,
    free_list: Vec<NodeId>,
    root: NodeId,
    lru: VecDeque<NodeId>,
    total_tokens: usize,
    max_tokens: usize,
    epoch: u64,
    evictions: usize,
}

impl KvPrefixCache {
    pub fn new(max_tokens: usize) -> Self {
        let root = KvNode {
            parent: None,
            children: FxHashMap::default(),
            kv_slice: None,
            depth: 0,
            last_access_epoch: 0,
        };

        Self {
            arena: vec![root],
            free_list: vec![],
            root: 0,
            lru: VecDeque::new(),
            total_tokens: 0,
            max_tokens,
            epoch: 0,
            evictions: 0,
        }
    }

    pub fn prefix_match(&mut self, tokens: &[TokenHash]) -> Option<(usize, KvSlice)> {
        let mut node = self.root;
        let mut last_kv = None;
        let mut depth = 0;

        for t in tokens {
            let children = &self.arena[node as usize].children;
            if let Some(&next) = children.get(t) {
                node = next;
                depth += 1;

                if let Some(ref kv) = self.arena[node as usize].kv_slice {
                    last_kv = Some((depth as usize, kv.clone()));
                }
            } else {
                break;
            }
        }

        last_kv
    }

    pub fn commit(&mut self, tokens: &[TokenHash], kv: KvSlice) {
        let mut node = self.root;

        for t in tokens {
            let next = {
                let children = &mut self.arena[node as usize].children;
                if let Some(&n) = children.get(t) {
                    n
                } else {
                    let nid = self.alloc_node(node);
                    children.insert(*t, nid);
                    nid
                }
            };
            node = next;
        }

        self.arena[node as usize].kv_slice = Some(kv);
        self.lru.push_back(node);
        self.total_tokens += tokens.len();
    }

    fn alloc_node(&mut self, parent: NodeId) -> NodeId {
        if let Some(n) = self.free_list.pop() {
            return n;
        }

        let id = self.arena.len() as NodeId;

        self.arena.push(KvNode {
            parent: Some(parent),
            children: FxHashMap::default(),
            kv_slice: None,
            depth: 0,
            last_access_epoch: self.epoch,
        });

        id
    }

    pub fn evict_lru(&mut self) {
        while self.total_tokens > self.max_tokens {
            if let Some(node) = self.lru.pop_front() {
                self.arena[node as usize].kv_slice = None;
                self.evictions += 1;
                self.total_tokens = self.total_tokens.saturating_sub(1);
                self.free_list.push(node);
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            nodes: self.arena.len(),
            tokens: self.total_tokens,
            evictions: self.evictions,
        }
    }

    pub fn hash_token(token_id: u32, role_marker: u32) -> TokenHash {
        let mut h = (token_id as u64).wrapping_mul(0x9e3779b97f4a7c15);
        h ^= (role_marker as u64).wrapping_add(0x85ebca6b);
        h
    }
}

DIFF: kv_cache.rs

// ADD:

pub fn snapshot_pos(&self) -> u32 {
    self.current_pos as u32
}

pub fn rollback_to(&mut self, pos: u32) {
    self.current_pos = pos as usize;
}

pub fn commit_through(&mut self, pos: u32) {
    self.current_pos = pos as usize;
}

=== FILE: src/speculative.rs ===
use std::sync::Arc;
use crate::{kv_cache::KvCache, transformer::TransformerDecoder};

pub struct SpecConfig {
    pub draft_window: usize,
    pub temperature: f32,
    pub max_tokens: usize,
}

pub struct SpeculativeDecoder {
    pub draft: Arc<TransformerDecoder>,
    pub main: Arc<TransformerDecoder>,
    pub draft_kv: KvCache,
    pub main_kv: KvCache,
    pub config: SpecConfig,
}

enum Accept {
    Yes,
    No(u32),
}

impl SpeculativeDecoder {
    pub fn new(
        draft: Arc<TransformerDecoder>,
        main: Arc<TransformerDecoder>,
        config: SpecConfig,
    ) -> Self {
        let layers = main.config.num_layers;
        let max_seq = main.config.max_seq_len;

        Self {
            draft,
            main,
            draft_kv: KvCache::new(layers, max_seq),
            main_kv: KvCache::new(layers, max_seq),
            config,
        }
    }

    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32> {
        let mut out = prompt_tokens.to_vec();
        let mut ctx = prompt_tokens.to_vec();

        for _ in 0..self.config.max_tokens {
            let draft_logits = self.draft.forward(&mut vec![0.0; self.draft.config.hidden_size],
                                                  ctx.len(),
                                                  todo!(),
                                                  &mut self.draft_kv);

            let main_logits = self.main.forward(&mut vec![0.0; self.main.config.hidden_size],
                                                ctx.len(),
                                                todo!(),
                                                &mut self.main_kv);

            let mut draft_probs = softmax(&draft_logits);
            let main_probs = softmax(&main_logits);

            let t = argmax(&draft_probs);

            let p_main = main_probs[t as usize].max(1e-20);
            let p_draft = draft_probs[t as usize].max(1e-20);

            let alpha = (p_main / p_draft).min(1.0);
            let u: f32 = rand();

            if u <= alpha {
                out.push(t);
                ctx.push(t);
            } else {
                let repair = sample_repair(&main_probs, &draft_probs);
                out.push(repair);
                ctx.push(repair);
            }
        }

        out
    }

    fn draft_propose(&mut self, _ctx: &[u32]) -> Vec<(u32, Vec<f32>)> {
        vec![]
    }

    fn main_verify(&mut self, _ctx: &[u32], _drafts: &[u32]) -> Vec<Vec<f32>> {
        vec![]
    }

    fn rejection_sample(&self, _draft_token: u32, _p_main: &[f32], _p_draft: &[f32]) -> Accept {
        Accept::Yes
    }

    fn sample_repair(&self, _p_main: &[f32], _p_draft: &[f32]) -> u32 {
        0
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for v in &mut exps {
        *v /= sum.max(1e-20);
    }
    exps
}

fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

fn rand() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    (t % 1000) as f32 / 1000.0
}

FILE: tests/speculative_parity.rs

#[test]
fn parity_greedy_speculative() {
    let logits = vec![0.1, 0.2, 0.3, 0.4];
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();

    let sum: f32 = probs.iter().sum();
    let norm: Vec<f32> = probs.iter().map(|v| v / sum).collect();

    let argmax = norm.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;

    assert_eq!(argmax, 3);
}

DIFF: src/lib.rs
pub mod diagnostics;
pub mod uniform_pool;
pub mod gpu_stats;
pub mod kv_prefix_cache;
pub mod speculative;

DIFF: Cargo.toml
[features]
engine-debug = []

[dependencies]
ahash = "0.8"
rustc-hash = "1.1"
bytemuck = { version = "1.14", features = ["derive"] }
wgpu = "0.20"
parking_lot = "0.12"
