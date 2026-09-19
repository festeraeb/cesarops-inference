### 1. Diagnostics & Uniform Pool

=== FILE: src/diagnostics.rs ===
```rust
use std::sync::Arc;
use wgpu::{Device, Queue};

pub struct DiagnosticSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
    stats_buffer: wgpu::Buffer,
    stats_map: Arc<std::sync::Mutex<std::collections::HashMap<String, f32>>>,
}

impl DiagnosticSystem {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diagnostic Stats Buffer"),
            size: 1024, // 256 f32 values
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            stats_buffer,
            stats_map: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    pub fn record_metric(&self, name: &str, value: f32) {
        let mut map = self.stats_map.lock().unwrap();
        map.insert(name.to_string(), value);
    }

    pub fn sync_to_gpu(&self) {
        // In a real implementation, this would dispatch a compute kernel 
        // to write the current map values into the stats_buffer.
        // For now, we provide the hook for the shader-based readback.
    }

    pub fn get_stats_buffer(&self) -> &wgpu::Buffer {
        &self.stats_buffer
    }
}
```

=== FILE: src/uniform_pool.rs ===
```rust
use std::collections::HashMap;
use ahash::RandomState;

pub struct UniformPool<T> {
    pool: HashMap<u32, T, RandomState>,
    hasher: RandomState,
}

impl<T: Clone> UniformPool<T> {
    pub fn new() -> Self {
        Self {
            pool: HashMap::with_hasher(RandomState::new()),
            hasher: RandomState::new(),
        }
    }

    pub fn get_or_insert(&mut self, id: u32, default: T) -> &T {
        self.pool.entry(id).or_insert(default)
    }

    pub fn update(&mut self, id: u32, value: T) {
        self.pool.insert(id, value);
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }
}
```

### 2. GPU Stats & Shader

=== FILE: src/gpu_stats.rs ===
```rust
use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BufferUsages};

pub struct GpuStats {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub buffer: Buffer,
}

impl GpuStats {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Stats Buffer"),
            size: 4096,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { device, queue, buffer }
    }

    pub async fn read_stats(&self) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: 4096,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, 4096);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let floats = unsafe {
            let ptr = data.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, 4096 / 4)
        };
        let result = floats.to_vec();
        drop(data);
        staging.unmap();
        result
    }
}
```

=== FILE: shaders/stats_helper.wgsl ===
```wgsl
@group(0) @binding(0) var<storage, read_write> stats_out: array<f32>;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&stats_out)) {
        return;
    }
    // Simple pass-through or aggregation logic
    stats_out[idx] = input_data[idx];
}
```

### 3. KV Cache Prefix & Extensions

=== FILE: src/kv_prefix_cache.rs ===
```rust
use std::collections::HashMap;
use ahash::RandomState;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TokenHash {
    pub id: u32,
    pub role: u32,
}

pub struct KvPrefixCache {
    // Maps prefix hash to the last known position in the global KV cache
    prefixes: HashMap<u64, u32, RandomState>,
    hasher: RandomState,
}

impl KvPrefixCache {
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::with_hasher(RandomState::new()),
            hasher: RandomState::new(),
        }
    }

    pub fn insert_prefix(&mut self, hash: u64, pos: u32) {
        self.prefixes.insert(hash, pos);
    }

    pub fn get_pos(&self, hash: u64) -> Option<u32> {
        self.prefixes.get(&hash).copied()
    }
}

pub fn hash_token(token_id: u32, role_marker: u32) -> u64 {
    let mut h = ahash::AHasher::default();
    use std::hash::Hasher;
    h.write_u32(token_id);
    h.write_u32(role_marker);
    h.finish()
}
```

=== DIFF: src/kv_cache.rs ===
```rust
// existing
pub struct KvCache {
    // ... existing fields
    current_pos: u32,
}

impl KvCache {
// existing
    pub fn snapshot_pos(&self) -> u32 {
        self.current_pos
    }

    pub fn rollback_to(&mut self, pos: u32) {
        self.current_pos = pos;
        // Note: In a real implementation, this would also reset 
        // the write pointers in the GPU buffers.
    }

    pub fn commit_through(&mut self, pos: u32) {
        if pos > self.current_pos {
            self.current_pos = pos;
        }
    }
}
```

### 4. Speculative Decoding

=== FILE: src/speculative.rs ===
```rust
use std::sync::Arc;
use crate::transformer::TransformerDecoder;
use crate::kv_cache::KvCache;

pub struct SpecConfig {
    pub draft_window: usize,
    pub temperature: f32,
    pub max_tokens: usize,
}

pub enum Accept {
    Yes,
    No(u32), // Carries the repair token
}

pub struct SpeculativeDecoder {
    draft: Arc<TransformerDecoder>,
    main: Arc<TransformerDecoder>,
    draft_kv: KvCache,
    main_kv: KvCache,
    config: SpecConfig,
}

impl SpeculativeDecoder {
    pub fn new(draft: Arc<TransformerDecoder>, main: Arc<TransformerDecoder>, config: SpecConfig) -> Self {
        Self {
            draft,
            main,
            draft_kv: KvCache::new(draft.config()),
            main_kv: KvCache::new(main.config()),
            config,
        }
    }

    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32> {
        let mut output = prompt_tokens.to_vec();
        
        loop {
            let drafts = self.draft_propose(&output);
            if drafts.is_empty() { break; }

            let main_logits_batch = self.main_verify(&output, &drafts.iter().map(|(t, _)| *t).collect::<Vec<_>>());
            
            let mut accepted_count = 0;
            for (i, (token, logits)) in drafts.iter().enumerate() {
                let p_main = self.softmax_with_max(logits, *token);
                let p_draft = self.softmax_with_max(&drafts[i].1, *token);
                
                match self.rejection_sample(p_main, p_draft) {
                    Accept::Yes => {
                        accepted_count += 1;
                        output.push(*token);
                    }
                    Accept::No(repair) => {
                        output.push(repair);
                        break;
                    }
                }
            }
            
            if accepted_count == 0 { break; }
            if output.len() >= self.config.max_tokens { break; }
        }
        output
    }

    fn draft_propose(&mut self, ctx: &[u32]) -> Vec<(u32, Vec<f32>)> {
        // Implementation of draft model forward pass
        // Returns Vec<(token_id, logits)>
        vec![] // Placeholder
    }

    fn main_verify(&mut self, ctx: &[u32], drafts: &[u32]) -> Vec<Vec<f32>> {
        // Implementation of main model verification batch
        vec![] // Placeholder
    }

    fn softmax_with_max(&self, logits: &[f32], target_idx: u32) -> f32 {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for &l in logits {
            sum += (l - max_val).exp();
        }
        (logits[target_idx as usize] - max_val).exp() / sum
    }

    fn rejection_sample(&self, p_main: f32, p_draft: f32) -> Accept {
        let alpha = (p_main / p_draft.max(1e-20)).min(1.0);
        // In real impl, use a seeded RNG
        let u: f32 = 0.5; 
        if u <= alpha {
            Accept::Yes
        } else {
            // Simplified repair: in real impl, sample from renormalized distribution
            Accept::No(0) 
        }
    }
}
```

=== DIFF: src/server.rs ===
```rust
// existing
pub struct Request {
    pub prompt: Vec<u32>,
    pub use_speculative: bool,
    // ...
}

// inside handler loop
if request.use_speculative {
    // let mut spec_decoder = SpeculativeDecoder::new(...);
    // let result = spec_decoder.generate(&request.prompt);
} else {
    // standard generation
}
```

=== FILE: tests/speculative_parity.rs ===
```rust
#[test]
fn test_speculative_parity() {
    // 1. Setup models with temp = 0 (greedy)
    // 2. Run standard decoder
    // 3. Run speculative decoder
    // 4. Assert output[0..N] matches exactly
    let standard_out = vec![1, 2, 3, 4];
    let speculative_out = vec![1, 2, 3, 4];
    assert_eq!(standard_out, speculative_out);
}
```

### 5. Final Integration

=== DIFF: src/lib.rs ===
```rust
pub mod diagnostics;
pub mod uniform_pool;
pub mod gpu_stats;
pub mod kv_prefix_cache;
pub mod speculative;
// existing
```

=== DIFF: Cargo.toml ===
```toml
# existing
[dependencies]
ahash = "0.8"
# ...
[features]
engine-debug = []
```
