You are an expert in WebGPU + Rust + wgpu cross-target compilation + WebSocket protocols. Build the MVP for the Web Compute Layer (WCL) — opportunistic browser-based WebGPU workers that act as stateless kernel accelerators for our Rust+wgpu inference engine.

## Strategic context

This is NOT distributed inference (Petals-style). It's NOT browser-only inference (WebLLM-style). This is:

> Server-authoritative inference engine farms out specific stateless kernel dispatches to volunteer browser tabs over WebSocket. Workers compute, return results, never see KV cache or prompts.

Niche check: nobody else in the LLM ecosystem is doing this. WebLLM puts the whole model in browser. Petals is Python+CUDA+full-shard. Cake is mobile sharding with state. WCL = stateless opportunistic accelerator pool. Realistic target: +10-25% utilization gain when the native GPU is saturated, NOT a latency reducer.

## Hardware + stack context

- Server: Rust + wgpu 0.20+, Pascal P100 / GTX 1070 native target
- Browser worker: wgpu compiled to wasm via `wasm-bindgen`, WebGPU backend
- Same WGSL shaders ship to both (wgpu's superpower)
- WebSocket transport (binary frames, NOT JSON for actual tensors)
- Trust model: LAN-only initially, signed-job + result-verify for internet later

## Deliverables

### 1. Worker capability handshake protocol

Binary message format (use rkyv or postcard for zero-copy):

```rust
// crate: cesarops-wcl-protocol (shared between server + worker)

#[derive(Serialize, Deserialize)]
pub enum WcMsg {
    Hello(WorkerCapabilities),
    JobAssign(Job),
    JobResult(JobResult),
    JobError { job_id: u64, reason: String },
    Goodbye,
}

pub struct WorkerCapabilities {
    pub gpu_name: String,
    pub max_workgroup: u32,
    pub max_storage_buffer_size: u64,
    pub fp16_support: bool,
    pub subgroup_support: bool,
    pub bandwidth_class: BandwidthClass,
}

pub struct Job {
    pub job_id: u64,
    pub kernel: KernelKind,
    pub inputs: Vec<TensorPayload>,    // binary tensor data
    pub workgroup_count: [u32; 3],
    pub push_constants: Vec<u8>,
}

pub struct JobResult {
    pub job_id: u64,
    pub outputs: Vec<TensorPayload>,
    pub elapsed_us: u64,
}

pub enum KernelKind {
    AttentionQK,        // QK^T for one head — small inputs, tractable bandwidth
    AttentionAV,        // AV for one head — V already cached at server, ship per-token weights
    SoftmaxRow,         // tiny rows, almost pure compute
    // matvec for full weight matrix is NOT in this list — bandwidth death.
}
```

Reasoning: only kernels where compute >> transfer go in this list.
Single-head attention QK^T at head_dim=128, M=4096 = 1 MB Q + 1 MB K
in, 4 MB scores out = 6 MB per offload, ~few hundred µs of compute,
~50ms over 100 Mbps WiFi. Worth it on heavily loaded servers, not
worth it on idle.

### 2. Server-side worker pool

```rust
// src/wcl/pool.rs

pub struct WorkerPool {
    workers: parking_lot::RwLock<HashMap<WorkerId, WorkerHandle>>,
    job_queue: crossbeam::channel::Sender<PendingJob>,
    rng: ThreadRng,
}

impl WorkerPool {
    pub fn new(listen_addr: &str) -> Self;

    pub async fn submit(&self, job: Job) -> Result<JobResult, OffloadError>;

    pub fn select_worker(&self) -> Option<WorkerId>;  // load-balanced or random

    pub fn fallback_to_native(job: Job, native_gpu: &GpuContext) -> JobResult;
}
```

Routing logic in scheduler: if `job.compute_weight > 0.6 * native_gpu_utilization`
and worker pool has capacity, offload. Otherwise dispatch native.

### 3. Browser worker (wasm) — minimal

```rust
// crates/cesarops-wcl-worker/src/lib.rs (wasm-bindgen target)

#[wasm_bindgen]
pub async fn start_worker(server_url: &str) -> Result<(), JsValue> {
    let device = init_webgpu().await?;
    let socket = WebSocket::connect(server_url).await?;

    // Handshake
    socket.send_binary(serialize(WcMsg::Hello(my_capabilities()))).await?;

    // Job loop
    while let Some(msg) = socket.recv().await {
        match msg {
            WcMsg::JobAssign(job) => {
                let result = execute_kernel(&device, &job).await?;
                socket.send_binary(serialize(WcMsg::JobResult(result))).await?;
            }
            WcMsg::Goodbye => break,
            _ => {}
        }
    }
    Ok(())
}

fn execute_kernel(device: &wgpu::Device, job: &Job) -> impl Future<Output = JobResult> {
    // Same WGSL as native, just dispatched via wgpu's WebGPU backend
    let pipeline = compile_pipeline(device, job.kernel);
    let bind_group = bind_inputs(device, &job.inputs);
    let mut encoder = device.create_command_encoder(...);
    let mut pass = encoder.begin_compute_pass(...);
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(job.workgroup_count[0], ...);
    drop(pass);
    queue.submit(once(encoder.finish()));
    queue.on_submitted_work_done().await;
    read_outputs(device, &job)
}
```

### 4. Static HTML page for opt-in

Single self-contained page that:
- Loads the wasm worker bundle
- Calls `start_worker(server_url)` on a button click
- Shows status: connected, jobs processed, bytes transferred, current
  GPU utilization (via WebGPU's optional perf APIs)
- "Disconnect" button cleanly tears down

```html
<!doctype html>
<html>
<head><title>cesarops worker</title></head>
<body>
  <h1>cesarops volunteer compute</h1>
  <button onclick="start()">Start contributing</button>
  <div id="status">Idle</div>
  <script type="module">
    import init, { start_worker } from './cesarops_wcl_worker.js';
    await init();
    window.start = () => start_worker('wss://your-server/wcl');
  </script>
</body>
</html>
```

### 5. Result verification (security)

Server periodically reruns ~1% of completed jobs on its own GPU,
compares outputs (tolerance: relative diff < 1e-4 for fp32). Workers
that exceed N% mismatch rate get permabanned. Show the verification
dispatcher + the threshold check + the ban list (in-memory HashSet
keyed by remote IP for now; cookie-based for browser sessions later).

### 6. Bandwidth budgeting

For each KernelKind, compute the per-job byte cost:

| Kernel | In bytes | Out bytes | Compute (P100 native) | Network breakeven |
|--------|----------|-----------|------------------------|-------------------|
| AttentionQK (1 head, M=4096, hd=128) | 2 MB Q + 2 MB K | 64 MB scores | ~500 µs | 100 Mbps gives ~70 ms latency, breakeven at ~150 ms compute = M=8192+ |
| AttentionAV (1 head) | 2 MB Q' + 2 MB V | 1 MB output | ~300 µs | ~50 ms latency, breakeven at ~100 ms = mostly NEVER worth it on 100 Mbps |
| SoftmaxRow | <1 KB | <1 KB | <10 µs | always net loss, drop from list |

Recommendation: ship AttentionQK only at MVP. Expand carefully based
on measurement.

### 7. CLI / activation

Server flag: `--wcl-listen 0.0.0.0:7100` to enable the WebSocket
endpoint. Off by default. Page served at `http://server:7100/` for
opt-in workers.

## Constraints

- Pascal P100 native + WebGPU browser worker compatibility
- wgpu 0.20+ on both sides
- ~800-1200 LOC total across protocol crate + server pool + wasm worker + static page
- MVP only: AttentionQK kernel offload. Others tracked but not shipped.
- No authentication beyond LAN-trust at MVP. Track JWT/HMAC for v2.
- naga must validate the WebGPU compile target as well as Vulkan
- Don't implement: load-balancing across heterogeneous workers, job
  prefetch, multi-job batching. All v2 work.

## Output

Six pieces:
1. `crates/cesarops-wcl-protocol/` — shared types, serialization (~150 LOC)
2. `src/wcl/pool.rs` + `src/wcl/server.rs` — server-side worker pool + WS endpoint (~300 LOC)
3. `src/wcl/scheduler_hook.rs` — routing decision in dispatch path (~100 LOC)
4. `crates/cesarops-wcl-worker/` — wasm worker crate (~250 LOC)
5. `static/wcl-worker.html` — opt-in page (~50 LOC)
6. `scripts/build_wcl_worker.sh` — wasm-pack build invocation

Plus a one-page memo on:
- The bandwidth math per kernel (table above expanded with measured numbers)
- Trust model assumptions (LAN-only at MVP, signed-jobs at v2)
- Why this is genuinely different from Petals/WebLLM/cake (the niche memo)
