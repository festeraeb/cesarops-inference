//! Per-kernel GPU vs CPU correctness probes.
//!
//! Each test feeds a deterministic input through one GPU shader, runs the
//! same math on the CPU, and asserts the GPU output matches within a tight
//! tolerance. These tests are the foundation for getting the GPU-resident
//! Gemma-4 forward pass right.
//!
//! Run with:
//!     cargo test --release --test gpu_kernel_correctness -- --ignored --nocapture
//!
//! Each test is `--ignored` because they need a real Vulkan adapter.

use std::sync::Arc;

use cesarops_inference::gemma4_gpu_pipelines::{
    AttnPush, EmbedLookupPush, Gemma4GpuPipelines, KvWritePush, LogitSoftcapPush,
    RmsNormPush, RmsNormWeightlessPush, RopePush, SiluMulPush, WeightedAccumPush,
};

const EPS: f32 = 1e-6;

// ────────────────────────────────────────────────────────────────────────────
// GPU bring-up helper
// ────────────────────────────────────────────────────────────────────────────

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Gpu {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no Vulkan adapter");
        let info = adapter.get_info();
        println!("GPU adapter: {} ({:?})", info.name, info.backend);

        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("kernel_correctness"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("request_device");
        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }
}

fn upload_f32(gpu: &Gpu, label: &str, data: &[f32]) -> wgpu::Buffer {
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

fn alloc_f32_rw(gpu: &Gpu, label: &str, n: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn upload_uniform<T: bytemuck::Pod>(gpu: &Gpu, label: &str, value: T) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(&value);
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buf, 0, bytes);
    buf
}

fn read_f32(gpu: &Gpu, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, (n * 4) as u64);
    let sub = gpu.queue.submit(std::iter::once(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::Maintain::WaitForSubmissionIndex(sub));
    let _ = rx.recv();
    let mapped = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out
}

fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut worst_idx = 0usize;
    let mut worst_err = 0.0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - e).abs();
        let allowed = atol + rtol * e.abs();
        if err > allowed {
            if err > worst_err {
                worst_err = err;
                worst_idx = i;
            }
        }
    }
    if worst_err > 0.0 {
        panic!(
            "{label}: GPU != CPU\n  worst at idx {worst_idx}: gpu={} cpu={} err={worst_err}\n  first 8 gpu: {:?}\n  first 8 cpu: {:?}",
            actual[worst_idx], expected[worst_idx],
            &actual[..actual.len().min(8)],
            &expected[..expected.len().min(8)],
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CPU references
// ────────────────────────────────────────────────────────────────────────────

fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32, plus_one: bool) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| {
            let gain = if plus_one { wi + 1.0 } else { wi };
            xi * inv_rms * gain
        })
        .collect()
}

fn cpu_rmsnorm_weightless(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    x.iter().map(|&xi| xi * inv_rms).collect()
}

fn cpu_silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

fn cpu_logit_softcap(x: &[f32], cap: f32) -> Vec<f32> {
    x.iter().map(|&v| cap * (v / cap).tanh()).collect()
}

fn cpu_rope_inplace(qk: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, base: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for p in 0..half {
            let exp = -2.0 * p as f32 / head_dim as f32;
            let theta = pos as f32 * base.powf(exp);
            let c = theta.cos();
            let s = theta.sin();
            let lo = h * head_dim + p;
            let hi = lo + half;
            let x0 = qk[lo];
            let x1 = qk[hi];
            qk[lo] = x0 * c - x1 * s;
            qk[hi] = x0 * s + x1 * c;
        }
    }
}

fn cpu_attention_one_head(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window_start: Option<usize>,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_stride = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let kv_off = kv_h * head_dim;
        let q_off = h * head_dim;

        let mut scores = Vec::with_capacity(seq_len);
        let mut max_s = f32::NEG_INFINITY;
        for p in 0..seq_len {
            let masked = window_start.map(|w| p < w).unwrap_or(false);
            if masked {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k_cache[p * kv_stride + kv_off + d];
            }
            let s = dot * scale;
            if s > max_s {
                max_s = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            } else {
                *s = (*s - max_s).exp();
                sum += *s;
            }
        }
        let inv = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for p in 0..seq_len {
                acc += scores[p] * inv * v_cache[p * kv_stride + kv_off + d];
            }
            out[q_off + d] = acc;
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn rmsnorm_matches_cpu_both_conventions() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let n = 2816usize; // Gemma-4 hidden_dim

    // Deterministic input — varying magnitudes to catch stability bugs.
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) / 100.0).sin() * 5.0 + 1.0)
        .collect();
    let w: Vec<f32> = (0..n)
        .map(|i| ((i as f32) / 13.0).cos() * 0.5)
        .collect();

    for plus_one in [false, true] {
        let x_buf = upload_f32(&gpu, "x", &x);
        let w_buf = upload_f32(&gpu, "w", &w);
        let y_buf = alloc_f32_rw(&gpu, "y", n);
        let push = upload_uniform(
            &gpu,
            "push",
            RmsNormPush {
                hidden_dim: n as u32,
                plus_one_flag: if plus_one { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
                eps: EPS,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            },
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm_bg"),
            layout: &pipes.rmsnorm.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm_pass"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipes.rmsnorm.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        let actual = read_f32(&gpu, &y_buf, n);

        let expected = cpu_rmsnorm(&x, &w, EPS, plus_one);
        assert_close(
            &actual,
            &expected,
            1e-4,
            1e-5,
            &format!("rmsnorm plus_one={plus_one}"),
        );
        println!(
            "rmsnorm plus_one={plus_one} ok (n={n}, sample y[0]={:.6})",
            actual[0]
        );
    }
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn rmsnorm_weightless_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let n = 512usize;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) / 7.0).sin() * 3.0).collect();

    let x_buf = upload_f32(&gpu, "x", &x);
    let y_buf = alloc_f32_rw(&gpu, "y", n);
    let push = upload_uniform(
        &gpu,
        "push",
        RmsNormWeightlessPush {
            hidden_dim: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            eps: EPS,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rmsw_bg"),
        layout: &pipes.rmsnorm_weightless.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsw"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.rmsnorm_weightless.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &y_buf, n);
    let expected = cpu_rmsnorm_weightless(&x, EPS);
    assert_close(&actual, &expected, 1e-4, 1e-5, "rmsnorm_weightless");
    println!("rmsnorm_weightless ok (n={n})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn silu_mul_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let n = 2112usize; // Gemma-4 dense FFN inner.

    let gate: Vec<f32> = (0..n).map(|i| ((i as f32) / 17.0).sin() * 2.0).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i as f32) / 23.0).cos()).collect();
    let g_buf = upload_f32(&gpu, "gate", &gate);
    let u_buf = upload_f32(&gpu, "up", &up);
    let y_buf = alloc_f32_rw(&gpu, "y", n);
    let push = upload_uniform(
        &gpu,
        "silu_push",
        SiluMulPush {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("silu_bg"),
        layout: &pipes.silu_mul.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: g_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: u_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.silu_mul.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &y_buf, n);
    let expected = cpu_silu_mul(&gate, &up);
    assert_close(&actual, &expected, 1e-4, 1e-5, "silu_mul");
    println!("silu_mul ok (n={n})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn rope_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);

    // Try both Gemma-4 head dims to make sure the stride loop works.
    for &head_dim in [256usize, 512].iter() {
        let n_heads = 4usize;
        let total = n_heads * head_dim;
        let qk: Vec<f32> = (0..total).map(|i| ((i as f32) / 11.0).sin()).collect();
        let pos = 7u32;
        let base = if head_dim == 512 { 1.0e6f32 } else { 1.0e4f32 };

        let qk_buf = upload_f32(&gpu, "qk", &qk);
        let push = upload_uniform(
            &gpu,
            "rope_push",
            RopePush {
                head_dim: head_dim as u32,
                pos,
                n_heads: n_heads as u32,
                _pad: 0,
                rope_base: base,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            },
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope_bg"),
            layout: &pipes.rope.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: qk_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: push.as_entire_binding() },
            ],
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipes.rope.pipeline);
            p.set_bind_group(0, Some(&bg), &[]);
            p.dispatch_workgroups(n_heads as u32, 1, 1);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        let actual = read_f32(&gpu, &qk_buf, total);

        let mut expected = qk.clone();
        cpu_rope_inplace(&mut expected, n_heads, head_dim, pos as usize, base);
        assert_close(
            &actual,
            &expected,
            1e-4,
            1e-5,
            &format!("rope head_dim={head_dim}"),
        );
        println!("rope head_dim={head_dim} ok");
    }
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn logit_softcap_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let n = 1024usize;
    let cap = 30.0f32;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) - 512.0) * 0.5).collect();
    let x_buf = upload_f32(&gpu, "x", &x);
    let y_buf = alloc_f32_rw(&gpu, "y", n);
    let push = upload_uniform(
        &gpu,
        "softcap_push",
        LogitSoftcapPush {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            cap,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softcap_bg"),
        layout: &pipes.logit_softcap.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softcap"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.logit_softcap.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &y_buf, n);
    let expected = cpu_logit_softcap(&x, cap);
    assert_close(&actual, &expected, 1e-4, 1e-5, "logit_softcap");
    println!("logit_softcap ok (n={n})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn embed_lookup_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let vocab = 256usize;
    let hidden = 2816usize;
    let table: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i as f32) / 137.0).sin() * 0.05)
        .collect();
    let token_id = 42u32;

    let table_buf = upload_f32(&gpu, "embed", &table);
    let h_buf = alloc_f32_rw(&gpu, "h", hidden);
    let push = upload_uniform(
        &gpu,
        "embed_push",
        EmbedLookupPush {
            token_id,
            hidden_dim: hidden as u32,
            embed_scale_flag: 1,
            _pad: 0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embed_bg"),
        layout: &pipes.embed_lookup.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: table_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: h_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embed"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.embed_lookup.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(((hidden + 255) / 256) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &h_buf, hidden);

    let scale = (hidden as f32).sqrt();
    let row = (token_id as usize) * hidden;
    let expected: Vec<f32> = table[row..row + hidden].iter().map(|v| v * scale).collect();
    assert_close(&actual, &expected, 1e-4, 1e-5, "embed_lookup");
    println!("embed_lookup ok (vocab={vocab} hidden={hidden})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn weighted_accum_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let n = 2816usize;
    let weight = 0.37f32;
    let src: Vec<f32> = (0..n).map(|i| ((i as f32) / 31.0).sin()).collect();
    let init: Vec<f32> = (0..n).map(|i| ((i as f32) / 19.0).cos() * 0.5).collect();

    let src_buf = upload_f32(&gpu, "src", &src);
    let y_buf = upload_f32(&gpu, "y_init", &init);
    let push = upload_uniform(
        &gpu,
        "wa_push",
        WeightedAccumPush {
            weight,
            len: n as u32,
            _pad0: 0,
            _pad1: 0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wa_bg"),
        layout: &pipes.weighted_accum.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wa"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.weighted_accum.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &y_buf, n);
    let expected: Vec<f32> = init
        .iter()
        .zip(src.iter())
        .map(|(&y0, &s)| y0 + weight * s)
        .collect();
    assert_close(&actual, &expected, 1e-4, 1e-5, "weighted_accum");
    println!("weighted_accum ok (n={n})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn kv_write_matches_cpu() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);

    let n_kv = 8usize;
    let head_dim = 256usize;
    let kv_dim = n_kv * head_dim;
    let max_seq = 16usize;

    let k_in: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.001).collect();
    let v_in: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * -0.002).collect();
    // Pre-fill caches with sentinel garbage to verify only the target slot
    // is written.
    let mut k_cache = vec![-9999.0f32; max_seq * kv_dim];
    let mut v_cache = vec![-9999.0f32; max_seq * kv_dim];
    let pos = 5usize;

    let k_in_buf = upload_f32(&gpu, "k_in", &k_in);
    let v_in_buf = upload_f32(&gpu, "v_in", &v_in);
    let k_cache_buf = upload_f32(&gpu, "k_cache", &k_cache);
    let v_cache_buf = upload_f32(&gpu, "v_cache", &v_cache);
    let push = upload_uniform(
        &gpu,
        "kvw_push",
        KvWritePush {
            pos: pos as u32,
            kv_dim: kv_dim as u32,
            _pad0: 0,
            _pad1: 0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kvw_bg"),
        layout: &pipes.kv_write.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: k_in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: v_in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_cache_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_cache_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kvw"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.kv_write.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(((kv_dim + 255) / 256) as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let k_out = read_f32(&gpu, &k_cache_buf, max_seq * kv_dim);
    let v_out = read_f32(&gpu, &v_cache_buf, max_seq * kv_dim);

    // CPU reference: write into the same slot, leave others sentinel.
    let off = pos * kv_dim;
    k_cache[off..off + kv_dim].copy_from_slice(&k_in);
    v_cache[off..off + kv_dim].copy_from_slice(&v_in);
    assert_close(&k_out, &k_cache, 0.0, 0.0, "kv_write K");
    assert_close(&v_out, &v_cache, 0.0, 0.0, "kv_write V");
    println!("kv_write ok (n_kv={n_kv} head_dim={head_dim} pos={pos})");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn attention_matches_cpu_global() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    // Gemma-4 global attention: head_dim=512, n_heads=16, n_kv_heads=8.
    let head_dim = 512usize;
    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_stride = n_kv_heads * head_dim;
    let seq_len = 7usize;

    let q: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| ((i as f32) / 41.0).sin())
        .collect();
    let k_cache: Vec<f32> = (0..seq_len * kv_stride)
        .map(|i| ((i as f32) / 53.0).cos())
        .collect();
    let v_cache: Vec<f32> = (0..seq_len * kv_stride)
        .map(|i| ((i as f32) / 67.0).sin() * 0.5)
        .collect();

    let q_buf = upload_f32(&gpu, "q", &q);
    let k_buf = upload_f32(&gpu, "k", &k_cache);
    let v_buf = upload_f32(&gpu, "v", &v_cache);
    let out_buf = alloc_f32_rw(&gpu, "out", n_heads * head_dim);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let push = upload_uniform(
        &gpu,
        "attn_push",
        AttnPush {
            head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            heads_per_kv: heads_per_kv as u32,
            seq_len: seq_len as u32,
            window_start: 0,
            use_swa: 0,
            _pad0: 0,
            scale,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bg"),
        layout: &pipes.attn.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.attn.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(n_heads as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &out_buf, n_heads * head_dim);
    let expected = cpu_attention_one_head(
        &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len, None,
    );
    assert_close(&actual, &expected, 1e-3, 1e-4, "attention global");
    println!("attention global ok");
}

#[tokio::test]
#[ignore = "needs Vulkan adapter"]
async fn attention_matches_cpu_swa() {
    let gpu = Gpu::new().await;
    let pipes = Gemma4GpuPipelines::new(&gpu.device);
    let head_dim = 256usize;
    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let heads_per_kv = n_heads / n_kv_heads;
    let kv_stride = n_kv_heads * head_dim;
    let seq_len = 32usize;
    let window_start = 24usize; // attend only positions 24..31

    let q: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| ((i as f32) / 11.0).sin())
        .collect();
    let k_cache: Vec<f32> = (0..seq_len * kv_stride)
        .map(|i| ((i as f32) / 13.0).cos())
        .collect();
    let v_cache: Vec<f32> = (0..seq_len * kv_stride)
        .map(|i| ((i as f32) / 17.0).sin() * 0.7)
        .collect();

    let q_buf = upload_f32(&gpu, "q", &q);
    let k_buf = upload_f32(&gpu, "k", &k_cache);
    let v_buf = upload_f32(&gpu, "v", &v_cache);
    let out_buf = alloc_f32_rw(&gpu, "out", n_heads * head_dim);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let push = upload_uniform(
        &gpu,
        "attn_push",
        AttnPush {
            head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            heads_per_kv: heads_per_kv as u32,
            seq_len: seq_len as u32,
            window_start: window_start as u32,
            use_swa: 1,
            _pad0: 0,
            scale,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        },
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_swa_bg"),
        layout: &pipes.attn.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: push.as_entire_binding() },
        ],
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn_swa"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipes.attn.pipeline);
        p.set_bind_group(0, Some(&bg), &[]);
        p.dispatch_workgroups(n_heads as u32, 1, 1);
    }
    gpu.queue.submit(std::iter::once(enc.finish()));
    let actual = read_f32(&gpu, &out_buf, n_heads * head_dim);
    let expected = cpu_attention_one_head(
        &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len, Some(window_start),
    );
    assert_close(&actual, &expected, 1e-3, 1e-4, "attention swa");
    println!("attention swa ok");
}


// ────────────────────────────────────────────────────────────────────────────
// IQ4 matvec verification against CPU dequant + dense matmul.
//
// We pull a real tensor out of the Gemma-4-26B-MoE GGUF, dequant it on the
// CPU into a dense fp32 weight matrix, run the GPU IQ4 matvec on the raw
// quant bytes, and check they agree to within typical fp32 round-off
// noise (rtol = 1e-2 because IQ4 is lossy and the codebook is integer-ish).
// ────────────────────────────────────────────────────────────────────────────

use cesarops_inference::bridge;
use cesarops_inference::hardware;
use cesarops_inference::iq4_pipeline::{
    iq4nl_shader_src, iq4xs_shader_src, Iq4MatvecPipeline, MatvecPush, QuantKind,
};
use cesarops_inference::loader;
use std::path::Path;

const GEMMA4_MOE: &str = "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf";

#[tokio::test]
#[ignore = "needs GGUF on disk + Vulkan adapter"]
async fn iq4xs_matvec_matches_cpu_reference() {
    let gpu = Gpu::new().await;
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(GEMMA4_MOE), &profile).expect("load gguf");

    // Use a real Gemma-4 IQ4_XS tensor: blk.0.attn_q.weight, shape [hidden, q_dim].
    let tensor_name = "blk.0.attn_q.weight";
    let region = weights
        .tensors
        .get(tensor_name)
        .expect("attn_q tensor present");
    assert_eq!(region.quant_type, 23, "expected IQ4_XS (qt=23)");
    let bytes = weights
        .tensor_bytes(tensor_name)
        .expect("attn_q bytes mapped");

    // GGUF stores weight matrices with shape [k_in, n_out] in inner-first
    // order. For the matvec shader we treat each row as one output and the
    // K axis as the input vector, so n_rows = shape[1] (=q_dim).
    let k_dim = region.shape[0]; // hidden = 2816
    let n_rows = region.shape[1]; // q_dim = 16 * 512 = 8192
    println!(
        "tensor {} shape=[{}, {}], bytes={}",
        tensor_name,
        k_dim,
        n_rows,
        bytes.len()
    );

    // CPU reference: dequant the whole tensor to f32, then dense matvec.
    let dense = bridge::dequant_iq4_xs(bytes, k_dim * n_rows);
    assert_eq!(dense.len(), k_dim * n_rows);
    let x: Vec<f32> = (0..k_dim).map(|i| ((i as f32) / 23.0).sin() * 0.1).collect();
    let mut y_cpu = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row_base = r * k_dim;
        let mut acc = 0.0f64;
        for c in 0..k_dim {
            acc += dense[row_base + c] as f64 * x[c] as f64;
        }
        y_cpu[r] = acc as f32;
    }

    // GPU path: upload raw IQ4_XS bytes, run matvec.
    let pipe = Iq4MatvecPipeline::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        QuantKind::Iq4Xs,
        iq4xs_shader_src(),
    );

    let w_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn_q_w"),
        size: ((bytes.len() + 3) & !3) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&w_buf, 0, bytes);
    let x_buf = upload_f32(&gpu, "x", &x);
    let y_buf = alloc_f32_rw(&gpu, "y", n_rows);
    let push_buf = pipe.make_push_buffer(MatvecPush {
        k: k_dim as u32,
        n_rows_total: n_rows as u32,
        row_offset: 0,
        _pad: 0,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipe.dispatch(&mut enc, &w_buf, &x_buf, &y_buf, &push_buf, n_rows as u32);
    gpu.queue.submit(std::iter::once(enc.finish()));
    let y_gpu = read_f32(&gpu, &y_buf, n_rows);

    // IQ4 is fundamentally lossy and the dot product accumulates k_dim=2816
    // small errors, so absolute tolerances need to match real-tensor magnitudes.
    let max_abs = y_cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "y range: cpu max |y|={:.4}, gpu sample={:.6}",
        max_abs, y_gpu[0]
    );
    let rtol = 5e-3f32;
    let atol = 5e-3 * max_abs.max(1.0);
    assert_close(&y_gpu, &y_cpu, rtol, atol, "iq4xs matvec");
    println!("iq4xs matvec ok (k={k_dim} n={n_rows}, ~{} elems)", k_dim * n_rows);
}

#[tokio::test]
#[ignore = "needs GGUF on disk + Vulkan adapter"]
async fn iq4nl_matvec_matches_cpu_reference() {
    let gpu = Gpu::new().await;
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(GEMMA4_MOE), &profile).expect("load gguf");

    let tensor_name = "blk.0.ffn_down.weight";
    let region = weights
        .tensors
        .get(tensor_name)
        .expect("ffn_down tensor present");
    assert_eq!(region.quant_type, 20, "expected IQ4_NL (qt=20)");
    let bytes = weights
        .tensor_bytes(tensor_name)
        .expect("ffn_down bytes mapped");

    let k_dim = region.shape[0]; // 2112 (intermediate)
    let n_rows = region.shape[1]; // 2816 (hidden)
    let dense = bridge::dequant_iq4_nl(bytes, k_dim * n_rows);
    assert_eq!(dense.len(), k_dim * n_rows);
    let x: Vec<f32> = (0..k_dim).map(|i| ((i as f32) / 13.0).sin() * 0.05).collect();
    let mut y_cpu = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row_base = r * k_dim;
        let mut acc = 0.0f64;
        for c in 0..k_dim {
            acc += dense[row_base + c] as f64 * x[c] as f64;
        }
        y_cpu[r] = acc as f32;
    }

    let pipe = Iq4MatvecPipeline::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        QuantKind::Iq4Nl,
        iq4nl_shader_src(),
    );
    let w_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ffn_down_w"),
        size: ((bytes.len() + 3) & !3) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&w_buf, 0, bytes);
    let x_buf = upload_f32(&gpu, "x", &x);
    let y_buf = alloc_f32_rw(&gpu, "y", n_rows);
    let push_buf = pipe.make_push_buffer(MatvecPush {
        k: k_dim as u32,
        n_rows_total: n_rows as u32,
        row_offset: 0,
        _pad: 0,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipe.dispatch(&mut enc, &w_buf, &x_buf, &y_buf, &push_buf, n_rows as u32);
    gpu.queue.submit(std::iter::once(enc.finish()));
    let y_gpu = read_f32(&gpu, &y_buf, n_rows);
    let max_abs = y_cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "y range: cpu max |y|={:.4}, gpu sample={:.6}",
        max_abs, y_gpu[0]
    );
    let rtol = 5e-3f32;
    let atol = 5e-3 * max_abs.max(1.0);
    assert_close(&y_gpu, &y_cpu, rtol, atol, "iq4nl matvec");
    println!("iq4nl matvec ok (k={k_dim} n={n_rows})");
}
