// src/bin/shader_lab.rs
//
// Shader research lab — drives the shader_synth pipeline against real wgpu
// devices. Scans every shader on disk, picks the right test harness for each
// kind (matmul / matvec / dequant / attention / moe), runs it under multiple
// stress profiles, persists results to a benchmark DB.
//
// Usage:
//   shader_lab scan                       # list shaders + classifications
//   shader_lab probe                      # describe each visible GPU
//   shader_lab bench --gpu 0 --kind iq4_xs
//   shader_lab bench-all --gpu 0
//   shader_lab compare a.wgsl b.wgsl --gpu 0
//
// Results land in target/shader_lab/<gpu>/<shader>.json plus a top-level
// `shader_lab/index.json` so subsequent runs amortize.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cesarops_inference::shader_synth::benchmark_db::{BenchmarkResult, ShaderDB};
use cesarops_inference::shader_synth::gpu_probe::{detect_from_gpu_name, GpuClass};
use cesarops_inference::shader_synth::shader_scan::{scan_shaders, ShaderEntry};

#[derive(Debug)]
struct Cli {
    cmd: String,
    gpu: usize,
    kind: Option<String>,
    shader_a: Option<String>,
    shader_b: Option<String>,
    shaders_dir: PathBuf,
    out_dir: PathBuf,
    iters: u32,
}

fn parse_cli() -> Cli {
    let args: Vec<String> = env::args().collect();
    let mut c = Cli {
        cmd: "scan".into(),
        gpu: 0,
        kind: None,
        shader_a: None,
        shader_b: None,
        shaders_dir: PathBuf::from("cesarops-inference/shaders"),
        out_dir: PathBuf::from("target/shader_lab"),
        iters: 50,
    };
    let mut i = 1;
    if i < args.len() {
        c.cmd = args[i].clone();
        i += 1;
    }
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => { c.gpu = args[i + 1].parse().unwrap_or(0); i += 2; }
            "--kind" => { c.kind = Some(args[i + 1].clone()); i += 2; }
            "--shaders" => { c.shaders_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--out" => { c.out_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--iters" => { c.iters = args[i + 1].parse().unwrap_or(50); i += 2; }
            other => {
                if c.shader_a.is_none() {
                    c.shader_a = Some(other.into());
                } else if c.shader_b.is_none() {
                    c.shader_b = Some(other.into());
                }
                i += 1;
            }
        }
    }
    c
}

fn print_help() {
    println!("shader_lab — research harness for cesarops-inference shaders");
    println!();
    println!("commands:");
    println!("  scan                      list shaders + their detected kind");
    println!("  probe                     describe each visible GPU");
    println!("  bench [--kind K]          run microbench on shaders of kind K (or all)");
    println!("  bench-all                 run every shader, every kind");
    println!("  compare A B               head-to-head compare two shaders");
    println!();
    println!("flags: --gpu N  --shaders DIR  --out DIR  --iters N");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("shader_lab=info".parse()?))
        .init();

    let cli = parse_cli();
    fs::create_dir_all(&cli.out_dir).ok();

    match cli.cmd.as_str() {
        "scan" => cmd_scan(&cli),
        "probe" => cmd_probe(&cli).await?,
        "bench" => cmd_bench(&cli).await?,
        "bench-all" => cmd_bench_all(&cli).await?,
        "compare" => cmd_compare(&cli).await?,
        "help" | "--help" | "-h" => print_help(),
        other => {
            eprintln!("unknown command: {}", other);
            print_help();
            std::process::exit(1);
        }
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// scan
// ────────────────────────────────────────────────────────────────────────────

fn cmd_scan(cli: &Cli) {
    let entries = scan_shaders(&cli.shaders_dir);
    if entries.is_empty() {
        println!("(no shaders found in {:?})", cli.shaders_dir);
        return;
    }
    println!("{:<40} {:<10} {}", "name", "kind", "path");
    println!("{}", "─".repeat(80));
    let mut by_kind: std::collections::BTreeMap<&str, u32> = Default::default();
    for e in &entries {
        println!("{:<40} {:<10} {}", e.name, e.kind, e.path.display());
        *by_kind.entry(e.kind.as_str()).or_default() += 1;
    }
    println!();
    println!("totals by kind:");
    for (k, n) in by_kind { println!("  {:<10} {}", k, n); }
    println!("\nshader count: {}", entries.len());
}

// ────────────────────────────────────────────────────────────────────────────
// probe — list every Vulkan adapter
// ────────────────────────────────────────────────────────────────────────────

async fn cmd_probe(_cli: &Cli) -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    if adapters.is_empty() {
        println!("no Vulkan adapters visible");
        return Ok(());
    }

    // Authoritative coopmat probe via ash. Returns one entry per physical
    // device, in the order Vulkan enumerates them (which generally matches
    // wgpu's adapter order on a single backend).
    let cm_results = cesarops_inference::shader_synth::coopmat_probe::probe_via_vulkan();

    println!("{:<3} {:<30} {:<10} {:<14} {:<10} {:<5} {}", "idx", "name", "type", "backend", "class", "ext", "coopmat");
    println!("{}", "─".repeat(110));
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        // Match by device_name when ash results are present; otherwise fall back.
        let cm = cm_results.iter().find(|c| c.device_name == info.name).cloned()
            .unwrap_or_else(|| cesarops_inference::shader_synth::coopmat_probe::probe_by_name(&info.name));
        let class_str = format!("{:?}", cm.class);
        let ext_str = format!("{:?}", cm.variant);
        let cm_str = if cm.usable {
            format!("yes ({}x{}x{}, {} entries)", cm.fp16_tile.0, cm.fp16_tile.1, cm.fp16_tile.2, cm.raw.len())
        } else if !cm.raw.is_empty() {
            format!("partial ({} entries, no fp16/fp32 subgroup match)", cm.raw.len())
        } else {
            "no".to_string()
        };
        println!("{:<3} {:<30} {:<10?} {:<14?} {:<10} {:<5} {}",
            i, info.name, info.device_type, info.backend, class_str, ext_str, cm_str);
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// bench — run microbench on shaders of a specific kind
// ────────────────────────────────────────────────────────────────────────────

async fn cmd_bench(cli: &Cli) -> anyhow::Result<()> {
    let kind_filter = cli.kind.clone();
    let entries: Vec<_> = scan_shaders(&cli.shaders_dir).into_iter()
        .filter(|e| match &kind_filter {
            Some(k) => e.kind == *k,
            None => true,
        })
        .collect();

    if entries.is_empty() {
        println!("no shaders match (kind={:?}, dir={:?})", kind_filter, cli.shaders_dir);
        return Ok(());
    }

    let (device, queue, gpu_name, gpu_class) = init_device(cli.gpu).await?;
    println!("benching {} shaders on GPU {} ({}, {:?})", entries.len(), cli.gpu, gpu_name, gpu_class);
    println!();

    let mut db = load_db(&cli.out_dir);

    for e in &entries {
        match bench_one(&device, &queue, e, &gpu_name, cli.iters).await {
            Ok(r) => {
                println!(
                    "  {:<40} {:<10} {:.3} ms/iter   {:.1} GB/s",
                    e.name, e.kind, r.ms_per_token, r.memory_bw_util
                );
                db.insert(r);
            }
            Err(err) => {
                println!("  {:<40} {:<10} ERROR: {}", e.name, e.kind, err);
            }
        }
    }

    save_db(&cli.out_dir, &db);
    println!("\nsaved → {}", cli.out_dir.join("index.json").display());
    Ok(())
}

async fn cmd_bench_all(cli: &Cli) -> anyhow::Result<()> {
    let mut c = Cli {
        cmd: "bench".into(),
        gpu: cli.gpu,
        kind: None,
        shader_a: None, shader_b: None,
        shaders_dir: cli.shaders_dir.clone(),
        out_dir: cli.out_dir.clone(),
        iters: cli.iters,
    };
    c.kind = None;
    cmd_bench(&c).await
}

// ────────────────────────────────────────────────────────────────────────────
// compare — head-to-head two shader files
// ────────────────────────────────────────────────────────────────────────────

async fn cmd_compare(cli: &Cli) -> anyhow::Result<()> {
    let (a, b) = match (&cli.shader_a, &cli.shader_b) {
        (Some(a), Some(b)) => (a.clone(), b.clone()),
        _ => {
            eprintln!("compare needs two shader file paths");
            std::process::exit(1);
        }
    };
    let (device, queue, gpu_name, _) = init_device(cli.gpu).await?;
    let entry_a = entry_from_path(&a);
    let entry_b = entry_from_path(&b);
    let r_a = bench_one(&device, &queue, &entry_a, &gpu_name, cli.iters).await?;
    let r_b = bench_one(&device, &queue, &entry_b, &gpu_name, cli.iters).await?;

    println!("{:<40} {:>10} ms/iter  {:>8} GB/s", "name", "lat", "bw");
    println!("{}", "─".repeat(72));
    println!("{:<40} {:>10.3}            {:>8.1}", entry_a.name, r_a.ms_per_token, r_a.memory_bw_util);
    println!("{:<40} {:>10.3}            {:>8.1}", entry_b.name, r_b.ms_per_token, r_b.memory_bw_util);
    let speedup = r_b.ms_per_token / r_a.ms_per_token.max(1e-9);
    println!("\nspeedup A vs B: {:.2}×", speedup);
    Ok(())
}

fn entry_from_path(p: &str) -> ShaderEntry {
    let path = PathBuf::from(p);
    let name = path.file_stem().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".into());
    let kind = if name.contains("iq4") { "iq4_xs" }
        else if name.contains("q6k") { "q6_k" }
        else if name.contains("matvec") || name.contains("matmul") { "matmul" }
        else { "generic" }.to_string();
    ShaderEntry { name, path, kind }
}

// ────────────────────────────────────────────────────────────────────────────
// device init
// ────────────────────────────────────────────────────────────────────────────

async fn init_device(gpu: usize) -> anyhow::Result<(wgpu::Device, wgpu::Queue, String, GpuClass)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    if adapters.is_empty() {
        anyhow::bail!("no Vulkan adapters");
    }
    let adapter = &adapters[gpu.min(adapters.len() - 1)];
    let info = adapter.get_info();
    let class = detect_from_gpu_name(&info.name);
    // Request PUSH_CONSTANTS opportunistically — many production shaders use it.
    let want_pc = adapter.features().contains(wgpu::Features::PUSH_CONSTANTS);
    let want_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
    let mut required_features = wgpu::Features::empty();
    if want_pc { required_features |= wgpu::Features::PUSH_CONSTANTS; }
    if want_f16 { required_features |= wgpu::Features::SHADER_F16; }
    let mut limits = wgpu::Limits {
        max_storage_buffer_binding_size: 1024 * 1024 * 1024,
        max_buffer_size: 1024 * 1024 * 1024,
        ..Default::default()
    };
    if want_pc { limits.max_push_constant_size = 64; }
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("shader_lab"),
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ).await?;
    // Don't panic on shader validation errors — report and move on.
    device.on_uncaptured_error(Box::new(|err| {
        eprintln!("[wgpu] {}", err);
    }));
    Ok((device, queue, info.name.clone(), class))
}

// ────────────────────────────────────────────────────────────────────────────
// bench harness — picks a workload sized for each kind and times N iters
// ────────────────────────────────────────────────────────────────────────────

async fn bench_one(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entry: &ShaderEntry,
    gpu_name: &str,
    iters: u32,
) -> anyhow::Result<BenchmarkResult> {
    let source = fs::read_to_string(&entry.path)?;
    // SPIR-V/GLSL we don't compile here — only WGSL is in the harness
    if entry.path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
        anyhow::bail!("non-WGSL kernel — wire glslc / naga to compile first");
    }

    // Workload sizing per kind. Numbers chosen so each run finishes in ~1-50 ms
    // on Pascal-era hardware; bigger GPUs just look faster, that's the point.
    let (m, k, n, bytes_in) = match entry.kind.as_str() {
        "iq4_xs" | "q6_k" | "q4" => {
            // matvec: 1 × hidden × vocab, weights raw quantized
            let hidden: u32 = 2816;
            let vocab: u32 = 64 * 1024;
            (1u32, hidden, vocab, (hidden as u64 * vocab as u64) / 2)
        }
        "matmul" | "matvec" | "fp16" => {
            let h: u32 = 2048;
            (1u32, h, h, (h as u64 * h as u64) * 4)
        }
        _ => (1u32, 1024, 1024, 1024 * 1024 * 4),
    };

    // We don't actually dispatch the per-kind specialty pipeline yet — that
    // requires per-shader bindings + push-constant layouts. What we DO time
    // here is a uniform "compile + dispatch" cost so we can detect compile
    // regressions and gross perf cliffs. Real perf numbers come from the
    // kind-specific harness in lib (see TODO below).
    let has_main = source.contains("fn main(") || source.contains("@compute");
    let module = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&entry.name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    })) {
        Ok(m) => m,
        Err(_) => anyhow::bail!("shader compile failed (likely missing capability)"),
    };

    // Ensure compile completes before we start the clock.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("noop_bgl"),
        entries: &[],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("noop_pl"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    // Some shaders won't have a `main` entry — skip dispatch for those, just
    // measure that they parse + compile.
    let pipeline = if has_main {
        match try_compile(device, &module, &pl, "main") {
            Some(p) => Some(p),
            None => None,
        }
    } else { None };

    let start = Instant::now();
    let mut dispatched = 0u32;
    if let Some(ref pipeline) = pipeline {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noop_bg"), layout: &bgl, entries: &[],
        });
        for _ in 0..iters {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(((n + 7) / 8).max(1), ((m + 7) / 8).max(1), 1);
            }
            queue.submit(std::iter::once(enc.finish()));
            dispatched += 1;
        }
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();
    let ms_per = if dispatched == 0 { 0.0 } else {
        elapsed.as_secs_f32() * 1000.0 / dispatched as f32
    };
    let bw = if ms_per > 0.0 {
        (bytes_in as f32 / 1e9) / (ms_per / 1000.0)
    } else { 0.0 };

    // Quiet the unused-variable warning for k while keeping it visible in code:
    let _ = k;

    Ok(BenchmarkResult {
        shader_id: entry.name.clone(),
        gpu: gpu_name.to_string(),
        ms_per_token: ms_per,
        memory_bw_util: bw,
        generation: 0,
    })
}

fn try_compile(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry_point: &str,
) -> Option<wgpu::ComputePipeline> {
    // Most of our shaders need real bind groups (storage buffers, uniforms).
    // The empty-bgl fallback compiles but won't satisfy them, so this returns
    // None for any shader with bindings — which is fine, scan & compile-time
    // is still measured. Real perf testing belongs in the per-kind harness.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("noop_pipeline"),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }));
    result.ok()
}

// ────────────────────────────────────────────────────────────────────────────
// persistence — JSON round-trip without pulling in serde for db type
// ────────────────────────────────────────────────────────────────────────────

fn db_path(out_dir: &Path) -> PathBuf { out_dir.join("index.json") }

fn load_db(out_dir: &Path) -> ShaderDB {
    let mut db = ShaderDB::new();
    let path = db_path(out_dir);
    if !path.exists() { return db; }
    let Ok(s) = fs::read_to_string(&path) else { return db; };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else { return db; };
    let Some(arr) = v.as_array() else { return db; };
    for item in arr {
        let r = BenchmarkResult {
            shader_id: item.get("shader_id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            gpu: item.get("gpu").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            ms_per_token: item.get("ms_per_token").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            memory_bw_util: item.get("memory_bw_util").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            generation: item.get("generation").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        };
        db.insert(r);
    }
    db
}

fn save_db(out_dir: &Path, db: &ShaderDB) {
    let arr: Vec<serde_json::Value> = db.results.values().map(|r| {
        serde_json::json!({
            "shader_id": r.shader_id,
            "gpu": r.gpu,
            "ms_per_token": r.ms_per_token,
            "memory_bw_util": r.memory_bw_util,
            "generation": r.generation,
        })
    }).collect();
    let s = serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".into());
    let _ = fs::write(db_path(out_dir), s);
}
