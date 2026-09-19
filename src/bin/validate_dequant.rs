// src/bin/validate_dequant.rs
//
// Cross-check GPU IQ4_XS / IQ4_NL matvec output against the CPU reference
// in `bridge::dequantize_tensor`. Pulls a real tensor from a GGUF file,
// generates a deterministic input vector, runs both paths, and prints
// max abs error, RMS, and the 5 largest mismatches.
//
// Usage:
//   validate_dequant --model PATH --tensor blk.0.attn_q.weight [--gpu 0] [--quant iq4_xs|iq4_nl]

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use cesarops_inference::bridge;
use cesarops_inference::loader;
use cesarops_inference::hardware;

// IQ4 codebook — must match `bridge::dequant_iq4_xs`.
const KVALUES: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
];

#[derive(Debug)]
struct Cli {
    model: PathBuf,
    tensor: String,
    gpu: usize,
    quant: String,           // "iq4_xs" or "iq4_nl" (auto-detected if empty)
    rows: u32,
    seed: u64,
    show_n: usize,
    shader_override: Option<PathBuf>,
}

fn parse_cli() -> Cli {
    let args: Vec<String> = env::args().collect();
    let mut c = Cli {
        model: PathBuf::new(),
        tensor: "blk.0.attn_q.weight".into(),
        gpu: 0,
        quant: String::new(),
        rows: 64,
        seed: 0xC0FFEE,
        show_n: 5,
        shader_override: None,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { c.model = PathBuf::from(&args[i + 1]); i += 2; }
            "--tensor" => { c.tensor = args[i + 1].clone(); i += 2; }
            "--gpu" => { c.gpu = args[i + 1].parse().unwrap_or(0); i += 2; }
            "--quant" => { c.quant = args[i + 1].clone(); i += 2; }
            "--rows" => { c.rows = args[i + 1].parse().unwrap_or(64); i += 2; }
            "--seed" => { c.seed = args[i + 1].parse().unwrap_or(0xC0FFEE); i += 2; }
            "--shader" => { c.shader_override = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(1); }
        }
    }
    if c.model.as_os_str().is_empty() {
        eprintln!("--model is required");
        std::process::exit(1);
    }
    c
}

fn print_help() {
    println!("validate_dequant — compare GPU matvec against CPU reference");
    println!();
    println!("--model PATH        GGUF model file");
    println!("--tensor NAME       tensor to validate (default: blk.0.attn_q.weight)");
    println!("--gpu N             Vulkan adapter index");
    println!("--quant TYPE        force iq4_xs or iq4_nl; default = auto");
    println!("--rows N            output rows to validate (caps weight rows tested)");
    println!("--seed S            input vector seed");
    println!("--shader PATH       override which WGSL shader to validate");
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushParams {
    k: u32,
    n_rows_total: u32,
    row_offset: u32,
    _pad: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = parse_cli();

    println!("Loading model: {:?}", cli.model);
    let profile = hardware::audit_system();
    let weights = loader::load(&cli.model, &profile)?;
    let region = weights.tensors.get(&cli.tensor)
        .ok_or_else(|| anyhow::anyhow!("tensor not found: {}", cli.tensor))?;
    let bytes = weights.tensor_bytes(&cli.tensor)
        .ok_or_else(|| anyhow::anyhow!("tensor bytes missing: {}", cli.tensor))?;

    let qtype = region.quant_type;
    let qname = match qtype {
        20 => "iq4_nl",
        23 => "iq4_xs",
        14 => "q6_k",
        12 => "q4_k",
        other => {
            eprintln!("tensor uses qtype {} which this validator doesn't drive yet", other);
            return Ok(());
        }
    };
    let quant = if cli.quant.is_empty() { qname.to_string() } else { cli.quant.clone() };
    println!("Tensor: {} shape={:?} qtype={} ({})", cli.tensor, region.shape, qtype, qname);

    // GGUF stores [ne0, ne1] = [k, n]. Each row of the matrix has K weights.
    let k = region.shape[0];
    let n_full = if region.shape.len() > 1 { region.shape[1] } else { 1 };
    let n_rows = (cli.rows as usize).min(n_full);
    println!("Validating {} rows × {} K", n_rows, k);

    let n_elements_partial = n_rows * k;
    // Compute the byte slice covering the first `n_rows` of the matrix.
    let bytes_per_row = match qtype {
        23 => (k + 255) / 256 * 136,
        20 => (k + 31) / 32 * 18,
        _ => unreachable!(),
    };
    let partial_bytes = &bytes[..bytes_per_row * n_rows];

    // CPU reference dequant of just the rows we care about, then a CPU matvec.
    let cpu_w = bridge::dequantize_tensor(partial_bytes, qtype, n_elements_partial);
    let x = make_input(k, cli.seed);

    let cpu_start = Instant::now();
    let mut cpu_y = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let mut acc = 0.0f64;
        for kk in 0..k {
            acc += (cpu_w[r * k + kk] as f64) * (x[kk] as f64);
        }
        cpu_y[r] = acc as f32;
    }
    let cpu_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;
    println!("CPU reference matvec: {:.2} ms ({} ops/row)", cpu_ms, k);

    // GPU path
    let (device, queue, gpu_name) = init_device(cli.gpu).await?;
    println!("GPU: {}", gpu_name);

    let shader_path = match cli.shader_override.as_ref() {
        Some(p) => p.clone(),
        None => match quant.as_str() {
            "iq4_xs" => PathBuf::from("cesarops-inference/shaders/matvec_iq4xs_correct.wgsl"),
            "iq4_nl" => PathBuf::from("cesarops-inference/shaders/matvec_iq4nl_correct.wgsl"),
            other => anyhow::bail!("no GPU shader for {}", other),
        },
    };
    println!("Shader: {}", shader_path.display());
    let shader_src = std::fs::read_to_string(&shader_path)?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("validate_dequant"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // Build buffers
    let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("W"),
        size: partial_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&w_buf, 0, partial_bytes);

    let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("X"),
        size: (k * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));

    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Y"),
        size: (n_rows * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // LUT (16 floats) packed into 4×vec4
    let mut lut_packed = [0f32; 16];
    lut_packed.copy_from_slice(&KVALUES);
    let lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lut"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&lut_buf, 0, bytemuck::cast_slice(&lut_packed));

    let push = PushParams { k: k as u32, n_rows_total: n_rows as u32, row_offset: 0, _pad: 0 };
    let push_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("push"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&push_buf, 0, bytemuck::bytes_of(&push));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("validate_bgl"),
        entries: &[
            bg_entry(0, true), bg_entry(1, true), bg_entry_rw(2),
            uniform_entry(3), uniform_entry(4),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("validate_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("validate_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("validate_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lut_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: push_buf.as_entire_binding() },
        ],
    });

    let gpu_start = Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_rows as u32, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));

    // Readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("validate_stage"),
        size: (n_rows * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc2.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, (n_rows * 4) as u64);
    queue.submit(std::iter::once(enc2.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        device.poll(wgpu::Maintain::Poll);
        if rx.try_recv().is_ok() { break; }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
    let mapped = slice.get_mapped_range();
    let gpu_y: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    let gpu_ms = gpu_start.elapsed().as_secs_f32() * 1000.0;
    println!("GPU matvec: {:.2} ms", gpu_ms);

    // Compare
    let mut diffs: Vec<(usize, f32, f32, f32)> = (0..n_rows).map(|r| {
        let c = cpu_y[r];
        let g = gpu_y[r];
        (r, c, g, (c - g).abs())
    }).collect();
    let max_abs = diffs.iter().map(|d| d.3).fold(0.0f32, f32::max);
    let mean_abs = diffs.iter().map(|d| d.3).sum::<f32>() / n_rows as f32;
    let rms = (diffs.iter().map(|d| d.3 * d.3).sum::<f32>() / n_rows as f32).sqrt();
    let max_cpu = cpu_y.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
    let rel = max_abs / max_cpu;

    println!();
    println!("max_abs_err  = {:.6e}", max_abs);
    println!("mean_abs_err = {:.6e}", mean_abs);
    println!("rms_err      = {:.6e}", rms);
    println!("max_rel_err  = {:.4}%", rel * 100.0);

    diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!("largest {} mismatches:", cli.show_n);
    for (r, c, g, d) in diffs.iter().take(cli.show_n) {
        println!("  row {:>5}  cpu={:>13.4}  gpu={:>13.4}  diff={:>10.4e}", r, c, g, d);
    }

    if rel < 1e-3 { println!("\n[PASS] relative error within 0.1%"); }
    else if rel < 1e-2 { println!("\n[WARN] relative error 0.1%–1%"); }
    else { println!("\n[FAIL] relative error >1% — shader and CPU reference disagree"); }
    Ok(())
}

fn make_input(k: usize, seed: u64) -> Vec<f32> {
    // Deterministic LCG, scaled to a small dynamic range so partial sums
    // don't blow up on long-K matvecs.
    let mut s = seed;
    let mut out = vec![0.0f32; k];
    for v in out.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) & 0xFFFFFF;
        *v = (u as f32 / 16_777_216.0 - 0.5) * 0.1;
    }
    out
}

async fn init_device(gpu: usize) -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, String)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    if adapters.is_empty() { anyhow::bail!("no Vulkan adapters"); }
    let adapter = &adapters[gpu.min(adapters.len() - 1)];
    let info = adapter.get_info();
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("validate_dequant"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 1024 * 1024 * 1024,
                max_buffer_size: 1024 * 1024 * 1024,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ).await?;
    Ok((Arc::new(device), Arc::new(queue), info.name))
}

fn bg_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bg_entry_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    bg_entry(binding, false)
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
