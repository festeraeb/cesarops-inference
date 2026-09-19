// src/bin/run_gemma4.rs
//
// Single-GPU smoke test for Gemma-4-26B-MoE.
//
// Usage:
//   run_gemma4 --model PATH [--gpu 0] [--prompt-tokens 2,5466,3489,...]
//              [--max-new 16] [--temperature 0.0]
//
// Token IDs are passed pre-tokenized as a comma-separated list because the
// existing tokenizer is Qwen-tuned. A real gemma4 vocab loader is its own
// task.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use cesarops_inference::gemma4_runner::{Gemma4Runner, SamplingParams};

fn parse_token_list(s: &str) -> Vec<u32> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u32>().ok())
        .collect()
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    cesarops_inference::pamp::init_from_env();

    let args: Vec<String> = env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut gpu: usize = 0;
    let mut prompt_tokens = vec![2u32]; // BOS only
    let mut max_new: usize = 16;
    let mut temperature: f32 = 0.0;
    let mut top_p: f32 = 1.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--gpu" => {
                gpu = args[i + 1].parse()?;
                i += 2;
            }
            "--prompt-tokens" => {
                prompt_tokens = parse_token_list(&args[i + 1]);
                i += 2;
            }
            "--max-new" => {
                max_new = args[i + 1].parse()?;
                i += 2;
            }
            "--temperature" => {
                temperature = args[i + 1].parse()?;
                i += 2;
            }
            "--top-p" => {
                top_p = args[i + 1].parse()?;
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "run_gemma4 --model PATH [--gpu N] \
                     [--prompt-tokens 2,1234,...] [--max-new 16] \
                     [--temperature 0.0] [--top-p 1.0]"
                );
                return Ok(());
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(1);
            }
        }
    }

    let Some(model) = model else {
        eprintln!("--model is required");
        std::process::exit(1);
    };

    println!("Loading model: {:?}", model);
    let load_start = Instant::now();
    let runner = Gemma4Runner::load(&model, gpu)
        .map_err(|e| anyhow::anyhow!("Gemma4Runner::load: {e}"))?;
    println!(
        "Loaded in {:.1}s. layers={} vocab={} softcap={}",
        load_start.elapsed().as_secs_f32(),
        runner.layers.len(),
        runner.vocab_size,
        runner.logit_softcap,
    );

    println!("Prompt tokens: {:?}", prompt_tokens);
    println!(
        "Generating up to {} new tokens (T={}, top_p={})",
        max_new, temperature, top_p
    );

    let params = SamplingParams {
        max_new_tokens: max_new,
        temperature,
        top_p,
        stop_tokens: vec![1, 2], // Gemma-4 EOS=1, BOS=2
    };

    let gen_start = Instant::now();
    let out = runner
        .generate(&prompt_tokens, &params)
        .map_err(|e| anyhow::anyhow!("generate: {e}"))?;
    let gen_secs = gen_start.elapsed().as_secs_f32();

    println!(
        "Generated {} tokens in {:.2}s ({:.2} tok/s)",
        out.len(),
        gen_secs,
        out.len() as f32 / gen_secs.max(1e-6),
    );
    println!("Output token ids: {:?}", out);
    Ok(())
}
