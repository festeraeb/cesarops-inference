use clap::Parser;
use reqwest::redirect::Policy;
use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

#[derive(Parser, Debug)]
#[command(name = "production-satellite-downloader")]
#[command(about = "Production Satellite Downloader (Rust launcher + EDL fetch helper)")]
struct Args {
    /// Run Earthdata-authenticated fetch for a single URL (EDL + redirect aware).
    #[arg(long)]
    edl_fetch_url: Option<String>,

    /// Output path for --edl-fetch-url.
    #[arg(long)]
    output: Option<String>,

    /// Any remaining args are forwarded to the python package runner.
    #[arg(trailing_var_arg = true)]
    passthrough: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    if let Some(url) = args.edl_fetch_url {
        let output = match args.output {
            Some(o) => o,
            None => {
                eprintln!("--output is required when using --edl-fetch-url");
                return ExitCode::from(2);
            }
        };
        return match edl_fetch_to_file(&url, &output).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("earthdata fetch failed: {}", e);
                ExitCode::from(1)
            }
        };
    }

    let script = "/codebase/repos/wreckhunter2000-1/scripts/production_satellite_downloader.py";
    if !Path::new(script).exists() {
        eprintln!("production_satellite_downloader.py not found at {}", script);
        return ExitCode::from(1);
    }

    let mut cmd = Command::new("python3");
    cmd.arg(script);
    for arg in &args.passthrough {
        cmd.arg(arg);
    }

    match cmd.status() {
        Ok(status) => {
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(status.code().unwrap_or(1) as u8)
            }
        }
        Err(e) => {
            eprintln!("failed to launch production downloader: {}", e);
            ExitCode::from(1)
        }
    }
}

async fn edl_fetch_to_file(url: &str, output: &str) -> Result<(), String> {
    // EDL auth requires cookie persistence + redirects through urs -> DAAC/S3.
    let client = reqwest::Client::builder()
        .cookie_store(true)
        .redirect(Policy::limited(10))
        .build()
        .map_err(|e| e.to_string())?;

    let user = env::var("NASA_EARTHDATA_USERNAME")
        .or_else(|_| env::var("EARTHDATA_USERNAME"))
        .map_err(|_| "NASA_EARTHDATA_USERNAME/EARTHDATA_USERNAME not set".to_string())?;
    let pass = env::var("NASA_EARTHDATA_PASSWORD")
        .or_else(|_| env::var("EARTHDATA_PASSWORD"))
        .map_err(|_| "NASA_EARTHDATA_PASSWORD/EARTHDATA_PASSWORD not set".to_string())?;

    // Seed EDL session and cookies.
    let login_resp = client
        .post("https://urs.earthdata.nasa.gov/login")
        .form(&[("username", user), ("password", pass)])
        .send()
        .await
        .map_err(|e| format!("EDL login request failed: {}", e))?;
    if !login_resp.status().is_success() {
        return Err(format!("EDL login returned {}", login_resp.status()));
    }

    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("fetch request failed: {}", e))?
        .error_for_status()
        .map_err(|e| format!("fetch status error: {}", e))?
        .bytes()
        .await
        .map_err(|e| format!("failed reading response body: {}", e))?;

    std::fs::write(output, &bytes)
        .map_err(|e| format!("failed writing output file {}: {}", output, e))?;
    Ok(())
}

