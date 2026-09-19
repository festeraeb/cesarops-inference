//! CUDA environment configuration — port of `tools/cuda_env.py`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const CUDA_CANDIDATES: &[&str] = &[
    "/usr/local/cuda",
    "/usr/local/cuda-13.2",
    "/usr/local/cuda-13.0",
    "/usr/local/cuda-12.2",
    "/usr/local/cuda-11.8",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaConfig {
    pub cuda_path: PathBuf,
    pub cuda_bin: PathBuf,
}

/// Resolve CUDA install directory from env or known candidates.
pub fn resolve_cuda_path(env_cuda_path: Option<&str>) -> Result<PathBuf, String> {
    if let Some(p) = env_cuda_path.map(str::trim).filter(|s| !s.is_empty()) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("CUDA_PATH set but not found: {p}"));
    }
    for candidate in CUDA_CANDIDATES {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(
        "CUDA toolkit not found. Set CUDA_PATH to a valid CUDA installation directory \
         or install CUDA 13.2/13.0/12.2/11.8."
            .into(),
    )
}

/// Build config from resolved CUDA path (does not mutate process env).
pub fn cuda_config_from_path(cuda_path: PathBuf) -> CudaConfig {
    CudaConfig {
        cuda_bin: cuda_path.join("bin"),
        cuda_path,
    }
}

/// Idempotent CUDA PATH setup for subprocess spawning.
pub fn configure_cuda_environment() -> Result<CudaConfig, String> {
    let env_path = std::env::var("CUDA_PATH").ok();
    let selected = resolve_cuda_path(env_path.as_deref())?;
    let config = cuda_config_from_path(selected.clone());
    std::env::set_var("CUDA_PATH", &config.cuda_path);
    let sep = if cfg!(windows) { ";" } else { ":" };
    let current = std::env::var("PATH").unwrap_or_default();
    let bin = config.cuda_bin.to_string_lossy();
    if !current.split(sep).any(|p| p == bin.as_ref()) {
        std::env::set_var("PATH", format!("{bin}{sep}{current}"));
    }
    Ok(config)
}

pub fn path_has_cuda_bin(path: &Path) -> bool {
    path.join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" }).exists()
        || path.join("bin").exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_from_env_when_exists() {
        let tmp = std::env::temp_dir().join("cesarops_cuda_test");
        std::fs::create_dir_all(tmp.join("bin")).unwrap();
        let cfg = cuda_config_from_path(tmp.clone());
        assert_eq!(cfg.cuda_path, tmp);
    }

    #[test]
    fn rejects_missing_cuda() {
        assert!(resolve_cuda_path(Some("/nonexistent/cuda/path/cesarops-test")).is_err());
    }
}
