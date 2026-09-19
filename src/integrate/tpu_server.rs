//! TPU inference server types — port of `wreckhunter/tpu_server.py` (API surface).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TpuServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: Option<String>,
}

impl Default for TpuServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 5001,
            model_path: None,
        }
    }
}

impl TpuServerConfig {
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn infer_route(&self) -> String {
        format!("http://{}:{}/infer", self.host, self.port)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TpuHealthResponse {
    pub status: String,
    pub used_tpu: bool,
    #[serde(default)]
    pub model: Option<String>,
}

pub fn validate_infer_payload(body: &serde_json::Value) -> Result<(), &'static str> {
    if body.get("image_base64").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).is_none() {
        return Err("image_base64 required");
    }
    Ok(())
}

pub fn default_meta() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_route_default_port() {
        let c = TpuServerConfig::default();
        assert!(c.infer_route().contains(":5001/infer"));
    }
}
