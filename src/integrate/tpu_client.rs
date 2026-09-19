//! TPU remote client — port of `utils/tpu_client.py` (request/response types).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TpuInferRequest {
    pub image_base64: String,
    pub meta: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TpuInferResponse {
    pub glint_score: f32,
    pub jitter_score: f32,
    pub pass: bool,
    pub took_ms: f32,
    pub used_tpu: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TpuClientConfig {
    pub server_url: String,
}

impl Default for TpuClientConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:5001".into(),
        }
    }
}

impl TpuClientConfig {
    pub fn infer_endpoint(&self) -> String {
        format!("{}/infer", self.server_url.trim_end_matches('/'))
    }

    pub fn offline_pass_response() -> TpuInferResponse {
        TpuInferResponse {
            glint_score: 0.0,
            jitter_score: 0.0,
            pass: true,
            took_ms: 0.0,
            used_tpu: false,
            error: Some("server_unreachable".into()),
        }
    }

    pub fn build_request(image_base64: &str, meta: HashMap<String, serde_json::Value>) -> TpuInferRequest {
        TpuInferRequest {
            image_base64: image_base64.to_string(),
            meta,
        }
    }
}

pub fn parse_infer_response(json: &str) -> Result<TpuInferResponse, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_infer_url() {
        let c = TpuClientConfig {
            server_url: "http://10.0.0.55:5001/".into(),
        };
        assert_eq!(c.infer_endpoint(), "http://10.0.0.55:5001/infer");
    }

    #[test]
    fn offline_fallback_passes() {
        let r = TpuClientConfig::offline_pass_response();
        assert!(r.pass);
        assert!(!r.used_tpu);
    }
}
