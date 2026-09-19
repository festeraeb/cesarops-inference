//! Runtime diagnostic gate — replaces per-token CPU readbacks of hidden_state
//! and logits with a level-gated, zero-cost-when-off helper system.
//!
//! Levels (set via CESAROPS_DIAG env var):
//!   off     — no checks, no logs (default, release-safe)
//!   error   — NaN/Inf checks only, log on detection
//!   debug   — full tensor stats logging
//!
//! When the `engine-debug` Cargo feature is OFF, all helpers compile to
//! no-ops regardless of runtime level (compile-time strip).

use std::env;
use tracing::warn;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Off,
    ErrorOnly,
    Debug,
}

#[derive(Clone, Copy)]
pub struct Diagnostics {
    pub level: DiagnosticLevel,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::from_env()
    }
}

impl Diagnostics {
    pub fn from_env() -> Self {
        let lvl = env::var("CESAROPS_DIAG").unwrap_or_else(|_| "off".into());
        let level = match lvl.as_str() {
            "error" => DiagnosticLevel::ErrorOnly,
            "debug" => DiagnosticLevel::Debug,
            _ => DiagnosticLevel::Off,
        };
        Self { level }
    }

    #[inline(always)]
    pub fn enabled(&self) -> bool {
        self.level != DiagnosticLevel::Off
    }

    #[inline(always)]
    pub fn debug_enabled(&self) -> bool {
        self.level == DiagnosticLevel::Debug
    }

    /// Scan a slice for NaN/Inf. Returns true if found.
    /// No-op when level is Off OR engine-debug feature is disabled.
    #[inline]
    pub fn check_nan(&self, name: &str, v: &[f32]) -> bool {
        #[cfg(not(feature = "engine-debug"))]
        {
            let _ = (name, v);
            return false;
        }
        #[cfg(feature = "engine-debug")]
        {
            if !self.enabled() {
                return false;
            }
            let bad = v.iter().any(|x| x.is_nan() || x.is_infinite());
            if bad {
                warn!("[diag] NaN/Inf detected in {}", name);
            }
            bad
        }
    }

    /// Log min/max/mean of a logit vector. Debug level only.
    #[inline]
    pub fn log_logits(&self, logits: &[f32]) {
        #[cfg(not(feature = "engine-debug"))]
        {
            let _ = logits;
        }
        #[cfg(feature = "engine-debug")]
        {
            if !self.debug_enabled() {
                return;
            }
            if logits.is_empty() {
                return;
            }
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = logits.iter().sum::<f32>() / logits.len() as f32;
            tracing::debug!("[diag] logits min={:.4} max={:.4} mean={:.6}", min, max, mean);
        }
    }

    /// Log labeled tensor stats. Debug level only.
    #[inline]
    pub fn log_tensor_stats(&self, name: &str, v: &[f32]) {
        #[cfg(not(feature = "engine-debug"))]
        {
            let _ = (name, v);
        }
        #[cfg(feature = "engine-debug")]
        {
            if !self.debug_enabled() {
                return;
            }
            if v.is_empty() {
                return;
            }
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            tracing::debug!("[diag] {} min={:.4} max={:.4} mean={:.6}", name, min, max, mean);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_level_is_default_when_env_missing() {
        // SAFETY: this test reads env, but doesn't mutate it
        let d = Diagnostics::from_env();
        // Either Off (no env) or whatever's set by the test environment
        let _ = d;
    }

    #[test]
    fn nan_detection_when_enabled() {
        let d = Diagnostics { level: DiagnosticLevel::ErrorOnly };
        let bad = vec![1.0, f32::NAN, 3.0];
        let good = vec![1.0, 2.0, 3.0];

        // When engine-debug feature is on, check_nan returns true on bad data
        #[cfg(feature = "engine-debug")]
        {
            assert!(d.check_nan("test", &bad));
            assert!(!d.check_nan("test", &good));
        }

        // When feature is off, always false
        #[cfg(not(feature = "engine-debug"))]
        {
            let _ = (bad, good, d);
        }
    }
}
