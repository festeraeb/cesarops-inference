//! Dump key GGUF metadata fields for Gemma-4-26B-MoE so we can compare
//! against the official Gemma-4 reference implementation.

use std::path::Path;

use cesarops_inference::hardware;
use cesarops_inference::loader::{self, GgufValue};

const MODEL: &str = "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf";

#[test]
#[ignore = "metadata-only quick lookup"]
fn dump_gemma4_metadata() {
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(MODEL), &profile).expect("load gguf");

    // Print every gemma4.* metadata key so we can map our runner against
    // the source of truth.
    let mut keys: Vec<&String> = weights.metadata.keys().collect();
    keys.sort();
    for k in keys {
        if !k.starts_with("gemma4.") && !k.contains("logit") && !k.contains("scale")
            && !k.contains("softcap") && !k.contains("embedding") {
            continue;
        }
        let v = &weights.metadata[k];
        let summary = match v {
            GgufValue::U32(x) => format!("u32: {x}"),
            GgufValue::I32(x) => format!("i32: {x}"),
            GgufValue::F32(x) => format!("f32: {x}"),
            GgufValue::U64(x) => format!("u64: {x}"),
            GgufValue::Bool(x) => format!("bool: {x}"),
            GgufValue::Str(s) if s.len() > 80 => format!("str: {:?}…", &s[..80]),
            GgufValue::Str(s) => format!("str: {s:?}"),
            GgufValue::Array(arr) => format!("array(len={})", arr.len()),
            GgufValue::Other => "other".into(),
        };
        println!("{k:<60} {summary}");
    }
}
