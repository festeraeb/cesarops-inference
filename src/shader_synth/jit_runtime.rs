//! Vulkan/wgpu JIT shader runtime — compile, cache, swap live
//!
//! Compiles shader variants at runtime, swaps pipelines live,
//! selects based on GPU class + workload type.

use std::collections::HashMap;
use std::process::Command;

/// Pipeline cache — maps shader key to compiled pipeline handle
pub struct PipelineCache {
    pub map: HashMap<String, Vec<u8>>, // key -> SPIR-V bytes
}

impl PipelineCache {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn insert(&mut self, key: &str, spv: Vec<u8>) {
        self.map.insert(key.to_string(), spv);
    }

    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.map.get(key)
    }

    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
}

/// JIT compile GLSL to SPIR-V at runtime
pub fn compile_glsl_to_spirv(glsl_source: &str) -> Result<Vec<u8>, String> {
    let tmp_glsl = "/tmp/jit_shader.comp";
    let tmp_spv = "/tmp/jit_shader.spv";

    std::fs::write(tmp_glsl, glsl_source)
        .map_err(|e| format!("write: {}", e))?;

    let status = Command::new("glslc")
        .args([tmp_glsl, "-o", tmp_spv])
        .output()
        .map_err(|e| format!("glslc: {}", e))?;

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        return Err(format!("glslc failed: {}", stderr));
    }

    std::fs::read(tmp_spv).map_err(|e| format!("read spv: {}", e))
}

/// Select the best pipeline key for a given GPU + workload combination
pub fn select_pipeline_key(gpu_class: &str, workload: &str) -> String {
    match (gpu_class, workload) {
        ("pascal", "iq4_xs") => "matvec_iq4xs_pascal_noshared".to_string(),
        ("pascal", "q6_k") => "matvec_q6k_pascal".to_string(),
        ("pascal", "fp16") => "matmul_fp16_tiled".to_string(),
        ("turing", "iq4_xs") => "matvec_iq4xs_turing_subgroup".to_string(),
        ("turing", "fp16") => "matmul_fp16_tensorcore".to_string(),
        ("turing", "q4") => "matvec_q4_turing_shuffle".to_string(),
        _ => "fallback_generic".to_string(),
    }
}

/// Swap a pipeline in the cache (hot-reload)
pub fn swap_pipeline(cache: &mut PipelineCache, key: &str, new_spv: Vec<u8>) {
    cache.insert(key, new_spv);
}
