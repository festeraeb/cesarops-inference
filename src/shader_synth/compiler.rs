use std::process::Command;
use super::gpu_probe::GpuClass;

#[derive(Debug)]
pub struct ValidationResult {
    pub spirv_ok: bool,
    pub validation_log: String,
    pub subgroup_safe: bool,
    pub memory_safe: bool,
}

pub fn compile_glsl_to_spv(glsl_path: &str, spv_out: &str) -> bool {
    let status = Command::new("glslc")
        .arg(glsl_path)
        .arg("-o")
        .arg(spv_out)
        .status();
    matches!(status, Ok(s) if s.success())
}

pub fn validate_spirv(path: &str) -> (bool, String) {
    let output = Command::new("spirv-val").arg(path).output();
    match output {
        Ok(o) => {
            let log = String::from_utf8_lossy(&o.stderr).to_string();
            (o.status.success(), log)
        }
        Err(e) => (false, format!("spirv-val failed: {:?}", e)),
    }
}

pub fn validate_subgroups(shader_src: &str, gpu: &GpuClass) -> bool {
    match gpu {
        GpuClass::Pascal => {
            !shader_src.contains("subgroup") &&
            !shader_src.contains("GL_KHR_shader_subgroup")
        }
        _ => true,
    }
}

pub fn validate_memory_patterns(shader: &str) -> bool {
    let bad_patterns = [
        "uint8_t data[]",
        "buffer.*void",
    ];
    !bad_patterns.iter().any(|p| shader.contains(p))
}

pub fn validate_shader(
    spirv_path: &str,
    src: &str,
    gpu: &GpuClass,
) -> ValidationResult {
    let (spirv_ok, log) = validate_spirv(spirv_path);
    let subgroup_ok = validate_subgroups(src, gpu);
    let memory_ok = validate_memory_patterns(src);

    ValidationResult {
        spirv_ok,
        validation_log: log,
        subgroup_safe: subgroup_ok,
        memory_safe: memory_ok,
    }
}
